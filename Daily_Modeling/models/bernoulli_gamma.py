"""
Bernoulli-Gamma GLM for site-specific daily rainfall downscaling.

For daily data, rainfall is modelled as a two-part process:
  1. Bernoulli (logistic regression): P(rain > 0 | X)
  2. Gamma GLM: E[rain | rain > 0, X]  with log link

This is the standard approach for daily precipitation
(Bano-Medina et al. 2021; Cannon 2008; Vaughan et al. 2022).

One model is fit per station using statsmodels.
"""

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Gamma as GammaFamily
    from statsmodels.genmod.families.links import Log as LogLink
    HAS_SM = True
except ImportError:
    HAS_SM = False


class BernoulliGammaGLM:
    """Two-part GLM: logistic (occurrence) + Gamma (amount).

    Uses L1-regularised (elastic-net) fits to prevent perfect separation
    and coefficient blow-up that afflict high-dimensional GLMs with
    limited samples.
    """

    def __init__(self, alpha: float = 1.0, L1_wt: float = 0.5):
        """Args:
            alpha: regularisation strength (higher = more regularisation).
            L1_wt: elastic-net mixing (1.0 = pure L1/lasso, 0.0 = pure L2/ridge).
        """
        if not HAS_SM:
            raise ImportError("statsmodels is required for BernoulliGammaGLM")
        self.alpha = alpha
        self.L1_wt = L1_wt
        self.logistic_model = None
        self.gamma_model = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliGammaGLM":
        """Fit both components.

        Args:
            X: (N, D) predictor matrix (already standardised).
            y: (N,)   daily rainfall in mm (>= 0).
        """
        import warnings
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        Xc = sm.add_constant(X)

        # Part 1: occurrence (y > 0) — regularised to prevent perfect separation
        binary = (y > 0).astype(float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*[Pp]erfect.*")
            try:
                self.logistic_model = sm.GLM(
                    binary, Xc, family=sm.families.Binomial()
                ).fit_regularized(
                    alpha=self.alpha, L1_wt=self.L1_wt,
                    maxiter=200, cnvrg_tol=1e-6,
                )
            except Exception:
                # Fallback: plain fit with fewer iterations
                self.logistic_model = sm.GLM(
                    binary, Xc, family=sm.families.Binomial()
                ).fit(disp=False, maxiter=50)

        # Part 2: amount | rain > 0 — regularised Gamma
        rain_mask = y > 0
        if rain_mask.sum() < 10:
            self.gamma_model = None
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    self.gamma_model = sm.GLM(
                        y[rain_mask], Xc[rain_mask],
                        family=GammaFamily(link=LogLink()),
                    ).fit_regularized(
                        alpha=self.alpha, L1_wt=self.L1_wt,
                        maxiter=200, cnvrg_tol=1e-6,
                    )
                except Exception:
                    self.gamma_model = sm.GLM(
                        y[rain_mask], Xc[rain_mask],
                        family=GammaFamily(link=LogLink()),
                    ).fit(disp=False, maxiter=50)

        self._fitted = True
        return self

    def predict(
        self, X: np.ndarray, return_components: bool = False,
    ) -> np.ndarray:
        """Predict expected daily rainfall E[y] = P(rain>0) * E[y|rain>0].

        If return_components is True, returns (prob, amount, combined).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        Xc = sm.add_constant(np.asarray(X, dtype=np.float64))

        prob = self.logistic_model.predict(Xc)
        if self.gamma_model is not None:
            amount = self.gamma_model.predict(Xc)
        else:
            amount = np.zeros_like(prob)

        combined = prob * amount
        # Clip to reasonable range (prevent Gamma explosion via log link)
        max_rain = 1000.0  # mm/day - physical upper bound
        combined = np.clip(combined, 0, max_rain)
        amount = np.clip(amount, 0, max_rain)
        if return_components:
            return prob, amount, combined
        return combined

    def predict_sample(self, X: np.ndarray, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Draw one stochastic sample per row (for distributional evaluation)."""
        if rng is None:
            rng = np.random.RandomState()
        prob, amount, _ = self.predict(X, return_components=True)
        occurs = rng.binomial(1, np.clip(prob, 0, 1))
        # For Gamma samples: shape = 1/dispersion, scale = mu * dispersion
        if self.gamma_model is not None:
            disp = getattr(self.gamma_model, "scale", 1.0)
            if disp is None:
                disp = 1.0
            shape = 1.0 / max(disp, 1e-6)
            scale = amount * max(disp, 1e-6)
            gamma_draw = rng.gamma(shape, scale)
        else:
            gamma_draw = np.zeros_like(amount)
        return occurs * gamma_draw


def flatten_features_numpy(
    climate: np.ndarray,
    local_dem: np.ndarray,
    regional_dem: np.ndarray,
    month_onehot: np.ndarray,
    center_pixel_only: bool = True,
) -> np.ndarray:
    """Flatten feature arrays into (N, D) for GLM input.

    When *center_pixel_only* is True (default), only the centre pixel of
    the 3×3 climate, local-DEM, and regional-DEM grids is used.  This
    reduces dimensionality from ~165 to ~29, dramatically reducing
    overfitting for stations with limited samples.
    """
    N = climate.shape[0]
    if center_pixel_only:
        # climate: (N, C, H, W) -> centre pixel -> (N, C)
        if climate.ndim == 4:
            ch, cw = climate.shape[2] // 2, climate.shape[3] // 2
            clim = climate[:, :, ch, cw]  # (N, C)
        else:
            clim = climate.reshape(N, -1)
        # DEM: (N, H, W) -> centre pixel -> (N, 1)
        if local_dem.ndim == 3:
            lh, lw = local_dem.shape[1] // 2, local_dem.shape[2] // 2
            ld = local_dem[:, lh, lw].reshape(N, 1)
        else:
            ld = local_dem.reshape(N, -1)
        if regional_dem.ndim == 3:
            rh, rw = regional_dem.shape[1] // 2, regional_dem.shape[2] // 2
            rd = regional_dem[:, rh, rw].reshape(N, 1)
        else:
            rd = regional_dem.reshape(N, -1)
        parts = [clim, ld, rd, month_onehot.reshape(N, -1)]
    else:
        parts = [
            climate.reshape(N, -1),
            local_dem.reshape(N, -1),
            regional_dem.reshape(N, -1),
            month_onehot.reshape(N, -1),
        ]
    out = np.hstack(parts).astype(np.float64)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out
