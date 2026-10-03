"""Stage 5: pooled and site-specific baselines for the LAND model.

Baseline definitions only; they are fit and scored by evaluate.py in this
package using the same s2_dataset splits as the LAND ensembles.

Test stations have no pre-2017 records, so test-set baselines are pooled
(trained across stations, like LAND) or trivially computable (persistence).
Site-specific baselines are evaluated per LOSO fold on train stations only.

Features mirror the model inputs: current-period climate channels (patch
mean and station's center cell), DEM channel means, month one-hot, and the
lag vector (antecedent rainfall + availability masks).
"""

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_features(bundle, indices):
    """Flat feature matrix (mm-independent) for pooled baselines."""
    indices = np.asarray(indices, dtype=int)
    climate = bundle.arrays["climate"][indices]
    block = int(bundle.metadata["climate_block"])
    current = climate[:, :block]  # current-period channels only
    ch, cw = current.shape[2] // 2, current.shape[3] // 2
    dem = bundle.arrays["station_dem_idx"][indices]
    features = [
        current.mean(dim=(2, 3)),                        # patch-mean channels
        current[:, :, ch, cw],                           # center-cell channels
        bundle.arrays["local_dem"][dem].mean(dim=(2, 3)),
        bundle.arrays["regional_dem"][dem].mean(dim=(2, 3)),
        bundle.arrays["month"][indices],
        bundle.arrays["lag"][indices],
    ]
    return torch.cat(features, dim=1).numpy()


def pooled_mean(bundle, train_idx, test_idx):
    mean = bundle.arrays["target"][train_idx].numpy().mean()
    return np.full(len(test_idx), mean)


def month_climatology(bundle, train_idx, test_idx):
    """Pooled month-of-year mean across train stations (no spatial info)."""
    y_train = bundle.arrays["target"][train_idx].numpy()
    months = bundle.arrays["month"][train_idx].argmax(dim=1).numpy()
    mean_by_month = np.array([
        y_train[months == m].mean() if (months == m).any() else y_train.mean()
        for m in range(12)
    ])
    test_months = bundle.arrays["month"][test_idx].argmax(dim=1).numpy()
    return mean_by_month[test_months]


def persistence(bundle, train_idx, test_idx):
    """Previous period's observed rainfall; falls back to train mean when the
    lag period is missing (record start/gap)."""
    lag = bundle.arrays["lag"][test_idx]
    lag_max = int(bundle.metadata["lag_max"])
    scale = float(bundle.stats["target_scale"])
    rain_prev = lag[:, 0].numpy() * scale
    has_prev = lag[:, lag_max].numpy() > 0.5
    fallback = bundle.arrays["target"][train_idx].numpy().mean()
    return np.where(has_prev, rain_prev, fallback)


def ols(bundle, train_idx, test_idx):
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(build_features(bundle, train_idx), bundle.arrays["target"][train_idx].numpy())
    return np.clip(model.predict(build_features(bundle, test_idx)), 0, None)


def tweedie_glm(bundle, train_idx, test_idx):
    """Pooled analog of the original LAND site's per-station gamma GLM."""
    y = np.clip(bundle.arrays["target"][train_idx].numpy(), 1e-2, None)
    model = make_pipeline(StandardScaler(), TweedieRegressor(power=2, link="log", max_iter=1000))
    model.fit(build_features(bundle, train_idx), y)
    return np.clip(model.predict(build_features(bundle, test_idx)), 0, None)


def gbm(bundle, train_idx, test_idx):
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
    model.fit(build_features(bundle, train_idx), bundle.arrays["target"][train_idx].numpy())
    return np.clip(model.predict(build_features(bundle, test_idx)), 0, None)


def station_climatology(bundle, val_idx):
    """In-sample month-of-year mean of the held-out station's own record.

    Optimistic (fit on the same weeks it predicts) -- the standard "own
    climatology" reference for spatial holdout comparison.
    """
    y = bundle.arrays["target"][val_idx].numpy()
    months = bundle.arrays["month"][val_idx].argmax(dim=1).numpy()
    pred = np.array([
        y[months == m].mean() if (months == m).any() else y.mean()
        for m in range(12)
    ])
    return pred[months]


TEST_BASELINES = {
    "pooled_mean": pooled_mean,
    "month_climatology": month_climatology,
    "persistence": persistence,
    "ols": ols,
    "tweedie_glm": tweedie_glm,
    "gbm": gbm,
}
