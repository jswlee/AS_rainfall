"""
Loss functions for daily rainfall models.

- LogMSELoss          MSE in log-space
- TweedieLoss         Tweedie deviance (compound Poisson-Gamma)
- GammaNLL            Gamma negative log-likelihood
- BernoulliGammaNLL   Bernoulli + Gamma NLL (daily data with zeros)

The ``get_criterion()`` factory maps CLI loss-type strings to classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogMSELoss(nn.Module):
    """MSE in log-space: MSE(log(1 + pred), log(1 + true)).

    Down-weights extreme events relative to raw MSE, giving the model a
    better signal for the bulk of the (zero-inflated) distribution.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(
            torch.log1p(pred.clamp(min=0)),
            torch.log1p(target.clamp(min=0)),
        )


class TweedieLoss(nn.Module):
    """Tweedie deviance loss with power *p* in (1, 2).

    The Tweedie distribution is a natural fit for zero-inflated continuous
    data like daily rainfall.  p=1.5 is a common default.

    D(y, mu) = 2 * [ y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) ]
    """
    def __init__(self, p: float = 1.5, mu_max: float | None = None, loss_cap: float | None = None):
        super().__init__()
        assert 1.0 < p < 2.0, "Tweedie power p must be in (1, 2)"
        self.p = p
        self.mu_max = mu_max
        self.loss_cap = loss_cap

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mu_max is None:
            mu = pred.clamp(min=1e-6)
        else:
            mu = pred.clamp(min=1e-6, max=float(self.mu_max))
        y = target.clamp(min=0)
        p = self.p
        dev = 2 * (
            torch.pow(y, 2 - p) / ((1 - p) * (2 - p))
            - y * torch.pow(mu, 1 - p) / (1 - p)
            + torch.pow(mu, 2 - p) / (2 - p)
        )
        if self.loss_cap is not None:
            dev = dev.clamp(max=float(self.loss_cap))
            fill = float(self.loss_cap)
        else:
            fill = 0.0
        dev = torch.where(torch.isfinite(dev), dev, torch.full_like(dev, fill))
        return dev.mean()


class GammaNLL(nn.Module):
    """Negative log-likelihood of a Gamma distribution.

    The model outputs two raw scalars per sample (raw_alpha, raw_beta).
    Softplus is applied here (not in the model) for AMP stability.

    Parameterisation: alpha = shape, beta = scale, mean = alpha * beta.
    NLL = lgamma(alpha) - alpha*log(alpha/beta) ... using the standard form:
        NLL = lgamma(alpha) - (alpha-1)*log(y) + y/beta + alpha*log(beta)

    Only applied to samples where y > 0.  Loss computed in float32.

    Args:
        rainfall_weight: if True, up-weight each wet sample by log1p(y) so that
            heavy-rain events receive stronger gradient signal (Fix I).
    """
    EPS = 1e-6

    def __init__(self, rainfall_weight: bool = False):
        super().__init__()
        self.rainfall_weight = rainfall_weight

    def forward(self, raw_params: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raw_params = raw_params.float()
        y = targets.float().view(-1)

        alpha = F.softplus(raw_params[:, 0]) + self.EPS
        beta  = F.softplus(raw_params[:, 1]) + self.EPS

        mask = y > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)

        a, b, y_pos = alpha[mask], beta[mask], y[mask].clamp(min=self.EPS)
        # Gamma NLL with scale parameterisation (mean = alpha * beta)
        # -log p(y) = lgamma(a) + a*log(b) - (a-1)*log(y) + y/b
        nll = torch.lgamma(a) + a * torch.log(b) - (a - 1) * torch.log(y_pos) + y_pos / b
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if self.rainfall_weight:
            # Fix I: log1p-weight heavy rain samples
            w = torch.log1p(y_pos).detach()
            w = w / w.mean().clamp(min=1e-8)
            nll = nll * w
        return nll.mean()


class BernoulliGammaNLL(nn.Module):
    """Combined Bernoulli (rain/no-rain) + Gamma (amount | rain > 0) NLL.

    Model outputs 3 raw values per sample: (logit_p, raw_alpha, raw_beta).
    Softplus is applied here (not in the model forward) so the loss is
    numerically stable under AMP (float16) -- softplus is linear for large
    inputs while exp overflows float16 at ~11.1.

    Loss is computed entirely in float32 regardless of AMP dtype.

    Args:
        dry_wet_ratio: ratio of dry days to wet days in the training set.
            Used as pos_weight in BCE to penalize false alarms (Fix A).
            Default 1.0 (balanced). Pass n_dry/n_wet from training split.
        lambda_bce: weight on the Bernoulli occurrence loss relative to
            the Gamma amount loss (Fix C). Default 1.0 (equal weighting).
            Increase (>1) to penalize occurrence errors more heavily.
        rainfall_weight: if True, up-weight wet-day Gamma loss by
            log1p(y) so heavy-rain events get more gradient signal (Fix I).

    Numerical stability notes:
    - alpha and beta are clamped to [EPS, MAX_PARAM] to prevent lgamma explosion
    - lgamma(alpha) explodes as alpha -> 0 and grows slowly for large alpha
    - y/beta explodes if beta -> 0
    - Logits are clamped to prevent BCE overflow
    """
    EPS = 1e-4  # Increased from 1e-6 for better stability
    MAX_PARAM = 100.0  # Prevent extreme alpha/beta values
    MAX_LOGIT = 10.0  # Clamp logits to prevent BCE overflow

    def __init__(
        self,
        dry_wet_ratio: float = 1.0,
        lambda_bce: float = 1.0,
        rainfall_weight: bool = False,
    ):
        super().__init__()
        self.dry_wet_ratio = float(dry_wet_ratio)
        self.lambda_bce = float(lambda_bce)
        self.rainfall_weight = rainfall_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Force float32 -- lgamma / log are unstable in float16
        outputs = outputs.float()
        y = targets.float().view(-1)

        # Clamp logits to prevent BCE overflow
        logit_p  = outputs[:, 0].clamp(-self.MAX_LOGIT, self.MAX_LOGIT)
        # Clamp alpha/beta to reasonable range to prevent lgamma/division explosion
        alpha    = F.softplus(outputs[:, 1]).clamp(self.EPS, self.MAX_PARAM)
        beta     = F.softplus(outputs[:, 2]).clamp(self.EPS, self.MAX_PARAM)

        is_rain = (y > 0).float()

        # Fix A: pos_weight penalises false alarms proportional to class imbalance
        pw = torch.tensor(self.dry_wet_ratio, device=logit_p.device, dtype=torch.float32)
        # Fix C: lambda_bce scales the occurrence loss relative to the amount loss
        loss_prob = self.lambda_bce * F.binary_cross_entropy_with_logits(
            logit_p, is_rain, pos_weight=pw, reduction="none"
        )

        # Gamma NLL component (Gamma PDF with scale parameterisation)
        # NLL = lgamma(alpha) + alpha*log(beta) - (alpha-1)*log(y) + y/beta
        target_safe = y.clamp(min=self.EPS)
        loss_gamma = (
            torch.lgamma(alpha)
            + alpha * torch.log(beta)
            - (alpha - 1) * torch.log(target_safe)
            + target_safe / beta
        )

        # Fix I: log1p-weight heavy rain samples in the Gamma term
        if self.rainfall_weight:
            w = torch.log1p(target_safe).detach()
            w = w / w.mean().clamp(min=1e-8)
            loss_gamma = loss_gamma * w

        # Only apply Gamma loss on wet days
        total = loss_prob + is_rain * loss_gamma

        # Failsafe: clamp extreme values instead of zeroing (preserves gradients)
        total = total.clamp(max=50.0)  # Cap individual sample loss
        total = torch.where(torch.isfinite(total), total, torch.full_like(total, 10.0))
        return total.mean()


def get_criterion(loss_type: str = "mse", **kwargs) -> nn.Module:
    """Factory for loss functions.

    Args:
        loss_type: one of 'mse', 'log_mse', 'tweedie', 'gamma', 'bernoulli_gamma'.
        **kwargs: passed to the loss constructor.
            For 'bernoulli_gamma': dry_wet_ratio, lambda_bce, rainfall_weight.
            For 'gamma': rainfall_weight.
            For 'tweedie': p, mu_max, loss_cap.
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "log_mse":
        return LogMSELoss()
    elif loss_type == "tweedie":
        return TweedieLoss(**kwargs)
    elif loss_type == "gamma":
        return GammaNLL(**kwargs)
    elif loss_type == "bernoulli_gamma":
        return BernoulliGammaNLL(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
