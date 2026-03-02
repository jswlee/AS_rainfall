"""
Loss functions for daily rainfall models.

- MSE (standard)
- Gamma negative log-likelihood
- Bernoulli-Gamma negative log-likelihood (for daily data with zeros)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GammaNLL(nn.Module):
    """Negative log-likelihood of a Gamma distribution.

    The model outputs two raw scalars per sample (raw_alpha, raw_beta).
    Softplus is applied here (not in the model) for AMP stability.

    Parameterisation: alpha = shape, beta = scale, mean = alpha * beta.
    NLL = lgamma(alpha) - alpha*log(alpha/beta) ... using the standard form:
        NLL = lgamma(alpha) - (alpha-1)*log(y) + y/beta + alpha*log(beta)

    Only applied to samples where y > 0.  Loss computed in float32.
    """
    EPS = 1e-6

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
        return nll.mean()


class BernoulliGammaNLL(nn.Module):
    """Combined Bernoulli (rain/no-rain) + Gamma (amount | rain > 0) NLL.

    Model outputs 3 raw values per sample: (logit_p, raw_alpha, raw_beta).
    Softplus is applied here (not in the model forward) so the loss is
    numerically stable under AMP (float16) -- softplus is linear for large
    inputs while exp overflows float16 at ~11.1.

    Loss is computed entirely in float32 regardless of AMP dtype.
    """
    EPS = 1e-6

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Force float32 -- lgamma / log are unstable in float16
        outputs = outputs.float()
        y = targets.float().view(-1)

        logit_p  = outputs[:, 0]
        alpha    = F.softplus(outputs[:, 1]) + self.EPS  # shape > 0
        beta     = F.softplus(outputs[:, 2]) + self.EPS  # scale > 0

        is_rain = (y > 0).float()

        # Bernoulli component (stable: takes raw logit)
        loss_prob = F.binary_cross_entropy_with_logits(
            logit_p, is_rain, reduction="none"
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

        # Only apply Gamma loss on wet days
        total = loss_prob + is_rain * loss_gamma

        # Failsafe: zero out any inf/nan that might slip through
        total = torch.where(torch.isfinite(total), total, torch.zeros_like(total))
        return total.mean()
