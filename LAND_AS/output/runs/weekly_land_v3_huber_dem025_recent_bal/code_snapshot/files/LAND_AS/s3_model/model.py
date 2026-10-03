"""LAND network architectures and loss functions.

Defines the multi-branch model consumed by engine.fit(): a climate CNN over
the reanalysis patch (+ lagged climate), a DEM CNN, tabular/month embeddings,
and a lagged-rainfall branch, fused into a Gamma distribution head, a
Bernoulli-Gamma hurdle head, or a scalar Huber head. Input shapes come from
s2_dataset.data.model_metadata(); hyperparameters come from s4_train
tuning/CLI.
"""
import torch
from torch import nn
from torch.nn import functional as F


MODEL_HP_KEYS = ("climate_units", "dem_units", "month_units", "hidden_units", "dropout", "dem_size",
                 "lightweight")


class LAND(nn.Module):
    output_kind = "gamma"

    def __init__(self, climate_shape, dem_channels, lag_dim=0, climate_units=120, dem_units=64,
                 month_units=32, hidden_units=256, dropout=0.3, dem_size=10, output_units=2,
                 lightweight=False):
        super().__init__()
        channels, height, width = climate_shape
        if climate_units % channels:
            raise ValueError("climate_units must be divisible by the number of climate channels")
        self.dem_size = dem_size
        # lightweight (Daily_Modeling style): single-layer branches with
        # dropout/2 applied inside each branch, and a single-hidden-layer
        # fusion head. Full model: deeper branches, dropout in the head only.
        branch_drop = dropout / 2.0 if lightweight else 0.0
        # Pool to at most 3x3 after the conv so the flatten layer stays small
        # when the reanalysis patch is larger than 3x3 (identity for 3x3).
        climate_pool = max(1, min(3, height - 2))
        climate_layers = [
            nn.Conv2d(channels, climate_units, 3, groups=channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(climate_pool),
            nn.Flatten(),
            nn.Linear(climate_units * climate_pool ** 2, climate_units),
            nn.ReLU(),
        ]
        if lightweight:
            climate_layers.append(nn.Dropout(branch_drop))
        self.climate = nn.Sequential(*climate_layers)
        # Flattened conv output keeps the spatial layout of the local/regional
        # DEMs (station sits at patch center), matching the original model.
        dem_flat = dem_units * (dem_size - 2) ** 2
        dem_layers = [
            nn.Conv2d(2 * dem_channels, dem_units, 3, groups=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(dem_flat, dem_units),
            nn.ReLU(),
        ]
        if lightweight:
            dem_layers.append(nn.Dropout(branch_drop))
        else:
            dem_layers += [nn.Linear(dem_units, dem_units), nn.ReLU()]
        self.dem = nn.Sequential(*dem_layers)
        month_layers = [nn.Linear(12, month_units), nn.ReLU()]
        if lightweight:
            month_layers.append(nn.Dropout(branch_drop))
        self.month = nn.Sequential(*month_layers)
        fusion_in = climate_units + dem_units + month_units + lag_dim
        head_layers = [
            nn.Linear(fusion_in, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if not lightweight:
            head_layers += [
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        head_layers.append(nn.Linear(hidden_units, output_units))
        self.head = nn.Sequential(*head_layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_auxiliary(self, features):
        local_dem = F.interpolate(features["local_dem"], size=(self.dem_size, self.dem_size), mode="bilinear", align_corners=False)
        regional_dem = F.interpolate(features["regional_dem"], size=(self.dem_size, self.dem_size), mode="bilinear", align_corners=False)
        return self.dem(torch.cat([local_dem, regional_dem], dim=1)), self.month(features["month"])

    def forward(self, features):
        climate = self.climate(features["climate"])
        dem, month = self.encode_auxiliary(features)
        return self.head(torch.cat([climate, dem, month, features["lag"]], dim=1))


class HuberLAND(LAND):
    output_kind = "huber"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, output_units=1)


class BernoulliGammaLAND(LAND):
    """Hurdle variant for zero-inflated targets: Bernoulli rain occurrence
    plus a Gamma amount on wet periods. Raw outputs are
    (logit_p, raw_alpha, raw_scale); see bernoulli_gamma_nll."""

    output_kind = "bern_gamma"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, output_units=3)


def build_model(hp, metadata):
    model_type = hp.get("model_type", "gamma")
    kwargs = {key: hp[key] for key in MODEL_HP_KEYS if key in hp}
    common = {
        "climate_shape": metadata["climate_shape"],
        "dem_channels": metadata["dem_channels"],
        "lag_dim": metadata["lag_dim"],
    }
    if model_type == "gamma":
        return LAND(**common, **kwargs)
    if model_type == "huber":
        return HuberLAND(**common, **kwargs)
    if model_type == "bern_gamma":
        return BernoulliGammaLAND(**common, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}. Use 'gamma', 'huber', or 'bern_gamma'.")


def gamma_mean(raw):
    return F.softplus(raw[:, 0]) * F.softplus(raw[:, 1])


def prediction_mean(model, raw):
    if model.output_kind == "gamma":
        return gamma_mean(raw)
    if model.output_kind == "bern_gamma":
        return torch.sigmoid(raw[:, 0]) * F.softplus(raw[:, 1]) * F.softplus(raw[:, 2])
    return F.softplus(raw[:, 0])


def training_loss(model, raw, target, rainfall_weight=False, huber_delta=0.5, loss_type=None,
                  dry_wet_ratio=1.0, lambda_bce=1.0):
    if model.output_kind == "gamma":
        return gamma_nll(raw, target, rainfall_weight)
    if model.output_kind == "bern_gamma":
        return bernoulli_gamma_nll(raw, target, dry_wet_ratio, lambda_bce, rainfall_weight)
    target = target.reshape(-1)
    losses = F.huber_loss(prediction_mean(model, raw), target, delta=huber_delta, reduction="none")
    if loss_type in (None, "huber"):
        return losses.mean()
    if loss_type == "huber_weighted":
        weight = torch.log1p(target.clamp(min=0)).detach()
        return (losses * weight / weight.mean().clamp(min=1e-8)).mean()
    raise ValueError(f"Unknown scalar loss_type: {loss_type!r}")


def gamma_nll(raw, target, rainfall_weight=False):
    alpha = F.softplus(raw[:, 0]) + 1e-6
    scale = F.softplus(raw[:, 1]) + 1e-6
    target = target.reshape(-1)
    # A Gamma puts no mass on exactly 0: drop dry weeks from the amount loss
    # rather than clamping them to 1e-6, which distorts the shape fit.
    wet = target > 0
    if not wet.any():
        return raw.sum() * 0
    alpha, scale, y = alpha[wet], scale[wet], target[wet].clamp(min=1e-6)
    nll = torch.lgamma(alpha) + alpha * torch.log(scale) - (alpha - 1) * torch.log(y) + y / scale
    if rainfall_weight:
        weight = torch.log1p(y).detach()
        nll = nll * (weight / weight.mean().clamp(min=1e-8))
    return nll.mean()


def bernoulli_gamma_nll(raw, target, dry_wet_ratio=1.0, lambda_bce=1.0, rainfall_weight=False):
    """Bernoulli occurrence + Gamma amount NLL for zero-inflated targets.

    Same NLL convention as Daily_Modeling's BernoulliGammaNLL: every sample
    contributes BCE(logit_p, 1{y>0}) -- with ``dry_wet_ratio`` (n_dry/n_wet
    of the training fold) as pos_weight, scaled by ``lambda_bce`` -- and wet
    samples additionally contribute the Gamma amount NLL.
    """
    target = target.reshape(-1)
    logit_p = raw[:, 0]
    alpha = F.softplus(raw[:, 1]) + 1e-6
    scale = F.softplus(raw[:, 2]) + 1e-6
    wet = target > 0
    pos_weight = torch.as_tensor(float(dry_wet_ratio), dtype=torch.float32, device=raw.device)
    bce = F.binary_cross_entropy_with_logits(
        logit_p, wet.float(), pos_weight=pos_weight, reduction="none"
    )
    amount = torch.zeros_like(bce)
    if wet.any():
        y = target[wet].clamp(min=1e-6)
        nll = (torch.lgamma(alpha[wet]) + alpha[wet] * torch.log(scale[wet])
               - (alpha[wet] - 1) * torch.log(y) + y / scale[wet])
        if rainfall_weight:
            weight = torch.log1p(y).detach()
            nll = nll * (weight / weight.mean().clamp(min=1e-8))
        amount[wet] = nll
    return (lambda_bce * bce + amount).mean()
