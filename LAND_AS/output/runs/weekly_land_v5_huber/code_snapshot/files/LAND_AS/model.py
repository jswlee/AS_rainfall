import torch
from torch import nn
from torch.nn import functional as F


MODEL_HP_KEYS = ("climate_units", "dem_units", "month_units", "hidden_units", "dropout", "dem_size")


class LAND(nn.Module):
    output_kind = "gamma"

    def __init__(self, climate_shape, dem_channels, lag_dim=0, climate_units=120, dem_units=64,
                 month_units=32, hidden_units=256, dropout=0.3, dem_size=10, output_units=2):
        super().__init__()
        channels, height, width = climate_shape
        if climate_units % channels:
            raise ValueError("climate_units must be divisible by the number of climate channels")
        self.dem_size = dem_size
        # Pool to at most 3x3 after the conv so the flatten layer stays small
        # when the reanalysis patch is larger than 3x3 (identity for 3x3).
        climate_pool = max(1, min(3, height - 2))
        self.climate = nn.Sequential(
            nn.Conv2d(channels, climate_units, 3, groups=channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(climate_pool),
            nn.Flatten(),
            nn.Linear(climate_units * climate_pool ** 2, climate_units),
            nn.ReLU(),
        )
        # Flattened conv output keeps the spatial layout of the local/regional
        # DEMs (station sits at patch center), matching the original model.
        dem_flat = dem_units * (dem_size - 2) ** 2
        self.dem = nn.Sequential(
            nn.Conv2d(2 * dem_channels, dem_units, 3, groups=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(dem_flat, dem_units),
            nn.ReLU(),
            nn.Linear(dem_units, dem_units),
            nn.ReLU(),
        )
        self.month = nn.Sequential(nn.Linear(12, month_units), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(climate_units + dem_units + month_units + lag_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, output_units),
        )
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
    raise ValueError(f"Unknown model_type: {model_type!r}. Use 'gamma' or 'huber'.")


def gamma_mean(raw):
    return F.softplus(raw[:, 0]) * F.softplus(raw[:, 1])


def prediction_mean(model, raw):
    if model.output_kind == "gamma":
        return gamma_mean(raw)
    return F.softplus(raw[:, 0])


def training_loss(model, raw, target, rainfall_weight=False, huber_delta=0.5):
    if model.output_kind == "gamma":
        return gamma_nll(raw, target, rainfall_weight)
    return F.huber_loss(prediction_mean(model, raw), target.reshape(-1), delta=huber_delta)


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
