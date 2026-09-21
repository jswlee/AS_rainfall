import torch
from torch import nn
from torch.nn import functional as F


class LAND(nn.Module):
    def __init__(self, climate_shape, dem_channels, climate_units=120, dem_units=64,
                 month_units=32, hidden_units=256, dropout=0.3):
        super().__init__()
        channels, height, width = climate_shape
        if climate_units % channels:
            raise ValueError("climate_units must be divisible by the number of climate channels")
        self.climate = nn.Sequential(
            nn.Conv2d(channels, climate_units, 3, groups=channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(climate_units * (height - 2) * (width - 2), climate_units),
            nn.ReLU(),
        )
        self.dem = nn.Sequential(
            nn.Conv2d(2 * dem_channels, dem_units, 3, groups=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dem_units, dem_units),
            nn.ReLU(),
        )
        self.month = nn.Sequential(nn.Linear(12, month_units), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(climate_units + dem_units + month_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 2),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features):
        climate = self.climate(features["climate"])
        local_dem = F.interpolate(features["local_dem"], size=(10, 10), mode="bilinear", align_corners=False)
        regional_dem = F.interpolate(features["regional_dem"], size=(10, 10), mode="bilinear", align_corners=False)
        dem = self.dem(torch.cat([local_dem, regional_dem], dim=1))
        month = self.month(features["month"])
        return self.head(torch.cat([climate, dem, month], dim=1))


def gamma_mean(raw):
    return F.softplus(raw[:, 0]) * F.softplus(raw[:, 1])


def gamma_nll(raw, target):
    alpha = F.softplus(raw[:, 0]) + 1e-6
    scale = F.softplus(raw[:, 1]) + 1e-6
    target = target.reshape(-1).clamp(min=1e-6)
    return (torch.lgamma(alpha) + alpha * torch.log(scale) - (alpha - 1) * torch.log(target) + target / scale).mean()
