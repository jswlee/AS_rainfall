"""
LAND - Location-Agnostic Neural Downscaler (PyTorch).

Adapted from Hatanaka et al. (2025) for daily rainfall in American Samoa.
Architecture (see paper Figure 2):
  - Climate branch:   Conv2d (grouped) -> BN -> ReLU  (x2 dense stages)
  - DEM branch:       local + regional DEMs resized to dem_patch_size, stacked as a
                      2-channel image -> Conv2d -> BN -> ReLU (x2 dense stages)
                      This matches the paper which uses a single 2-channel 10x10 DEM image.
  - Month branch:     Dense -> BN -> ReLU (x2)
  - Concatenate -> Dense(na) -> BN -> ReLU -> Dense(nb) -> BN -> ReLU -> Dropout -> Output

Output head is configurable:
  - 'mse':            single scalar with softplus (non-negative prediction, trained with MSE)
  - 'softplus':       single scalar with softplus (non-negative, for Tweedie loss)
  - 'gamma':          two outputs (raw_alpha, raw_beta) for Gamma NLL loss
  - 'bernoulli_gamma': three outputs (logit_p, raw_alpha, raw_beta) for BG NLL loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# Canonical DEM patch size used in the stacked Conv2d branch.
# Both local and regional DEMs are bilinearly resized to this before stacking.
_DEM_PATCH_SIZE = 10  # matches paper (10x10)


class LANDModel(nn.Module):
    def __init__(
        self,
        climate_shape: Tuple[int, int, int] = (15, 3, 3),
        local_dem_shape: Tuple[int, int] = (3, 3),
        regional_dem_shape: Tuple[int, int] = (3, 3),
        num_month_features: int = 12,
        climate_units: int = 1020,
        dem_units: int = 64,
        temporal_units: int = 16,
        na: int = 512,
        nb: int = 128,
        dropout_rate: float = 0.3,
        climate_processing: str = "conv2d",
        output_head: str = "mse",
        # Legacy HP names kept for backward compat (ignored, dem_units used instead)
        local_dem_units: int = 64,
        regional_dem_units: int = 32,
    ):
        super().__init__()
        self.climate_processing = climate_processing
        self.output_head = output_head
        self.local_dem_shape = local_dem_shape
        self.regional_dem_shape = regional_dem_shape

        # --- Climate branch ---
        in_ch = climate_shape[0]
        if climate_processing == "conv2d":
            if climate_units % in_ch != 0:
                raise ValueError(
                    f"climate_units ({climate_units}) must be divisible by "
                    f"num_climate_vars ({in_ch}) for grouped conv."
                )
            self.climate_conv = nn.Conv2d(
                in_ch, climate_units, kernel_size=3, padding=0, groups=in_ch,
            )
            cu = climate_units
        else:
            flat = in_ch * climate_shape[1] * climate_shape[2]
            self.climate_fc = nn.Linear(flat, climate_units)
            cu = climate_units
        self.clim_bn1 = nn.BatchNorm1d(cu)
        self.clim_fc2 = nn.Linear(cu, cu)
        self.clim_bn2 = nn.BatchNorm1d(cu)
        self._cu = cu

        # --- Stacked DEM branch (paper: 2-channel 10x10 image -> Conv2d) ---
        # Local and regional DEMs are resized to _DEM_PATCH_SIZE x _DEM_PATCH_SIZE
        # and stacked along the channel dim before this conv.
        p = _DEM_PATCH_SIZE
        self.dem_conv = nn.Conv2d(2, dem_units, kernel_size=3, padding=1)
        dem_flat = dem_units * p * p
        self.dem_bn1 = nn.BatchNorm1d(dem_flat)
        self.dem_fc2 = nn.Linear(dem_flat, dem_units)
        self.dem_bn2 = nn.BatchNorm1d(dem_units)
        self._dem_units = dem_units

        # --- Month branch ---
        self.mo_fc1 = nn.Linear(num_month_features, temporal_units)
        self.mo_bn1 = nn.BatchNorm1d(temporal_units)
        self.mo_fc2 = nn.Linear(temporal_units, temporal_units)
        self.mo_bn2 = nn.BatchNorm1d(temporal_units)

        # --- Dense head ---
        combined = cu + dem_units + temporal_units
        self.fc_a = nn.Linear(combined, na)
        self.bn_a = nn.BatchNorm1d(na)
        self.fc_b = nn.Linear(na, nb)
        self.bn_b = nn.BatchNorm1d(nb)
        self.dropout = nn.Dropout(dropout_rate)

        # --- Output ---
        if output_head in ("mse", "softplus"):
            n_out = 1
        elif output_head == "gamma":
            n_out = 2
        elif output_head == "bernoulli_gamma":
            n_out = 3
        else:
            raise ValueError(f"Unknown output_head: {output_head!r}")
        self.out = nn.Linear(nb, n_out)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = features["climate"].size(0)
        p = _DEM_PATCH_SIZE

        # Climate
        c = features["climate"]
        if self.climate_processing == "conv2d":
            c = self.climate_conv(c).view(B, -1)
        else:
            c = self.climate_fc(c.view(B, -1))
        c = F.relu(self.clim_bn1(c))
        c = F.relu(self.clim_bn2(self.clim_fc2(c)))

        # Stacked DEM: resize each to p×p, stack as 2-channel image, Conv2d
        ld = features["local_dem"]
        rd = features["regional_dem"]
        if ld.dim() == 2:
            h = int(ld.shape[1] ** 0.5)
            ld = ld.view(B, 1, h, h)
        elif ld.dim() == 3:
            ld = ld.unsqueeze(1)
        if rd.dim() == 2:
            h = int(rd.shape[1] ** 0.5)
            rd = rd.view(B, 1, h, h)
        elif rd.dim() == 3:
            rd = rd.unsqueeze(1)
        ld = F.interpolate(ld, size=(p, p), mode="bilinear", align_corners=False)
        rd = F.interpolate(rd, size=(p, p), mode="bilinear", align_corners=False)
        dem = torch.cat([ld, rd], dim=1)  # (B, 2, p, p)
        dem = self.dem_conv(dem).view(B, -1)   # (B, dem_units * p * p)
        dem = F.relu(self.dem_bn1(dem))
        dem = F.relu(self.dem_bn2(self.dem_fc2(dem)))

        # Month
        mo = features["temporal"]
        mo = F.relu(self.mo_bn1(self.mo_fc1(mo)))
        mo = F.relu(self.mo_bn2(self.mo_fc2(mo)))

        # Fusion
        x = torch.cat([c, dem, mo], dim=1)
        x = F.relu(self.bn_a(self.fc_a(x)))
        x = F.relu(self.bn_b(self.fc_b(x)))
        x = self.dropout(x)
        x = self.out(x)

        if self.output_head in ("mse", "softplus"):
            x = F.softplus(x)
        # gamma / bernoulli_gamma: return raw logits (loss handles softplus internally)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_land_model(hyperparams: dict, metadata: dict) -> LANDModel:
    """Factory: build a LAND model from hyperparams + data metadata."""
    climate_shape = tuple(metadata.get("climate_shape", (15, 3, 3)))
    local_dem_shape = tuple(metadata.get("local_dem_shape", (3, 3)))
    regional_dem_shape = tuple(metadata.get("regional_dem_shape", (3, 3)))
    num_month = int(metadata.get("num_month_features", 12))

    # dem_units: use explicit key if present, else fall back to average of legacy keys
    dem_units = hyperparams.get(
        "dem_units",
        (hyperparams.get("local_dem_units", 64) + hyperparams.get("regional_dem_units", 32)) // 2,
    )

    return LANDModel(
        climate_shape=climate_shape,
        local_dem_shape=local_dem_shape,
        regional_dem_shape=regional_dem_shape,
        num_month_features=num_month,
        climate_units=hyperparams.get("climate_units", 1020),
        dem_units=dem_units,
        temporal_units=hyperparams.get("temporal_units", 16),
        na=hyperparams.get("na", 512),
        nb=hyperparams.get("nb", 128),
        dropout_rate=hyperparams.get("dropout_rate", 0.3),
        climate_processing=hyperparams.get("climate_processing", "conv2d"),
        output_head=hyperparams.get("output_head", "mse"),
    )
