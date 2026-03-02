"""
LAND - Location-Agnostic Neural Downscaler (PyTorch).

Adapted from Hatanaka et al. (2025) for daily rainfall in American Samoa.
Architecture (see paper Figure 2):
  - Climate branch:  Conv2d (grouped) -> BN -> ReLU  (x2 dense stages)
  - Local DEM branch: Flatten -> Dense -> BN -> ReLU  (x2)
  - Regional DEM branch: same pattern
  - Month branch: Dense -> BN -> ReLU (x2)
  - Concatenate -> Dense(na) -> BN -> ReLU -> Dense(nb) -> BN -> ReLU -> Dropout -> Output

Output head is configurable:
  - 'mse':      single scalar with softplus (non-negative prediction, trained with MSE)
  - 'softplus':  single scalar with softplus (non-negative, for Tweedie loss)
  - 'gamma':     two outputs (log_alpha, log_beta) for Gamma NLL loss
  - 'bernoulli_gamma': three outputs (logit_p, log_alpha, log_beta) for BG NLL loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LANDModel(nn.Module):
    def __init__(
        self,
        climate_shape: Tuple[int, int, int] = (15, 3, 3),
        local_dem_shape: Tuple[int, int] = (3, 3),
        regional_dem_shape: Tuple[int, int] = (3, 3),
        num_month_features: int = 12,
        climate_units: int = 1020,
        local_dem_units: int = 64,
        regional_dem_units: int = 32,
        temporal_units: int = 16,
        na: int = 512,
        nb: int = 128,
        dropout_rate: float = 0.3,
        climate_processing: str = "conv2d",
        output_head: str = "mse",
    ):
        super().__init__()
        self.climate_processing = climate_processing
        self.output_head = output_head

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

        # --- Local DEM branch ---
        ld_flat = local_dem_shape[0] * local_dem_shape[1]
        self.ld_fc1 = nn.Linear(ld_flat, local_dem_units)
        self.ld_bn1 = nn.BatchNorm1d(local_dem_units)
        self.ld_fc2 = nn.Linear(local_dem_units, local_dem_units)
        self.ld_bn2 = nn.BatchNorm1d(local_dem_units)

        # --- Regional DEM branch ---
        rd_flat = regional_dem_shape[0] * regional_dem_shape[1]
        self.rd_fc1 = nn.Linear(rd_flat, regional_dem_units)
        self.rd_bn1 = nn.BatchNorm1d(regional_dem_units)
        self.rd_fc2 = nn.Linear(regional_dem_units, regional_dem_units)
        self.rd_bn2 = nn.BatchNorm1d(regional_dem_units)

        # --- Month branch ---
        self.mo_fc1 = nn.Linear(num_month_features, temporal_units)
        self.mo_bn1 = nn.BatchNorm1d(temporal_units)
        self.mo_fc2 = nn.Linear(temporal_units, temporal_units)
        self.mo_bn2 = nn.BatchNorm1d(temporal_units)

        # --- Dense head ---
        combined = cu + local_dem_units + regional_dem_units + temporal_units
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
        # Climate
        c = features["climate"]
        if self.climate_processing == "conv2d":
            c = self.climate_conv(c).view(c.size(0), -1)
        else:
            c = self.climate_fc(c.view(c.size(0), -1))
        c = F.relu(self.clim_bn1(c))
        c = F.relu(self.clim_bn2(self.clim_fc2(c)))

        # Local DEM
        ld = features["local_dem"].view(c.size(0), -1)
        ld = F.relu(self.ld_bn1(self.ld_fc1(ld)))
        ld = F.relu(self.ld_bn2(self.ld_fc2(ld)))

        # Regional DEM
        rd = features["regional_dem"].view(c.size(0), -1)
        rd = F.relu(self.rd_bn1(self.rd_fc1(rd)))
        rd = F.relu(self.rd_bn2(self.rd_fc2(rd)))

        # Month
        mo = features["temporal"]
        mo = F.relu(self.mo_bn1(self.mo_fc1(mo)))
        mo = F.relu(self.mo_bn2(self.mo_fc2(mo)))

        # Fusion
        x = torch.cat([c, ld, rd, mo], dim=1)
        x = F.relu(self.bn_a(self.fc_a(x)))
        x = F.relu(self.bn_b(self.fc_b(x)))
        x = self.dropout(x)
        x = self.out(x)

        if self.output_head in ("mse", "softplus"):
            x = F.softplus(x)
        # gamma / bernoulli_gamma: return raw logits (loss handles exp internally)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_land_model(hyperparams: dict, metadata: dict) -> LANDModel:
    """Factory: build a LAND model from hyperparams + data metadata."""
    climate_shape = tuple(metadata.get("climate_shape", (15, 3, 3)))
    local_dem_shape = tuple(metadata.get("local_dem_shape", (3, 3)))
    regional_dem_shape = tuple(metadata.get("regional_dem_shape", (3, 3)))
    num_month = int(metadata.get("num_month_features", 12))

    return LANDModel(
        climate_shape=climate_shape,
        local_dem_shape=local_dem_shape,
        regional_dem_shape=regional_dem_shape,
        num_month_features=num_month,
        climate_units=hyperparams.get("climate_units", 1020),
        local_dem_units=hyperparams.get("local_dem_units", 64),
        regional_dem_units=hyperparams.get("regional_dem_units", 32),
        temporal_units=hyperparams.get("temporal_units", 16),
        na=hyperparams.get("na", 512),
        nb=hyperparams.get("nb", 128),
        dropout_rate=hyperparams.get("dropout_rate", 0.3),
        climate_processing=hyperparams.get("climate_processing", "conv2d"),
        output_head=hyperparams.get("output_head", "mse"),
    )
