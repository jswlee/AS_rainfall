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


_DEFAULT_DEM_PATCH_SIZE = 10


class LANDModel(nn.Module):
    def __init__(
        self,
        climate_shape: Tuple[int, int, int] = (16, 3, 3),
        local_dem_shape: Tuple[int, int] = (3, 3),
        regional_dem_shape: Tuple[int, int] = (3, 3),
        dem_patch_size: int = _DEFAULT_DEM_PATCH_SIZE,
        num_month_features: int = 12,
        climate_units: int = 1020,
        dem_units: int = 64,
        temporal_units: int = 16,
        na: int = 512,
        nb: int = 128,
        dropout_rate: float = 0.3,
        climate_processing: str = "conv2d",
        output_head: str = "mse",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.climate_processing = climate_processing
        self.output_head = output_head
        self.local_dem_shape = local_dem_shape
        self.regional_dem_shape = regional_dem_shape
        self.dem_patch_size = int(dem_patch_size)

        _bn = lambda n: nn.BatchNorm1d(n) if use_batch_norm else nn.Identity()

        # --- Climate branch ---
        in_ch = climate_shape[0]
        H, W = climate_shape[1], climate_shape[2]
        if climate_processing == "conv2d" and H >= 3 and W >= 3:
            if climate_units % in_ch != 0:
                raise ValueError(
                    f"climate_units ({climate_units}) must be divisible by "
                    f"num_climate_vars ({in_ch}) for grouped conv."
                )
            self.climate_conv = nn.Conv2d(
                in_ch, climate_units, kernel_size=3, padding=0, groups=in_ch,
            )
            conv_out_h, conv_out_w = H - 2, W - 2
            self.climate_stem = nn.Sequential(
                self.climate_conv,
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(climate_units * conv_out_h * conv_out_w, climate_units),
            )
            cu = climate_units
        else:
            # Linear path (also auto-selected for grids too small for conv)
            if climate_processing == "conv2d":
                self.climate_processing = "linear"
            flat = in_ch * H * W
            self.climate_stem = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, climate_units),
            )
            cu = climate_units
        self.climate_body = nn.Sequential(
            _bn(cu),
            nn.ReLU(),
            nn.Linear(cu, cu),
            _bn(cu),
            nn.ReLU(),
        )
        self._cu = cu

        # --- Stacked DEM branch (paper: 2-channel 10x10 image -> Conv2d) ---
        # Local and regional DEMs are resized to dem_patch_size x dem_patch_size
        # and stacked along the channel dim before this conv.
        p = self.dem_patch_size
        if dem_units % 2 != 0:
            raise ValueError(f"dem_units ({dem_units}) must be even for groups=2 DEM conv.")
        self.dem_conv = nn.Conv2d(2, dem_units, kernel_size=3, padding=0, groups=2)
        dem_flat = dem_units * (p - 2) * (p - 2)
        self.dem_stack = nn.Sequential(
            self.dem_conv,
            nn.ReLU(),
            nn.Flatten(),
            _bn(dem_flat),
            nn.ReLU(),
            nn.Linear(dem_flat, dem_units),
            _bn(dem_units),
            nn.ReLU(),
        )
        self._dem_units = dem_units

        # --- Month branch ---
        self.month_stack = nn.Sequential(
            nn.Linear(num_month_features, temporal_units),
            _bn(temporal_units),
            nn.ReLU(),
            nn.Linear(temporal_units, temporal_units),
            _bn(temporal_units),
            nn.ReLU(),
        )

        # --- Dense head ---
        combined = cu + dem_units + temporal_units
        self.fusion_stack = nn.Sequential(
            nn.Linear(combined, na),
            _bn(na),
            nn.ReLU(),
            nn.Linear(na, nb),
            _bn(nb),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

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
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = self.dem_patch_size

        # Climate
        c = features["climate"]
        c = self.climate_stem(c)
        c = self.climate_body(c)

        # Stacked DEM: resize each to p×p, stack as 2-channel image, Conv2d
        ld = features["local_dem"]
        rd = features["regional_dem"]
        B = ld.size(0)
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
        dem = self.dem_stack(dem)

        # Month
        mo = features["temporal"]
        mo = self.month_stack(mo)

        # Fusion
        x = torch.cat([c, dem, mo], dim=1)
        x = self.fusion_stack(x)
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

    # dem_units: use explicit key
    dem_units = hyperparams.get("dem_units", 64)

    return LANDModel(
        climate_shape=climate_shape,
        local_dem_shape=local_dem_shape,
        regional_dem_shape=regional_dem_shape,
        dem_patch_size=int(hyperparams.get("dem_patch_size", _DEFAULT_DEM_PATCH_SIZE)),
        num_month_features=num_month,
        climate_units=hyperparams.get("climate_units", 1020),
        dem_units=dem_units,
        temporal_units=hyperparams.get("temporal_units", 16),
        na=hyperparams.get("na", 512),
        nb=hyperparams.get("nb", 128),
        dropout_rate=hyperparams.get("dropout_rate", 0.3),
        climate_processing=hyperparams.get("climate_processing", "conv2d"),
        output_head=hyperparams.get("output_head", "mse"),
        use_batch_norm=hyperparams.get("use_batch_norm", False),
    )
