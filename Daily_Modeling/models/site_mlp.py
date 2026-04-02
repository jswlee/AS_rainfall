"""Site-specific MLP for daily rainfall prediction.

Architecture (adapted from paper Section 3b):
  - LayerNorm (replaces BatchNorm for stability with variable batch sizes)
  - Adaptive sizing: smaller networks for stations with fewer samples
  - Supports pretrain-on-all / fine-tune-per-station workflow
  - Configurable loss: MSE, log-MSE, or Tweedie

Input features: flattened reanalysis (CxHxW) + DEM + month one-hot.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SiteMLP(nn.Module):
    """Site-specific MLP with configurable hidden sizes and LayerNorm."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = (512, 512, 512),
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_size) -> (batch, 1) non-negative prediction."""
        h = self.backbone(x)
        return F.softplus(self.out(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GLUBlock(nn.Module):
    """Gated Linear Unit block (Dauphin et al., 2017).

    Splits a linear projection into two halves: one passes through a sigmoid
    gate and element-wise multiplies the other.  This provides multiplicative
    feature gating without recurrent overhead.

        GLU(x) = (W₁x + b₁) ⊙ σ(W₂x + b₂)

    The linear layer projects to 2 × output_dim, then chunks into value/gate.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.linear(x)  # (batch, output_dim * 2)
        value, gate = projected.chunk(2, dim=-1)  # each (batch, output_dim)
        return self.dropout(self.norm(value * torch.sigmoid(gate)))


class SiteGLU(nn.Module):
    """Site-specific model with a GLU gating layer followed by an MLP head.

    The first layer is a Gated Linear Unit (hidden_sizes[0] output dim) which
    provides adaptive, data-dependent feature selection — the same benefit as
    LSTM gating but without recurrent overhead.  The remaining layers
    (hidden_sizes[1:]) are a
    standard MLP head with LayerNorm + ReLU + Dropout.

    When hidden_sizes has only one element the GLU output feeds directly into
    the output layer (no MLP head layers).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = (256, 256),
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        # First layer: GLU gating
        self.glu = GLUBlock(input_size, hidden_sizes[0], dropout_rate)

        # MLP head on top of GLU output
        head_layers = []
        prev = hidden_sizes[0]
        for h in hidden_sizes[1:]:
            head_layers.append(nn.Linear(prev, h))
            head_layers.append(nn.LayerNorm(h))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(dropout_rate))
            prev = h
        self.head = nn.Sequential(*head_layers) if head_layers else nn.Identity()
        self.out = nn.Linear(prev, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_size) -> (batch, 1) non-negative prediction."""
        h = self.glu(x)     # (batch, hidden_sizes[0]) — gated
        h = self.head(h)    # MLP head
        return F.softplus(self.out(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



def build_model(
    arch_type: str,
    input_size: int,
    hidden_sizes: List[int],
    dropout_rate: float,
) -> nn.Module:
    """Factory: return SiteMLP or SiteGLU based on *arch_type*."""
    if arch_type == "glu":
        return SiteGLU(input_size, hidden_sizes, dropout_rate)
    return SiteMLP(input_size, hidden_sizes, dropout_rate)


def flatten_features(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten climate + local_dem + regional_dem + temporal into a single vector."""
    parts = []
    for key in ("climate", "local_dem", "regional_dem", "temporal"):
        t = features[key]
        parts.append(t.view(t.size(0), -1))
    return torch.cat(parts, dim=1)


def compute_input_size(climate_shape: Tuple, local_dem_shape: Tuple,
                       regional_dem_shape: Tuple, num_month: int = 12) -> int:
    """Compute the flattened input size for SiteMLP."""
    import math
    c = math.prod(climate_shape)
    ld = math.prod(local_dem_shape)
    rd = math.prod(regional_dem_shape)
    return c + ld + rd + num_month


def adaptive_hidden_sizes(n_train: int, base_hidden: List[int] = None) -> List[int]:
    """Return hidden-layer sizes scaled to the station's training set size.

    Preserves the **depth** (number of layers) of *base_hidden* and only
    scales down the **width** of each layer when the station has very few
    training samples.  This respects per-station tuned architectures (which
    may already be 1- or 2-layer) while still guarding against overfitting.

    Width scaling heuristic (applied per-layer):
      n_train < 150   →  cap each layer at 64
      n_train < 300   →  cap each layer at 128
      n_train < 600   →  cap each layer at 256
      n_train >= 600  →  use *base_hidden* as-is

    If *base_hidden* is None, falls back to [512, 512, 512].
    """
    if base_hidden is None:
        base_hidden = [512, 512, 512]
    if n_train >= 600:
        return list(base_hidden)
    if n_train < 150:
        cap = 64
    elif n_train < 300:
        cap = 128
    else:
        cap = 256
    return [min(h, cap) for h in base_hidden]


def create_site_mlp(metadata: dict, hyperparams: Optional[dict] = None,
                    n_train: Optional[int] = None) -> SiteMLP:
    """Factory: build a SiteMLP from data metadata + optional HP overrides.

    If *n_train* is given, hidden sizes are adapted to the station's sample
    count (unless hidden_sizes is explicitly provided in *hyperparams*).
    """
    from Daily_Modeling.config import MLP_DEFAULT_HP
    hp = {**MLP_DEFAULT_HP, **(hyperparams or {})}
    input_size = compute_input_size(
        metadata.get("climate_shape", (15, 3, 3)),
        metadata.get("local_dem_shape", (3, 3)),
        metadata.get("regional_dem_shape", (3, 3)),
        metadata.get("num_month_features", 12),
    )
    if n_train is not None and "hidden_sizes" not in (hyperparams or {}):
        hidden = adaptive_hidden_sizes(n_train, hp["hidden_sizes"])
    else:
        hidden = hp["hidden_sizes"]
    return SiteMLP(
        input_size=input_size,
        hidden_sizes=hidden,
        dropout_rate=hp["dropout_rate"],
    )
