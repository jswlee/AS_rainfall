"""
PyTorch Dataset and DataLoader utilities for the assembled NPZ.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Daily_Modeling import config


def crop_dem_patch(
    patch: torch.Tensor,
    target_size: int,
    stride: int,
) -> torch.Tensor:
    """Crop and subsample a DEM patch from a fine-resolution base.

    Args:
        patch: (..., H, W) tensor — the full max-size DEM patch.
        target_size: desired output grid dimension (e.g. 3 for 3x3).
        stride: subsampling stride in pixels (= km_per_cell when base is 1km).

    Returns:
        (..., target_size, target_size) tensor.
    """
    h, w = patch.shape[-2], patch.shape[-1]
    center_h, center_w = h // 2, w // 2
    half = target_size // 2
    # Build indices centred on the middle of the base patch
    rows = [center_h + (i - half) * stride for i in range(target_size)]
    cols = [center_w + (i - half) * stride for i in range(target_size)]
    # Clamp to valid range
    rows = [max(0, min(r, h - 1)) for r in rows]
    cols = [max(0, min(c, w - 1)) for c in cols]
    return patch[..., rows, :][..., cols]


class RainfallDataset(Dataset):
    """Index-based dataset over shared tensors (avoids data duplication)."""

    def __init__(
        self,
        tensors: Dict[str, torch.Tensor],
        indices: np.ndarray,
        target_scale: Optional[float] = None,
        dem_crop_config: Optional[dict] = None,
    ):
        self.tensors = tensors
        self.indices = indices
        self.target_scale = float(target_scale) if target_scale else None
        self.feature_keys = [k for k in tensors if k != "targets"]
        # dem_crop_config: {"local_patch_size": int, "local_km": int,
        #                   "regional_patch_size": int, "regional_km": int}
        self.dem_crop = dem_crop_config

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        di = self.indices[idx]
        features = {k: self.tensors[k][di] for k in self.feature_keys}
        # Runtime DEM cropping if configured
        if self.dem_crop is not None:
            if "local_patch_size" in self.dem_crop:
                features["local_dem"] = crop_dem_patch(
                    features["local_dem"],
                    self.dem_crop["local_patch_size"],
                    self.dem_crop["local_km"],
                )
            if "regional_patch_size" in self.dem_crop:
                features["regional_dem"] = crop_dem_patch(
                    features["regional_dem"],
                    self.dem_crop["regional_patch_size"],
                    self.dem_crop["regional_km"],
                )
        target = self.tensors["targets"][di]
        if self.target_scale and self.target_scale > 0:
            target = target / self.target_scale
        return features, target


def load_tensors_from_npz(
    npz_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """Load the assembled NPZ into GPU/CPU tensors + numpy metadata.

    Returns:
        tensors: dict of torch tensors {climate, local_dem, regional_dem, temporal, targets}
        metadata: dict of numpy arrays {stations, years, months, days, variables}
    """
    npz_path = npz_path or (config.ASSEMBLED_DIR / "daily_dataset.npz")
    z = np.load(str(npz_path), allow_pickle=True)

    tensors = {
        "climate": torch.from_numpy(z["reanalysis_patches"].astype(np.float32)).to(device),
        "local_dem": torch.from_numpy(z["dem_local_raw"].astype(np.float32)).to(device),
        "regional_dem": torch.from_numpy(z["dem_regional_raw"].astype(np.float32)).to(device),
        "temporal": torch.from_numpy(z["month_onehot"].astype(np.float32)).to(device),
        "targets": torch.from_numpy(z["rainfall_mm_raw"].astype(np.float32)).to(device),
    }
    metadata = {
        "stations": z["stations"],
        "years": z["years"],
        "months": z["months"],
        "days": z["days"],
        "variables": z["variables"] if "variables" in z.files else np.array([]),
    }
    return tensors, metadata


def normalize_tensors(
    tensors: Dict[str, torch.Tensor],
    train_indices: np.ndarray,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Normalize features using train-only statistics.  Returns updated tensors + stats dict."""
    stats: dict = {}
    device = tensors["climate"].device

    # Climate: per-channel z-score
    climate = tensors["climate"]
    c = climate.shape[1]
    means = torch.zeros(c, device=device)
    stds = torch.ones(c, device=device)
    train_clim = climate[train_indices]
    for i in range(c):
        vals = train_clim[:, i].reshape(-1)
        mask = torch.isfinite(vals)
        if mask.any():
            means[i] = vals[mask].mean()
            s = vals[mask].std()
            stds[i] = s if s > 0 else 1.0
    tensors["climate"] = (climate - means[None, :, None, None]) / stds[None, :, None, None]
    stats["climate_mean"] = means.cpu().numpy()
    stats["climate_std"] = stds.cpu().numpy()

    # DEM: global z-score per scale
    for key in ("local_dem", "regional_dem"):
        dem = tensors[key]
        train_vals = dem[train_indices].reshape(-1)
        mask = torch.isfinite(train_vals)
        m = train_vals[mask].mean() if mask.any() else torch.tensor(0.0)
        s = train_vals[mask].std() if mask.any() else torch.tensor(1.0)
        s = s if s > 0 else torch.tensor(1.0)
        tensors[key] = (dem - m) / s
        stats[f"{key}_mean"] = float(m)
        stats[f"{key}_std"] = float(s)

    # Target scale (train-only std in mm)
    raw_targets = tensors["targets"][train_indices]
    target_std = float(raw_targets.std())
    stats["target_std_mm"] = target_std
    print(f"Train-only target std: {target_std:.4f} mm")

    return tensors, stats


def print_normalization_report(
    tensors: Dict[str, torch.Tensor],
    stats: dict,
    split_indices: Dict[str, np.ndarray],
    variable_names: Optional[list] = None,
):
    """Print a detailed report of normalized feature statistics for transparency.

    Call this after normalize_tensors() to verify the data going into the models.
    """
    print("\n" + "=" * 70)
    print("  NORMALIZATION REPORT - Feature Statistics After Normalisation")
    print("=" * 70)

    # --- Climate per-channel stats ---
    climate = tensors["climate"]
    n_ch = climate.shape[1]
    print(f"\n  Climate reanalysis: shape={tuple(climate.shape)}")
    print(f"  {'Channel':<35s} {'mean':>8s} {'std':>8s} {'min':>10s} {'max':>10s} {'NaN%':>7s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*7}")
    for i in range(n_ch):
        vals = climate[:, i].reshape(-1)
        finite = vals[torch.isfinite(vals)]
        vname = variable_names[i] if variable_names and i < len(variable_names) else f"ch{i}"
        nan_pct = 100.0 * (1 - len(finite) / len(vals)) if len(vals) > 0 else 0
        if len(finite) > 0:
            print(f"  {vname:<35s} {finite.mean():>8.4f} {finite.std():>8.4f} "
                  f"{finite.min():>10.3f} {finite.max():>10.3f} {nan_pct:>6.2f}%")
        else:
            print(f"  {vname:<35s} {'nan':>8s} {'nan':>8s} {'nan':>10s} {'nan':>10s} {nan_pct:>6.2f}%")

    # --- Normalization params used ---
    if "climate_mean" in stats and "climate_std" in stats:
        print(f"\n  Climate normalisation params (train-only):")
        cm = stats["climate_mean"]
        cs = stats["climate_std"]
        for i in range(len(cm)):
            vname = variable_names[i] if variable_names and i < len(variable_names) else f"ch{i}"
            print(f"    {vname:<35s} mean={cm[i]:>12.4f}  std={cs[i]:>12.4f}")

    # --- DEM stats ---
    for key in ("local_dem", "regional_dem"):
        dem = tensors[key]
        flat = dem.reshape(-1)
        finite = flat[torch.isfinite(flat)]
        nan_pct = 100.0 * (1 - len(finite) / len(flat)) if len(flat) > 0 else 0
        print(f"\n  {key}: shape={tuple(dem.shape)}")
        if len(finite) > 0:
            print(f"    After norm:  mean={finite.mean():.4f}  std={finite.std():.4f}  "
                  f"min={finite.min():.3f}  max={finite.max():.3f}  NaN={nan_pct:.2f}%")
        m_key, s_key = f"{key}_mean", f"{key}_std"
        if m_key in stats:
            print(f"    Norm params: mean={stats[m_key]:.4f}  std={stats[s_key]:.4f}")

    # --- Temporal ---
    temp = tensors["temporal"]
    print(f"\n  temporal: shape={tuple(temp.shape)}")
    print(f"    min={temp.min():.4f}  max={temp.max():.4f}  (not normalised, one-hot)")

    # --- Targets ---
    targets = tensors["targets"]
    print(f"\n  targets (raw mm): shape={tuple(targets.shape)}")
    print(f"    mean={targets.mean():.4f}  std={targets.std():.4f}  "
          f"min={targets.min():.4f}  max={targets.max():.4f}")
    if "target_std_mm" in stats:
        print(f"    Train-only std (divisor for target normalisation): {stats['target_std_mm']:.4f} mm")

    # --- Per-split stats ---
    print(f"\n  Per-split target statistics (raw mm):")
    print(f"  {'Split':<15s} {'N':>7s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'%zero':>7s}")
    print(f"  {'-'*15} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for name, idx in split_indices.items():
        if len(idx) == 0:
            print(f"  {name:<15s} {0:>7d}")
            continue
        y = targets[idx]
        pz = 100.0 * float((y == 0).float().mean())
        print(f"  {name:<15s} {len(idx):>7d} {y.mean():>8.2f} {y.std():>8.2f} "
              f"{y.min():>8.2f} {y.max():>8.2f} {pz:>6.1f}%")

    print("\n" + "=" * 70 + "\n")


def make_dataloaders(
    tensors: Dict[str, torch.Tensor],
    split_indices: Dict[str, np.ndarray],
    target_scale: Optional[float] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    dem_crop_config: Optional[dict] = None,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for each split.

    If *dem_crop_config* is provided, DEM patches will be cropped/subsampled
    at runtime to the specified (patch_size, km_per_cell) combos.
    """
    loaders: Dict[str, DataLoader] = {}
    for name, idx in split_indices.items():
        if len(idx) == 0:
            continue
        ds = RainfallDataset(tensors, idx, target_scale=target_scale,
                             dem_crop_config=dem_crop_config)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders
