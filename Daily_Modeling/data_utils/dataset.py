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


def precompute_dem_crops(
    tensors: Dict[str, torch.Tensor],
    dem_crop_config: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """Batch-crop DEM tensors upfront so ``__getitem__`` is pure indexing.

    Works on both per-station DEM tables (S_stations, C, H, W) and the legacy
    per-sample layout.  Returns a shallow copy of *tensors* with ``local_dem``
    and ``regional_dem`` replaced by their cropped versions.
    """
    if dem_crop_config is None:
        return tensors
    out = dict(tensors)  # shallow copy; non-DEM keys share storage
    if "local_patch_size" in dem_crop_config:
        out["local_dem"] = crop_dem_patch(
            tensors["local_dem"],
            dem_crop_config["local_patch_size"],
            dem_crop_config["local_km"],
        )
    if "regional_patch_size" in dem_crop_config:
        out["regional_dem"] = crop_dem_patch(
            tensors["regional_dem"],
            dem_crop_config["regional_patch_size"],
            dem_crop_config["regional_km"],
        )
    return out


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
        # station_dem_idx present => per-station DEM storage
        self._dem_indexed = "station_dem_idx" in tensors
        _exclude = {"targets", "station_dem_idx"}
        if self._dem_indexed:
            _exclude |= {"local_dem", "regional_dem"}
        self.feature_keys = [k for k in tensors if k not in _exclude]
        self.dem_crop = dem_crop_config

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        di = self.indices[idx]
        features = {k: self.tensors[k][di] for k in self.feature_keys}
        # Per-station DEM: look up via station_dem_idx
        if self._dem_indexed:
            si = int(self.tensors["station_dem_idx"][di])
            features["local_dem"] = self.tensors["local_dem"][si]
            features["regional_dem"] = self.tensors["regional_dem"][si]
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


class FlatDataset(Dataset):
    """Wraps RainfallDataset to return a single flattened feature vector.

    Used by site-specific MLP/GLU models that expect a 1-D input.
    """

    def __init__(self, base: RainfallDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        feats, target = self.base[idx]
        parts = [feats[k].view(-1) for k in ("climate", "local_dem", "regional_dem", "temporal")]
        return torch.cat(parts), target


def get_dataset_metadata(
    tensors: Dict[str, torch.Tensor],
    dem_crop_config: Optional[dict] = None,
) -> dict:
    """Extract shape metadata from loaded tensors.

    Works with both per-station DEM tables (``station_dem_idx`` present) and
    the legacy per-sample layout.

    Args:
        tensors: dict of torch tensors from :func:`load_tensors_from_npz`.
        dem_crop_config: optional DEM crop configuration; if provided,
            overrides local/regional DEM shapes in the returned metadata.

    Returns:
        dict with keys: climate_shape, local_dem_shape, regional_dem_shape,
        num_month_features, num_climate_vars.
    """
    c = tensors["climate"]
    # For per-station DEM tables shape is (S, C, H, W); per-sample is (N, C, H, W).
    # shape[1:] gives (C, H, W) in both cases — correct for model metadata.
    meta = {
        "climate_shape": tuple(c.shape[1:]),
        "local_dem_shape": tuple(tensors["local_dem"].shape[1:]),
        "regional_dem_shape": tuple(tensors["regional_dem"].shape[1:]),
        "num_month_features": int(tensors["temporal"].shape[1]),
        "num_climate_vars": int(c.shape[1]),
    }
    if dem_crop_config is not None:
        ld = tensors["local_dem"]
        n_dem_bands = ld.shape[1] if ld.dim() >= 3 else 1
        if "local_patch_size" in dem_crop_config:
            lp = dem_crop_config["local_patch_size"]
            meta["local_dem_shape"] = (n_dem_bands, lp, lp) if n_dem_bands > 1 else (lp, lp)
        if "regional_patch_size" in dem_crop_config:
            rp = dem_crop_config["regional_patch_size"]
            meta["regional_dem_shape"] = (n_dem_bands, rp, rp) if n_dem_bands > 1 else (rp, rp)
    return meta


def load_tensors_from_npz(
    npz_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """Load the assembled NPZ into GPU/CPU tensors + numpy metadata.

    Supports two DEM storage layouts:

    * **Per-station** (new, compact): NPZ contains ``station_dem_idx`` mapping
      each sample to a row in ``dem_local_raw`` / ``dem_regional_raw`` which
      have shape ``(S_stations, C, H, W)``.
    * **Per-sample** (legacy): ``dem_local_raw`` has shape ``(N_samples, C, H, W)``
      and no ``station_dem_idx`` key is present.

    Returns:
        tensors: dict of torch tensors.
            Always: climate, local_dem, regional_dem, temporal, targets.
            Per-station layout only: station_dem_idx (int32 on CPU/device).
        metadata: dict of numpy arrays {stations, years, months, days, variables}.
    """
    if npz_path is None:
        npz_path = config.ASSEMBLED_DIR / "daily_dataset_station_centered.npz"
    z = np.load(str(npz_path), allow_pickle=True)

    tensors = {
        "climate": torch.from_numpy(z["reanalysis_patches"].astype(np.float32)).to(device),
        "local_dem": torch.from_numpy(z["dem_local_raw"].astype(np.float32)).to(device),
        "regional_dem": torch.from_numpy(z["dem_regional_raw"].astype(np.float32)).to(device),
        "temporal": torch.from_numpy(z["month_onehot"].astype(np.float32)).to(device),
        "targets": torch.from_numpy(z["rainfall_mm_raw"].astype(np.float32)).to(device),
    }
    if "station_dem_idx" in z.files:
        tensors["station_dem_idx"] = torch.from_numpy(
            z["station_dem_idx"].astype(np.int64)
        ).to(device)
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

    # Climate: per-channel z-score
    climate = tensors["climate"]
    device = climate.device
    c = climate.shape[1]
    means = torch.zeros(c, device=device)
    stds = torch.ones(c, device=device)
    train_clim = climate[train_indices]
    for i in range(c):
        vals = train_clim[:, i].reshape(-1)
        mask = torch.isfinite(vals)
        if mask.any():
            means[i] = vals[mask].mean()
            stds[i] = vals[mask].std()
    tensors["climate"] = (climate - means[None, :, None, None]) / stds[None, :, None, None]
    stats["climate_mean"] = means.cpu().numpy()
    stats["climate_std"] = stds.cpu().numpy()

    # DEM: per-channel z-score (elevation, slope, aspect have different scales)
    # IMPORTANT: Exclude ocean (-1) values from normalization (match paper's land_mean/land_std)
    #
    # Per-station layout: dem tensors are (S_stations, C, H, W).  We compute
    # stats over all station patches (not per-sample train indices) because each
    # station appears exactly once in the DEM table.  For the legacy per-sample
    # layout we still slice by train_indices.
    dem_per_station = "station_dem_idx" in tensors
    for key in ("local_dem", "regional_dem"):
        dem = tensors[key]  # (S, C, H, W) per-station  OR  (N, C, H, W) / (N, H, W) legacy
        if dem.dim() == 3:
            # Legacy single-band: (N, H, W)
            src = dem if dem_per_station else dem[train_indices]
            train_vals = src.reshape(-1)
            mask = train_vals > 0
            m = train_vals[mask].mean() if mask.any() else torch.tensor(0.0)
            s = train_vals[mask].std() if mask.any() else torch.tensor(1.0)
            tensors[key] = (dem - m) / s
            stats[f"{key}_mean"] = float(m)
            stats[f"{key}_std"] = float(s)
        else:
            # Multi-band: (S or N, n_bands, H, W)
            n_bands = dem.shape[1]
            means = torch.zeros(n_bands, device=dem.device)
            stds = torch.ones(n_bands, device=dem.device)
            # Per-station: use all station patches; legacy: restrict to train rows
            train_dem = dem if dem_per_station else dem[train_indices]
            for b in range(n_bands):
                train_vals = train_dem[:, b].reshape(-1)
                mask = train_vals > -0.5
                if mask.any():
                    means[b] = train_vals[mask].mean()
                    stds[b] = train_vals[mask].std().clamp(min=1e-6)
            tensors[key] = (dem - means[None, :, None, None]) / stds[None, :, None, None]
            stats[f"{key}_mean"] = means.cpu().numpy()
            stats[f"{key}_std"] = stds.cpu().numpy()

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
            m_val, s_val = stats[m_key], stats[s_key]
            if np.ndim(m_val) == 0:
                # Single-band scalar
                print(f"    Norm params: mean={float(m_val):.4f}  std={float(s_val):.4f}")
            else:
                # Multi-band: one value per band
                band_names = ["elev", "slope", "sin_aspect", "cos_aspect"]
                for b, (mv, sv) in enumerate(zip(m_val, s_val)):
                    bname = band_names[b] if b < len(band_names) else f"band{b}"
                    print(f"    Norm params [{bname}]: mean={float(mv):.4f}  std={float(sv):.4f}")

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


class _DatasetStub:
    """Minimal stub so ``len(loader.dataset)`` works for :class:`InMemoryLoader`."""
    def __init__(self, n: int):
        self._n = n
    def __len__(self) -> int:
        return self._n


class InMemoryLoader:
    """Fast batch iterator for in-memory tensors.

    Replaces :class:`DataLoader` when all data already lives in CPU/GPU tensors
    and no subprocess workers are needed.  Instead of calling ``__getitem__``
    *batch_size* times and collating, it performs a single
    ``tensor[batch_indices]`` per feature key — typically 10–50× faster on the
    main thread.
    """

    def __init__(
        self,
        tensors: Dict[str, torch.Tensor],
        indices: np.ndarray,
        batch_size: int = 256,
        shuffle: bool = False,
        target_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        # When *device* is given, stage all tensors on that device once so
        # every batch is assembled via on-device indexing (no CPU→GPU copy).
        if device is not None:
            tensors = {k: v.to(device) for k, v in tensors.items()}
        self.tensors = tensors
        self.indices = torch.from_numpy(np.asarray(indices)).long()
        if device is not None:
            self.indices = self.indices.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_scale = float(target_scale) if target_scale else None
        # station_dem_idx present => per-station DEM storage
        self._dem_indexed = "station_dem_idx" in tensors
        _exclude = {"targets", "station_dem_idx"}
        if self._dem_indexed:
            _exclude |= {"local_dem", "regional_dem"}
        self.feature_keys = [k for k in tensors if k not in _exclude]
        self.dataset = _DatasetStub(len(indices))

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.indices))
            idx = self.indices[perm]
        else:
            idx = self.indices
        for start in range(0, len(idx), self.batch_size):
            bi = idx[start : start + self.batch_size]
            features = {k: self.tensors[k][bi] for k in self.feature_keys}
            # Per-station DEM: expand patches via station_dem_idx
            if self._dem_indexed:
                si = self.tensors["station_dem_idx"][bi]  # (B,)
                features["local_dem"] = self.tensors["local_dem"][si]
                features["regional_dem"] = self.tensors["regional_dem"][si]
            targets = self.tensors["targets"][bi]
            if self.target_scale and self.target_scale > 0:
                targets = targets / self.target_scale
            yield features, targets


def make_dataloaders(
    tensors: Dict[str, torch.Tensor],
    split_indices: Dict[str, np.ndarray],
    target_scale: Optional[float] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    dem_crop_config: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for each split.

    If *dem_crop_config* is provided, DEM patches will be cropped/subsampled
    at runtime to the specified (patch_size, km_per_cell) combos.

    When ``num_workers=0`` and ``dem_crop_config is None`` (i.e. tensors are
    already pre-cropped), an :class:`InMemoryLoader` is used instead of
    :class:`DataLoader` for significantly faster batch iteration.

    If *device* is given (e.g. ``torch.device("cuda")``), tensors are staged
    on that device inside :class:`InMemoryLoader` so batch assembly is pure
    on-device indexing with zero CPU→GPU transfer per step.
    """
    use_fast = int(num_workers) == 0 and dem_crop_config is None

    loaders = {}
    for name, idx in split_indices.items():
        if len(idx) == 0:
            continue

        if use_fast:
            loaders[name] = InMemoryLoader(
                tensors, idx,
                batch_size=batch_size,
                shuffle=(name == "train"),
                target_scale=target_scale,
                device=device,
            )
        else:
            ds = RainfallDataset(tensors, idx, target_scale=target_scale,
                                 dem_crop_config=dem_crop_config)
            use_persistent = bool(persistent_workers) and int(num_workers) > 0
            loaders[name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(name == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=use_persistent,
                prefetch_factor=(prefetch_factor if int(num_workers) > 0 else None),
            )
    return loaders
