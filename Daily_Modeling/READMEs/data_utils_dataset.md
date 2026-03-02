# `data_utils/dataset.py` — PyTorch Dataset, DataLoaders, and Normalisation

## Purpose
Provides the PyTorch `Dataset` and `DataLoader` infrastructure for all models. Handles loading the assembled NPZ into GPU tensors, train-only normalisation (z-score), runtime DEM cropping, and a detailed normalisation report for transparency. This is the bridge between raw numpy data and PyTorch training loops.

## Relation to the Deep Downscaling Paper
The paper (Section 3a) normalises inputs using training-set statistics (z-score per reanalysis channel, global z-score for DEM). Targets are divided by the training-set standard deviation. This module implements that normalisation exactly. The runtime DEM cropping is our extension — the paper uses fixed DEM patches, but we generate max-size patches and crop at runtime to support multi-resolution HP tuning.

## Line-by-Line Walkthrough

### `crop_dem_patch()` (lines 15–39)
```python
def crop_dem_patch(patch, target_size, stride):
    h, w = patch.shape[-2], patch.shape[-1]
    center_h, center_w = h // 2, w // 2
    half = target_size // 2
    rows = [center_h + (i - half) * stride for i in range(target_size)]
    cols = [center_w + (i - half) * stride for i in range(target_size)]
    rows = [max(0, min(r, h - 1)) for r in rows]
    cols = [max(0, min(c, w - 1)) for c in cols]
    return patch[..., rows, :][..., cols]
```
**This is the key function for multi-resolution DEM support.** Given a max-size base patch (e.g. 11×11 @ 1 km), it extracts a smaller patch at a different resolution by:
1. Finding the centre pixel of the base.
2. Computing row/column indices at `stride` intervals (stride = km_per_cell when base is 1 km).
3. Clamping indices to valid range.
4. Fancy-indexing to extract the subsampled grid.

Example: cropping a 3×3 @ 2 km patch from an 11×11 @ 1 km base:
- Centre is pixel (5, 5).
- Rows = [5 + (0-1)*2, 5 + (1-1)*2, 5 + (2-1)*2] = [3, 5, 7].
- Result: pixels at (3,3), (3,5), (3,7), (5,3), (5,5), (5,7), (7,3), (7,5), (7,7) — a 3×3 grid spaced 2 km apart.

### `RainfallDataset` (lines 42–83)
```python
class RainfallDataset(Dataset):
    def __init__(self, tensors, indices, target_scale=None, dem_crop_config=None):
```

**`__init__`**: Stores references to the shared tensor dict (no copy), the sample indices for this split, an optional target scaling factor, and an optional DEM crop config.

**`__getitem__` (lines 63–83)**:
```python
def __getitem__(self, idx):
    di = self.indices[idx]
    features = {k: self.tensors[k][di] for k in self.feature_keys}
    if self.dem_crop is not None:
        if "local_patch_size" in self.dem_crop:
            features["local_dem"] = crop_dem_patch(
                features["local_dem"],
                self.dem_crop["local_patch_size"],
                self.dem_crop["local_km"],
            )
        ...
    target = self.tensors["targets"][di]
    if self.target_scale and self.target_scale > 0:
        target = target / self.target_scale
    return features, target
```
1. Indexes into the shared tensors using the pre-computed index for this split.
2. If DEM crop config is set, crops local and regional DEM patches from the max-size base.
3. Divides the target by `target_scale` (train-set std in mm) — this is how target normalisation works.
4. Returns `(features_dict, target_scalar)`.

The features dict has keys: `climate`, `local_dem`, `regional_dem`, `temporal`. The LAND model consumes this dict directly; the MLP wraps it in `_FlatDataset` to flatten.

### `load_tensors_from_npz()` (lines 86–113)
```python
def load_tensors_from_npz(npz_path=None, device=torch.device("cpu")):
```
Loads `daily_dataset.npz` and converts numpy arrays to PyTorch tensors on the specified device (CPU or CUDA). Returns:
- `tensors`: dict of `{climate, local_dem, regional_dem, temporal, targets}` — all float32.
- `metadata`: dict of `{stations, years, months, days, variables}` — numpy arrays for split computation.

Note: the entire dataset is loaded onto GPU at once. For ~100k samples with 15×3×3 climate + 11×11 local DEM + 25×25 regional DEM, this is ~2–3 GB of GPU memory.

### `normalize_tensors()` (lines 116–159)
```python
def normalize_tensors(tensors, train_indices):
```
Normalises features in-place using train-only statistics:

1. **Climate (per-channel z-score)**: For each of the 15 channels, computes mean and std from only the training samples, then normalises the entire dataset. `(x - mean) / std`. NaN/Inf values are excluded from mean/std computation.

2. **DEM (global z-score per scale)**: One mean/std for all local DEM pixels, one for all regional. This is simpler than per-pixel normalisation but sufficient since DEM values are elevation in metres with a consistent scale.

3. **Target std**: Computes `target_std_mm` from training targets only. This is NOT applied to the tensors here — it's passed to `RainfallDataset` which divides targets by it on-the-fly. This design means the raw targets are always accessible.

Returns the modified tensors dict and a stats dict containing all normalisation parameters (for saving and later de-normalisation).

### `print_normalization_report()` (lines 162–242)
Prints a detailed formatted report of post-normalisation statistics: per-channel climate stats, normalisation parameters, DEM stats, temporal stats, target stats, and per-split target breakdowns. This is called by training scripts for transparency and debugging.

### `make_dataloaders()` (lines 245–271)
```python
def make_dataloaders(tensors, split_indices, target_scale=None,
                     batch_size=256, num_workers=0, dem_crop_config=None):
```
Creates one `DataLoader` per split (train, val_spatial, test_spatial, etc.). The train loader shuffles; all others don't. `num_workers=0` means data loading happens in the main process — fine for in-memory data but could be parallelised for disk-backed data.

The `dem_crop_config` is passed through to `RainfallDataset`, ensuring all loaders apply the same DEM cropping.

## Data Manipulation
- **In-place normalisation**: `normalize_tensors()` modifies the tensors dict in-place. This means calling it twice would double-normalise. The design assumes it's called exactly once.
- **Train-only statistics**: Normalisation parameters are computed from training indices only. This prevents data leakage from val/test sets.
- **Target normalisation**: Division by `target_std_mm` happens at DataLoader time, not in `normalize_tensors()`. This preserves raw targets for metric computation.
- **DEM cropping**: Applied per-sample in `__getitem__`. This is computationally cheap (fancy indexing on small arrays) but happens on every access, including repeated epochs.

## Architecture Decisions
- **Shared tensor storage**: All splits index into the same tensor arrays. This avoids duplicating the dataset for each split, saving GPU memory. The cost is indirect indexing, but this is negligible.
- **Dict-based features**: Returning features as a dict `{climate, local_dem, regional_dem, temporal}` allows the LAND model to process each modality in its own branch. The MLP's `_FlatDataset` adapter flattens this.
- **Runtime cropping over pre-cropping**: Cropping at DataLoader time means a single dataset file supports all DEM configurations. Pre-cropping would require regenerating the dataset for each HP trial.

## Areas of Improvement
- **num_workers > 0**: For larger datasets, setting `num_workers=2–4` with `pin_memory=True` would overlap data loading with GPU computation. Currently the bottleneck is GPU computation, not data loading, so this isn't critical.
- **Pre-crop DEM once per epoch**: Instead of cropping in every `__getitem__` call, the DEM could be pre-cropped once when the DataLoader is created (since it's per-station, not per-day). This would save repeated work.
- **Mixed precision**: Converting tensors to float16 for GPU storage would halve memory usage. The training loop would need `torch.cuda.amp` for mixed-precision training.
- **Lazy loading**: The current approach loads everything into GPU RAM upfront. For larger datasets, a memory-mapped approach (e.g. numpy memmap or HDF5) would scale better.
