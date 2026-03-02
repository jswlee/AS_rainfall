# `scripts/06_train_land.py` — Train the LAND Model

## Purpose
Sixth pipeline step. Trains the Location-Agnostic Neural Downscaling (LAND) model using the best hyperparameters from step 04 (or defaults). Evaluates on all four held-out splits (val_spatial, test_spatial, val_temporal, test_temporal). Saves the trained model, metrics, scatter plots, per-station metrics, and training history.

## Relation to the Deep Downscaling Paper
This directly implements the LAND model from Hatanaka et al. (2025) Section 3a. LAND is "location-agnostic" — a single model handles all stations by conditioning on local/regional DEM and month encoding. The paper trains LAND on all train-group stations in training years, then evaluates on:
- **Spatial generalisation**: held-out stations (tests if the model can downscale at new locations).
- **Temporal generalisation**: train stations in future years (tests if the model handles climate change).

Our implementation adds DEM patch cropping from tuned hyperparameters and the `resolve_dem_crop()` fix that translates Optuna index-based DEM configs to explicit (patch_size, km_per_cell) values.

## Line-by-Line Walkthrough

### `_get_metadata()` (lines 39–47)
Extracts tensor shapes for model construction. Same as in tuning script.

### `predict()` (lines 50–59)
```python
@torch.no_grad()
def predict(model, loader, device):
    for features, tgt in loader:
        features = {k: torch.nan_to_num(v.to(device)) for k, v in features.items()}
        out = model(features)
```
Inference function. Key detail: `torch.nan_to_num` replaces any remaining NaNs with 0 before feeding to the model. This is a safety net — NaNs should have been handled during feature building, but edge cases (coastal DEM pixels) might slip through.

### `main()` (lines 62–203)

**Lines 74–92: Load data + splits**
Standard pattern: load tensors, compute data-driven year boundaries, assign station groups, compute 5-way spatio-temporal split, normalise with train-only stats.

**Lines 94–95: Normalisation report**
```python
print_normalization_report(tensors, stats, splits, variable_names=var_names)
```
Prints the full normalisation report to stdout (and log file when run via pipeline). This is a transparency measure — you can verify post-hoc that normalisation was correct.

**Lines 97–100: Split heatmap**
Saves a station × year heatmap visualising which cells belong to which split. Colour-coded: blue=train, green=val_spatial, red=test_spatial, light green=val_temporal, orange=test_temporal.

**Lines 102–109: Load hyperparameters**
```python
if args.hp_dir:
    hp = json.loads((Path(args.hp_dir) / "best_hyperparameters.json").read_text())
else:
    hp = dict(config.LAND_DEFAULT_HP)
```
Loads tuned HPs from the JSON produced by step 04, or falls back to hardcoded defaults in `config.py`.

**Lines 111–119: DEM crop config (BUG FIX)**
```python
dem_crop = config.resolve_dem_crop(hp)
```
This was a critical bug fix. Previously, the code checked `if "local_dem_patch" in hp:` — but Optuna's `study.best_params` only saves `suggest_*` parameter names (e.g. `local_dem_cfg: 2`), not derived values. The `resolve_dem_crop()` function handles both index-based and explicit keys, ensuring the training script always gets the correct DEM crop config.

Without this fix, the model trained on full 11×11 and 25×25 DEM patches while the tuner had evaluated 3×3 patches — a complete train/tune mismatch that caused the LAND test_spatial R² to drop from 0.22 to -0.03.

**Lines 124–147: Train**
```python
loaders = make_dataloaders(tensors, splits, target_scale=target_scale,
                           batch_size=hp.get("batch_size", 256), dem_crop_config=dem_crop)
model = create_land_model(hp, metadata).to(device)
history = train_model(model, loaders["train"], loaders[val_key], device, ...)
```
Creates DataLoaders with DEM cropping, builds the model from HP + metadata, trains with early stopping. Uses `val_spatial` for early stopping (the hardest split — unseen stations in future years).

**Lines 149–158: Save model + HP + architecture**
Saves the trained model weights, the hyperparameters used, normalisation stats, station group assignments, and a copy of the model architecture source code (`land.py`). This makes each run self-documenting.

**Lines 164–194: Evaluation**
For each of the four evaluation splits:
1. Runs inference.
2. De-normalises predictions and targets (`× target_scale`).
3. Computes RMSE, MAE, MBE, R², Spearman correlation.
4. Computes baseline (mean predictor) metrics for comparison.
5. Saves scatter plot, predictions NPZ, and per-station metrics JSON.

## Data Manipulation
- **Normalisation**: Per-channel z-score for climate (mean/std from train set). Global z-score for DEM (single mean/std across all pixels from train). Target std computed from train but targets are NOT pre-normalised — division by `target_scale` happens in `RainfallDataset.__getitem__`.
- **DEM cropping**: At DataLoader time, the 11×11/25×25 base patches are cropped to the tuned (patch_size, km_per_cell) using `crop_dem_patch()`. This is a strided centre-crop: for a 3×3 @ 2km crop from an 11×11 @ 1km base, it takes pixels at stride 2 centred on the middle pixel.
- **De-normalisation**: Predictions are multiplied by `target_scale` to convert back to mm before computing metrics.

## Architecture Decisions
- **Single model for all stations**: The paper's key insight — DEM patches provide location information, so one model handles all stations. This enables spatial generalisation.
- **val_spatial for early stopping**: Using the hardest split for early stopping prevents overfitting to train stations. If val_spatial is empty (unlikely), falls back to val_temporal.
- **Architecture snapshot**: Copying `land.py` into the output directory ensures you can always reconstruct the exact model architecture used for a run, even if the code changes later.

## Areas of Improvement
- **Learning rate scheduling**: The current implementation uses constant LR with early stopping. A cosine annealing or OneCycleLR schedule could improve convergence speed and final performance.
- **Gradient accumulation**: For small batch sizes (64), gradient accumulation over 2–4 steps would give the effective gradient quality of bs=256 while fitting in GPU memory.
- **Ensemble**: Training 3–5 LAND models with different random seeds and averaging predictions is a simple way to reduce variance. The paper doesn't do this but it's standard practice.
- **Batch size for training speed**: The default is 256 which is reasonable. If tuning selects bs=64, training will be ~4× slower with minimal benefit. Consider clamping the training batch size to ≥128.
