# `scripts/04_tune_land.py` — Hyperparameter Tuning for the LAND Model

## Purpose
Fourth pipeline step. Uses Optuna's Tree-structured Parzen Estimator (TPE) to search over LAND model hyperparameters, including architecture sizes, learning rate, regularisation, batch size, and DEM patch resolution. Saves the best hyperparameters as JSON for use by the training script (step 06).

## Relation to the Deep Downscaling Paper
Hatanaka et al. (2025) tune LAND hyperparameters but do not describe the specific search algorithm. We use Optuna TPE, which is a Bayesian optimisation approach well-suited to mixed continuous/categorical/integer search spaces. The paper's LAND architecture has separate branches for climate (Conv2D), local DEM (FC), regional DEM (FC), and temporal (FC) inputs, merged through two fully-connected layers (Na, Nb). All of these branch widths and fusion layer sizes are tuned here.

The DEM patch size and resolution are **not** tuned in the paper (they use fixed 3×3 patches), but we extend the approach by including DEM patch configuration as a hyperparameter. This is encoded as an integer index into candidate lists defined in `config.py`.

## Line-by-Line Walkthrough

### `_get_metadata()` (lines 27–44)
```python
def _get_metadata(tensors, dem_crop_config=None):
```
Extracts tensor shapes needed to construct the LAND model. If `dem_crop_config` is provided, overrides the DEM shapes with the cropped sizes — this is critical because the model's FC input dimensions depend on DEM patch size.

### `objective()` (lines 47–106) — The Optuna Trial Function

**Lines 50–60: DEM patch HPs**
```python
local_idx = trial.suggest_int("local_dem_cfg", 0, len(local_candidates) - 1)
regional_idx = trial.suggest_int("regional_dem_cfg", 0, len(regional_candidates) - 1)
lp, lk = local_candidates[local_idx]
rp, rk = regional_candidates[regional_idx]
```
Selects a (patch_size, km_per_cell) combo for both local and regional DEM. The index maps to tuples like `(3, 2)` = 3×3 grid at 2 km spacing = 6 km total box. A `dem_crop_config` dict is built and passed to the dataloader so patches are cropped from the max-size base at runtime.

**Lines 64–80: Architecture HPs**
```python
"climate_units": trial.suggest_int("climate_units", num_cv * 34, num_cv * 102, step=num_cv),
"na": trial.suggest_int("na", 256, 2048, step=128),
"nb": trial.suggest_int("nb", 64, 256, step=32),
```
- `climate_units` is the width of the climate Conv2D branch output. The range is scaled by `num_climate_vars` (15) to ensure each variable gets a proportional number of units. The `step=num_cv` constraint ensures divisibility.
- `na` and `nb` are the two fusion layers that merge all branches. These are the most important capacity parameters.
- `local_dem_units`, `regional_dem_units`, `temporal_units` are the FC branch widths.
- `dropout_rate` is searched from 0.1 to 0.5.
- `learning_rate` and `weight_decay` are searched on log scales.
- `batch_size` is categorical: [64, 128, 256, 512].

**Lines 82–84: Divisibility enforcement**
```python
if hp["climate_units"] % num_cv != 0:
    hp["climate_units"] = (hp["climate_units"] // num_cv) * num_cv
```
The LAND model's Conv2D branch reshapes the climate tensor, requiring the output width to be divisible by the number of climate variables.

**Lines 86–105: Train and evaluate**
```python
loaders = make_dataloaders(tensors, splits, target_scale=target_scale,
                           batch_size=hp["batch_size"], dem_crop_config=dem_crop)
model = create_land_model(hp, metadata).to(device)
history = train_model(model, loaders["train"], loaders[val_key], device, ...)
```
Each trial trains a full LAND model with early stopping (patience=30, max 200 epochs). The objective value is the best validation loss (MSE on normalised targets). Uses `val_spatial` if available, falling back to `val_temporal`.

### `main()` (lines 109–185) — Study Setup

**Lines 122–134: Data-driven splits**
```python
train_yr, val_yr, test_yr = compute_year_boundaries(years)
groups = assign_station_groups(...)
splits = spatiotemporal_split(stations, years, groups, ...)
tensors, stats = normalize_tensors(tensors, splits["train"])
```
Year boundaries are computed from data (70/20/10 split of samples chronologically). Stations are assigned to train/val/test groups. Normalisation uses train-only statistics.

**Lines 145–156: Optuna study**
```python
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(..., n_trials=args.n_trials)
```
TPE sampler with fixed seed for reproducibility. MedianPruner can early-terminate unpromising trials (though our training loop doesn't report intermediate values, so pruning is limited).

**Lines 161–171: Save enriched best params**
```python
best_hp = dict(study.best_params)
dem_crop = config.resolve_dem_crop(best_hp)
if dem_crop is not None:
    best_hp["local_dem_patch"] = dem_crop["local_patch_size"]
    ...
```
**Critical**: `study.best_params` only contains Optuna's `suggest_*` parameter names (e.g. `local_dem_cfg: 2`). We enrich it with the resolved DEM values (`local_dem_patch: 3`, `local_dem_km: 2`) so the training script can use either form. The `resolve_dem_crop()` function in `config.py` handles both index-based and explicit keys.

### `_save_tuning_visuals()` (lines 188–231)
Generates four Optuna plots: HP importance (fANOVA-based), optimisation history, parallel coordinate, and slice plots. The HP importance plot is especially useful — it shows which hyperparameters most affect validation loss.

## Data Manipulation
- **Spatio-temporal split**: Same 5-way split as training (train, val_spatial, test_spatial, val_temporal, test_temporal).
- **Normalisation**: z-score on climate channels (per-channel mean/std from train), global z-score on DEM, target std computed from train.
- **DEM cropping**: Applied at DataLoader time via `crop_dem_patch()` in `dataset.py`. The max-size 11×11/25×25 patches are subsampled and cropped to the trial's chosen (patch_size, km_per_cell).

## Architecture Decisions
- **Single study, no pruning integration**: Each trial trains to convergence. This is expensive but gives reliable objective values. Integration with Optuna's `report()` for epoch-level pruning would speed things up.
- **TPE over grid/random search**: TPE is more sample-efficient for the 13-dimensional search space, focusing on promising regions after ~20 random trials.
- **DEM as integer index**: Encoding DEM configs as integer indices (`0–4` for local, `0–3` for regional) is a pragmatic choice since the (patch_size, km_per_cell) combos are discrete and non-ordinal. TPE treats these as ordinal, which is imperfect but works.

## Areas of Improvement
- **Tuning speed**: Each trial takes 5–15 minutes on GPU. With 60 trials, that's 5–15 hours total. Key bottlenecks:
  - **Batch size**: The search includes bs=64, which is very slow on GPU. Consider raising the minimum to 128 or 256 — GPU utilisation is low with small batches.
  - **Epoch-level pruning**: Integrating `trial.report(val_loss, epoch)` and using `MedianPruner` would kill bad trials after a few epochs instead of training to convergence.
  - **Reduced max epochs during tuning**: Using `MAX_EPOCHS=100` (instead of 200) for tuning is common practice — the relative ranking of HPs is usually stable after 100 epochs.
  - **Subset training**: Training on a random 50% subset of train data during tuning would halve trial time with minimal impact on HP selection quality.
- **DEM pre-screening**: Running a quick 10-trial study over just DEM configs (fixing architecture to defaults), then fixing the best DEM for a full architecture search, would reduce the search space from 13D to 11D.
- **Parallel trials**: Optuna supports `n_jobs=-1` for parallel trial execution across multiple GPUs.
