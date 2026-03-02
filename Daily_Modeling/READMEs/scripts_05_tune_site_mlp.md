# `scripts/05_tune_site_mlp.py` — Hyperparameter Tuning for the Site-Specific MLP

## Purpose
Fifth pipeline step. Uses Optuna TPE to search over Site MLP hyperparameters: hidden layer sizes, dropout, learning rate, weight decay, batch size, and DEM patch resolution. Unlike LAND tuning, each trial trains on a **subset of stations** (5 train stations) and averages their validation losses. Saves the best hyperparameters as JSON for step 08.

## Relation to the Deep Downscaling Paper
The paper's site-specific MLP (Section 3b) is a 3-hidden-layer network with softplus activation. The paper tunes per-station, but our approach shares a single HP search across stations for efficiency — the tuned HPs serve as defaults that are then used with adaptive sizing per station (see step 08). The paper uses MSE loss throughout; we fixed tuning to MSE after discovering that mixing loss types (MSE/log-MSE/Tweedie) in a single study produces incomparable objective scales.

## Line-by-Line Walkthrough

### `_FlatDataset` (lines 32–41)
```python
class _FlatDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        feats, target = self.base[idx]
        parts = [feats[k].view(-1) for k in ("climate", "local_dem", "regional_dem", "temporal")]
        return torch.cat(parts), target
```
Wraps `RainfallDataset` to return a single flattened feature vector instead of a dict. The MLP expects a 1-D input (unlike LAND which expects separate branches). The flattening order is: climate (15×3×3=135), local_dem (varies with crop), regional_dem (varies), temporal (12) = total ~156–781 depending on DEM crop.

### `objective()` (lines 54–111) — The Optuna Trial Function

**Lines 55–65: DEM patch HPs**
Same as LAND tuning — selects from candidate (patch_size, km_per_cell) combos and builds `dem_crop` config.

**Lines 67–80: Architecture HPs**
```python
"hidden_sizes": [
    trial.suggest_categorical("h1", [128, 256, 512]),
    trial.suggest_categorical("h2", [128, 256, 512]),
    trial.suggest_categorical("h3", [128, 256, 512]),
],
"dropout_rate": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
"learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
"weight_decay": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
"batch_size": trial.suggest_categorical("bs", [64, 128, 256]),
```
- Three hidden layers, each independently chosen from {128, 256, 512}. This allows asymmetric architectures (e.g. 512→128→128).
- Loss is **fixed to MSE** for tuning (line 80). Previously `loss_type` was an HP, but different losses produce different objective scales, making cross-loss comparison invalid within a single Optuna study. The HP importance plot showed loss_type at 0.97, starving all other HPs of exploration budget.

**Lines 82–88: Input size computation**
```python
input_size = compute_input_size(
    climate_shape=metadata["climate_shape"],
    local_dem_shape=(lp, lp),
    regional_dem_shape=(rp, rp),
    num_month=metadata["num_month_features"],
)
```
The MLP's first layer width depends on the cropped DEM sizes. `compute_input_size` multiplies out the shapes: `prod(15,3,3) + prod(lp,lp) + prod(rp,rp) + 12`.

**Lines 90–111: Train on subset of stations**
```python
for st in train_stations[:5]:  # sample 5 for speed
    sp = station_proportional_split(stations, years, months, days, st)
    ...
    model = SiteMLP(input_size, hp["hidden_sizes"], hp["dropout_rate"]).to(device)
    hist = train_model(model, tl, vl, device, epochs=100, patience=15, ...)
    val_losses.append(min(hist["val_loss"]))
return float(np.mean(val_losses))
```
For each of 5 train stations:
1. Computes a per-station chronological 70/20/10 split.
2. Creates `_FlatDataset` instances with DEM cropping.
3. Trains a fresh MLP for up to 100 epochs with patience=15.
4. Records the best validation loss.

The objective is the mean val loss across the 5 stations. Stations with <50 train or <10 val samples are skipped.

### `main()` (lines 114–174) — Study Setup

**Lines 125–138: Data-driven splits + normalisation**
Same pattern as LAND tuning. Data-driven year boundaries, station group assignment, normalisation with train-only stats.

**Lines 140: Select train stations**
```python
train_stations = sorted([s for s, r in groups.items() if r == "train"])
```
Only stations assigned to the "train" group are used for tuning. Val/test stations are held out.

**Lines 154–164: Save enriched best params**
Same as LAND — enriches `study.best_params` with resolved DEM values via `config.resolve_dem_crop()`.

## Data Manipulation
- **Per-station proportional split**: Each station's data is sorted chronologically, then split 70/20/10 into train/val/test. This is different from the LAND split (which uses year ranges globally). The proportional split ensures every station gets a validation set even if it only has data for a narrow year range.
- **Subset evaluation**: Only 5 stations are evaluated per trial for speed. This introduces noise but is a standard Optuna practice for expensive objectives.
- **DEM cropping**: Same runtime crop as LAND tuning — the `dem_crop_config` is passed through `RainfallDataset.__getitem__`.

## Architecture Decisions
- **Shared HP search**: One Optuna study across all stations (not per-station tuning). This is pragmatic — with ~18 train stations and 30+ trials each, per-station tuning would take days.
- **Fixed MSE loss**: After discovering the loss_type dominance issue (importance=0.97), loss was removed from the search. Alternative losses can be tested manually via `--loss-type` flag on step 08.
- **5-station subset**: Balances trial speed (~3 min) vs signal quality. More stations would reduce noise but increase trial time proportionally.

## Areas of Improvement
- **Tuning speed**: Each trial trains 5 separate MLPs sequentially. Key speedups:
  - **Raise min batch size**: bs=64 underutilises the GPU. Setting minimum to 128 would help.
  - **Reduce epochs during tuning**: 100 epochs with patience=15 is reasonable but could be tightened to 60/10.
  - **Parallel station training**: The 5 station models are independent — they could be trained in parallel on a single GPU using separate CUDA streams, or across multiple GPUs.
  - **Fewer hidden size options**: The 3³=27 combinations for hidden sizes could be reduced to 5–10 pre-defined architectures.
- **Adaptive sizing during tuning**: Currently tuning ignores adaptive sizing (uses fixed hidden sizes). Since step 08 applies adaptive sizing at train time, the tuned HPs may not match the actual architecture used for small stations.
- **Cross-validation within tuning**: Instead of 5 fixed stations, randomly sampling 5 different stations per trial would give a less biased estimate.
