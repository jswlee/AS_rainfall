# `scripts/08_train_site_mlp.py` — Train Site-Specific MLPs

## Purpose
Eighth pipeline step. Trains one MLP per station using tuned hyperparameters from step 05. Supports optional shared pretraining (off by default, per paper), adaptive network sizing based on station sample count, configurable loss function (MSE default), and DEM patch cropping from tuned HP.

## Relation to the Deep Downscaling Paper
Hatanaka et al. (2025) Section 3b describes the site-specific MLP: a 3-hidden-layer network with softplus activations trained independently per station using MSE loss. Our implementation adds several extensions:
- **Adaptive sizing**: Smaller networks for stations with fewer training samples (not in paper).
- **Optional pretraining**: A shared backbone trained on all stations' data, then fine-tuned per station (not in paper — off by default).
- **LayerNorm**: Replaces BatchNorm for stability with variable batch sizes across stations.
- **DEM multi-resolution**: Cropping DEM patches from max-size bases based on tuned HP (extends paper).

## Line-by-Line Walkthrough

### `_FlatDataset` (lines 42–50)
```python
class _FlatDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        feats, target = self.base[idx]
        parts = [feats[k].view(-1) for k in ("climate", "local_dem", "regional_dem", "temporal")]
        return torch.cat(parts), target
```
Wraps `RainfallDataset` to flatten the feature dict into a single 1-D tensor. The MLP expects a flat input, unlike LAND which has separate branch inputs. The DEM crop happens inside `RainfallDataset.__getitem__` before flattening.

### `_pretrain_backbone()` (lines 53–83)
```python
def _pretrain_backbone(tensors, all_train_indices, target_scale, input_size,
                       hidden, dropout, lr, wd, bs, loss_type, device,
                       epochs=80, patience=20, dem_crop_config=None):
```
Optional shared pretraining:
1. Collects all train indices from all stations into one dataset.
2. Splits 90/10 for pretrain train/val.
3. Trains a single MLP on the combined data.
4. Returns the pretrained model — its weights are used to initialise per-station models (if architectures match).

This is **off by default** (`--pretrain` flag must be explicitly passed). The paper does not use pretraining. When enabled, per-station models that match the pretrained architecture are initialised from pretrained weights and fine-tuned at 0.1× LR. Stations with adapted (smaller) architectures train from scratch.

### `main()` (lines 86–293)

**Lines 86–97: CLI arguments**
```python
parser.add_argument("--loss-type", default="mse", choices=["mse", "log_mse", "tweedie"])
parser.add_argument("--pretrain", action="store_true")
```
- `--loss-type` defaults to MSE (per paper). `log_mse` and `tweedie` are available for manual ablation.
- `--pretrain` is opt-in. The paper trains each station independently.

**Lines 136–152: Load hyperparameters**
```python
if args.hp_dir:
    hp = json.loads((Path(args.hp_dir) / "best_hyperparameters.json").read_text())
    hidden = [hp.get("h1", 512), hp.get("h2", 512), hp.get("h3", 512)]
    dropout = hp.get("dropout", 0.3)
    lr = hp.get("lr", 1e-4)
    ...
    loss_type = hp.get("loss_type", args.loss_type)
```
Reads tuned HP from the JSON produced by step 05. Falls back to defaults from `config.py` if no HP dir is provided. The `loss_type` is taken from the HP file if present, otherwise from CLI.

**Lines 154–162: DEM crop config (BUG FIX)**
```python
dem_crop = config.resolve_dem_crop(hp)
```
Same critical fix as step 06 — translates Optuna index-based DEM configs to explicit (patch_size, km_per_cell) values. Without this fix, the MLP would train on full 11×11 DEM patches while the tuner evaluated cropped 3×3 patches.

**Lines 164–169: Input size computation**
```python
input_size = compute_input_size(
    climate_shape=metadata["climate_shape"],
    local_dem_shape=ld_shape,
    regional_dem_shape=rd_shape,
    num_month=metadata["num_month_features"],
)
```
Computes the flattened input dimension based on cropped DEM sizes. This must match what `_FlatDataset` produces.

**Lines 177–194: Phase 1 — Optional pretraining**
Only runs if `--pretrain` is passed. Collects training indices from all stations with ≥50 samples, concatenates them, and trains a shared backbone. Saves to `pretrained_backbone.pth`.

**Lines 196–269: Phase 2 — Per-station fine-tuning**

For each station:
1. **Proportional split** (line 204): chronological 70/20/10 using `station_proportional_split`.
2. **Adaptive sizing** (line 211):
   ```python
   stn_hidden = adaptive_hidden_sizes(n_train, hidden)
   ```
   - <200 samples → `[128, 128]`
   - 200–500 samples → `[256, 256, 256]`
   - ≥500 samples → use tuned hidden sizes (e.g. `[512, 128, 128]`)
3. **Dataset creation** (lines 213–219): Creates `_FlatDataset` with DEM cropping for train and val. Val falls back to the last 20% of train if no val data.
4. **Weight initialisation** (lines 221–228): If pretrained backbone exists AND the station's architecture matches the backbone's, loads pretrained weights and uses 0.1× LR. Otherwise trains from scratch.
5. **Training** (lines 230–235): Trains with early stopping.
6. **Evaluation** (lines 240–269): Runs inference on test split, de-normalises, computes metrics.

**Lines 271–288: Aggregate metrics**
Concatenates all per-station test predictions. Computes aggregate RMSE, MAE, MBE, R², Spearman. Saves scatter plot and predictions NPZ.

## Data Manipulation
- **Per-station proportional split**: Chronological 70/20/10 per station. Different from LAND's global year-based split.
- **DEM cropping**: Applied at DataLoader time in `RainfallDataset.__getitem__` via `crop_dem_patch()`.
- **De-normalisation**: Predictions (normalised) are multiplied by `target_scale` to get mm.
- **Adaptive sizing**: Network architecture varies per station based on sample count. This is transparent to the training loop since `SiteMLP` accepts any `hidden_sizes` list.

## Architecture Decisions
- **One model per station**: Unlike LAND, each station gets its own trained model. This gives maximum per-station accuracy but cannot generalise to new locations.
- **LayerNorm over BatchNorm**: LayerNorm normalises across features within a single sample, making it stable regardless of batch size. This is important because small stations may have batch sizes as small as 50.
- **Softplus activation**: The paper specifies softplus (smooth ReLU), which avoids dead neurons and is continuous. Implemented in `SiteMLP.__init__`.
- **Adaptive sizing defaults off during tuning**: Tuning uses fixed architecture; adaptive sizing only applies during training. This is a known gap — tuned HPs may not be optimal for the adapted architectures.

## Areas of Improvement
- **Tuning speed / batch size**: The main bottleneck is the per-station sequential training loop. With ~18 stations, training takes ~30–60 minutes. Parallelising across stations (or training all small-station models in a single batch) would help.
- **Per-station HP tuning**: The paper tunes per-station, but our implementation uses shared HPs + adaptive sizing. Per-station tuning would be better but much slower (~18× more Optuna studies).
- **Ensemble**: Training 3 random seeds per station and averaging would reduce variance, especially for small stations.
- **Transfer learning**: The pretrain→finetune approach could be improved by only freezing the first N layers during fine-tuning, rather than fine-tuning all weights at reduced LR.
- **Early stopping metric**: Currently uses validation MSE loss for early stopping. Using a rainfall-specific metric (e.g. skill score relative to climatology) might select better models.
