# Model Training Summary — `land_weekly_gamma_mse_cv3both_n100_old_BEST`

**Model family:** LAND (Location-Agnostic Neural Downscaler, PyTorch)  
**Task:** Weekly rainfall regression for American Samoa  
**Output head:** `gamma` (Gamma NLL)  
**Climate processing:** `conv2d`  
**Lightweight mode:** false (full 2-stage architecture)

---

## Final hyperparameters

| Hyperparameter | Value |
|---|---|
| Local DEM config | 0 |
| Regional DEM config | 2 |
| Climate units (`climate_units`) | 120 |
| DEM units (`dem_units`) | 64 |
| DEM patch size (`dem_patch_size`) | 9 |
| Temporal units (`temporal_units`) | 16 |
| `na` (fusion layer size) | 240 |
| `nb` (pre-output layer size) | 32 |
| Dropout rate | 0.40 |
| Learning rate | 1.68e-05 |
| Weight decay | 6.79e-04 |
| Batch size | 128 |
| Loss type | `gamma` |
| Output head | `gamma` |
| Tweedie `p` | 1.5 |
| Batch normalization | false |

---

## Architecture at a glance

- **Climate branch:** grouped Conv2d over a `(16, 3, 3)` variable patch
- **DEM branch:** local + regional DEMs resized to `9 × 9`, stacked and passed through a grouped Conv2d
- **Month branch:** dense encoding of 12 month features
- **Fusion head:** `Dense(240) → ReLU → Dense(32) → ReLU → Dropout → Linear(2)` for the two Gamma parameters

Full architecture code is preserved in `model_architecture.py`.

---

## Cross-validation summary

**CV folds completed:** 3 / 3

| Metric | Mean | Std. dev. |
|---|---|---|
| MAE  | 31.63 | 3.98 |
| RMSE | 44.07 | 8.12 |

### Per-fold validation metrics

| Fold | MSE | RMSE | MAE | MBE | R² | Spearman r | 98th %ile true (mm) | 98th %ile pred (mm) | 98th %ile rel. bias | CSI (50 mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold 0 | 1138.86 | 33.75 | 26.36 | +2.35 | 0.690 | 0.818 | 226.42 | 168.27 | −0.257 | 0.620 |
| Fold 1 | 2871.36 | 53.59 | 35.98 | −3.50 | 0.453 | 0.698 | 293.88 | 206.35 | −0.298 | 0.673 |
| Fold 2 | 2014.99 | 44.89 | 32.56 | −0.84 | 0.369 | 0.675 | 217.17 | 208.47 | −0.040 | 0.629 |

*The 98th percentile relative bias (`pctl_rel_bias`) shows the model tends to under-predict the heaviest weekly rainfall events, especially in fold 1 where the high-end deficit is ~30%.*

---

## Station split

| Fold | Stations |
|---|---|
| **Train** | `aasufou80`, `aasufou90`, `airport5101`, `airport80`, `aua`, `aunuu`, `aunuu_UH`, `fagaitua`, `iliili`, `malaeimi`, `maloata`, `masefau`, `matatula`, `mt_alava`, `pioa_afono`, `satala`, `vaipito2000`, `vaipito_res` |
| **Validation** | `aoloafou`, `malaeimi_1691`, `poloa_UH`, `siufaga_WRCC`, `vaipito_UH` |
| **Test** | `aasu_UH`, `afono_UH`, `toa_ridge_WRCC` |

---

## Normalization

- Target standard deviation: **75.07 mm**
- Feature z-score statistics are stored in `normalization_stats.json` for climate variables and both DEM scales.

---

## Training artifacts

- `hyperparameters.json` — final selected hyperparameters
- `model_architecture.py` — complete PyTorch model definition
- `normalization_stats.json` — mean/std used for standardization
- `station_groups.json` — train/validation/test station assignments
- `cv_summary.json` — aggregate CV MAE/RMSE
- `architecture_land*.png` — model architecture diagrams
- `fold_*/` — per-fold outputs:
  - `metrics_cv_val.json` — validation fold metrics
  - `scatter_cv_val.png` — observed vs. predicted scatter
  - `training_history_seed*.png` — loss curves for each of 5 random seeds

---

## Summary

This is the retained “best” weekly LAND ensemble. It uses a full (non-lightweight) architecture with a Gamma NLL head and was trained with a low learning rate and a moderate dropout rate. Cross-validation performance is stable but variable across folds, with the most difficult fold (fold 1) showing higher RMSE (53.6 mm) and a large under-prediction of extreme weekly rainfall. The final ensemble combines all 15 trained members for inference.
