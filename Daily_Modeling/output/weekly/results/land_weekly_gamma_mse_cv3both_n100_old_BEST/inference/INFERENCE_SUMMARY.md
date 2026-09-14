# Inference Summary — `land_weekly_gamma_mse_cv3both_n100_old_BEST`

**Model:** LAND (Location-Agnostic Neural Downscaler, PyTorch)  
**Output head:** `gamma` (Gamma NLL)  
**Ensemble:** 15 members (3 CV folds × 5 seeds)  
**Split strategy:** `both` (spatial + temporal holdout)  
**Wet/dry threshold:** 1.0 mm  
**Target scale:** 75.07 mm

---

## Test-set regression metrics

| Metric | All samples (`test_all`) | Spatial test (`test_spatial`) | Temporal test (`test_temporal`) |
|---|---|---|---|
| **MSE**  | 1261.14  | 1245.14  | 1367.49  |
| **RMSE** | 35.51    | 35.29    | 36.98    |
| **MAE**  | 25.70    | 25.66    | 25.97    |
| **MBE**  | +4.20    | +6.05    | −8.15    |
| **R²**   | 0.533    | 0.537    | 0.509    |
| **Spearman r** | 0.734 | 0.736 | 0.782 |
| **Spearman p** | 3.17e-67 | 3.88e-59 | 1.29e-11 |

All correlations are highly significant.  
The model explains roughly 51–54% of the variance on the unseen test sets, with RMSE/MAE around 25–37 mm.

---

## Wet/dry classification (threshold = 1.0 mm)

| Metric | All samples (`test_all`) | Spatial test (`test_spatial`) | Temporal test (`test_temporal`) |
|---|---|---|---|
| Observed wet | 375 | 324 | 51 |
| Predicted wet | 390 | 339 | 51 |
| True positives (TP) | 375 | 324 | 51 |
| False positives (FP) | 15 | 15 | 0 |
| False negatives (FN) | 0 | 0 | 0 |
| True negatives (TN) | 0 | 0 | 0 |
| **POD (recall)** | 1.000 | 1.000 | 1.000 |
| **FAR** | 0.038 | 0.044 | 0.000 |
| **Frequency bias** | 1.040 | 1.046 | 1.000 |
| **CSI** | 0.962 | 0.956 | 1.000 |
| **ETS** | 0.000 | 0.000 | NaN |
| **HSS** | 0.000 | 0.000 | NaN |
| Wet RMSE (mm) | 35.05 | 34.73 | 36.98 |
| Wet MAE (mm)  | 25.20 | 25.09 | 25.97 |
| Wet MBE (mm)  | +2.84 | +4.57 | −8.15 |
| Wet R²        | 0.541 | 0.545 | 0.509 |
| Wet Spearman r | 0.744 | 0.744 | 0.782 |
| Wet mean observed (mm) | 59.65 | 60.40 | 54.90 |
| Wet mean predicted (mm) | 62.49 | 64.97 | 46.76 |

*Notes:*
- The model detects nearly all observed wet events (POD = 1.0).
- `ETS` and `HSS` are 0 because there are no dry samples in these test sets.
- The temporal split shows no false positives/negatives and no dry days, causing the equitable threat score (`ETS`) and Heidke skill score (`HSS`) to be undefined (`NaN`).

---

## Key artifacts

- `inference_manifest.json` — ensemble member list and metadata
- `metrics_test_*.json` — regression metrics for the three test partitions
- `wetdry_metrics_test_*.json` — wet/dry classification metrics for the three partitions
- `wetdry_eval_test_*.png` — wet/dry evaluation plots for each partition

---

## Interpretation

The gamma-output LAND ensemble performs consistently across spatial and temporal test holdouts.  
It reliably identifies wet weeks (≥1 mm), but shows a modest positive bias on the spatial test set and a negative bias on the temporal test set.  
Predictive accuracy (R² ≈ 0.51–0.54) indicates the model captures roughly half of the weekly rainfall variance, with a slight edge on the spatial split.
