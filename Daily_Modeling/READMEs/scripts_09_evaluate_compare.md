# `scripts/09_evaluate_compare.py` — Cross-Model Evaluation and Comparison

## Purpose
Ninth and final pipeline step. Loads saved predictions from all three models (LAND, Bernoulli-Gamma, Site MLP), computes standardised metrics, and produces comparison visualisations: metrics table, per-model scatter plots, combined overlay scatter, per-station bar charts, and seasonal breakdown.

## Relation to the Deep Downscaling Paper
Hatanaka et al. (2025) Section 4 presents a side-by-side comparison of LAND vs the Bernoulli-Gamma baseline. Our implementation adds the site-specific MLP as a third model and includes Wasserstein distance (earth mover's distance) as an additional distributional metric. The paper evaluates on multiple metrics including RMSE, correlation, and distributional similarity — we follow that approach.

## Line-by-Line Walkthrough

### `_load_predictions()` (lines 33–40)
```python
def _load_predictions(run_dir: Path):
    for name in ("predictions_test_spatial.npz", "predictions_test.npz", "predictions.npz"):
        p = run_dir / name
        if p.exists():
            z = np.load(str(p), allow_pickle=True)
            return z["y_true"], z["y_pred"], z.get("stations", np.array([]))
    return None, None, None
```
Tries multiple filenames because different models save predictions under different names:
- LAND saves `predictions_test_spatial.npz` (from the 5-way split).
- Bernoulli-Gamma and Site MLP save `predictions.npz` (aggregated across stations).

### `main()` (lines 43–185)

**Lines 54–68: Load predictions from all models**
```python
model_dirs = {
    "LAND": Path(args.land_dir),
    "Bernoulli-Gamma": Path(args.glm_dir),
    "Site MLP": Path(args.mlp_dir),
}
for name, d in model_dirs.items():
    yt, yp, st = _load_predictions(d)
```
Iterates over the three model directories. Skips any model whose predictions file is missing (with a warning).

**Lines 74–89: Compute metrics**
```python
m = compute_metrics(data["y_true"], data["y_pred"])
m["wasserstein"] = compute_wasserstein(data["y_true"], data["y_pred"])
bl = baseline_mean_metrics(data["y_true"])
```
For each model computes: RMSE, MAE, MBE (mean bias error), R², Spearman rank correlation, and Wasserstein distance. Also adds a "Baseline (mean)" row — the performance of always predicting the mean observed rainfall.

**Lines 91–97: Save metrics table**
Saves as both CSV and JSON, plus a formatted PNG table via `plot_model_comparison_table`.

**Lines 99–125: Scatter plots**
1. Individual scatter plots per model (observed vs predicted).
2. Combined overlay scatter with colour coding: LAND=steelblue, Bernoulli-Gamma=coral, Site MLP=seagreen. The diagonal red dashed line is the 1:1 perfect prediction line. Points below the line indicate underprediction.

**Lines 127–139: Per-station comparison**
```python
for metric_name in ("rmse", "mae", "r2"):
    plot_per_station_comparison(station_metrics_all, metric_name=metric_name, ...)
```
Grouped bar charts showing each model's RMSE/MAE/R² per station. This is the most informative visualisation — it reveals whether one model dominates across all stations or if different models excel in different locations.

**Lines 141–183: Seasonal breakdown**
Attempts to compute LAND metrics separately for dry season (May–Oct) and wet season (Nov–Apr). This requires aligning test predictions with month metadata from the assembled dataset. The paper analyses seasonal performance because tropical rainfall has strong seasonality — models may perform well in the dry season (predicting zeros) but poorly during intense wet-season events.

## Data Manipulation
- **De-normalised predictions**: All predictions loaded here are already in mm (de-normalisation happened in the training scripts).
- **Wasserstein distance**: Measures the "earth mover's distance" between the predicted and observed rainfall distributions. Lower is better. This captures distributional mismatch that point metrics like RMSE might miss.
- **Seasonal alignment**: The script re-loads the assembled NPZ to get month labels, then re-computes splits to identify which test indices correspond to which months. This is fragile — it assumes the test predictions are aligned with the splits.

## Architecture Decisions
- **Model-agnostic**: The comparison script doesn't import any model code — it only reads prediction NPZs. This means it works with any model that saves predictions in the expected format.
- **Baseline included**: The mean-predictor baseline provides a floor. Any model with R² < 0 is doing worse than simply predicting the average — a clear signal of failure.
- **Graceful degradation**: If a model's predictions are missing, it's skipped rather than crashing the script.

## Areas of Improvement
- **CDF / QQ plots**: Quantile-quantile plots would more precisely show distributional mismatch (e.g. heavy-tail underprediction) than scatter plots alone.
- **Skill scores**: Relative metrics like RMSE Skill Score (1 - RMSE_model/RMSE_baseline) would normalise across stations with different rainfall variability.
- **Spatial maps**: Plotting per-station R² on a map of American Samoa would reveal geographic patterns in model performance (e.g. windward vs leeward stations).
- **Confidence intervals**: Bootstrap confidence intervals on aggregate metrics would indicate whether differences between models are statistically significant.
- **Consistent test sets**: The three models may evaluate on slightly different test sets (LAND uses spatial test, GLM/MLP use temporal test). The comparison should ideally align them on the same test samples.
