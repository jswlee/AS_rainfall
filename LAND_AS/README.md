# LAND_AS: Weekly Rainfall Downscaling for American Samoa

`LAND_AS` adapts the Location-Agnostic Neural Downscaler (LAND) from
`LocationAgnosticNeuralDownscaling` to weekly station rainfall in American Samoa.
The package currently supports two controlled variants of the same architecture:

1. `gamma`: the original-style distributional LAND model used by `weekly_land_v5`.
2. `huber`: a v5-sized scalar model that predicts positive weekly rainfall and is
   trained with Huber or rainfall-weighted Huber loss.

The purpose of this directory is not merely to maximize a leaderboard score. It
is to determine whether location-agnostic spatial feature learning is useful for
a very small, spatially heterogeneous rainfall problem, and to compare it fairly
against simpler climatological and tabular baselines.

## 1. Scientific problem

The target is total weekly rainfall at a rain-gauge station. The model receives
only information that should transfer across locations:

- current-week atmospheric reanalysis around the station;
- local and regional topography around the station;
- calendar month;
- recent observed rainfall at the same station.

This is a downscaling problem because the model must infer a point-scale station
total from coarse atmospheric fields plus local geographic context. It is also a
spatial-transfer problem: the final test stations are absent from model fitting.

## 2. Why the evaluation is strict

The current test split is deliberately harder than random row splitting.

- Training stations: all stations not listed in `config.TEST_STATIONS`.
- Training years: `year <= 2016`.
- Test stations: `aasu_UH`, `afono_UH`, `aunuu_UH`, `poloa_UH`, `vaipito_UH`.
- Test years: `year > 2016`.
- Validation: leave-one-station-out over training stations only.

Thus test predictions require generalization to both unseen locations and a
later climate period. Randomly mixing stations or years would overstate
performance because nearby station-years are highly correlated.

The split is implemented in `LAND_AS/config.py` and `LAND_AS/data.py`:

- `TRAIN_YEAR_END = 2016`
- `TEST_STATIONS = ["aasu_UH", "afono_UH", "aunuu_UH", "poloa_UH", "vaipito_UH"]`
- `_station_roles()` enforces station-role separation
- `_split()` applies the year cutoff
- `cv_folds(..., mode="loso")` creates 21 training-station folds

No test sample is used for architecture selection, objective selection,
checkpoint selection, blend-weight selection, or calibration.

## 3. Data pipeline

### 3.1 Raw inputs

`LAND_AS.prepare` calls the shared `Daily_Modeling` data builders. It expects:

- station metadata: `raw_data/AS/station_locations.csv`
- daily station rainfall CSVs: `raw_data/AS/final_rainfall_per_station/`
- daily reanalysis NetCDFs: `raw_data/AS/climate_variables_daily_1980-2024/`
- terrain raster: `raw_data/AS/DEM/10m_tutuila_3band.tif`

Station metadata supplies latitude, longitude, elevation, source, and record
bounds. Rainfall CSVs are converted to millimeters when needed.

### 3.2 Feature-cache construction

`prepare()` creates two cached feature files under `LAND_AS/data/features/`:

```text
LAND_AS/data/features/reanalysis_daily.npz
LAND_AS/data/features/dem.npz
```

The reanalysis cache contains station-centered daily `3 x 3` patches for 15
atmospheric variables. The DEM cache contains station-level local and regional
terrain patches with four channels:

1. elevation
2. slope
3. sine aspect
4. cosine aspect

The maximum stored terrain extents are:

- local DEM: `11 x 11` at 1-km spacing
- regional DEM: `25 x 25` at 1-km spacing

Runtime hyperparameters can crop or subsample these patches.

### 3.3 Weekly assembly

`Daily_Modeling.data_utils.assemble_dataset.assemble(..., freq="weekly")` then
joins reanalysis, terrain, rainfall, and calendar fields and aggregates daily
samples to ISO weeks.

Each weekly sample contains:

- 15 weekly mean atmospheric channels;
- 15 weekly within-week standard-deviation channels;
- month one-hot vector;
- weekly rainfall total;
- station DEM index;
- station, year, month, and week-start day metadata.

The result is:

```text
LAND_AS/data/weekly_dataset.npz
```

Rebuild it with:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.prepare `
  --start-date 1980-01-01 `
  --end-date 2024-12-31 `
  --rebuild
```

### 3.4 Runtime lag and normalization

`LAND_AS.data.load_data()` adds lag information at load time.

- `LAG_WEEKS = 3` prior weeks are materialized in the raw lag array.
- For each lag, the model receives prior observed rainfall and a validity flag.
- Missing lag weeks are zero-filled and flagged rather than treated as observed
  dry weeks.
- Reanalysis lag blocks are also materialized, but the selected v5
  hyperparameters use `climate_lag_weeks=0`, so only current-week atmosphere is
  used.
- The selected v5 hyperparameters use `rain_lag_weeks=2`, so only the first two
  rainfall lags remain in the runtime feature vector.

The default bundle uses only the pre-2017 training split for normalization:

- atmospheric channels: per-channel train mean/std;
- DEM channels: land-pixel mean/std;
- target: standard deviation of training rainfall.

For cross-validation, `normalized_bundle()` rebuilds these statistics using each
fold's own training rows. This mirrors `Daily_Modeling`'s fold-local
normalization and prevents a temporal or LOSO validation fold from influencing
its own feature scaling. New checkpoints write `seed_<N>_normalization.json`;
evaluation and OOF blending use that per-checkpoint marker to select fold-local
scaling. Older checkpoints without the marker retain their original
global-training normalization for backward compatibility.

The run-level `normalization.json` and `split.json` record the all-training-row
settings for reproduction.

## 4. LAND_AS architecture

The implementation is in `LAND_AS/model.py`.

### 4.1 Branch design

The model processes each input family independently:

- atmospheric patch: grouped `3 x 3` convolution, ReLU, adaptive pooling,
  flatten, linear encoder;
- local/regional DEM: grouped `3 x 3` convolution, ReLU, flatten, two linear
  layers;
- month one-hot: linear encoder;
- lag vector: concatenated directly into the fusion head.

The original LAND uses the same general idea: independent atmospheric, DEM, and
month branches followed by a dense head. The local implementation adds recent
rainfall lag features and adapts dimensions to weekly American Samoa inputs.

### 4.2 Gamma output head

`model_type: "gamma"` produces two raw outputs. Softplus transforms them into a
positive Gamma concentration and scale. The point prediction used for metrics
is the Gamma mean:

```text
E[y] = concentration x scale
```

Gamma is reasonable for weekly rainfall because weekly totals are nonnegative
and right-skewed, and almost all retained weekly totals are positive. It also
provides a distributional formulation rather than only a scalar regression
output.

The Gamma negative log-likelihood ignores exact-zero weeks because a Gamma
distribution has no mass at exactly zero. With `rainfall_weight=true`, wetter
weeks receive larger weights proportional to `log1p(target)`, normalized within
the batch.

### 4.3 Scalar Huber head

`model_type: "huber"` preserves every v5 branch and dimension but changes the
output to one scalar transformed by softplus. Training uses Huber loss on the
normalized target.

Huber is used because it sits between MSE and MAE:

- small residual errors receive approximately quadratic penalties;
- large residuals receive approximately linear penalties;
- the model is less dominated by rare extreme weeks than MSE;
- unlike pure MAE, it retains some curvature for typical errors.

Huber is therefore a useful controlled alternative to Gamma NLL when the main
question is whether Gamma's distributional likelihood, rather than the feature
extractor, is responsible for the result.

### 4.4 Rainfall-weighted Huber

`loss_type: "huber_weighted"` applies ordinary Huber elementwise loss but
multiplies each sample by a detached `log1p(target)` weight normalized within
the batch. The target here is the model's normalized target, so the weight is a
relative wetness weight rather than an interpretable millimeter scale.

The intended effect is to counteract scalar Huber's tendency to underpredict
the upper tail without making the objective fully MSE-like.

### 4.5 Checkpoint monitor

Training histories always record validation MAE and MSE. The `monitor` setting
selects which one controls early stopping and checkpoint retention:

- `monitor: "mae"`: optimize typical absolute error;
- `monitor: "mse"`: optimize squared error, which is more sensitive to large
  misses and can improve RMSE/extreme-week behavior.

This affects checkpoint selection, not the form of the training loss.

### 4.6 Why Gamma remains ahead of Bernoulli-Gamma

A Bernoulli-Gamma/hurdle model would learn separate occurrence and positive-amount
components. That is appropriate for zero-inflated daily rainfall, but weekly
American Samoa totals are rarely exactly dry:

- training rows: 197 exact zeros in 6,686 samples (2.95%);
- test rows: 17 exact zeros in 1,188 samples (1.43%).

The current Gamma loss already excludes those rare dry weeks from the amount fit.
A Bernoulli occurrence head would therefore receive sparse weekly supervision and
add another output, threshold, and calibration decision for a phenomenon that is
not currently the dominant error source. `Daily_Modeling` makes the same
practical distinction: Bernoulli-Gamma is the daily default, while ordinary Gamma
is the weekly default. Its weekly tuning also favored Gamma over Bernoulli-Gamma.
Bernoulli-Gamma remains a valid controlled challenger, but it should not replace
Gamma without leakage-free validation showing an improvement.

## 5. Baselines and why they are included

The baseline code is in:

```text
LAND_AS/baselines/models.py
LAND_AS/baselines/evaluate.py
```

All learned test baselines are pooled across stations, like LAND. This matters
because the test stations have no pre-2017 observations and therefore cannot
support legitimate per-station test models.

The flattened baseline feature matrix mirrors the neural model inputs:

- spatial mean of the 30 current-week atmospheric channels;
- center-cell values of those channels;
- mean of each local DEM channel;
- mean of each regional DEM channel;
- month one-hot;
- rainfall lags and lag-validity masks.

This provides a fair test of whether the neural spatial convolutions add value
beyond conventional regression on the same information.

### 5.1 `pooled_mean`

Predicts the mean of all training weeks for every test sample.

Purpose:

- lower-bound benchmark;
- tests whether a constant climatological mean is already adequate;
- reveals whether other models explain variance around a common climatology.

It is important because rainfall forecasting can appear useful while doing no
better than a global mean.

### 5.2 `month_climatology`

Predicts the pooled month-of-year mean across all training stations.

Purpose:

- captures seasonal rainfall cycle;
- has no station-specific or atmospheric information;
- separates seasonal climatology from dynamical prediction.

This is the simplest realistic seasonal forecast reference.

### 5.3 `persistence`

Predicts the previous observed weekly rainfall total. If the previous week is
missing, it falls back to the pooled training mean.

Purpose:

- tests short-term autocorrelation;
- is highly relevant for weekly prediction;
- does not use atmosphere or terrain directly.

Persistence is a standard forecast baseline but is weak here because the test
is about spatial transfer and the target period occurs years after fitting.

### 5.4 `ridge`

A standardized-feature linear regression trained on all pre-test training
samples.

Purpose:

- strong linear benchmark;
- tests additive effects of atmosphere, terrain, seasonality, and antecedent
  rainfall;
- shows whether complicated nonlinear spatial features are necessary.

Ridge is especially important in this problem because it performs strongly on
the held-out test stations.

### 5.5 `tweedie_glm`

A pooled log-linked Tweedie generalized linear model with `power=2`, chosen as
an approximately Gamma-like positive continuous regression model.

Purpose:

- closer statistical analog to the Gamma neural model;
- retains a skew-aware distributional assumption;
- is still linear in the learned feature space.

It is a useful intermediate between ordinary linear regression and the Gamma
LAND architecture.

### 5.6 `gbm`

A pooled `HistGradientBoostingRegressor` trained on the same flattened inputs.

Purpose:

- nonlinear tabular benchmark;
- captures interactions and thresholds without requiring learned spatial
  feature extractors;
- represents a pragmatic operational alternative to a neural model.

It is important because GBM is competitive with the neural ensemble while being
simpler to train and interpret.

### 5.7 `station_climatology`

This baseline is only evaluated inside LOSO folds and is optimistic by design:
it computes each validation station's month-of-year means from the same weeks
it predicts.

Purpose:

- approximates the value of knowing a station's own climatology;
- provides a spatial-holdout reference rather than a legitimate test model;
- helps quantify whether the neural model is learning station identity versus
  transferable spatial relationships.

It is saved in `fold_metrics.json` but is not a valid final test baseline.

## 6. Baseline output files

Regenerate all baseline metrics and include every evaluated run/blend with:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.baselines.evaluate `
  --all-runs `
  --folds
```

Outputs:

```text
LAND_AS/output/baselines/
├── test_metrics.json              # pooled test-set baselines
├── test_metrics_by_station.json   # strict JSON: baselines + runs/blends by station
├── test_metrics_by_station.csv    # long-form table for Excel/pandas
├── test_metrics_by_station.md     # stations x models metric tables
├── test_predictions.npz           # baseline predictions only
├── model_metrics.json             # overall metrics for runs and blends
├── model_predictions.npz          # aligned predictions for runs and blends
└── fold_metrics.json              # optimistic LOSO station climatology
```

`test_metrics_by_station.json` is the main apples-to-apples comparison file. It
uses the same observed values, station labels, and metric function for every
model.

## 7. v5 tuning

The retained Optuna study is:

```text
LAND_AS/output/tuning/weekly_land_v5/study.db
```

Best trial: trial 5, objective value approximately `1560.18`. The study used
spatial cross-validation and minimized validation MSE. Selected parameters:

```json
{
  "climate_lag_weeks": 0,
  "rain_lag_weeks": 2,
  "climate_multiplier": 13,
  "dem_units": 64,
  "month_units": 64,
  "hidden_units": 320,
  "dropout": 0.30000000000000004,
  "batch_size": 128,
  "learning_rate": 0.00011476582119489201,
  "weight_decay": 0.000187422109855557,
  "dem_size": 10,
  "local_dem_cfg": 4,
  "regional_dem_cfg": 9,
  "climate_patch": 3,
  "rainfall_weight": true
}
```

`climate_multiplier=13` expands to `climate_units=390` because the model has 30
current-week atmospheric channels and no climate lag. The persisted run
hyperparameters already contain `climate_units=390`.

Inspect the study with `LAND_AS/notebooks/02_tuning_eda.ipynb`.

### New controlled tuning rounds

`LAND_AS.tune` now supports the `Daily_Modeling`-inspired controls that matter
most here: spatial versus temporal validation folds, mean/median fold
aggregation, Gamma versus scalar Huber heads, weighted losses, Huber delta, and
station-balanced sampling.

A fast spatial screen, using the same three-station-group style as the retained
v5 study:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.tune `
  --study weekly_land_v6_gamma_kfold_mse `
  --model-type gamma `
  --rainfall-weight `
  --opt-metric mse `
  --cv-mode kfold --folds 3 --fold-agg median `
  --search-space core `
  --trials 40 --epochs 500 --patience 50 --min-epochs 30
```

A temporal screen for the current weighted-Huber loss:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.tune `
  --study weekly_land_v6_huber_rw_temporal_mse `
  --model-type huber `
  --loss-type huber_weighted `
  --huber-delta 0.5 `
  --opt-metric mse `
  --cv-mode temporal --folds 3 --fold-agg median `
  --search-space core `
  --trials 40 --epochs 500 --patience 50 --min-epochs 30
```

After a study finishes, train its selected configuration under the standard
21-fold LOSO protocol:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train `
  --study weekly_land_v6_huber_rw_temporal_mse `
  --trial 36 `
  --run weekly_land_v6_huber_rw_t36 `
  --seeds 3 --epochs 500 --patience 50 --workers 4
```

The completed v6 studies are:

| Study | CV mode | Raw winner | Fold-normalized winner | Trained run |
|---|---|---:|---:|---|
| `weekly_land_v6_gamma_kfold_mse` | spatial 3-fold | trial 18 | trial 18 | `weekly_land_v6_gamma_kfold_t18` |
| `weekly_land_v6_gamma_temporal_mse` | temporal 3-fold | trial 12 | trial 17 | `weekly_land_v6_gamma_temporal_mse` (trial 12) |
| `weekly_land_v6_huber_rw_temporal_mse` | temporal 3-fold | trial 22 | trial 36 | `weekly_land_v6_huber_rw_temporal_mse` (trial 22), `weekly_land_v6_huber_rw_t36` |

`--cv-mode both` is available but expensive: it combines 21 LOSO folds with the
requested number of temporal folds. Prefer it only for finalist validation, not
broad Optuna search.

Do not compare raw Optuna objectives across `cv-mode` values. Spatial groups and
temporal blocks have different validation variance, station coverage, and
difficulty, so a temporal MSE around 3,200 is not directly worse than a spatial
MSE around 1,600. Use `--opt-metric mse_ratio` for a dimensionless score equal
to validation MSE divided by the fold's train-mean climatology MSE, or evaluate
finalists under one common validation protocol.

If a post-hoc fold-normalized ranking selects a non-default Optuna trial, train it
explicitly with `--trial N` instead of relying on the raw-objective best trial.

## 8. Training and evaluation commands

### 8.1 Original Gamma v5

The v5 run is complete and should remain frozen. Its training entry point was
`LAND_AS.train`, which delegates fold-level work to `LAND_AS.parallelize`.

Reconstructed command:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train `
  --study weekly_land_v5 `
  --run weekly_land_v5 `
  --seeds 3 `
  --epochs 500 `
  --patience 50 `
  --workers 4
```

Evaluate:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.evaluate `
  --run weekly_land_v5
```

### 8.2 Ordinary v5-sized Huber

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber `
  --loss-type huber `
  --monitor mae `
  --huber-delta 0.5 `
  --seeds 3 `
  --epochs 500 `
  --patience 50 `
  --workers 4

.\venv\Scripts\python.exe -m LAND_AS.evaluate `
  --run weekly_land_v5_huber
```

### 8.3 Rainfall-weighted Huber

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_rw `
  --loss-type huber_weighted `
  --monitor mae `
  --seeds 3 `
  --epochs 500 `
  --patience 50 `
  --workers 4

.\venv\Scripts\python.exe -m LAND_AS.evaluate `
  --run weekly_land_v5_huber_rw
```

### 8.4 Rainfall-weighted Huber with MSE monitoring

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_rw_msemon `
  --loss-type huber_weighted `
  --monitor mse `
  --seeds 3 `
  --epochs 500 `
  --patience 50 `
  --workers 4

.\venv\Scripts\python.exe -m LAND_AS.evaluate `
  --run weekly_land_v5_huber_rw_msemon
```

### 8.5 Other controlled variants

The script also supports:

```powershell
# More MSE-like Huber loss
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_d1 `
  --huber-delta 1.0 `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# Equal expected station representation in training batches
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_balanced `
  --balanced-stations `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# Resume an existing run and add two seeds
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber `
  --seeds 5 `
  --epochs 500 --patience 50 --workers 4
```

Each variant should use a new run name so prior experiments remain comparable.

## 9. Blend workflow

`LAND_AS/blend_v5_huber.py` creates held-out predictions for every LOSO fold,
averages seeds within each fold, scans a Huber weight from 0 to 1, and selects
that weight using only training-station validation predictions. The selected
weight is then applied to the already-evaluated test predictions.

Example:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.blend_v5_huber `
  --gamma-run weekly_land_v5 `
  --huber-run weekly_land_v5_huber `
  --objective mse `
  --output v5_gamma_huber_cv_mse
```

A blend output contains:

```text
selection.json
oof_metrics.json
oof_predictions.npz
oof_weight_grid.csv
test_metrics.json
test_predictions.npz
experiment.json
code_snapshot/
```

The test set is not used to select the weight.

## 10. Current experiments and results

### 10.1 Completed model runs

| Run | Change from v5 | Selection monitor |
|---|---|---|
| `weekly_land_v5` | Original Gamma NLL | v5-era MAE early stopping |
| `weekly_land_v5_huber` | Scalar softplus output + Huber (`delta=0.5`) | MAE |
| `weekly_land_v5_huber_d025` | Scalar Huber, `delta=0.25` | MAE |
| `weekly_land_v5_huber_d1` | Scalar Huber, `delta=1.0` | MAE |
| `weekly_land_v5_huber_d2` | Scalar Huber, `delta=2.0` | MAE |
| `weekly_land_v5_huber_rw` | `huber_weighted` loss | MAE |
| `weekly_land_v5_huber_rw_msemon` | `huber_weighted` loss | MSE |
| `weekly_land_v6_gamma_kfold_t18` | Retuned Gamma from spatial k-fold trial 18 | MSE |
| `weekly_land_v6_gamma_temporal_mse` | Retuned Gamma from temporal raw-MSE trial | MSE |
| `weekly_land_v6_huber_rw_temporal_mse` | Retuned weighted Huber, raw temporal best | MSE |
| `weekly_land_v6_huber_rw_t36` | Retuned weighted Huber, fold-normalized temporal trial 36 | MSE |

The seven v5 variants use the same v5 atmospheric, terrain, month, and lag
feature settings. The four v6 finalists use tuned hyperparameters but are all
trained and evaluated under the standard 21-fold LOSO protocol.

### 10.2 Overall test metrics

Metrics below evaluate each retained checkpoint with the normalization
statistics saved for that run. New runs will instead use per-checkpoint
fold-local normalization. Baselines are rebuilt on the corrected training-station
scaler.

| Model | RMSE | MAE | Bias | R2 | Spearman | 98th-pct bias | CSI >=50 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge baseline | 49.353 | 35.902 | +1.021 | 0.480 | 0.683 | -22.32% | 0.646 |
| GBM baseline | 50.365 | 34.293 | -6.641 | 0.458 | 0.671 | -23.48% | 0.632 |
| Tweedie GLM baseline | 55.322 | 35.528 | -12.044 | 0.346 | 0.724 | -14.93% | 0.660 |
| Gamma v5 | 50.829 | 36.067 | +0.101 | 0.448 | 0.661 | -23.16% | 0.642 |
| Huber (`delta=0.5`) | 50.746 | 34.970 | -6.927 | 0.450 | 0.664 | -25.55% | 0.636 |
| Huber (`delta=0.25`) | 50.695 | 35.929 | +1.266 | 0.451 | 0.665 | -25.65% | 0.628 |
| Huber (`delta=1.0`) | 50.873 | 36.152 | +0.734 | 0.447 | 0.656 | -23.85% | 0.632 |
| Huber (`delta=2.0`) | 50.838 | 36.170 | +0.562 | 0.448 | 0.657 | -25.33% | 0.636 |
| Weighted Huber | 50.495 | 35.561 | -0.653 | 0.455 | 0.666 | -24.15% | 0.632 |
| Weighted Huber + MSE monitor | 50.373 | 35.749 | +0.827 | 0.458 | 0.667 | -24.20% | 0.632 |
| v6 Gamma spatial k-fold t18 | 51.035 | 35.412 | -4.480 | 0.444 | 0.666 | -28.31% | 0.636 |
| v6 Gamma temporal | 50.702 | 35.433 | -3.091 | 0.451 | 0.676 | -28.73% | 0.652 |
| v6 weighted Huber temporal | 50.285 | 35.846 | -0.143 | 0.460 | 0.668 | -26.26% | 0.642 |
| v6 weighted Huber temporal t36 | 49.584 | 35.612 | +1.719 | 0.475 | 0.675 | -22.76% | 0.644 |
| Gamma + Huber blend | 50.353 | 35.017 | -4.594 | 0.458 | 0.667 | -26.00% | 0.638 |
| Gamma + Huber `d025` blend | 50.498 | 35.802 | +0.463 | 0.455 | 0.665 | -24.09% | 0.635 |
| Gamma + Huber `d1` blend | 50.565 | 35.849 | +0.271 | 0.454 | 0.663 | -23.61% | 0.635 |
| Gamma + Huber `d2` blend | 50.568 | 35.853 | +0.213 | 0.454 | 0.664 | -23.77% | 0.637 |
| Gamma + weighted-Huber blend | 50.521 | 35.772 | -0.073 | 0.455 | 0.665 | -23.91% | 0.638 |
| Gamma + weighted-Huber-MSE blend | 50.469 | 35.793 | +0.282 | 0.456 | 0.665 | -24.05% | 0.636 |

### 10.3 Interpretation

- `weekly_land_v5` remains the clean Gamma reference: nearly unbiased and the
  best neural model for high-end magnitude and CSI.
- Ordinary Huber improves typical MAE but develops a consistent negative bias
  and underpredicts extremes more strongly.
- The `delta=0.25`, `1.0`, and `2.0` Huber sweep confirms that the loss change
  mostly moves the bias/extreme tradeoff rather than producing a large overall
  accuracy gain.
- Rainfall-weighted Huber mostly removes that negative bias and improves RMSE,
  supporting the hypothesis that the scalar Huber objective needed more wet-week
  influence.
- MSE checkpoint selection further improves RMSE and produces the best neural
  standalone squared-error profile.
- The original Gamma/Huber blend is effectively tied with GBM for RMSE and has
  lower MAE, but remains negatively biased and weak at the 98th percentile.
- Ridge and GBM are not merely sanity checks: they are competitive alternatives
  and should remain in every comparison table.
- The tuned spatial Gamma finalist does not improve on v5 and has more negative
  bias, suggesting that the k-fold search is not producing a better transferable
  configuration.
- The tuned temporal Gamma candidate improves ranking and CSI but still has a
  material negative bias and weak upper-tail magnitude.
- The fold-normalized temporal weighted-Huber trial (`t36`) is the strongest
  neural RMSE/R2 candidate so far, though Ridge remains slightly better on both
  aggregate metrics.
- No neural model currently dominates the comparison across all metrics.

The leading candidates depend on the intended objective:

- lowest neural MAE: `v5_gamma_huber_cv_mse` or `weekly_land_v5_huber`;
- best neural RMSE/R2 balance: `weekly_land_v6_huber_rw_t36`;
- strongest neural CSI: `weekly_land_v6_gamma_temporal_mse`;
- best v5 calibration balance: `weekly_land_v5_huber_rw_msemon`;
- simplest competitive model: `ridge`;
- lowest tabular-baseline MAE: `gbm`.

## 11. Per-station differences

Overall scores hide station-level differences. Use either of these files:

```text
LAND_AS/output/baselines/test_metrics_by_station.md
LAND_AS/output/baselines/test_metrics_by_station.csv
```

The Markdown file provides a station-by-model table for each metric, while the
CSV is the complete long-form table for filtering/pivoting. The canonical
machine-readable source remains:

```text
LAND_AS/output/baselines/test_metrics_by_station.json
```

This file now contains every baseline, every evaluated run, and every blend. It
includes, for each test station:

- MSE/RMSE/MAE;
- bias;
- R2;
- Spearman correlation;
- sample count;
- observed standard deviation;
- predicted standard deviation.

Use `LAND_AS/notebooks/04_results_comparison.ipynb` to inspect station heatmaps,
per-station errors, observed/predicted variance, QQ plots, and time-series
behavior.

## 12. Low-end rainfall audit

`LAND_AS/notebooks/01_data_prep_eda.ipynb` contains a dedicated low-end
distribution diagnostic. It compares the pre-2017 training rows, post-2016 test
stations, and 734 unused post-2016 "bridge" rows for the two WRCC training
stations (`siufaga_WRCC` and `toa_ridge_WRCC`).

Key findings:

- Training has 197/6,686 exact-zero weeks (2.95%); test has 17/1,188 (1.43%).
- The post-2016 bridge group has only 2/734 exact-zero weeks (0.27%).
- The same two bridge stations had a 0.73% pre-2017 zero rate, so the same-site
  temporal shift is much smaller than the difference between the training pool
  and bridge/test periods.
- Training zeros are concentrated in a few legacy stations. `pioa_afono`,
  `vaipito2000`, `aunuu`, `fagaitua`, and `vaipito_res` contribute about 66% of
  the exact-zero training weeks.
- `aunuu` is especially heterogeneous: the legacy daily gauge has a 12.4%
  weekly zero rate, while the nearby modern `aunuu_UH` record has no observed
  zero weeks in 51 complete weeks.
- Weekly aggregation keeps only complete seven-day weeks and is not the obvious
  source of the discrepancy. The raw `vaipito2000` record contributes another
  772 complete weeks before the 1980 reanalysis start date; those are audited
  separately as `pre_1980_excluded` and are not model rows.

The implication is that the low-end discrepancy is primarily a station/source
composition issue, with a smaller temporal component, rather than an isolated
preprocessing bug. Historical daily gauges have much wider heterogeneity in
zero rates, units, record length, and reporting resolution than the modern UH
test stations. The strict split therefore asks the model to extrapolate across
both location and observing network.

The same notebook also evaluates whether a dry-week occurrence head is likely
to fix this. Leakage-free LOSO wet/dry classifiers achieve ROC AUC around 0.88,
but dry-event precision/recall remain constrained by the rare-event rate.
Probability-scaling the existing amount predictions improves OOF MAE by only
about 0.2-0.3 mm and leaves predicted dry-week rainfall near 20-27 mm. This
supports keeping Bernoulli-Gamma/hurdle models as controlled challengers rather
than treating occurrence as the dominant error source.

Promotion should therefore include low-end diagnostics, not just pooled MAE:

- predicted probability below 1, 5, and 10 mm;
- mean prediction on observed dry weeks;
- dry-event precision/recall;
- conditional bias and MAE by observed-rainfall bins;
- station/source-group sensitivity.

## 13. Historical experiments not retained

Earlier exploratory code and outputs were removed from the active tree. The
main lessons were:

- adding a daily temporal atmospheric encoder changed many variables at once
  and did not produce a defensible replacement for v5;
- full-data retraining without a validation split removed the ability to select
  an independent best epoch;
- the resulting models improved some ranking or MAE metrics but worsened bias,
  RMSE, and extreme-rainfall calibration;
- the controlled v5-sized Huber experiments were retained because they changed
  fewer assumptions and produced more interpretable comparisons.

Those deleted experiments are not represented in the current output tree and
should not be cited as active model candidates.

## 14. Reproducibility

Training, evaluation, and blending write source snapshots:

```text
<run>/code_snapshot/
<run>/evaluation/code_snapshot/
<blend>/code_snapshot/
```

Each manifest records:

- source file SHA-256 hashes;
- command line;
- Python executable/version;
- relevant package versions;
- Git commit and dirty status;
- dataset path, size, and SHA-256.

Run metadata also includes:

- `hyperparameters.json`
- `normalization.json`
- `split.json`
- `experiment.json` for v5 Huber variants
- fold histories and checkpoints

The original `weekly_land_v5` predates the snapshot system, so it has
`reproduction.json` documenting the reconstructed command and Optuna provenance.

## 15. Notebook guide

```text
LAND_AS/notebooks/
├── 01_data_prep_eda.ipynb          # raw/prepared data + low-end shift audit
├── 02_tuning_eda.ipynb             # v5 Optuna study and split diagnostics
├── 03_training_eda.ipynb           # LOSO histories and run evaluation
├── 04_results_comparison.ipynb     # baselines, runs, and blends together
└── 05_land_style_figures.ipynb     # LAND-style figures adapted to American Samoa
```

The fifth notebook adapts figure ideas from the original repository's
`results/plots.ipynb`, `main_results_I.ipynb`, `main_results_II.ipynb`,
`results_III.ipynb`, `topography_alignment.ipynb`, and `rah_comparison.ipynb`.
It uses American Samoa data rather than the Hawaii-specific map/GCM files.

## 16. Output map

```text
LAND_AS/output/
├── baselines/   # pooled baselines + aligned model comparisons
├── blends/      # leakage-free Gamma/Huber blend outputs
├── figures/     # notebook-generated diagnostic figures
├── runs/        # retained LOSO ensembles and evaluations
└── tuning/      # retained Optuna studies
```

The main entry points are:

```text
LAND_AS/prepare.py            # build feature caches and weekly NPZ
LAND_AS/data.py               # split, lag features, crops, loaders, LOSO
LAND_AS/model.py              # Gamma and Huber LAND heads/losses
LAND_AS/engine.py             # fit, predict, metrics helpers
LAND_AS/parallelize.py        # fold/seed orchestration
LAND_AS/train.py              # Gamma study/hyperparameter training
LAND_AS/train_v5_huber.py     # controlled v5 scalar-Huber experiments
LAND_AS/evaluate.py           # ensemble evaluation
LAND_AS/blend_v5_huber.py     # OOF-selected Gamma/Huber blend
LAND_AS/baselines/models.py   # baseline implementations
LAND_AS/baselines/evaluate.py # baseline and aligned-model metrics
```

`LAND_AS/next-steps.md` documents the recommended future experiments and the
validation controls required before any new candidate is promoted.

## 17. Practical decision rule

A candidate should not be promoted on one metric. Compare at least:

- RMSE and MAE;
- mean bias;
- R2 and Spearman rank correlation;
- 98th-percentile predicted/observed ratio;
- CSI above 50 mm;
- low-end rates below 1/5/10 mm and mean prediction on observed dry weeks;
- conditional error by observed-rainfall bins;
- station-level errors;
- predicted variance by station.

A neural candidate should also be checked against Ridge and GBM, because both
are strong enough in this dataset to serve as legitimate alternatives rather
than trivial baselines.
