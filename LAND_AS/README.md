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
- `cv_folds(..., mode="loso")` creates 19 training-station folds (21 before
  the rainfall QC exclusions described in section 3.1)

No test sample is used for architecture selection, objective selection,
checkpoint selection, blend-weight selection, or calibration.

## 3. Data pipeline

### 3.1 Raw inputs

`LAND_AS.prepare` calls the data builders vendored under
`LAND_AS/daily_modeling/` (copied from `Daily_Modeling` so this package is
self-contained). It expects:

- station metadata: `raw_data/AS/station_locations.csv`
- daily station rainfall CSVs: `raw_data/AS/final_rainfall_per_station/`
- daily reanalysis NetCDFs: `raw_data/AS/climate_variables_daily_1980-2024/`
- terrain raster: `raw_data/AS/DEM/10m_tutuila_3band.tif`

Raw data is located at `<repo root>/raw_data/AS` (monorepo layout) or
`LAND_AS/raw_data/AS` (standalone layout) — whichever exists.

Station metadata supplies latitude, longitude, elevation, source, and record
bounds. Rainfall CSVs are converted to millimeters when needed.

Two data-quality rules are applied at load time by
`LAND_AS.daily_modeling.data_utils.load_raw.load_daily_rainfall` (rules live
in `LAND_AS/daily_modeling/config.py`, evidence in
`eda_scripts/rainfall_*.py`):

- `aunuu` and `vaipito2000` are excluded entirely. `aunuu` reports in 0.1-inch
  increments (its minimum nonzero daily value is 2.54 mm), so drizzle days read
  as zero; `vaipito2000`'s record collapses over time (1970s-90s daily median
  of 0 and a weekly mean about half of co-located `vaipito_res`/`vaipito_UH`).
- Three `afono_UH` flat-zero runs (2022-08-14..2022-10-05,
  2022-10-18..2022-11-16, 2024-07-12..2024-08-15) are masked to missing: the
  gauge recorded exactly 0.000 for 30-53 consecutive days while neighbouring
  stations recorded 3-6 mm/day, i.e. gauge-offline stored as zero, not drought.

The assembled `weekly_dataset.npz` therefore contains 24 stations and 7,982
weekly samples. The pre-QC dataset is preserved as
`LAND_AS/data/weekly_dataset_pre_qc.npz`.

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

`LAND_AS.daily_modeling.data_utils.assemble_dataset.assemble(..., freq="weekly")` then
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

- training rows: 143 exact zeros in 6,078 samples (2.35%);
- test rows: 3 exact zeros in 1,170 samples (0.26%).

(Counts are post-QC; before QC they were 197/6,686 (2.95%) and 17/1,188
(1.43%) -- see section 12.)

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
LOSO protocol (19 folds after the rainfall QC exclusions):

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

`--cv-mode both` is available but expensive: it combines 19 LOSO folds with the
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

### v7: broad architecture search on cleaned data

The post-QC study is:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.tune `
  --study weekly_land_v7_huber_rw_temporal_broad `
  --model-type huber `
  --loss-type huber_weighted `
  --opt-metric mse_ratio `
  --cv-mode temporal --folds 3 --fold-agg median `
  --search-space broad `
  --trials 40 --epochs 500 --patience 50 --min-epochs 30
```

Why each choice:

- **`--search-space broad` is the point of the study.** All previous studies
  (v5, all three v6) searched only the `core` space: learning rate, weight
  decay, rainfall lag, and DEM crop configurations. Model width
  (`climate_units`, `dem_units`, `month_units`, `hidden_units`), `dropout`,
  `batch_size`, and `dem_size` have never been retuned since the original v5
  study. Post-QC, Ridge still beats every neural run, which suggests the v5
  architecture's capacity/regularization tradeoff — not its optimizer settings
  — is the largest unexplored source of the gap.
- **`--model-type huber --loss-type huber_weighted`**: the weighted scalar
  Huber family produced the best neural models in both the pre-QC and post-QC
  leaderboards (`t36` twice). Re-litigating the Gamma/Huber head question costs
  trials better spent on architecture. Gamma remains represented by the
  retained `weekly_land_v5_qc` run and the `_qc` blend.
- **`--cv-mode temporal`**: the test task is spatial *and* temporal transfer.
  Temporal folds within pre-2017 were the protocol that produced t36, the best
  neural model under both dataset versions; spatial k-fold produced a finalist
  that did not improve on v5. LOSO (19 folds/trial) and `both` (22 folds/trial)
  are correct but too expensive for a 40-trial search — reserve them for
  finalist validation.
- **`--opt-metric mse_ratio`**: validation MSE divided by the fold's
  train-mean climatology MSE. Fold difficulty varies enormously across
  stations/years, so raw MSE rewards lucky fold assignments; the ratio is
  dimensionless, comparable across folds and protocols, and also sets the
  checkpoint monitor to MSE, matching the squared-error target of the
  comparison.
- **`--fold-agg median`**: each trial trains one seed per fold, so individual
  fold scores are seed-noisy; the median keeps a single bad fold or seed from
  dominating a 3-fold aggregate.
- **`--trials 40 --epochs 500 --patience 50 --min-epochs 30`**: same budget and
  early-stopping contract as the v5/v6 studies so results are comparable; the
  Optuna SQLite store makes the study resumable and incremental.

Sequencing caution: `load_data()` reads `weekly_dataset.npz` at process start.
Do not run this study while the npz is swapped out for a `QC_EXCLUDE_STATIONS`
ablation (see `next-steps.md` 1.2), or the study and the ablation run will
silently tune on different data.

When the study finishes, prefer the fold-normalized ranking in `trials.csv`
over the raw-objective best trial, then train the winner under LOSO:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train `
  --study weekly_land_v7_huber_rw_temporal_broad `
  --trial <N> `
  --run weekly_land_v7_huber_rw_t<N>_qc `
  --seeds 3 --epochs 500 --patience 50 --workers 4
```

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

### 8.6 Post-QC retrains

The three leading configurations were retrained on the cleaned dataset after
the QC rebuild (section 3.1). Pre-QC `output/baselines/` metrics were first
copied to `output/baselines_pre_qc/` (the evaluation rewrites
`model_metrics.json`), and `LAND_AS.baselines.evaluate --folds` was rerun so
the baselines refit on the cleaned training rows.

```powershell
# Gamma v5 (study best trial, equivalent to the retained v5 config)
.\venv\Scripts\python.exe -m LAND_AS.train `
  --study weekly_land_v5 --run weekly_land_v5_qc `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# Weighted Huber + MSE monitor (v5-sized)
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_rw_msemon_qc `
  --loss-type huber_weighted --monitor mse `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# v6 temporal-study trial 36 (best neural configuration)
.\venv\Scripts\python.exe -m LAND_AS.train `
  --study weekly_land_v6_huber_rw_temporal_mse --trial 36 `
  --run weekly_land_v6_huber_rw_t36_qc `
  --seeds 3 --epochs 500 --patience 50 --workers 4

.\venv\Scripts\python.exe -m LAND_AS.evaluate --run weekly_land_v5_qc
.\venv\Scripts\python.exe -m LAND_AS.evaluate --run weekly_land_v5_huber_rw_msemon_qc
.\venv\Scripts\python.exe -m LAND_AS.evaluate --run weekly_land_v6_huber_rw_t36_qc
```

Each `_qc` run skips existing checkpoints, so never reuse a pre-QC run name for
a post-QC retrain — the fold count and row alignment differ.

## 9. Blend workflow

`LAND_AS/blend_v5_huber.py` creates held-out predictions for every LOSO fold,
averages seeds within each fold, scans a Huber weight from 0 to 1, and selects
that weight using only training-station validation predictions. The selected
weight is then applied to the already-evaluated test predictions.

Example (post-QC members):

```powershell
.\venv\Scripts\python.exe -m LAND_AS.blend_v5_huber `
  --gamma-run weekly_land_v5_qc `
  --huber-run weekly_land_v5_huber_rw_msemon_qc `
  --objective mse `
  --output v5_qc_gamma_huber_cv_mse
```

Blend members must be trained on the same dataset version: the OOF fold count
and row alignment are checked at blend time, so mixing pre- and post-QC runs
fails or silently produces misaligned predictions.

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
trained and evaluated under the standard LOSO protocol.

All runs listed above predate the rainfall QC (section 3.1): they were trained
on the 21-train-station / 6,686-week dataset and evaluated on the 1,188-week
test set that still contained the `afono_UH` offline-as-zero weeks. The three
leading configurations were retrained on the cleaned dataset with `_qc` run
names (19 training folds, 6,078 training weeks, 1,170 test weeks):

| Run | Change from v5 | Selection monitor |
|---|---|---|
| `weekly_land_v5_qc` | Original Gamma NLL on cleaned data | v5-era MAE early stopping |
| `weekly_land_v5_huber_rw_msemon_qc` | `huber_weighted` loss on cleaned data | MSE |
| `weekly_land_v6_huber_rw_t36_qc` | v6 temporal-study trial 36 config on cleaned data | MSE |

The post-QC Gamma+Huber blend is `v5_qc_gamma_huber_cv_mse` (OOF-selected Huber
weight 0.288, i.e. mostly Gamma). Do not mix pre- and post-QC checkpoints,
predictions, or metrics in comparisons: normalization statistics, lag-week
alignment, and test observations all differ.

### 10.2 Overall test metrics

Metrics below evaluate each retained checkpoint with the normalization
statistics saved for that run. New runs will instead use per-checkpoint
fold-local normalization. Baselines are rebuilt on the corrected training-station
scaler.

#### Post-QC test metrics (current dataset)

These metrics were computed on the cleaned 1,170-week test set by models
trained on the cleaned 19-station / 6,078-week training set. Baselines were
refit on the cleaned training rows.

| Model | RMSE | MAE | Bias | R2 | Spearman | 98th-pct bias | CSI >=50 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge baseline | 48.368 | 34.714 | -2.067 | 0.500 | 0.702 | -22.95% | 0.662 |
| GBM baseline | 48.811 | 33.502 | -4.675 | 0.490 | 0.709 | -27.66% | 0.648 |
| Tweedie GLM baseline | 55.217 | 35.398 | -11.494 | 0.348 | 0.727 | -13.96% | 0.654 |
| Persistence | 91.254 | 65.143 | +0.020 | -0.781 | 0.109 | -0.33% | 0.397 |
| Pooled mean | 68.402 | 51.546 | +1.738 | -0.001 | - | -70.56% | 0.541 |
| Month climatology | 67.223 | 50.408 | +1.964 | 0.034 | 0.211 | -63.88% | 0.541 |
| Gamma v5 (`_qc`) | 50.589 | 35.910 | +0.764 | 0.453 | 0.672 | -20.08% | 0.653 |
| Weighted Huber + MSE monitor (`_qc`) | 50.618 | 36.097 | +1.760 | 0.452 | 0.668 | -22.36% | 0.639 |
| v6 weighted Huber temporal t36 (`_qc`) | 49.317 | 35.457 | +2.539 | 0.480 | 0.686 | -20.37% | 0.659 |
| Gamma + weighted-Huber blend (`_qc`) | 50.300 | 35.707 | +1.051 | 0.459 | 0.674 | -21.32% | 0.649 |

#### Pre-QC test metrics (historical)

The table below was computed on the pre-QC test set (1,188 weeks, including the
`afono_UH` offline-as-zero weeks) by models trained on the pre-QC training set.
Keep it for relative comparisons between the old runs only; it is not
comparable with the post-QC table because both the models and the test rows
differ.

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

Post-QC interpretation (cleaned dataset):

- The QC did not close the neural-vs-baseline gap; it widened it. Every model
  improved because ~14 fake-dry `afono_UH` weeks that all models badly missed
  were removed, but Ridge gained more than the neural models (49.35 to 48.37
  RMSE versus t36's 49.58 to 49.32). Ridge now leads the best neural model by
  ~0.95 mm RMSE and leads on MAE, R2, and Spearman too.
- `weekly_land_v6_huber_rw_t36_qc` remains the best neural RMSE/R2 model and is
  nearly unbiased relative to the other neural runs; it also has the best
  neural 98th-percentile bias and CSI.
- The `_qc` Gamma/Huber blend selected a Huber weight of 0.288 (mostly Gamma)
  on OOF, versus the pre-QC blend family that leaned more evenly; the blend is
  still mid-pack and does not beat its members decisively.
- GBM retains the lowest MAE (33.50) but the largest negative bias (-4.68);
  Ridge's bias shrank post-QC (-2.07). Neural models are now slightly
  positively biased (+0.76 to +2.54).
- Conditional on the cleaned data, the ordering did not change: weighted-Huber
  formulations still dominate plain Huber/Gamma among neural runs, temporal-CV
  tuning still beats spatial k-fold tuning, and tabular baselines still beat
  every neural model on aggregate metrics. This strengthens the conclusion that
  the remaining gap is about what the neural spatial features add over pooled
  tabular regression, not about dirty targets.

Pre-QC interpretation (kept because the mechanism lessons still hold):

- `weekly_land_v5` was the clean Gamma reference: nearly unbiased and the best
  neural model for high-end magnitude and CSI.
- Ordinary Huber improved typical MAE but developed a consistent negative bias
  and underpredicted extremes more strongly.
- The `delta=0.25`, `1.0`, and `2.0` Huber sweep confirmed that the loss change
  mostly moves the bias/extreme tradeoff rather than producing a large overall
  accuracy gain.
- Rainfall-weighted Huber mostly removed that negative bias and improved RMSE,
  supporting the hypothesis that the scalar Huber objective needed more
  wet-week influence.
- MSE checkpoint selection further improved RMSE and produced the best neural
  standalone squared-error profile.
- The tuned spatial Gamma finalist did not improve on v5 and had more negative
  bias, suggesting that the k-fold search was not producing a better
  transferable configuration; the temporal studies produced the strongest
  candidates (t36), which is why temporal CV is the default for new studies.

The leading candidates on the cleaned dataset depend on the intended objective:

- best overall RMSE/R2: `ridge` (48.37 / 0.500);
- best neural RMSE/R2: `weekly_land_v6_huber_rw_t36_qc` (49.32 / 0.480);
- lowest MAE: `gbm` (33.50);
- best neural CSI: `weekly_land_v6_huber_rw_t36_qc` (0.659);
- best neural calibration (bias, 98th-pct): `weekly_land_v5_qc`.

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

Key findings (pre-QC numbers; see below for the QC outcome):

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
  zero weeks in 51 complete weeks. The deeper audit in
  `eda_scripts/rainfall_npz_and_artifact_checks.py` shows why: `aunuu` reports
  in 0.1-inch increments (minimum nonzero daily value = 2.54 mm), so any day
  with less than ~1.3 mm of rain reads as zero.
- `vaipito2000` shows a record collapse: 0% zero days and a ~30 mm/day mean in
  1958-60 (implausible), then daily medians of exactly 0 with 51-70% zero days
  through the 1970s-90s, and a weekly mean (~40 mm) about half of co-located
  `vaipito_res`/`vaipito_UH` (~82 mm).
- Nearly all test-set zeros are an `afono_UH` artifact: 14 of its 15 zero weeks
  fall inside three flat-zero runs of 30-53 consecutive days during which
  neighbouring gauges recorded 3-6 mm/day and shared none of the "dry" weeks.
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

QC actions applied (see section 3.1): `aunuu` and `vaipito2000` removed, and the
three `afono_UH` flat-zero runs masked to missing. On the rebuilt dataset:
training 143/6,078 exact-zero weeks (2.35%), test 3/1,170 (0.26%). The residual
train excess lives mostly in `pioa_afono`, `fagaitua`, `vaipito_res`, `satala`,
`aasufou80`, and `malaeimi_1691` (~90% of remaining zeros), whose zeros largely
co-occur with neighbouring stations and look real but are inflated by 0.01-inch
reporting floors; whether to remove more is the station-sensitivity question in
`next-steps.md` section 1.2.

`pioa_afono` is the lead ablation candidate and deserves the most scrutiny:

- it contributes the most remaining zero weeks (34) at a ~51% daily zero rate;
- it shows the strongest *multi-day accumulation* signature in the network:
  the first wet day after a run of zeros exceeds 2x the median wet day 32% of
  the time and 4x 18% of the time — the pattern produced when a gauge goes
  un-read for days and its stored water is dumped into a single later reading.
  Some of its "dry weeks" are therefore likely missed-observation bookkeeping
  rather than real drought;
- but its zeros co-occur with neighbours at rates far above independence and
  its weekly means are not biased low (unlike `vaipito2000`), so many of its
  dry weeks are real — which is why it is tested as a controlled exclusion
  (`QC_EXCLUDE_STATIONS` + rebuild, 666 weeks / one LOSO fold lost) rather than
  removed outright.

The ablation has now been run (`weekly_land_v6_huber_rw_t36_qc_nopioa_v2`,
same t36 config, 18 folds): test RMSE worsened 49.32 to 50.02, MAE 35.46 to
35.77, and R2 fell 0.480 to 0.465 — far above the ~0.01 mm run-to-run noise
measured by an accidental identical-config rerun. Excluding `pioa_afono` loses
more signal than artifact, so it is retained. The conclusion generalizes
cautiously to the other co-occurring legacy gauges: further exclusions should
be individually justified, not batched.

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
├── baselines/        # pooled baselines + aligned model comparisons (post-QC)
├── baselines_pre_qc/ # preserved pre-QC baseline metrics (historical)
├── blends/           # leakage-free Gamma/Huber blend outputs
├── figures/          # notebook-generated diagnostic figures
├── runs/             # retained LOSO ensembles and evaluations
└── tuning/           # retained Optuna studies
```

The main entry points are:

```text
LAND_AS/prepare.py            # build feature caches and weekly NPZ
LAND_AS/daily_modeling/       # vendored Daily_Modeling data builders + QC config
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
