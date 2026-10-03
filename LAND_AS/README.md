# LAND_AS: Weekly Rainfall Downscaling for American Samoa

`LAND_AS` adapts the Location-Agnostic Neural Downscaler (LAND) from
`LocationAgnosticNeuralDownscaling` to station rainfall in American Samoa.
The package supports three output heads on the same architecture:

1. `gamma`: the original-style distributional LAND head (Gamma NLL).
2. `huber`: a scalar model with the same default architecture that predicts
   positive weekly rainfall, trained with Huber or rainfall-weighted Huber loss.
3. `bern_gamma`: a Bernoulli occurrence + Gamma amount hurdle head, used for the
   zero-inflated daily dataset (section 8.7).

The package is organized by pipeline stage: `s1_prepare` (raw data to dataset
NPZ), `s2_dataset` (splits, lags, normalization, CV folds, loaders),
`s3_model` (architectures, losses, metrics), `s4_train` (Optuna tuning and
LOSO ensemble training), and `s5_evaluate` (run evaluation and baselines).

The purpose of this directory is to determine whether location-agnostic spatial feature
learning is useful for a very small, spatially heterogeneous rainfall problem, and to compare
it fairly against simpler climatological and tabular baselines.

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
- Test stations: `aasu_UH`, `aunuu_UH`, `poloa_UH`, `vaipito_UH`.
- Test years: `year > 2016`.
- Validation: leave-one-station-out over training stations only.

Thus test predictions require generalization to both unseen locations and a
later climate period.

The split is implemented in `LAND_AS/config.py` and `LAND_AS/s2_dataset/data.py`:

- `TRAIN_YEAR_END = 2016`
- `TEST_STATIONS = ["aasu_UH", "aunuu_UH", "poloa_UH", "vaipito_UH"]`
- `_station_roles()` enforces station-role separation
- `_split()` applies the year cutoff
- `cv_folds(..., mode="loso")` creates one fold per training station on the
  current dataset (23 stations load after the `aunuu`/`afono_UH` exclusions;
  the vaipito2000 windowing and GML_SMO addition are in section 3.1)
- `cv_folds(..., mode="loso_recent")` is a stricter LOSO variant: each fold's
  validation is restricted to the held-out station's years >=
  `config.LOSO_RECENT_YEAR_START` (2010), so early stopping measures
  unseen-station *and* late-era generalization, matching the test protocol.
  Train indices are identical to `loso`.

No test sample is used for architecture selection, objective selection,
checkpoint selection, or calibration.

## 3. Data pipeline

### 3.1 Raw inputs

`LAND_AS.s1_prepare.prepare` runs the data builders in `LAND_AS/s1_prepare/`
(`load_raw.py` → `build_features.py` → `assemble_dataset.py`). It expects:

- station metadata: `raw_data/AS/station_locations.csv`
- daily station rainfall CSVs: `raw_data/AS/final_rainfall_per_station/`
- daily reanalysis NetCDFs: `raw_data/AS/climate_variables_daily_1980-2024/`
- terrain raster: `raw_data/AS/DEM/10m_tutuila_3band.tif`

Raw data is located at `<repo root>/raw_data/AS` (monorepo layout) or
`LAND_AS/raw_data/AS` (standalone layout) — whichever exists.

Station metadata supplies latitude, longitude, elevation, source, and record
bounds. Rainfall CSVs are converted to millimeters when needed.

Two station-cleaning rules are applied at load time, before any tuning or
training, by `LAND_AS.s1_prepare.load_raw.load_daily_rainfall`. The rules and
the evidence behind them live in `LAND_AS/s1_prepare/config.py`
(`QC_EXCLUDE_STATIONS`, `QC_VALID_DATE_RANGES`, `QC_MASK_DATE_RANGES`). The
supporting numbers and figures are reproduced by
`notebooks/00_qc_evidence.ipynb` (figures saved to
`output/figures/qc_evidence/`):

- `aunuu` and `afono_UH` are excluded entirely (`load_raw.py` returns `None`
  for names in `QC_EXCLUDE_STATIONS`). `aunuu` reports strictly in 0.1-inch
  increments: its minimum nonzero daily value is 2.54 mm and 100% of nonzero
  values are exact 0.1-inch multiples, so drizzle days read as zero (59% of
  days, 12.4% of weeks) -- and the same site is covered by the modern
  `aunuu_UH` gauge (0.254 mm resolution), so the exclusion loses no unique
  location. `afono_UH` recorded exactly 0.000 for 30-53 consecutive days in
  three runs (2022-08-14..10-05, 2022-10-18..11-16, 2024-07-12..08-15) while
  its nearest gauges with data (toa_ridge_WRCC, vaipito_UH, siufaga_WRCC,
  GML_SMO) recorded real rainfall and shared none of those "dry" spells --
  gauge-offline stored as zero. The record is inconsistent enough that the
  station is dropped outright rather than masked; since `afono_UH` was a
  held-out *test* station, this shrinks the test set to four stations
  (`aasu_UH`, `aunuu_UH`, `poloa_UH`, `vaipito_UH`), it does not touch the
  training pool.
- `vaipito2000` is **partially** retained via `QC_VALID_DATE_RANGES`, keeping
  only 1976-01-01..1989-09-30 (4,374 daily rows). Rows before 1976 are
  dropped because the early era (~17 mm/day mean in 1958-69, vs 5.7 in
  1970-91) is uncorroborated -- no other station has pre-1970 data to check
  it against. Rows after 1989-09-30 are dropped because where `vaipito2000`
  overlaps `vaipito_res` (1989-90) the records are identical (weekly
  r = 1.000) -- the same site reported twice, so the overlap belongs to
  `vaipito_res`. See `notebooks/00_qc_evidence.ipynb` §2 for the
  corroboration scan, cadence/quantization forensics, and overlap test.

`pioa_afono` was audited as a third exclusion candidate and **retained**: a
controlled ablation run showed removing it costs more signal than artifact
(see section 11).

`GML_SMO` (NOAA GML Samoa Observatory, Cape Matatula; -14.2474, -170.5644,
42 masl, daily inches, 1976-2024) was added after those cleaning rules were
established. The station had
a valid `GML_SMO.csv` but was invisible to the pipeline because station
discovery is metadata-driven: `load_all_station_rainfall` iterates over
`station_locations.csv`, and no `GML_SMO` row existed. After adding the row:

- weekly comparison against the co-located `matatula` gauge over 46
  overlapping complete weeks (2001-2002): Pearson r = 0.989, mean absolute
  difference 4.8 mm/week on ~48 mm means -- consistent and suitable;
- `raw_data/AS/fill_gml_smo_missing.py` audited the NOAA hourly files
  (`raw_data/missing_rainfall/`, field 14 = precipitation intensity, -99 =
  missing) and found zero recoverable days: every day with all 24 valid hours
  was already present, and 2019-2020 are genuine full-year outages. The
  original file is preserved as `GML_SMO_OLD.csv`.

The assembled `weekly_dataset.npz` therefore contains 25 stations and 9,445
weekly samples (GML_SMO contributes 1,463 station-weeks; it is a train-pool
station, not a test station, so it adds a 20th LOSO fold). Note GML_SMO is the
driest train station (37.7 mm/week mean vs 47-104 elsewhere) and ~16% of
training rows; `--balanced-stations` exists partly to counteract that weight.

The previous dataset snapshot is preserved as
`LAND_AS/data/weekly_dataset_OLD.npz`. Passing `--daily` to the same
command instead writes `data/daily_dataset.npz`: one sample per station-day
(69,384 rows), 15 current-day climate channels (no within-week std block),
and no aggregation.

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

`LAND_AS.s1_prepare.assemble_dataset.assemble(..., freq="weekly")` then
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
.\venv\Scripts\python.exe -m LAND_AS.s1_prepare.prepare `
  --start-date 1980-01-01 `
  --end-date 2024-12-31 `
  --rebuild
```

### 3.4 Runtime lag and normalization

`LAND_AS.s2_dataset.data.load_data()` adds lag information at load time.

- `LAG_WEEKS = 3` prior weeks are materialized in the raw lag array.
- For each lag, the model receives prior observed rainfall and a validity flag.
- Missing lag weeks are zero-filled and flagged rather than treated as observed
  dry weeks.
- Reanalysis lag blocks are also materialized. `climate_lag_weeks` (0-3,
  tuned) selects how many prior-week blocks are appended to the conv input
  channels; `rain_lag_weeks` (0-3, tuned) selects how many rainfall lags
  remain in the runtime lag vector. The two depths are sampled independently
  by Optuna, so mixed combinations like climate=1/rain=0 are covered; the
  only coupling is that `use_lag=0` forces `rain_lag_weeks=0`.

The default bundle uses only the pre-2017 training split for normalization:

- atmospheric channels: per-channel train mean/std;
- DEM channels: land-pixel mean/std;
- target: standard deviation of training rainfall.

For cross-validation, `normalized_bundle()` rebuilds these statistics using each
fold's own training rows. This fold-local normalization prevents a temporal
or LOSO validation fold from influencing
its own feature scaling. New checkpoints write `seed_<N>_normalization.json`;
evaluation uses that per-checkpoint marker to select fold-local
scaling. Older checkpoints without the marker retain their original
global-training normalization for backward compatibility.

The run-level `normalization.json` and `split.json` record the all-training-row
settings for reproduction.

## 4. LAND_AS architecture

The implementation is in `LAND_AS/s3_model/model.py`.

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

`model_type: "huber"` preserves every branch and dimension of the Gamma model
but changes the output to one scalar transformed by softplus. Training uses
Huber loss on the normalized target.

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

(Counts measured on the cleaned dataset; before cleaning they were
197/6,686 (2.95%) and 17/1,188 (1.43%) -- see section 11.)

The current Gamma loss already excludes those rare dry weeks from the amount fit.
A Bernoulli occurrence head would therefore receive sparse weekly supervision and
add another output, threshold, and calibration decision for a phenomenon that is
not currently the dominant error source. The package therefore applies the
same distinction internally: ordinary Gamma is the weekly default and weekly
tuning favored it over Bernoulli-Gamma.
Bernoulli-Gamma remains a valid controlled challenger, but it should not replace
Gamma without leakage-free validation showing an improvement. On daily data, about 43% of training days are exactly dry, so
`model_type: "bern_gamma"` is implemented for that comparison.

### 4.7 Architecture ablation switches

Three hyperparameters toggle structural choices without a separate model class
(all searchable in `--search-space broad`):

- `lightweight` (0/1): a slimmer build -- single linear
  stage in the DEM and month branches with `dropout/2` inside each branch, and
  a single-hidden-layer fusion head. Default 0 keeps the deeper branches.
- `use_lag` (0/1): 0 sets `rain_lag_weeks=0`, dropping the antecedent-rainfall
  input entirely (the lag vector becomes width-0). Tests whether the lag branch
  earns its parameters or just leaks station identity.
- `dem_elev_only` (0/1): 1 sets `dem_channels=1`, slicing the DEM inputs to the
  elevation channel only (dropping slope, sin/cos aspect) in
  `RainDataset`; the DEM conv input narrows accordingly.

## 5. Baselines and why they are included

The baseline code is in:

```text
LAND_AS/s5_evaluate/baselines/models.py
LAND_AS/s5_evaluate/baselines/evaluate.py
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

Inputs: training targets only; no features.

Purpose:

- lower-bound benchmark;
- tests whether a constant climatological mean is already adequate;
- reveals whether other models explain variance around a common climatology.

### 5.2 `month_climatology`

Predicts the pooled month-of-year mean across all training stations.

Inputs: month-of-year of each sample + training targets grouped by month.

Purpose:

- captures seasonal rainfall cycle;
- has no station-specific or atmospheric information;
- separates seasonal climatology from dynamical prediction.

This is the simplest realistic seasonal forecast reference.

### 5.3 `persistence`

Predicts the previous observed weekly rainfall total. If the previous week is
missing, it falls back to the pooled training mean.

Inputs: the lag-1 observed rainfall value and its validity flag from the lag
vector; nothing else -- no atmosphere, terrain, or seasonality. (The lag
vector holds `rain_lag_weeks` prior values followed by their validity flags;
persistence reads the first of each.)

Purpose:

- tests short-term autocorrelation;
- is highly relevant for weekly prediction;
- does not use atmosphere or terrain directly.

Persistence is a standard forecast baseline but is weak here because the test
is about spatial transfer and the target period occurs years after fitting.

### 5.4 `ols`

A standardized-feature linear regression trained on all pre-test training
samples (`StandardScaler` + `sklearn.LinearRegression`, i.e. ordinary least
squares, no L2 penalty).

Inputs: the flattened feature vector above -- patch-mean and center-cell
values of the 30 current-week climate channels, means of the 4 local and 4
regional DEM channels, month one-hot, and rainfall lags + validity flags.

Purpose:

- strong linear benchmark;
- tests additive effects of atmosphere, terrain, seasonality, and antecedent
  rainfall;
- shows whether complicated nonlinear spatial features are necessary.

OLS is especially important in this problem because it performs strongly on
the held-out test stations.

### 5.5 `tweedie_glm`

A pooled log-linked Tweedie generalized linear model with `power=2`, chosen as
an approximately Gamma-like positive continuous regression model.

Inputs: same flattened feature vector as `ols`; predictions are clipped at
zero and training targets floored at 0.01 mm for the log link.

Purpose:

- closer statistical analog to the Gamma neural model;
- retains a skew-aware distributional assumption;
- is still linear in the learned feature space.

It is a useful intermediate between ordinary linear regression and the Gamma
LAND architecture.

### 5.6 `gbm`

A pooled `HistGradientBoostingRegressor` (400 trees, learning rate 0.05)
trained on the same flattened inputs.

Inputs: same flattened feature vector as `ols`.

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

Inputs: month-of-year + the held-out station's own target values (in-sample).

Purpose:

- approximates the value of knowing a station's own climatology;
- provides a spatial-holdout reference rather than a legitimate test model;
- helps quantify whether the neural model is learning station identity versus
  transferable spatial relationships.

It is saved in `fold_metrics.json` but is not a valid final test baseline.

## 6. Baseline output files

Regenerate all baseline metrics and include every evaluated run with:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.s5_evaluate.baselines.evaluate `
  --all-runs `
  --folds
```

Outputs:

```text
LAND_AS/output/baselines/
├── test_metrics.json              # pooled test-set baselines
├── test_metrics_by_station.json   # strict JSON: baselines + runs by station
├── test_metrics_by_station.csv    # long-form table for Excel/pandas
├── test_metrics_by_station.md     # stations x models metric tables
├── test_predictions.npz           # baseline predictions only
├── model_metrics.json             # overall metrics for runs
├── model_predictions.npz          # aligned predictions for runs
└── fold_metrics.json              # optimistic LOSO station climatology
```

`test_metrics_by_station.json` is the main apples-to-apples comparison file. It
uses the same observed values, station labels, and metric function for every
model.

## 7. Tuning

Hyperparameter search lives in `LAND_AS.s4_train.tune` (Optuna, SQLite-backed
so studies are resumable). The current weekly study is:

```text
LAND_AS/output/tuning/weekly_land_v3_huber_rw_temporal_broad/
```

It was run on the post-cleaning dataset (23 stations, `afono_UH`/`aunuu`
excluded, `vaipito2000` windowed, `GML_SMO` included) with `--search-space
broad`. Best trial: 34, `mse_ratio = 0.529`. Selected parameters:

```json
{
  "climate_lag_weeks": 0,
  "rain_lag_weeks": 0,
  "climate_multiplier": 15,
  "dem_units": 32,
  "month_units": 16,
  "hidden_units": 128,
  "dropout": 0.16065140657974433,
  "batch_size": 512,
  "learning_rate": 0.0006417206952582433,
  "weight_decay": 0.0003402795493710169,
  "dem_size": 7,
  "local_dem_cfg": 1,
  "regional_dem_cfg": 4,
  "lightweight": 1,
  "use_lag": 1,
  "dem_elev_only": 0
}
```

Note the ablation switches landed on the lighter end: `lightweight=1` and
`rain_lag_weeks=0` (`use_lag=1` leaves the switch on but the sampled depth is
zero), so the winning config is effectively a current-week atmosphere +
terrain model with no antecedent-rainfall input.

The `--search-space broad` flag searches model width (`climate_multiplier`,
`dem_units`, `month_units`, `hidden_units`), `dropout` over `[0.1, 0.7]`,
`batch_size` over `[64 … 2048]`, optimizer settings, DEM crop configs
(`local_dem_cfg`/`regional_dem_cfg` index `DEM_LOCAL_CANDIDATES` /
`DEM_REGIONAL_CANDIDATES` in `s1_prepare/config.py`), and the three
architecture-ablation switches of section 4.7 (`lightweight`, `use_lag`,
`dem_elev_only`). `climate_lag_weeks` and `rain_lag_weeks` are sampled
independently over `0..LAG_WEEKS` — all asymmetric combinations are reachable.

Command used:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.s4_train.tune `
  --study weekly_land_v3_huber_rw_temporal_broad `
  --model-type huber --loss-type huber_weighted --huber-delta 0.5 `
  --opt-metric mse_ratio --cv-mode temporal --folds 3 --fold-agg median `
  --search-space broad --trials 40 --epochs 500 --patience 50 --min-epochs 30
```

Why these choices:

- **`--opt-metric mse_ratio`**: validation MSE divided by the fold's
  train-mean climatology MSE — dimensionless and comparable across folds of
  different difficulty, where raw MSE rewards lucky fold assignments.
- **`--cv-mode temporal`**: the test task is spatial *and* temporal transfer;
  temporal folds inside the training era approximate that. `loso` /
  `loso_recent` / `both` are correct but too expensive for a 40-trial
  search — reserve them for final training and finalist checks.
- **`--fold-agg median`**: one seed per fold, so the median keeps a single
  bad fold from dominating the aggregate.
- **`--model-type huber --loss-type huber_weighted`**: the weighted scalar
  head has been the strongest neural variant on this dataset; the Gamma head
  is covered by its own earlier studies.

### Trial selection and retraining

Do not compare raw Optuna objectives across `cv-mode` values — spatial and
temporal folds differ in difficulty. Within a study, prefer the
fold-normalized ranking over the raw-objective best:
`python -m LAND_AS.s5_evaluate.evaluate --study NAME` scores top trials by
raw objective, mean per-fold rank, and worst-fold score and prints the
composite winner. `train.py --study NAME` (no `--trial`) applies the same
robust selection automatically; pass `--trial N` to override.

```powershell
.\venv\Scripts\python.exe -m LAND_AS.s5_evaluate.evaluate --study weekly_land_v3_huber_rw_temporal_broad

.\venv\Scripts\python.exe -m LAND_AS.s4_train.train `
  --study weekly_land_v3_huber_rw_temporal_broad `
  --cv-mode loso --balanced-stations `
  --run weekly_land_v3_huber_loso_bal `
  --seeds 3 --epochs 500 --patience 50 --workers 4
```

Sequencing caution: `load_data()` reads `weekly_dataset.npz` at process
start, so never run tuning while that file is being rebuilt.

## 8. Training and evaluation commands

### 8.1 Weekly training

`LAND_AS.s4_train.train` trains one model per CV fold (LOSO by default) and
per seed, then writes an ensemble prediction. Key flags:

- `--study NAME` pulls hyperparameters from an Optuna study; without
  `--trial N` it applies the composite robust-trial selector (section 7).
- `--cv-mode {loso,loso_recent}`: `loso` holds out each training station on
  all its years; `loso_recent` holds out the same stations but validates only
  on their post-`LOSO_RECENT_YEAR_START` (2010) weeks, matching the
  unseen-station + late-era test protocol. Train indices are identical.
- `--balanced-stations` equalizes per-station sampling weight in batches.
- `--hyperparameters path/to.json` runs a fixed config merged over
  `config.DEFAULTS` (no study needed).
- `--daily` switches everything to the daily dataset and `bern_gamma` head.

Current runs:

```powershell
# LOSO, station-balanced
.\venv\Scripts\python.exe -m LAND_AS.s4_train.train `
  --study weekly_land_v3_huber_rw_temporal_broad `
  --cv-mode loso --balanced-stations `
  --run weekly_land_v3_huber_loso_bal `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# loso_recent variant (early stopping on the held-out station's recent years)
.\venv\Scripts\python.exe -m LAND_AS.s4_train.train `
  --study weekly_land_v3_huber_rw_temporal_broad `
  --cv-mode loso_recent --balanced-stations `
  --run weekly_land_v3_huber_recent_bal `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# evaluate + refit baselines on the same train rows
.\venv\Scripts\python.exe -m LAND_AS.s5_evaluate.evaluate --run <run>
.\venv\Scripts\python.exe -m LAND_AS.s5_evaluate.baselines.evaluate --all-runs --folds
```

### 8.2 Daily Bernoulli-Gamma

`--daily` switches the pipeline to `data/daily_dataset.npz`: one sample per
station-day, 15 current-day climate channels (no within-week std block), and
`config.LAG_DAYS = 7` daily lag depth, so the `climate_lag_weeks` /
`rain_lag_weeks` hyperparameters count days. Daily totals are zero-inflated
(~43% dry training days vs ~2% dry weeks), so the daily head is
`--model-type bern_gamma`: Bernoulli occurrence + Gamma amount NLL
(`bernoulli_gamma_nll` in `s3_model/model.py`)
with a fold-local `pos_weight = n_dry / n_wet` and `--lambda-bce` scaling the
occurrence term. The point prediction is `sigmoid(p) * alpha * scale`.

```powershell
# build the daily NPZ (reuses the shared feature caches)
.\venv\Scripts\python.exe -m LAND_AS.s1_prepare.prepare --daily

# tune
.\venv\Scripts\python.exe -m LAND_AS.s4_train.tune `
  --daily --study daily_land_v1_bern_gamma `
  --model-type bern_gamma --rainfall-weight `
  --opt-metric mse_ratio `
  --cv-mode temporal --folds 3 --fold-agg median `
  --search-space broad `
  --trials 40 --epochs 500 --patience 50 --min-epochs 30

# train / evaluate / baselines (train auto-picks the robust trial)
.\venv\Scripts\python.exe -m LAND_AS.s4_train.train `
  --daily --study daily_land_v1_bern_gamma --run daily_land_v1_bern_gamma `
  --seeds 3 --workers 2
.\venv\Scripts\python.exe -m LAND_AS.s5_evaluate.evaluate --daily --run daily_land_v1_bern_gamma
.\venv\Scripts\python.exe -m LAND_AS.s5_evaluate.baselines.evaluate --daily --all-runs --folds
```

Daily baselines write to `output/baselines_daily/` (CSI threshold 25 mm/day
vs 50 mm/week). Runs record `dataset_freq` in `hyperparameters.json`;
`--all-runs` only pulls runs matching the requested frequency, and
train/evaluate raise when the flag disagrees with a run's `dataset_freq`.

## 9. Current experiments and results

### 9.1 Completed model runs

All current runs are trained on the post-cleaning dataset (23 stations,
`afono_UH`/`aunuu` excluded, `vaipito2000` windowed to 1976-01-01..
1989-09-30, `GML_SMO` included) under the strict split of section 2 — 4
held-out test stations (`aasu_UH`, `aunuu_UH`, `poloa_UH`, `vaipito_UH`),
post-2016 years.

| Run | Config | CV mode | Sampling |
|---|---|---|---|
| `weekly_land_v3_huber_loso_bal` | v3 study trial-34 HPs (lightweight, no rain/climate lag) | `loso` | station-balanced |
| `weekly_land_v3_huber_loso_bal_2` | same HPs, rerun | `loso` | station-balanced |
| `weekly_land_v3_huber_recent_bal` | same HPs | `loso_recent` | station-balanced |
| `weekly_land_v3_huber_recent_bal_2` | same HPs, rerun | `loso_recent` | station-balanced |
| `weekly_land_v3_huber_recent_bal_lowlr` | same HPs, `--learning-rate 9.13e-5`, 1000 epochs | `loso_recent` | station-balanced |
| `weekly_land_v3_huber_rw_recent_broad` | raw-objective best trial (lags enabled: rain=3, climate=3; `dem_elev_only`) | `loso_recent` | station-balanced |

Baselines (`s5_evaluate/baselines/models.py`) are refit on the same training
rows: OLS, GBM, Tweedie GLM, persistence, pooled mean, month climatology.

### 9.2 Overall test metrics

Evaluated on the held-out test set (4 stations x post-2016 weeks) by
`evaluate --run` and `baselines.evaluate --all-runs`; identical observed
values, station labels, and metric function for every model.

| Model | RMSE | MAE | Bias | R2 | Spearman | 98th-pct bias | CSI >=50 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| GBM baseline | 49.175 | 33.181 | -12.23 | 0.462 | 0.707 | -26.6% | 0.637 |
| OLS baseline | 49.379 | 36.378 | +0.79 | 0.458 | 0.680 | -24.6% | 0.644 |
| `weekly_land_v3_huber_recent_bal` | 49.479 | 35.871 | +2.70 | 0.456 | 0.685 | -26.5% | 0.639 |
| `weekly_land_v3_huber_loso_bal` | 49.484 | 35.841 | +2.11 | 0.456 | 0.687 | -27.2% | 0.642 |
| `weekly_land_v3_huber_recent_bal_lowlr` | 49.568 | 36.448 | +4.05 | 0.454 | 0.679 | -23.5% | 0.655 |
| `weekly_land_v3_huber_loso_bal_2` | 49.857 | 35.998 | -0.07 | 0.447 | 0.669 | -27.7% | 0.646 |
| `weekly_land_v3_huber_recent_bal_2` | 49.865 | 36.291 | +1.89 | 0.447 | 0.674 | -25.0% | 0.654 |
| `weekly_land_v3_huber_rw_recent_broad` | 50.941 | 34.834 | -9.29 | 0.423 | 0.699 | -32.7% | 0.650 |
| Tweedie GLM baseline | 59.391 | 37.166 | -15.40 | 0.216 | 0.701 | -22.8% | 0.617 |
| Month climatology | 66.461 | 48.653 | -6.02 | 0.018 | 0.182 | -66.4% | 0.538 |
| Pooled mean | 67.341 | 49.816 | -6.11 | -0.008 | - | -73.5% | 0.538 |
| Persistence | 88.379 | 64.313 | -0.13 | -0.737 | 0.122 | -0.6% | 0.396 |

Per-station RMSE (`output/baselines/test_metrics_by_station.md`):

| Station | OLS | GBM | LAND range (5 runs) |
|---|---:|---:|---:|
| `aasu_UH` | **49.58** | 52.92 | 52.82–54.16 |
| `aunuu_UH` | **34.10** | 37.80 | 34.16–39.34 |
| `poloa_UH` | 50.00 | 50.78 | **49.16**–49.46 |
| `vaipito_UH` | 50.89 | **46.34** | 48.29–50.14 |

LAND leads only on `poloa_UH` (by <1 mm). The spread on `aunuu_UH`
(34.2–39.3 mm across identically configured runs) is the largest
seed-sensitivity in the table: that station has only ~51 test weeks, so a
single seed can move its RMSE by 5 mm.

### 9.3 Interpretation

- **The neural model does not beat tuned tabular baselines.** GBM, OLS, and
  the best LAND ensemble sit within ~0.7 mm RMSE of each other (49.2–49.9),
  with R² ~0.45–0.46. This is the central empirical result: at ~20 training
  stations on one orographically complex island, the location-agnostic CNN
  branches add no measurable skill over pooled tabular regression.
- **The tuned winner went minimal.** The v3 broad search landed on
  `lightweight=1`, `rain_lag_weeks=0`, `climate_lag_weeks=0`, dropout ~0.16 —
  the Optuna basin itself migrated away from the heavier original
  architecture toward a thin current-week model, consistent with the
  small-spatial-support interpretation.
- **`loso` vs `loso_recent` made no difference** on this config (49.484 vs
  49.479 RMSE); the early-stopping protocol change was neutral here. The two
  `_2` reruns land at 49.86 in both CV modes, so identical configs span
  ~0.4 mm RMSE — real differences below that are not meaningful.
- **Lowering the learning rate ~7x (`_lowlr`) did not help** (49.57 RMSE):
  slightly better extremes (98th-pct bias -23.5%, best neural CSI 0.655) at
  the cost of a larger positive bias (+4.05 mm), i.e. a tradeoff, not a gain.
- **Bias structure differs by model class.** LAND runs are near-unbiased
  (+2.1 / -0.1 mm), GBM underpredicts systematically (-12.2 mm bias, best
  MAE 33.2 but worst 98th-pct error), OLS sits between. Model choice
  trades calibration for typical-case accuracy.
- **Tweedie underperforms on this split** (0.216 R2): with only 4 test
  stations and post-2016 years the GLM's strong negative bias (-15.4 mm)
  dominates. Earlier splits where Tweedie nearly tied LAND had more test
  stations and different era coverage — split composition matters.

The leading candidates on the current dataset:

- best overall RMSE/R2: `gbm` (49.17 / 0.462), statistically tied with
  OLS and the LAND ensembles;
- lowest MAE: `gbm` (33.18), at the cost of a -12.2 mm bias;
- best CSI: `weekly_land_v3_huber_recent_bal_lowlr` (0.655), within noise of
  the other LAND runs;
- per-station wins: none decisive — LAND's only lead is `poloa_UH` by <1 mm.

## 10. Per-station differences

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

This file now contains every baseline and every evaluated run. It
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

## 11. Low-end rainfall audit

`LAND_AS/notebooks/01_data_prep_eda.ipynb` contains a dedicated low-end
distribution diagnostic. It compares the pre-2017 training rows, post-2016 test
stations, and 734 unused post-2016 "bridge" rows for the two WRCC training
stations (`siufaga_WRCC` and `toa_ridge_WRCC`).

Key findings (raw-data audit, before the cleaning rules of section 3.1):

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
  through the 1970s-90s. Its 1989-90 overlap with `vaipito_res` is identical
  (weekly r = 1.000), so it is the same site's earlier record; the earlier
  "half the co-located mean" framing was era-confounded.
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

QC actions applied (see section 3.1): `aunuu` and `afono_UH` removed entirely
(the latter's three flat-zero runs were gauge-offline artifacts), and
`vaipito2000` windowed to 1976-01-01..1989-09-30 (its uncorroborated 1958-69
block and its identical-to-`vaipito_res` overlap dropped). On the audit's
rebuilt dataset: training 143/6,078 exact-zero weeks (2.35%), test 3/1,170
(0.26%). The residual
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

A controlled exclusion ablation was run (same tuned Huber config, LOSO folds,
`pioa_afono` dropped from `QC_EXCLUDE_STATIONS` + rebuild, losing 666 weeks /
one fold): test RMSE worsened 49.32 to 50.02, MAE 35.46 to
35.77, and R2 fell 0.480 to 0.465 — far above the ~0.4 mm run-to-run noise
seen between identical-config reruns. Excluding `pioa_afono` loses
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

## 12. Historical experiments not retained

Earlier exploratory code and outputs were removed from the active tree. The
main lessons were:

- adding a daily temporal atmospheric encoder changed many variables at once
  and did not produce a defensible replacement for the weekly model;
- full-data retraining without a validation split removed the ability to select
  an independent best epoch;
- the resulting models improved some ranking or MAE metrics but worsened bias,
  RMSE, and extreme-rainfall calibration;
- the controlled scalar-Huber experiments were retained because they changed
  fewer assumptions and produced more interpretable comparisons.

Those deleted experiments are not represented in the current output tree and
should not be cited as active model candidates.

## 13. Reproducibility

Training and evaluation write source snapshots:

```text
<run>/code_snapshot/
<run>/evaluation/code_snapshot/
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
- fold histories and checkpoints

## 14. Notebook guide

```text
LAND_AS/notebooks/
├── 00_qc_evidence.ipynb            # figures proving each station-cleaning rule
├── viz.ipynb                       # data coverage/zero-rate/maps/features/architecture/results tour
├── 01_data_prep_eda.ipynb          # raw/prepared data + low-end shift audit
├── 02_tuning_eda.ipynb             # Optuna study and split diagnostics
├── 03_training_eda.ipynb           # LOSO histories and run evaluation
├── 04_results_comparison.ipynb     # baselines and runs together
└── 05_land_style_figures.ipynb     # LAND-style figures adapted to American Samoa
```

The fifth notebook adapts figure ideas from the original repository's
`results/plots.ipynb`, `main_results_I.ipynb`, `main_results_II.ipynb`,
`results_III.ipynb`, `topography_alignment.ipynb`, and `rah_comparison.ipynb`.
It uses American Samoa data rather than the Hawaii-specific map/GCM files.

## 15. Output map

```text
LAND_AS/output/
├── baselines/        # pooled baselines + aligned model comparisons
├── baselines_daily/  # same layout for --daily runs (CSI 25 mm/day)
├── baselines_OLD/    # preserved metrics from a prior dataset snapshot
├── figures/          # notebook-generated diagnostic figures
├── runs/             # retained LOSO ensembles and evaluations
└── tuning/           # retained Optuna studies

LAND_AS/scripts/      # one-off analysis helpers
└── dm_baseline_check.py      # refit pooled baselines on a legacy weekly
                              # dataset under its own station/year split
```

The main entry points are:

```text
LAND_AS/s1_prepare/prepare.py              # build feature caches + weekly/daily NPZ
LAND_AS/s1_prepare/load_raw.py             # raw station/climate/DEM ingestion + QC config
LAND_AS/s1_prepare/build_features.py       # feature cache construction (reused by both freqs)
LAND_AS/s1_prepare/assemble_dataset.py     # NPZ assembly (weekly + daily)
LAND_AS/s2_dataset/data.py                 # split, lag features, crops, loaders, CV folds
LAND_AS/s3_model/model.py                  # LAND architecture + Gamma/Huber/BernGamma heads
LAND_AS/s3_model/engine.py                 # fit, predict, metrics helpers
LAND_AS/s4_train/tune.py                   # Optuna studies (broad space incl. ablations)
LAND_AS/s4_train/train.py                  # LOSO/loso_recent ensemble training
LAND_AS/s4_train/parallelize.py            # fold/seed orchestration
LAND_AS/s5_evaluate/evaluate.py            # ensemble eval + robust trial selector
LAND_AS/s5_evaluate/baselines/models.py    # baseline implementations
LAND_AS/s5_evaluate/baselines/evaluate.py  # baseline and aligned-model metrics
```

`LAND_AS/next-steps.md` documents the recommended future experiments and the
validation controls required before any new candidate is promoted.

## 16. Practical decision rule

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

A neural candidate should also be checked against OLS and GBM, because both
are strong enough in this dataset to serve as legitimate alternatives rather
than trivial baselines.
