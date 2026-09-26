# LAND_AS: Weekly Rainfall Downscaling for American Samoa

`LAND_AS` adapts the Location-Agnostic Neural Downscaler (LAND) to weekly station rainfall in American Samoa. The current reference model is **`weekly_land_v5`**. It is the strongest evaluated model for the strict spatiotemporal test set and should remain the baseline for future changes.

## Why this setup is used

The weekly problem is much smaller than the original monthly Hawaii application:

- About 21 usable pre-test training locations.
- Multiple years per station, but limited independent terrain realizations.
- Rainfall is highly skewed, with occasional extreme weekly totals.
- Train and test data differ in both station and time period.

That makes spatial CV and an untouched test set more important than adding model capacity.

## Data and split

`LAND_AS.prepare` writes:

```text
LAND_AS/data/weekly_dataset.npz
```

The active v5-era representation contains:

- Weekly reanalysis mean and standard-deviation channels
- Local and regional DEM patches
- Month one-hot features
- Previous-week rainfall and missing-lag masks
- Weekly station rainfall totals

The split is strict:

- Training: non-test stations, years `<= 2016`
- Test: configured test stations, years `> 2016`
- Validation: leave-one-station-out over training stations only

Configured test stations are in `LAND_AS/config.py`.

## v5 model

`weekly_land_v5` uses the original-style Gamma LAND architecture:

- Grouped convolution over reanalysis channels
- DEM convolution over local and regional terrain
- Month and rainfall-lag features
- Dense head producing two Gamma parameters
- Gamma negative log-likelihood training loss
- Rainfall-weighted Gamma loss (`rainfall_weight: true`)
- Ensemble over 21 LOSO folds and three seeds

Frozen hyperparameters are stored at:

```text
LAND_AS/output/runs/weekly_land_v5/hyperparameters.json
```

Retrospective run metadata, including the inferred training command and Optuna trial details, is stored at:

```text
LAND_AS/output/runs/weekly_land_v5/reproduction.json
```

### v5 tuning

The Optuna study is:

```text
LAND_AS/output/tuning/weekly_land_v5/study.db
```

Its selected trial was **trial 5**, minimizing validation MSE:

- `climate_multiplier`: 13
- `climate_lag_weeks`: 0
- `climate_patch`: 3
- `rain_lag_weeks`: 2
- `learning_rate`: `1.1476582119489201e-4`
- `weight_decay`: `1.87422109855557e-4`
- `rainfall_weight`: `true`

With 30 current-week channels and zero climate lag, `climate_multiplier=13` gives `climate_units=390`, matching the saved training hyperparameters.

The study database does not record the exact CLI invocation. A plausible reconstruction is:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.tune `
  --study weekly_land_v5 `
  --trials 36 `
  --folds 3 `
  --epochs 500 `
  --patience 50 `
  --min-epochs 30 `
  --opt-metric mse
```

The study contains 36 numbered trials, 35 complete and one failed.

### v5 training command

The confirmed training entry point is `LAND_AS/train.py`, which delegates fold execution to `LAND_AS/parallelize.py`. The run output confirms 21 LOSO folds and seeds 42, 43, and 44.

Effective command:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train `
  --study weekly_land_v5 `
  --run weekly_land_v5 `
  --seeds 3 `
  --epochs 500 `
  --patience 50 `
  --workers 4
```

Omitting `--workers` uses the same default cap of four workers when enough CPU workers are available.

### v5 evaluation

```powershell
.\venv\Scripts\python.exe -m LAND_AS.evaluate `
  --run weekly_land_v5
```

Current test metrics:

- RMSE: `50.829 mm`
- MAE: `36.067 mm`
- Bias: `0.101 mm`
- \(R^2\): `0.4482`
- Spearman: `0.6613`
- 98th-percentile bias: `-23.16%`
- CSI above 50 mm: `0.6416`

Metrics and predictions are in:

```text
LAND_AS/output/runs/weekly_land_v5/evaluation/
```

## Reproducibility snapshots

New training runs write:

```text
<run>/code_snapshot/
├── manifest.json
├── command.txt
└── files/
```

The snapshot contains:

- Exact source files used by the run
- SHA-256 hash of each source file
- Command line
- Python executable/version
- Relevant package versions
- Git commit and dirty status
- Dataset path, size, and SHA-256 hash

Evaluation writes a separate source snapshot to:

```text
<run>/evaluation/code_snapshot/
```

Snapshots are not overwritten when resuming an interrupted run, preserving the code that produced earlier checkpoints.

## v5-sized Huber experiment

The next controlled experiment changes only the output head/objective:

- Same v5 architecture dimensions
- Same reanalysis, DEM, month, and lag inputs
- Same LOSO folds and seeds
- Same epochs/patience defaults
- Scalar `softplus` rainfall output
- Huber loss on normalized rainfall
- `huber_delta=0.5`

Run:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber `
  --seeds 3 `
  --epochs 500 `
  --patience 50 `
  --workers 4
```

Then evaluate:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.evaluate `
  --run weekly_land_v5_huber
```

The `rainfall_weight` setting is retained in the experiment configuration for provenance, but it only affects the Gamma loss and has no effect on the Huber head.

### Controlled Huber training variants

The v5 Huber script supports one-change-at-a-time variants:

```powershell
# Wet-week-weighted Huber loss
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_rw `
  --loss-type huber_weighted `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# MSE checkpoint selection
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_msemon `
  --monitor mse `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# More MSE-like Huber loss
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_d1 `
  --huber-delta 1.0 `
  --seeds 3 --epochs 500 --patience 50 --workers 4

# Equal station representation in training batches
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_balanced `
  --balanced-stations `
  --seeds 3 --epochs 500 --patience 50 --workers 4
```

Evaluate each run with:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.evaluate --run <run-name>
```

To add two seeds to the existing Huber ensemble:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber `
  --seeds 5 `
  --epochs 500 --patience 50 --workers 4
```

## Gamma/Huber blend

`LAND_AS/blend_v5_huber.py` constructs held-out predictions for every LOSO fold, averages the three seeds within each fold, and selects the Huber weight without using the test set. The selected weight is then applied to the saved test ensemble predictions.

Primary MSE-based run:

```powershell
.\venv\Scripts\python.exe -m LAND_AS.blend_v5_huber `
  --objective mse `
  --output v5_gamma_huber_cv_mse
```

Outputs are under:

```text
LAND_AS/output/blends/v5_gamma_huber_cv_mse/
├── code_snapshot/
├── experiment.json
├── selection.json
├── oof_metrics.json
├── oof_predictions.npz
├── oof_weight_grid.csv
├── test_metrics.json
└── test_predictions.npz
```

The OOF-selected Huber weight was `0.682`. Alternative recorded weights were `0.882` for MAE and `0.731` for the balanced score.

MSE-selected blend test metrics:

- RMSE: `50.363 mm`
- MAE: `35.008 mm`
- Bias: `-4.726 mm`
- \(R^2\): `0.4582`
- Spearman: `0.6672`
- 98th-percentile bias: `-25.90%`
- CSI above 50 mm: `0.6416`

## Current active scope

The active source intentionally contains only:

- Original Gamma LAND (`model_type: "gamma"`)
- Scalar Huber LAND (`model_type: "huber"`)
- The shared LOSO training/evaluation pipeline
- Reproducibility snapshots

The temporal/full-data experimental scripts were removed from the active code path. Their generated output remains under `LAND_AS/output/` for audit purposes, but those runs should not be treated as candidates against v5.

## Decision rule for future experiments

A candidate should be compared with `weekly_land_v5` using the untouched test set only after CV supports it. It should improve the overall profile, not just one metric:

- Lower RMSE
- Lower or comparable MAE
- Higher \(R^2\)
- Comparable or better bias
- Better high-percentile rainfall bias
- Comparable or better wet-event CSI
