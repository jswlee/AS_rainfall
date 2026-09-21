# LAND_AS

A compact, weekly-only implementation of the Location-Agnostic Neural Downscaler for American Samoa.

The package keeps the paper-style three-branch model: atmospheric fields, local/regional DEMs, and month. It uses a Gamma likelihood for weekly rainfall and spatiotemporal cross-validation. Raw NetCDF and DEM extraction delegates to the tested `Daily_Modeling` extraction functions; all weekly data, tuning studies, checkpoints, and evaluation outputs belong to `LAND_AS`.

## Pipeline

Run commands from the repository root.

```bash
python -m LAND_AS.prepare
python -m LAND_AS.tune --trials 50 --folds 3
python -m LAND_AS.train \
  --hyperparameters LAND_AS/output/tuning/weekly_land/best.json \
  --run weekly_land --folds 3 --seeds 3
python -m LAND_AS.evaluate --run weekly_land
```

`prepare` retains only complete Monday-Sunday station weeks. Daily atmospheric channels are reduced to weekly means and standard deviations, rainfall is summed, and the month is taken from the Monday starting each week.

## Layout

- `prepare.py`: raw extraction and weekly assembly
- `data.py`: loading, train-only normalization, station/year splits, and CV folds
- `model.py`: paper-style climate, DEM, and month branches with a Gamma head
- `tune.py`: Optuna search using mean fold validation MAE
- `train.py`: fold-by-seed ensemble training
- `evaluate.py`: temporal and spatial test metrics, predictions, and uncertainty
- `data/`: generated features and assembled weekly dataset
- `output/`: tuning studies, trained runs, and evaluation artifacts

The held-out test years and test stations are never used by tuning or early stopping. Evaluation produces JSON metrics and NPZ prediction arrays without embedding visualization in the pipeline.
