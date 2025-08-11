# Train_Ensemble

Simple cross-validated ensemble training for the LAND rainfall model. Trains multiple models per CV fold with different random seeds, evaluates, and aggregates predictions and metrics. Outputs are in physical units when the NPZ includes `rainfall_mm_std` metadata.

All paths are assumed project-root-relative (CWD at `AS_rainfall/`).

## Contents

- `Train_Ensemble/train_ensemble.py`
  - `run_ensemble_cv(...)` — High-level API to run K-fold CV ensemble training on the combined NPZ dataset.
  - `train_ensemble(...)` — Core routine that performs CV splitting, per-fold model training, evaluation, logging, and ensembling.
- `Train_Ensemble/utils.py`
  - Plotting helpers:
    - `plot_ensemble_test_predictions(...)`
    - `plot_individual_vs_ensemble(...)`
    - `plot_fold_ensemble_predictions(...)`
  - File I/O helpers:
    - `save_progress(...)`
    - `write_training_summary(...)`
    - `write_fold_summary(...)`
    - `write_test_predictions_csv(...)`
    - `write_ensemble_summary(...)`

## Upstream Dependencies

- Data loading: `Hyperparameter_Tuning/npz_data_utils.load_assembled_npz_data(...)`
- Model + hyperparameters: `Train_Best_Model/model_utils.{load_best_hyperparameters, build_model}`
- Training utilities: `Train_Best_Model/training.{train_model, evaluate_model, plot_training_history}`
- TF dataset builder: `Train_Best_Model/data_utils.create_tf_dataset(...)`

## Data & Units

- Default dataset: `ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz`
- If `metadata['rainfall_mm_std'] > 0`, charts and text outputs are de-standardized to millimeters.
- Otherwise, inches are used (legacy conversion ×100 for values from training units).

## API Overview

### run_ensemble_cv(...)
Defined in `Train_Ensemble/train_ensemble.py`.

Parameters (defaults shown):
- `output_dir='Train_Ensemble/output/simple_ensemble_cv'`
- `n_folds=15`, `n_models_per_fold=5`, `epochs=150`
- `test_indices_path='Hyperparameter_Tuning/output/test_indices.pkl'`
- `hp_dir='Hyperparameter_Tuning/output'`
- `npz_path='ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz'`

Behavior:
- Skips training if `ensemble_summary.txt` already exists in `output_dir`.
- Loads the combined NPZ and persisted test indices, loads best hyperparameters, then calls `train_ensemble(...)`.
- Prints final CV/test metrics (mm if available).

### train_ensemble(data, hyperparams, output_dir, hp_dir, ...)
Key points:
- Requires `hyperparams['batch_size']` (enforced, no default fallback).
- Combines original train+val into a CV pool; the held-out test set is fixed by `test_indices.pkl`.
- Uses `KFold(n_splits=n_folds, shuffle=True, random_state=42)`.
- Per fold: builds TF datasets, trains `n_models_per_fold` models with distinct seeds `[42 + i]`, saves `model.h5`, history plot, evaluation files, and writes per-model `training_summary.txt`.
- Fold ensemble prediction is the mean of per-model test predictions; writes `fold_summary.txt` and a fold ensemble plot.
- Aggregates CV metrics and computes a final test ensemble by averaging predictions from all models across all folds.
- Writes `test_predictions.csv` and `ensemble_summary.txt`; generates ensemble plots.
- Supports resume via `training_progress.pkl` when `resume_training=True`.

## Outputs

Created under `output_dir` (default `Train_Ensemble/output/simple_ensemble_cv/`):

- Per fold: `fold_{k}/`
  - Per model: `model_{i}/`
    - `model.h5`
    - `training_history.png`
    - `evaluation_metrics.csv` and `evaluation_metrics.txt`
    - `training_summary.txt`
  - `fold_summary.txt`
  - `fold_ensemble_predictions.png`

- Top-level:
  - `ensemble_summary.txt` (hyperparameters, per-fold metrics, averages, final test metrics, training time)
  - `test_predictions.csv` (mm or inches depending on metadata)
  - `ensemble_test_predictions.png`
  - `individual_vs_ensemble.png`
  - `training_progress.pkl` (resume metadata)

## Usage

Notebook/script usage:
```python
from Train_Ensemble.train_ensemble import run_ensemble_cv

results = run_ensemble_cv(
    output_dir='Train_Ensemble/output/simple_ensemble_cv',
    n_folds=10,
    n_models_per_fold=5,
    epochs=150,
    test_indices_path='Hyperparameter_Tuning/output/test_indices.pkl',
    hp_dir='Hyperparameter_Tuning/output',
    npz_path='ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz',
)
```

For Jupyter notebook example, see `4_Training.ipynb`.

## Notes & Tips

- Ensure the combined NPZ exists and, if available, includes `rainfall_mm_std` metadata for mm outputs.
- Keep `hp_dir` aligned with your latest tuning run so `load_best_hyperparameters(...)` resolves properly.
- Use `resume_training=True` to continue long runs safely; the trainer skips completed models/folds.
- Maintain consistent CV parameters (`n_folds`, `test_indices_path`) when comparing across runs.
