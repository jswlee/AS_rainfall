# Train_Best_Model

This module trains the best LAND model (selected via hyperparameter tuning) using K-fold cross-validation on the combined NPZ dataset. It saves per-fold metrics in physical units (millimeters), plots, and concise summaries, plus an overall CV summary.

It is designed to be imported and run from Jupyter notebooks or scripts, assuming the working directory is the project root `AS_rainfall/`.

## Contents

- `train_best_model_cv.py`
  - `config(...)` → returns overridable defaults:
    - `npz_path='ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz'`
    - `test_indices_path='Hyperparameter_Tuning/output/test_indices.pkl'`
    - `output_dir='Train_Best_Model/output/land_model_best_cv'`
    - `n_folds`, `epochs`, `batch_size`, `hp_dir`, `cv_seed`
  - `create_tf_datasets_from_arrays(split_dict, batch_size=..., drop_remainder=False)` → convenience to produce TF datasets from in-memory arrays.
  - `run_cv_training(overrides=None)` → orchestrates K-fold CV training:
    1. Loads combined NPZ via `Hyperparameter_Tuning.npz_data_utils.load_assembled_npz_data(...)` with a persisted test split.
    2. Builds a CV pool from original train+val, keeps the held-out test set fixed.
    3. For each fold, builds datasets, constructs a model with tuned hyperparameters, trains, evaluates on test, plots history, writes a fold summary.
    4. Aggregates per-fold metrics to `cv_fold_metrics.csv` and `cv_summary.csv` in `output_dir`.

- `model_utils.py`
  - `load_best_hyperparameters(base_output_dir)` → loads `current_best_hyperparameters.py` saved by tuning. Supports both the flat and older nested tuner layouts.
  - `build_model(data_metadata, hp_dir=..., hyperparams=None)` → builds and compiles the LAND model using the best hyperparameters:
    - Inputs: `climate` (V×H×W), `local_dem` (H×W), `regional_dem` (H×W), `month` (one-hot)
    - Several dense blocks with BatchNorm and Dropout
    - Optional residual connection when sizes match
    - Output activation uses a non-negative function (`relu` or `softplus`) to ensure physically valid rainfall predictions
    - Optimizer: AdamW with tuned learning rate and weight decay

- `training.py`
  - `cosine_decay_with_warmup(...)` → LR schedule helper.
  - `train_model(model, data, epochs, batch_size, output_dir, initial_lr, min_lr, warmup_epochs, patience)` → trains with ModelCheckpoint + EarlyStopping. Saves `best_weights.weights.h5` and history.
  - `evaluate_model(model, data, output_dir=None, rainfall_std=None, label_unit=None)` → computes metrics and de-standardizes to millimeters using `rainfall_mm_std` from the dataset metadata. Saves `evaluation_metrics.npy/csv` and a human-readable `evaluation_metrics.txt`.
  - `plot_training_history(history, output_dir, rainfall_std)` → converts loss/MAE to millimeters for the plot and saves `training_history.png`.

- `data_utils.py`
  - Legacy CSV-based loader and dataset builder (superseded by NPZ workflow). Kept for reference and experimentation.

## Assumptions & Paths

- CWD is the repo root; all defaults are project-root-relative.
- The combined NPZ exists at `ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz`.
- Best hyperparameters are present in `Hyperparameter_Tuning/output/land_model_cv_tuning/current_best_hyperparameters.py` (or the older nested path).

## Typical Notebook Usage

Minimal run with defaults:
```python
from Train_Best_Model.train_best_model_cv import run_cv_training
run_cv_training()
```

Customized config:
```python
from Train_Best_Model.train_best_model_cv import run_cv_training, config

cfg = config(
    npz_path='ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz',
    test_indices_path='Hyperparameter_Tuning/output/test_indices.pkl',
    output_dir='Train_Best_Model/output/land_model_best_cv',
    n_folds=10,
    epochs=150,
    batch_size=64,
    hp_dir='Hyperparameter_Tuning/output',
    cv_seed=42,
)
run_cv_training(cfg)
```

## Outputs

For each fold `fold_k/` under `output_dir`:
- `best_weights.weights.h5`
- `training_history.npy`
- `training_history.png` (loss/MAE in mm)
- `evaluation_metrics.npy/csv/txt` (metrics in mm and R²)
- `fold_summary.txt`

Aggregated at `output_dir/`:
- `cv_fold_metrics.csv` -> per-fold rows with `rmse_mm`, `mae_mm`, `mse_mm2`, `r2`
- `cv_summary.csv` -> averages across folds (in millimeters)

Refer to`/Users/jlee/Desktop/github/AS_rainfall/4_Training.ipynb` for example usage.

## Tips
- Ensure your `batch_size` and `epochs` align with the tuned results; large changes can drift from the validated configuration.
- `evaluate_model(...)` requires `rainfall_mm_std` in dataset metadata for correct physical-unit reporting; this is provided by the combined NPZ produced by `ML_Data_Preprocessing`.
- If `load_best_hyperparameters(...)` cannot find the file, verify your tuning output directory and project structure.
