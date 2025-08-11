# Hyperparameter_Tuning

This module contains the end-to-end workflow for tuning the LAND model hyperparameters via K-fold cross-validation on the assembled NPZ dataset. It is designed to be imported and driven from Jupyter notebooks or Python scripts, assuming the current working directory is the project root `AS_rainfall/`.

Key entry points are:
- `extended_hyperparameter_tuning.build_tunable_model(hp, data_metadata)`
- `extended_hyperparameter_tuning.config(...)`
- `tuning_core.run_tuning(config, build_model_fn)`
- `npz_data_utils.load_assembled_npz_data(npz_path, ...)`

The module writes all tuner artifacts, logs, summaries, and the current best hyperparameters to the directory set by `config()['output_dir']` (default: `Hyperparameter_Tuning/output`).

## Contents

- `extended_hyperparameter_tuning.py`
  - `build_tunable_model(hp, data_metadata)`: Builds a LAND-style Keras model whose architecture is parameterized by a Keras Tuner `HyperParameters` object. Includes non-negative `output_activation` (e.g., `softplus` or `relu`). Uses AdamW with configurable `learning_rate` and `weight_decay`.
  - `config(...)`: Returns a dict of overridable defaults used by tuning. Paths are project-root-relative by default:
    - `npz_path='ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz'`
    - `test_indices_path='Hyperparameter_Tuning/output/test_indices.pkl'`
    - `output_dir='Hyperparameter_Tuning/output'`
    - Other knobs: `max_trials`, `executions_per_trial`, `epochs`, `batch_size`, `n_folds`, `cv_seed`, `resume`.

- `tuning_core.py`
  - `run_tuning(config, build_model_fn)`: Orchestrates K-fold CV tuning using Keras Tuner (Bayesian). Handles:
    - Creating/splitting datasets with a persisted test set (via `test_indices_path`).
    - Building a CV-aware `HyperModel` wrapper so each trial is evaluated through CV.
    - Saving plots, CSVs, and progress. Respects `resume` to continue stopped runs.
  - Saves current best hyperparameters to: `output_dir/land_model_cv_tuning/current_best_hyperparameters.py`.
  - Provides `plot_hyperparameter_importance(tuner, output_dir)` with a graceful fallback when advanced tuner plots are unavailable.

- `npz_data_utils.py`
  - `load_assembled_npz_data(npz_path, test_indices_path=None, test_size=0.1, val_size=0.1, random_state=None)`:
    - Loads the combined NPZ produced by `ML_Data_Preprocessing`.
    - Returns a dict of arrays for `climate`, `local_dem`, `regional_dem`, `month`, `targets` with `train/val/test` splits, plus `metadata` including `rainfall_mm_std`.
    - Persists/loads `test_indices.pkl` to ensure a stable test split across runs.

## Assumptions

- CWD is project root; all paths in `config()` are relative to the repository root.
- Data format aligns with the combined NPZ produced by `ML_Data_Preprocessing`.
- Keras Tuner 1.4.x and TensorFlow 2.x are available.

## Typical Notebook Usage

Minimal run (uses defaults):
```python
from Hyperparameter_Tuning.extended_hyperparameter_tuning import config, build_tunable_model
from Hyperparameter_Tuning.tuning_core import run_tuning

cfg = config()
run_tuning(config=cfg, build_model_fn=build_tunable_model)
```

Explicit config:
```python
from Hyperparameter_Tuning.extended_hyperparameter_tuning import config, build_tunable_model
from Hyperparameter_Tuning.tuning_core import run_tuning

cfg = config(
    npz_path='ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz',
    test_indices_path='Hyperparameter_Tuning/output/test_indices.pkl',
    output_dir='Hyperparameter_Tuning/output',
    max_trials=200,
    executions_per_trial=1,
    epochs=150,
    batch_size=64,
    n_folds=10,
    cv_seed=42,
    resume=True,
)
run_tuning(cfg, build_model_fn=build_tunable_model)
```

To run hyperparameter tuning in a jupyter notebook, use the following example:
`3_Hypertuning.ipynb`

## Outputs

- `output/land_model_cv_tuning/` containing tuner trials, logs, and `current_best_hyperparameters.py`.
- Plots and CSVs summarizing CV performance.

## Tips

- Ensure the NPZ exists at `ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz`.
- With `resume=True`, you can interrupt and restart tuning safely.
