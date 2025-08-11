# AS_rainfall

End-to-end pipeline for assembling data, hyperparameter tuning, training, and ensembling rainfall prediction models.

This README shows how to set up and run the project step by step.

## 0) Quick start

```bash
# 1) Clone repo and enter folder
git clone https://github.com/jswlee/AS_rainfall.git
cd AS_rainfall

# 2) Create env
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip

# 3) Install
pip install -e .
```

## 1) Clone and create an environment

- Clone the repo and enter the folder:
  ```bash
  git clone https://github.com/jswlee/AS_rainfall.git
  cd AS_rainfall
  ```
- Create and activate a virtual environment (recommended):
  - venv
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```
  - or conda (good for geo libs like GDAL/cartopy)
    ```bash
    conda create -n as_rainfall python=3.9
    conda activate as_rainfall
    ```

## 2) Install the package

- Editable install from the repo root:
  ```bash
  pip install -e .
  ```

## 3) Prepare data
- Notebook-first approach: open `1_Rainfall_Data_Processing.ipynb` and `2_ML_Data_Preprocessing.ipynb`.
- Ensure the combined NPZ exists at:
  - `ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz`
- Ensure persisted test indices exist at:
  - `Hyperparameter_Tuning/output/test_indices.pkl`

## 4) Hyperparameter tuning (optional)
- Notebook-first approach: open `3_Hypertuning.ipynb`.
- Best hyperparameters are saved under `Hyperparameter_Tuning/output/...` and automatically loaded by training/ensemble code via `Train_Best_Model/model_utils.load_best_hyperparameters()`.

## 5) Train best model with K-fold CV
- Notebook-first approach: open `4_Training.ipynb`.

## 6) Train ensemble with cross-validation
- Notebook-first approach: see `4_Training.ipynb`.

Outputs (examples):
- Per-fold metrics/plots under `.../fold_k/`
- Top-level: `ensemble_summary.txt`, `test_predictions.csv`, `ensemble_test_predictions.png`, `individual_vs_ensemble.png`

## 7) Interpreting results
- Metrics (RMSE, MAE, R²) are reported in mm when `rainfall_mm_std` is present
- Training and evaluation artifacts per model/fold are saved alongside summaries.

## 8) Repository structure (high level)
- `ML_Data_Preprocessing/` — Build features and combined NPZ
- `Process_Rainfall_Data/` — Raw rainfall processing helpers
- `Hyperparameter_Tuning/` — Tuning scripts and outputs
- `Train_Best_Model/` — Best model training utilities and scripts
- `Train_Ensemble/` — Ensemble CV training utilities and scripts
- `requirements.txt`, `pyproject.toml`, `setup.cfg` — packaging and dependencies
