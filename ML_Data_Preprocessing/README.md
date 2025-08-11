# ML_Data_Preprocessing

This module builds the model-ready dataset used across tuning, single best-model training, and ensembling. It assembles a single combined NPZ (`full_training_data.npz`) containing:

- `reanalysis_patches`: [N, V, H, W] standardized climate variables (e.g., 16 variables on a 3×3 grid)
- `dem_local_*` and `dem_regional_*`: standardized DEM patches at two spatial scales
- `month_onehot`: [N, 12] one-hot month encoding
- `rainfall_mm_divstd`: [N] rainfall targets in millimeters divided by a global std
- `rainfall_mm_std`: float, the global std (standard deviation) used for de-standardization
- Metadata such as shapes and sizes

All paths are project-root-relative and this package assumes the notebook/script CWD is the repo root `AS_rainfall/`.

## Contents

- `assemble_training_data.py`
  - Class `TrainingDataAssembler`
    - Loads station rainfall CSVs (supports two schemas):
      - New schema: columns `['year_month', 'monthly_total_precip_in']`
      - Legacy schema: columns `['Year', 'Month', 'Rainfall']` (inches)
    - Builds a lookup of `(station, year, month) → rainfall (inches)` and aligns rainfall with precomputed reanalysis indices
    - Produces normalized rainfall in millimeters (`rainfall_mm_divstd`) and stores the global std (`rainfall_mm_std`) for proper unit restoration later
    - Assembles and saves a single NPZ with all modalities aligned
  - Function `assemble_from_precomputed(dem_npz_path=None, reanalysis_npz_path=None, out_dir=None, out_filename='full_training_data.npz')`
    - Consumes precomputed DEM and reanalysis NPZs
    - Aligns station/year/month keys across datasets
    - Computes global stats (min/max/std) for DEM scaling and rainfall std
    - Writes outputs under `ML_Data_Preprocessing/output/assembled_npz/`

- `build_dem_patches.py`
  - Generates standardized DEM patches at local and regional scales
  - Writes `dem_patches_all_standardized.npz` under `ML_Data_Preprocessing/output/dem_npz/`

- `build_reanalysis_features.py`
  - Extracts climate variables into small spatial patches per sample and standardizes them
  - Writes `reanalysis_features_all_standardized.npz` under `ML_Data_Preprocessing/output/reanalysis_npz/`

- `config.py`
  - Centralized paths for this module (e.g., rainfall input directory, output directories)

- `utils.py`
  - Helpers for filtering outliers, IO, and general preprocessing utilities used by the assembler

- `print_single_datapoint.py`
  - Introspection utility to print/visualize a single composite sample (useful for sanity checks)

- `extract_station_metadata.py`
  - Extracts core station attributes used by builders/assembler

## Expected Directory Layout

- Input rainfall CSVs (per-station) located under the configured rainfall directory in `config.py`
- Generated outputs (examples):
  - `ML_Data_Preprocessing/output/dem_npz/dem_patches_all_standardized.npz`
  - `ML_Data_Preprocessing/output/reanalysis_npz/reanalysis_features_all_standardized.npz`
  - `ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz`

## Usage Examples

Build reanalysis features then DEM patches, then assemble the combined NPZ:
```python
# 1) Build reanalysis features
!python ML_Data_Preprocessing/build_reanalysis_features.py

# 2) Build DEM patches
!python ML_Data_Preprocessing/build_dem_patches.py

# 3) Assemble the combined NPZ
!python ML_Data_Preprocessing/assemble_training_data.py
```

Programmatic assembly from a notebook:
Check `2_ML_Data_Preprocessing.ipynb` for an example.

## Notes and Conventions

- Rainfall in downstream training and evaluation is reported in physical units (mm or inches) by de-standardizing with `rainfall_mm_std`.
- The combined NPZ is the canonical dataset feeding `Hyperparameter_Tuning`, `Train_Best_Model`, and `Train_Ensemble`.
- Ensure station rainfall CSVs are complete; missing values are skipped during assembly (with warnings).
