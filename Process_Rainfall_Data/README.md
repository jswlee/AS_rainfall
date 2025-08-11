# Process_Rainfall_Data

This module contains utilities and scripts for transforming raw rainfall station measurements into clean monthly aggregates suitable for ML consumption. Outputs feed the `ML_Data_Preprocessing` module and ultimately the combined NPZ used for model training.

All paths are intended to be project-root-relative with the working directory set to the repository root `AS_rainfall/`.

## Contents

- `scripts/rainfall_daily_to_monthly.py`
  - CLI script to aggregate raw daily measurements into monthly totals per file.
  - Expects input CSVs with columns including at least `datetime` and `precip_in` (inches).
  - Produces per-file monthly CSVs with columns:
    - `year_month` (e.g., `2020-07`)
    - `monthly_total_precip_in` (sum in inches; if any daily value is NaN in a month, that month is recorded as NA)
  - Defaults:
    - Input: `raw_data/rainfall/`
    - Output: `Process_Rainfall_Data/output/monthly_rainfall/`
  - Usage:
    ```bash
    python Process_Rainfall_Data/scripts/rainfall_daily_to_monthly.py \
      --input_dir raw_data/rainfall \
      --output_dir Process_Rainfall_Data/output/monthly_rainfall
    ```

- `scripts/process_wide_format_rainfall.py`
  - Script to convert and clean rainfall datasets provided in wide formats into a consistent long/monthly format (see source for exact assumptions and schema handling).
  - Useful when historical datasets come in heterogeneous structures.

- `scripts/compare_rainfall_files.py`
  - Compares two rainfall directories (e.g., `raw_data/rainfall` vs `raw_data/rainfall_added`) file-by-file and date-by-date.
  - Normalizes date strings and reports:
    - Missing files between directories
    - Counts of common dates
    - Values with >0.5% differences and their factors
    - Dates only present in one side
  - Writes a detailed log to a path you specify inside the script or via wrapper.

- `scripts/rainfall_monthly_coverage_viz.py`
  - Visualization utility to examine monthly coverage over time per station (open the script for plotting details and usage).

- `figures/`
  - Static figures used for documentation or exploratory analysis.

- `output/`
  - Generated artifacts; excluded from code references in this README. Not required to exist before running the scripts.

## Data Conventions

- Physical units are inches for raw and monthly totals in this module. Downstream modules (e.g., `ML_Data_Preprocessing`) handle standardization and any conversions required for training.
- Monthly outputs prefer the schema:
  - `['year_month', 'monthly_total_precip_in']`
- Some legacy data sources may use:
  - `['Year', 'Month', 'Rainfall']` (inches)
- The `ML_Data_Preprocessing/assemble_training_data.py` assembler supports both schemas when building the combined NPZ.

## Typical Workflow

1) Aggregate raw daily files to monthly totals:
```bash
python Process_Rainfall_Data/scripts/rainfall_daily_to_monthly.py \
  --input_dir raw_data/rainfall \
  --output_dir Process_Rainfall_Data/output/monthly_rainfall
```

2) (Optional) Convert or clean wide-format historical datasets:
```bash
python Process_Rainfall_Data/scripts/process_wide_format_rainfall.py
```

3) (Optional) Compare two sources to reconcile differences:
```bash
python Process_Rainfall_Data/scripts/compare_rainfall_files.py
```

4) (Optional) Visualize coverage over time:
```bash
python Process_Rainfall_Data/scripts/rainfall_monthly_coverage_viz.py
```

Programmatic assembly from a notebook:
Check `1_Rainfall_Data_Processing.ipynb` for an example.

## Downstream Consumption
- The assembler will align `(station, year, month)` with reanalysis/DEM indices and embed rainfall into the combined NPZ.

## Tips

- Ensure consistent station naming in filenames (e.g., `<station>_monthly.csv`) so the assembler can match station IDs.
- When reconciling data sources, inspect the comparison log produced by `compare_rainfall_files.py` to decide which source to trust for discrepancies.
