# `scripts/01_build_features.py` — Build Intermediate Feature Files

## Purpose
First pipeline step. Extracts reanalysis climate patches and DEM elevation patches from raw data sources and saves them as compressed NumPy archives (`.npz`). These intermediate files decouple the expensive raw-data processing from the faster assembly/training steps.

## Relation to the Deep Downscaling Paper
Hatanaka et al. (2025) use a 3×3 grid of reanalysis variables centred on each station as input to the LAND model (Section 3a). This script implements that extraction. The paper also uses local and regional DEM patches to capture fine-scale topography that coarse reanalysis cannot resolve. Our implementation extends the paper by generating **max-size** DEM patches (11×11 @ 1 km local, 25×25 @ 1 km regional) so that smaller patch sizes can be cropped at runtime during hyperparameter tuning — the paper used fixed 3×3 patches.

## Line-by-Line Walkthrough

### Imports (lines 1–18)
```python
import argparse, numpy as np
from Daily_Modeling import config
from Daily_Modeling.data_utils.load_raw import load_station_metadata, discover_station_days
from Daily_Modeling.data_utils.build_features import (
    load_reanalysis_datasets, build_reanalysis_patches, build_dem_patches,
)
```
- `config` centralises all paths (DEM file, reanalysis directory, output directories) and hyperparameter defaults.
- `load_station_metadata()` reads `station_locations.csv` → `{name: {lat, lon, ...}}`.
- `discover_station_days()` scans each station's rainfall CSV to find all `(year, month, day)` tuples within the date range. This determines which reanalysis time-steps to extract.
- The three `build_features` functions do the heavy lifting.

### `main()` function (lines 21–57)

**Line 22–23: Output directory**
```python
output_dir = config.FEATURES_DIR  # Daily_Modeling/output/features/
output_dir.mkdir(parents=True, exist_ok=True)
```
Creates the output directory if it doesn't exist.

**Lines 25–26: Station metadata**
```python
meta = load_station_metadata()
```
Loads lat/lon for all ~26 American Samoa stations from `raw_data/station_locations.csv`.

**Lines 28–39: Reanalysis patch extraction**
```python
station_days = discover_station_days(meta, start_date=start_date, end_date=end_date)
datasets = load_reanalysis_datasets()
patches, stations, years, months, days, var_names = \
    build_reanalysis_patches(meta, station_days, datasets)
```
1. `discover_station_days` scans all rainfall CSVs to find which dates each station has data for (within the 1980–2024 window).
2. `load_reanalysis_datasets` opens all NetCDF files (air temperature, geopotential height, omega, humidity, wind, etc.) into memory.
3. `build_reanalysis_patches` loops over every (station, day) pair:
   - For each of the 15 derived climate variables (defined in `config.DAILY_VARIABLE_CONFIGS`), it extracts a 3×3 spatial patch centred on the station's nearest grid point.
   - Some channels are simple lookups at a pressure level (e.g. `hgt_500`), others are differences between levels (`air_temp_diff_1000_500`), and others are products of two fields (`zon_moist_750 = uwnd_750 × shum_750`).
   - Output shape: `(N, 15, 3, 3)` — N samples, 15 channels, 3×3 spatial.

The result is saved as `reanalysis_patches_daily.npz`.

**Lines 41–56: DEM patch extraction**
```python
dem = build_dem_patches(
    meta,
    local_cfg=config.DEM_MAX_LOCAL,      # {"patch_size": 11, "km_per_cell": 1}
    regional_cfg=config.DEM_MAX_REGIONAL, # {"patch_size": 25, "km_per_cell": 1}
)
```
- Extracts DEM elevation from the GeoTIFF (`DEM_Tut1.tif`) using rasterio.
- **Max-size patches**: Local is 11×11 @ 1 km (11 km box), regional is 25×25 @ 1 km (25 km box). These are the largest patches needed by any candidate in the HP search.
- At runtime, smaller patches (e.g. 3×3 @ 2 km) are cropped from these max-size bases — see `dataset.py:crop_dem_patch`.
- NaN pixels (ocean) are filled from the nearest valid land pixel via `_fill_nan_nearest`.
- Saved as `dem_patches.npz` with keys `dem_local_raw`, `dem_regional_raw`, `stations`.

**Lines 60–65: CLI entry point**
```python
parser.add_argument("--start-date", default="1980-01-01")
parser.add_argument("--end-date", default="2024-12-31")
```
Allows overriding the date range from the command line.

## Data Manipulation
- **No splits here** — this is pure feature extraction. The split logic comes in step 02/03.
- The alignment between reanalysis and rainfall is done in step 02 (`assemble_dataset.py`).
- Samples with missing reanalysis data for any channel on a given date are **skipped** (the `ok` flag in `build_reanalysis_patches`).

## Architecture Decisions
- **Intermediate NPZ files**: Separating feature extraction from assembly means you can re-run assembly (step 02) without re-extracting from raw NetCDFs (which takes ~30 minutes).
- **Max-size DEM generation**: Extract once at the finest resolution and largest extent, then crop at runtime. This avoids regenerating DEM patches every time you want to try a different patch size.
- **15 derived channels** (paper specifies 16 — we lack `pottmp_diff_1000_850` due to missing data).

## Areas of Improvement
- **Parallelise reanalysis extraction**: The inner loop over station-days is single-threaded. Using `multiprocessing` or `dask` could speed up the ~30-minute extraction.
- **Pre-filter station-days**: Currently extracts all station-days even if the rainfall CSV has missing data for some. A pre-join with rainfall would avoid wasted extraction.
- **Chunk-based NetCDF access**: Loading entire NetCDFs into RAM works for ~45 years of daily data but won't scale to hourly or higher spatial resolution. Chunk-based xarray access would fix this.
