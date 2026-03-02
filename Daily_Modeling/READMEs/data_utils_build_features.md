# `data_utils/build_features.py` — Reanalysis & DEM Feature Extraction

## Purpose
Core feature engineering module. Extracts 3×3 spatial patches of 15 derived climate variables from NCEP/NCAR reanalysis NetCDF files, and local/regional DEM elevation patches from a GeoTIFF. These are the raw features that feed into all three models (LAND, Site MLP, Bernoulli-Gamma).

## Relation to the Deep Downscaling Paper
Hatanaka et al. (2025) Section 3a specifies the input features:
- **15 reanalysis channels** (paper has 16 — we lack `pottmp_diff_1000_850` due to missing data): surface and upper-air temperature, geopotential height, vertical velocity, precipitable water, humidity, moisture transport (zonal/meridional × two levels), skin temperature, sea-level pressure.
- **3×3 spatial patch** centred on the nearest reanalysis grid point to each station. At ~2.5° resolution, this covers roughly a 7.5°×7.5° box (~750 km).
- **Local DEM** (fine-scale topography) and **Regional DEM** (broader island shape). The paper uses fixed 3×3 patches; our implementation generates max-size patches for multi-resolution HP tuning.

## Line-by-Line Walkthrough

### Reanalysis Helper Functions (lines 29–56)

**`_get_nc_path()`** — Resolves a variable config entry to a NetCDF file path. Handles both standard naming (`{base}.day.mean.nc`) and custom file paths.

**`_detect_time_dim()`** — Tries `valid_time` then `time` as the temporal dimension name. ERA5 uses `valid_time`; NCEP uses `time`.

**`_detect_lat_lon()`** — Tries `latitude`/`longitude` then `lat`/`lon`.

**`_detect_level_dim()`** — Tries `pressure_level`, `level`, `lev`, `plev`, `isobaricInhPa` for the vertical dimension.

These detection functions make the code robust to different NetCDF conventions.

### `_extract_spatial_patch()` (lines 58–77)
```python
def _extract_spatial_patch(arr_2d, lats, lons, lat, lon, patch_size):
    ci = int(np.argmin(np.abs(lats - lat)))
    cj = int(np.argmin(np.abs(lons - lon)))
    half = patch_size // 2
    patch = arr_2d[i0:i0 + patch_size, j0:j0 + patch_size]
```
Finds the nearest grid point to the station's lat/lon, then extracts a square patch. Edge handling: if the patch extends beyond the grid boundary, missing cells are filled with NaN.

### `load_reanalysis_datasets()` (lines 80–109)
Loads all unique NetCDF files into memory. Deduplicates by file path — e.g. `uwnd.day.mean.nc` is loaded once even though it's used for both `zon_moist_750` and `zon_moist_925`. Each dataset is fully loaded into RAM via `ds.load()` for fast subsequent access.

### `_NumpyCube` class (lines 115–151)
Performance optimisation. Pre-extracts the xarray Dataset into pure numpy arrays and builds a `date → index` lookup dict. This avoids repeated xarray `.sel()` calls (which are slow due to label alignment overhead) in the inner loop.

Key fields:
- `self.data`: shape `(T, lat, lon)` or `(T, level, lat, lon)`.
- `self.date2idx`: `{datetime.date: int}` for O(1) temporal lookup.
- `self.lats`, `self.lons`: coordinate arrays for spatial indexing.

### `build_reanalysis_patches()` (lines 154–307) — The Main Extraction Loop

**Lines 167–207: Channel spec compilation**
Pre-compiles a list of "channel specs" — tuples describing how to compute each of the 15 channels:
- `("simple", nc_path, level)` — direct lookup at a pressure level.
- `("diff", nc_path, level0, level1)` — difference between two pressure levels.
- `("multiply", wind_nc, hum_nc, level)` — product of two fields (moisture transport = wind × humidity).

This avoids dictionary lookups and string comparisons in the hot inner loop.

**Lines 226–307: Station-day loop**
For each `(station, year, month, day)`:
1. Gets the date as `datetime.date`.
2. Allocates a `(15, 3, 3)` float32 buffer.
3. For each channel, dispatches to the appropriate operation (simple/diff/multiply).
4. Extracts the 3×3 spatial patch via numpy slicing centred on the pre-computed grid indices.
5. If any channel is missing data for this date, the entire sample is skipped.

Progress is reported every 10,000 station-days.

### DEM Functions (lines 310–419)

**`_latlon_to_metres()` (lines 314–318)**
Converts km_per_cell to approximate degrees for American Samoa (~14°S). Uses `cos(14°)` for longitude scaling.

**`_fill_nan_nearest()` (lines 321–350)**
Fills NaN cells (ocean pixels) from the nearest valid land pixel using an expanding-ring search. This ensures coastal stations still get topographic information even if their centre pixel falls in the ocean.

**`extract_dem_patch()` (lines 353–380)**
Extracts a single DEM patch from an open rasterio source:
1. Computes the lat/lon of each patch cell based on `km_per_cell` spacing.
2. Converts lat/lon to raster row/col via `rasterio.transform.rowcol`.
3. Reads the elevation value; sets to NaN if it equals the raster's nodata value.
4. Applies `_fill_nan_nearest` to fill ocean pixels.

**`build_dem_patches()` (lines 383–419)**
Iterates over all stations, extracting local and regional DEM patches. Accepts optional `local_cfg`/`regional_cfg` overrides — step 01 passes `DEM_MAX_LOCAL` and `DEM_MAX_REGIONAL` for max-size generation.

## Data Manipulation
- **No normalisation** — raw physical values are preserved. Normalisation happens in `dataset.py`.
- **NaN handling**: Missing reanalysis data causes sample skipping. Missing DEM data (ocean) is filled from neighbours.
- **Coordinate systems**: Reanalysis uses regular lat/lon grids (~2.5° spacing). DEM uses high-resolution raster coordinates (~30m native, sampled at km-scale spacing).

## Architecture Decisions
- **Pre-compiled channel specs**: Converting the config dict into tuples before the inner loop gives a ~3× speedup over the naive approach of re-parsing config per sample.
- **NumpyCube**: Pre-extracting xarray to numpy eliminates the overhead of xarray's label-based indexing, giving ~10× speedup in the inner loop.
- **Per-pixel DEM extraction**: Rather than extracting a raster window, each patch cell is sampled individually. This is necessary because the cells are spaced at `km_per_cell` intervals (not necessarily aligned with raster pixels), but it's slow for large patches.

## Areas of Improvement
- **Vectorised DEM extraction**: Using `rasterio.features.rasterize` or `rasterio.warp.reproject` to extract the entire patch as a reprojected window would be much faster than per-pixel sampling.
- **Parallel station processing**: The station loop is single-threaded. `multiprocessing.Pool` over stations would give near-linear speedup.
- **Dask for lazy NetCDF**: For larger datasets, using dask-backed xarray would avoid loading entire NetCDFs into RAM.
- **Missing channel recovery**: Currently, a missing channel for a single date causes the entire sample to be skipped. Imputation (e.g. using the previous day's value) could recover these samples.
- **The 15 vs 16 channel gap**: The paper specifies 16 channels including `pottmp_diff_1000_850`. Adding this channel (if the data becomes available) would match the paper exactly.
