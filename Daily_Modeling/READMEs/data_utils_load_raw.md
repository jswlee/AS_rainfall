# `data_utils/load_raw.py` — Raw Data Loading Utilities

## Purpose
Provides functions to load raw rainfall CSVs and station metadata for American Samoa. This is the lowest-level data access layer — all other modules build on top of it. Handles CSV parsing, unit conversion (inches → mm), date extraction, and station discovery.

## Relation to the Deep Downscaling Paper
The paper (Section 2) describes the American Samoa rain gauge network: ~26 stations with varying record lengths (some starting in the 1960s, others only recent). This module reads that raw data. The inches-to-mm conversion reflects that the original NOAA/NWS data is in US customary units.

## Line-by-Line Walkthrough

### `load_station_metadata()` (lines 15–29)
```python
def load_station_metadata(path=None) -> Dict[str, dict]:
    path = path or config.STATION_METADATA_PATH
    df = pd.read_csv(path)
    df = df.rename(columns={"Station": "station_name", "LAT": "latitude", "LONG": "longitude"})
    df = df.dropna(subset=["station_name", "latitude", "longitude"])
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df = df.drop_duplicates(subset=["station_name"])
```
1. Reads `raw_data/station_locations.csv`.
2. Renames columns to a consistent schema.
3. Drops rows with missing station name, lat, or lon.
4. Ensures lat/lon are float (they may be stored as strings in some CSVs).
5. Deduplicates by station name (takes the first occurrence).
6. Returns `{station_name: {latitude: float, longitude: float, ...}}`.

### `load_daily_rainfall()` (lines 32–93)
```python
def load_daily_rainfall(station_name, rainfall_dir=None, source_unit=None) -> Optional[pd.DataFrame]:
```
Loads a single station's CSV. The function is defensive about column naming:

**Lines 52–65: Datetime column detection**
Tries several common column names (`datetime`, `date`, `time`, `dt`) case-insensitively. If none found, checks for pre-existing `year`/`month`/`day` columns. This flexibility handles inconsistent CSV formats across stations.

**Lines 72–81: Precipitation column detection**
Tries `precip_in`, `precip`, `precipitation`, `rainfall`, `rain`, `prcp`, `precip_mm`, `rainfall_mm`. This handles the variety of column names in the raw data.

**Line 86: Unit conversion**
```python
df["rainfall_mm"] = df["rainfall_mm"] * 25.4
```
**All rainfall files are assumed to be in inches** (the `rainfall_corrected_NEW` directory convention). The factor 25.4 converts inches to millimetres. This is hardcoded — a potential issue if mixed-unit files are ever added.

**Lines 88–93: Clean and return**
Selects only `[year, month, day, rainfall_mm]`, drops NaN rows, casts types, returns a clean DataFrame.

### `load_all_station_rainfall()` (lines 102–121)
```python
def load_all_station_rainfall(station_metadata=None, rainfall_dir=None, min_days=365):
```
Iterates over all stations from metadata, loads each one, and keeps only those with at least `min_days` records. Default threshold of 365 (1 year) filters out stations with too little data to be useful.

### `discover_station_days()` (lines 124–148)
```python
def discover_station_days(station_metadata, rainfall_dir=None,
                          start_date="1980-01-01", end_date="2024-12-31"):
```
Returns `{station_name: [(year, month, day), ...]}` for all available days within the date range. This is used by `01_build_features.py` to determine which reanalysis time steps to extract — we only need reanalysis data for days when a station has a rainfall observation.

**Lines 139–140: Date filtering**
```python
df["date"] = pd.to_datetime(df[["year", "month", "day"]])
df = df[(df["date"] >= start) & (df["date"] <= end)]
```
Constructs proper datetime objects and filters to the requested range.

## Data Manipulation
- **No normalisation or splitting** — this module only reads raw data.
- **Inches → mm conversion**: Applied universally to all stations. The factor 25.4 is exact.
- **NaN handling**: Rows with NaN rainfall or dates are dropped silently. The number of dropped rows is not reported (could be improved).
- **Deduplication**: Station metadata is deduplicated by name. If a station appears twice with different coordinates, only the first is kept.

## Architecture Decisions
- **Defensive column detection**: Rather than requiring exact column names, the functions try multiple variants. This trades strictness for robustness against inconsistent CSV formats.
- **Optional path overrides**: Every function accepts optional path arguments, defaulting to `config.py` values. This enables testing with alternative data directories.
- **No caching**: Each call to `load_daily_rainfall` re-reads the CSV. For the ~26-station dataset this is fine, but would be slow for larger networks.

## Areas of Improvement
- **Configurable unit conversion**: The hardcoded `× 25.4` should be driven by metadata (e.g. a `source_unit` column in `station_locations.csv`). The `source_unit` parameter exists but is unused.
- **Caching**: An LRU cache on `load_daily_rainfall` would avoid redundant reads when the same station is loaded multiple times (e.g. during assembly + EDA).
- **Validation**: No range checks on rainfall values. Negative values, or values >500 mm/day, should be flagged as potential data errors.
- **Logging dropped rows**: When NaN rows are dropped, the count should be logged for data quality monitoring.
