# `data_utils/assemble_dataset.py` — Dataset Assembly

## Purpose
Combines the intermediate feature files (reanalysis patches, DEM patches) with raw rainfall observations into a single `daily_dataset.npz` that all downstream models consume. Handles the alignment between reanalysis samples, DEM (per-station), and rainfall (per station-day), plus month one-hot encoding.

## Relation to the Deep Downscaling Paper
The paper's models require four input modalities per sample: climate reanalysis, local DEM, regional DEM, and temporal encoding. This module constructs exactly that structure. The paper doesn't describe an explicit assembly step — it's an engineering detail, but one that's critical for correctness.

## Line-by-Line Walkthrough

### `_month_onehot()` (lines 23–27)
```python
def _month_onehot(months: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(months), 12), dtype=np.float32)
    oh[np.arange(len(months)), months - 1] = 1.0
    return oh
```
Converts integer months (1–12) to one-hot vectors (N, 12). The `months - 1` converts to 0-indexed for array assignment. The paper uses month encoding as a temporal feature — one-hot is the simplest representation that doesn't impose ordinal relationships between months.

### `assemble()` (lines 30–144) — Main Assembly Function

**Lines 39–44: Path resolution**
```python
out_path = config.ASSEMBLED_DIR / "daily_dataset.npz"
reanalysis_npz = config.FEATURES_DIR / "reanalysis_patches_daily.npz"
dem_npz = config.FEATURES_DIR / "dem_patches.npz"
```
All paths default to config values but can be overridden for testing.

**Lines 48–60: Load reanalysis patches**
```python
rz = np.load(str(reanalysis_npz), allow_pickle=True)
re_patches = rz["patches"]          # (N, C, H, W) = (N, 15, 3, 3)
re_stations = rz["stations"]         # (N,)  object array of station names
re_years = rz["years"]               # (N,)  int32
re_months = rz["months"]
re_days = rz["days"]
var_names = rz["variables"]
```
The reanalysis NPZ defines the sample universe — every sample is a `(station, year, month, day)` tuple for which reanalysis data was successfully extracted.

**Lines 62–69: Load DEM patches**
```python
dz = np.load(str(dem_npz), allow_pickle=True)
dem_local_raw = dz["dem_local_raw"]      # (S, H, W) — one per station
dem_regional_raw = dz["dem_regional_raw"]
dem_station_names = dz["stations"]
dem_lookup = {str(s): i for i, s in enumerate(dem_station_names)}
```
DEM patches are per-station (not per-day). The `dem_lookup` dict maps station name → index in the DEM arrays for O(1) access.

**Lines 72–83: Load rainfall**
```python
rain_lookup: Dict[str, Dict[tuple, float]] = {}
for sname in sorted(station_meta):
    df = load_daily_rainfall(sname)
    d = {}
    for _, row in df.iterrows():
        d[(int(row["year"]), int(row["month"]), int(row["day"]))] = float(row["rainfall_mm"])
    rain_lookup[sname] = d
```
Builds a nested dictionary `{station: {(y,m,d): rainfall_mm}}` for O(1) rainfall lookup by station and date.

**Lines 86–113: Alignment loop**
```python
for i in range(N):
    st = str(re_stations[i])
    y, m, d = int(re_years[i]), int(re_months[i]), int(re_days[i])
    # Look up DEM index for this station
    di = dem_lookup.get(st)
    if di is None: continue
    dem_local[i] = dem_local_raw[di]
    dem_regional[i] = dem_regional_raw[di]
    # Look up rainfall for this station-day
    rain_val = rl.get((y, m, d))
    if rain_val is None: continue
    rainfall_mm[i] = rain_val
    keep[i] = True
```
For each reanalysis sample:
1. Look up the station's DEM arrays via `dem_lookup`.
2. Look up the rainfall value via `rain_lookup`.
3. If either is missing, skip the sample.
4. Copy the DEM arrays into the pre-allocated output arrays.
5. Mark the sample as kept.

The DEM arrays are **duplicated** for every day of the same station — this is memory-inefficient but simplifies the DataLoader (no join at training time).

**Lines 114–124: Filter to kept samples**
```python
idx = np.where(keep)[0]
re_patches = re_patches[idx]
dem_local = dem_local[idx]
...
```
Selects only the successfully aligned samples.

**Lines 126–143: One-hot + save**
Generates month one-hot from the filtered months array, then saves everything to a compressed NPZ with keys: `reanalysis_patches`, `dem_local_raw`, `dem_regional_raw`, `month_onehot`, `rainfall_mm_raw`, `stations`, `years`, `months`, `days`, `variables`.

## Data Manipulation
- **Inner join**: The alignment is effectively an inner join of reanalysis × DEM × rainfall on (station, year, month, day). Samples missing any component are dropped.
- **DEM duplication**: Each station's DEM is copied N_days times. For a station with 10,000 days and 11×11 DEM patches, this means 10,000 × (11×11 + 25×25) × 4 bytes ≈ 30 MB per station. Not ideal but manageable for ~26 stations.
- **No normalisation**: All values are raw (physical units). Normalisation is deferred to `dataset.py:normalize_tensors()` so train-only statistics can be used.

## Architecture Decisions
- **Single-file output**: One NPZ contains everything models need. This simplifies the DataLoader and ensures consistency (no risk of mismatched sample ordering between separate files).
- **Compressed NPZ**: `np.savez_compressed` uses zlib, typically achieving 3–5× compression on float32 arrays. The ~200 MB raw dataset compresses to ~60 MB.
- **Pre-allocated arrays**: The DEM and rainfall output arrays are pre-allocated to full size N, then filtered. This avoids slow list appending for large N.

## Areas of Improvement
- **Store DEM once per station**: Instead of duplicating DEM arrays per sample, store a station→DEM mapping and join at DataLoader time. Would reduce NPZ size by ~10×.
- **Parquet for metadata**: The (station, year, month, day) metadata could be stored as a Parquet DataFrame for efficient querying.
- **Streaming assembly**: For very large datasets, the entire reanalysis array might not fit in RAM. Chunk-based processing would fix this.
- **Rainfall iterrows**: Line 81 uses `df.iterrows()` which is slow for large DataFrames. `df.to_dict('records')` or vectorised operations would be faster.
