# `scripts/02_assemble_dataset.py` — Assemble the Unified Dataset

## Purpose
Second pipeline step. Combines the intermediate reanalysis patches, DEM patches, and raw rainfall CSVs into a single `daily_dataset.npz` that all downstream models consume. This is a thin wrapper around `data_utils/assemble_dataset.py:assemble()`.

## Relation to the Deep Downscaling Paper
The paper's LAND model (Section 3a) requires four input tensors per sample: (1) reanalysis climate patch, (2) local DEM, (3) regional DEM, and (4) month encoding. This script produces exactly that structure, plus the rainfall target. The paper does not describe an explicit assembly step, but our pipeline separates it for reproducibility and debuggability.

## Line-by-Line Walkthrough

### Lines 1–9: Docstring and import
```python
from Daily_Modeling.data_utils.assemble_dataset import assemble
```
The entire logic lives in `assemble_dataset.py`. This script is intentionally minimal — a single function call.

### Lines 14–15: `main()`
```python
def main():
    assemble()
```
Calls `assemble()` with default paths from `config.py`. The function:
1. Loads `reanalysis_patches_daily.npz` (from step 01) — shape `(N, 15, 3, 3)`.
2. Loads `dem_patches.npz` (from step 01) — per-station local/regional DEM arrays.
3. Loads each station's rainfall CSV to build a `{station: {(y,m,d): mm}}` lookup.
4. Iterates over every reanalysis sample, looks up the matching DEM (by station name) and rainfall (by station + date).
5. Drops samples where either DEM or rainfall is missing.
6. Generates month one-hot encoding `(N, 12)`.
7. Saves everything into `daily_dataset.npz`.

### Lines 18–19: Entry point
Standard `if __name__ == "__main__"` guard.

## Data Manipulation
- **Alignment**: The reanalysis patches define the sample universe. For each reanalysis sample `(station, year, month, day)`, the script looks up:
  - The DEM arrays via station name (DEM is static per station, not per day).
  - The rainfall via `(year, month, day)` key.
- **Dropping**: Samples without matching DEM or rainfall are dropped. The typical drop rate is <1% (stations with DEM issues or rainfall gaps).
- **Month one-hot**: Integer months (1–12) are converted to 12-dimensional binary vectors.

## Architecture Decisions
- **Single NPZ**: All features + targets + metadata in one file simplifies downstream loading. The `load_tensors_from_npz()` function in `dataset.py` reads this directly into GPU tensors.
- **Raw rainfall preserved**: The rainfall is stored in millimetres (not normalised). Normalisation happens later in `dataset.py:normalize_tensors()` so that train-only statistics are computed after splitting.
- **DEM arrays are repeated per sample**: The same station's DEM is stored N times (once per sample day). This is memory-inefficient but makes the DataLoader simpler — no join at training time.

## Areas of Improvement
- **Store DEM per-station, not per-sample**: The NPZ currently duplicates DEM arrays across all days for the same station. Storing them once per station and joining at DataLoader time would reduce file size by ~10×.
- **Parquet or HDF5 instead of NPZ**: NPZ doesn't support lazy loading. HDF5 with chunked arrays would allow loading subsets without reading the entire file into RAM.
- **Rainfall conversion**: The inches→mm conversion is hardcoded in `load_raw.py`. This could be configurable per station if mixed-unit sources are ever added.
