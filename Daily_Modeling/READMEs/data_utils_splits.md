# `data_utils/splits.py` — Spatio-Temporal Cross-Validation Splits

## Purpose
Implements all data splitting strategies used across the pipeline: spatio-temporal splits (for LAND), temporal-only splits, per-station proportional splits (for site-specific models), data-driven year boundary computation, station group assignment, and split heatmap visualisation. This is the most methodologically important module — incorrect splits would invalidate all model evaluations.

## Relation to the Deep Downscaling Paper
Hatanaka et al. (2025) Section 3d describes the spatio-temporal cross-validation scheme:
- **Spatial generalisation**: Hold out entire stations that the model never sees during training. Tests whether LAND can downscale at new locations.
- **Temporal generalisation**: Use chronological year splits so the model never sees future data during training. Tests robustness to climate change.
- **Combined**: The test set includes held-out stations in future years — the hardest evaluation.

Our implementation produces five named splits: `train`, `val_spatial`, `test_spatial`, `val_temporal`, `test_temporal`. The paper uses a similar scheme but doesn't name the splits the same way.

For site-specific models (MLP, Bernoulli-Gamma), the paper trains per-station with chronological splits. Our `station_proportional_split` implements this as a 70/20/10 chronological split per station.

## Line-by-Line Walkthrough

### `compute_station_year_ranges()` (lines 30–41)
```python
def compute_station_year_ranges(stations, years):
    for s in np.unique(stations):
        mask = stations == s
        yrs = years[mask].astype(int)
        ranges[str(s)] = (int(yrs.min()), int(yrs.max()))
```
Returns `{station_name: (min_year, max_year)}` from the actual data. This is used to ensure that stations assigned to val/test roles actually have data in the val/test year ranges. Without this check, a station with data only from 1980–1995 could be assigned to the test group (2009–2024) and produce an empty test set.

### `compute_year_boundaries()` (lines 44–75)
```python
def compute_year_boundaries(years, train_frac=0.70, val_frac=0.20):
    yr = np.sort(years.astype(int))
    n = len(yr)
    train_end_idx = int(n * train_frac) - 1
    val_end_idx = int(n * (train_frac + val_frac)) - 1
    train_end_year = int(yr[train_end_idx])
    val_end_year = int(yr[val_end_idx])
```
Computes chronological year cutoffs so that approximately 70% of samples fall in train years, 20% in val years, and 10% in test years. This is **data-driven** — the actual boundaries depend on the sample distribution across years. Stations with more historical data shift the boundaries earlier.

Returns three `(start_year, end_year)` tuples (inclusive). Also prints a summary showing exact sample counts and percentages.

**Design decision**: The fractions are by sample count, not by year count. This means if early years have fewer stations, more years are allocated to training to reach 70% of samples.

### `assign_station_groups()` (lines 80–129)
```python
def assign_station_groups(station_names, n_val=5, n_test=3, seed=42,
                          station_year_ranges=None, val_years=None, test_years=None):
```
Deterministically assigns each station to `train`, `val`, or `test`:

1. **Eligibility filtering** (lines 97–110): If `station_year_ranges` is provided, only stations whose data overlaps the val/test year ranges are eligible for those roles. This prevents assigning a short-record station to a test period it has no data for.

2. **Deterministic shuffling** (lines 112–119): Uses `np.random.RandomState(seed)` to shuffle eligible stations, then picks the first `n_test` for test, the first `n_val` (excluding test) for val, and the rest for train.

With ~26 stations, the default assignment is: 3 test, 5 val, ~18 train. The paper uses a similar ratio.

**Key property**: The assignment is deterministic given the seed and station list. Changing the station list (e.g. adding a new station) will change all assignments — this is a limitation.

### `spatiotemporal_split()` (lines 134–177)
```python
def spatiotemporal_split(stations, years, station_groups,
                         train_years, val_years, test_years):
```
The core split function for LAND. Creates five index arrays by combining station roles and year ranges:

| Split | Stations | Years | Purpose |
|-------|----------|-------|---------|
| `train` | train group | train years | Model training |
| `val_spatial` | val group | val years | Spatial + temporal generalisation |
| `test_spatial` | test group | test years | Hardest evaluation |
| `val_temporal` | train group | val years | Temporal-only generalisation |
| `test_temporal` | train group | test years | Temporal-only generalisation |

The implementation uses boolean mask arrays:
```python
train_mask = (roles == "train") & (yr >= train_years[0]) & (yr <= train_years[1])
```
Each mask is applied to `np.arange(n)` to get index arrays.

### `temporal_split()` (lines 182–198)
Simple year-based split (train/val/test) without station grouping. Not currently used in the pipeline but available for simpler experiments.

### `station_temporal_split()` (lines 203–220)
Per-station year-based split. Filters to a single station, then splits by year ranges. Not currently used — superseded by `station_proportional_split`.

### `station_proportional_split()` (lines 223–260)
```python
def station_proportional_split(stations, years, months, days, target_station,
                                train_frac=0.70, val_frac=0.20):
    mask = np.array([str(s) == target_station for s in stations])
    idx = np.where(mask)[0]
    yr = years[idx].astype(int)
    mo = months[idx].astype(int)
    dy = days[idx].astype(int)
    date_order = np.lexsort((dy, mo, yr))
    sorted_idx = idx[date_order]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": sorted_idx[:n_train],
        "val": sorted_idx[n_train:n_train + n_val],
        "test": sorted_idx[n_train + n_val:],
    }
```
Used by site-specific models (MLP, Bernoulli-Gamma). For a single station:
1. Finds all indices belonging to that station.
2. Sorts by date using `np.lexsort` (sorts by day within month within year).
3. Takes the first 70% as train, next 20% as val, remainder as test.

**Key property**: Every station with sufficient data gets all three splits, regardless of its year range. A station with data only from 2000–2015 still gets a 70/20/10 split within that range. This is different from the global year-based split used for LAND.

### `plot_split_heatmap()` (lines 265–353)
Creates a station × year heatmap showing which cells belong to which split. Uses a custom 7-colour colormap:
- White = no data
- Blue = train
- Green = val_spatial
- Red = test_spatial
- Light green = val_temporal
- Orange = test_temporal
- Grey = unused (val/test station in train years — these samples are neither trained on nor evaluated)

## Data Manipulation
- **Index-based**: All split functions return numpy arrays of integer indices, not the data itself. This allows the same split to be applied to multiple tensor arrays.
- **Chronological ordering**: Both year-based and proportional splits respect temporal ordering — no future data leaks into training.
- **Deterministic**: All randomness (station assignment) uses a fixed seed.

## Architecture Decisions
- **Five splits, not three**: The 5-way split (with separate spatial and temporal evaluation) provides more diagnostic information than a simple train/val/test split. It reveals whether performance drops are due to spatial or temporal generalisation failure.
- **Data-driven year boundaries**: Rather than hardcoding year ranges, computing them from data ensures consistent train/val/test proportions regardless of data additions.
- **Proportional per-station splits**: For site-specific models, this guarantees every station gets evaluation data. A global year-based split might leave some short-record stations with no test data.

## Areas of Improvement
- **Stratified station assignment**: Currently stations are assigned randomly (seeded). Stratifying by elevation, location (windward/leeward), or record length would produce more representative val/test sets.
- **Blocked time series CV**: The current single chronological split doesn't account for temporal autocorrelation. Blocked k-fold CV (e.g. 5 non-overlapping year blocks) would give more robust performance estimates.
- **Leave-one-station-out CV**: For spatial generalisation, LOSO CV would evaluate on every station, not just the 3 held-out ones. More expensive but more informative.
- **Gap between train and val years**: There's no temporal gap between the last train year and the first val year. A 1–2 year gap would prevent leakage from lagged temporal autocorrelation.
- **Station assignment stability**: Adding or removing a station changes all assignments. A hash-based assignment (e.g. `hash(station_name) % 10`) would be stable under station list changes.
