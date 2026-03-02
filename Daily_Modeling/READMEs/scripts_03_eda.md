# `scripts/03_eda.py` — Exploratory Data Analysis

## Purpose
Third pipeline step. Produces visual and statistical summaries of the assembled dataset, broken down by spatio-temporal splits. Outputs include rainfall histograms, monthly seasonality curves, station sample counts, per-station rainfall distributions, a rainfall summary CSV, reanalysis variable summaries, and station group assignments.

## Relation to the Deep Downscaling Paper
The paper (Section 2) describes the American Samoa study region and its rainfall characteristics — high spatial variability driven by orographic effects, strong wet/dry seasonality, and a long-tailed daily rainfall distribution. This EDA script quantifies all of those properties for our specific dataset, allowing us to verify that the data matches expectations before modelling. The split-level breakdown also lets us check for distribution shift between train/val/test — a concern the paper addresses through spatio-temporal cross-validation.

## Line-by-Line Walkthrough

### Imports (lines 1–24)
```python
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz
from Daily_Modeling.data_utils.splits import assign_station_groups, spatiotemporal_split, ...
from Daily_Modeling.utils.visualization import plot_rainfall_histograms, ...
```
- Loads the assembled dataset as PyTorch tensors (CPU-only for EDA).
- Uses the same splitting logic as training to ensure the EDA reflects what models will see.

### `main()` function (lines 27–115)

**Lines 31–36: Load data**
```python
tensors, meta = load_tensors_from_npz(device=torch.device("cpu"))
stations = meta["stations"]
years = meta["years"]
months = meta["months"]
rain_mm = tensors["targets"].numpy()
```
Loads the full dataset onto CPU. Extracts stations/years/months as numpy arrays for split computation and plotting.

**Lines 43–46: Compute splits**
```python
yr_ranges = compute_station_year_ranges(stations, years)
groups = assign_station_groups(unique_stations, station_year_ranges=yr_ranges)
splits = spatiotemporal_split(stations, years, groups)
```
Uses the default config year ranges. Stations are assigned to train/val/test groups deterministically (seeded RNG). The splits produce 5 index arrays: `train`, `val_spatial`, `test_spatial`, `val_temporal`, `test_temporal`.

**Lines 48–63: Visual outputs**
Four visualisation calls:
1. `plot_rainfall_histograms` — Per-split rainfall distributions (log scale, shows heavy tail).
2. `plot_monthly_seasonality` — Monthly mean rainfall per split (shows wet season Nov–Mar).
3. `plot_station_sample_counts` — Bar chart of samples per station per split.
4. `plot_per_station_histograms` — Individual histograms for each of the ~26 stations.

**Lines 65–83: Rainfall summary table**
Computes per-split statistics: N, mean, std, min, percentiles (p50, p90, p95, p99), max, and percent zero days. Saved as `rainfall_summary.csv`. The high percentage of zero-rain days (~50–70%) is characteristic of tropical Pacific islands — this motivates the Bernoulli-Gamma model.

**Lines 85–107: Reanalysis variable means**
Computes per-channel, per-split means and standard deviations for all 15 reanalysis variables. Then calculates the **test–train shift** — large shifts could indicate temporal non-stationarity in climate variables, which would degrade model performance.

**Lines 109–113: Station group summary**
Prints which stations were assigned to train, val, and test groups.

## Data Manipulation
- **Splits are computed from scratch** using the same logic as training scripts. This ensures the EDA reflects what models actually see.
- **No normalisation** — all statistics are computed on raw values (mm, K, Pa, etc.) for interpretability.
- The `core_splits` dict only includes `train`, `val`, `test` (not the 5-way spatio-temporal split) for cleaner visualisations.

## Architecture Decisions
- **CPU-only**: EDA doesn't need GPU — keeps the script simple and runnable on any machine.
- **Split-aware**: By computing statistics per split, we can detect if the held-out data has a significantly different rainfall distribution, which would indicate a problem with the split strategy.

## Areas of Improvement
- **Spatial plots**: A map of American Samoa showing station locations coloured by their train/val/test role would be informative. The paper includes such a figure.
- **Correlation analysis**: Computing pairwise correlations between reanalysis variables and rainfall could identify the most predictive features.
- **Temporal autocorrelation**: Checking lag-1 autocorrelation in rainfall series would quantify how much temporal dependence exists — relevant for deciding whether to add temporal context to the models.
- **Runtime**: The per-station histogram generation is slow for many stations. Could be parallelised.
