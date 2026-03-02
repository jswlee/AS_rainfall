# `scripts/03b_inspect_dataset.py` — Comprehensive Dataset Inspection

## Purpose
Supplementary EDA step. Produces a full visual and statistical audit of `daily_dataset.npz` so anyone can verify the data being fed into models is correct. Covers raw NPZ structure, NaN audit, feature distributions, reanalysis channel correlations, sample DEM/reanalysis patches, per-station DEM summaries, and a side-by-side normalisation verification.

## Relation to the Deep Downscaling Paper
The paper does not describe a dataset inspection step, but this is essential engineering practice for reproducibility. The normalisation verification (step 9 in this script) directly validates that the z-score normalisation applied to reanalysis channels and DEM patches matches what the LAND model expects. Incorrect normalisation would silently degrade model performance.

## Line-by-Line Walkthrough

### Imports (lines 1–44)
Loads visualisation utilities for NaN audits, feature distributions, DEM patch visualisations, reanalysis patch heatmaps, and normalisation comparison plots. Uses both raw numpy loading and the PyTorch-based `load_tensors_from_npz` for normalisation verification.

### `main()` function (lines 47–219)

**Step 1 (lines 56–79): NPZ structure summary**
Opens the raw NPZ and iterates over every key, printing shape, dtype, value range, and NaN count. This is the first sanity check — if shapes are wrong, everything downstream will break. Saved to `npz_structure.txt`.

**Step 2 (lines 95–103): NaN audit**
Calls `plot_nan_audit()` which creates a bar chart showing the number of NaN values per array. Reanalysis patches may have NaNs at domain boundaries; DEM patches may have NaNs over ocean (filled by `_fill_nan_nearest` but worth checking). Zero NaN counts for rainfall confirm no missing targets slipped through.

**Step 3 (lines 105–113): Raw feature distributions**
Histograms of all features before normalisation. Climate values span physical units (K for temperature, Pa for pressure, etc.). DEM values are in metres. Rainfall is in mm with a heavy right tail.

**Step 4 (lines 115–122): Per-channel reanalysis distributions**
One histogram per reanalysis channel (all 15), showing the raw value distributions. Useful for spotting channels with unusual ranges or bimodal distributions.

**Step 5 (lines 124–131): Reanalysis correlation**
Pearson correlation heatmap across the 15 channels. Highly correlated channels (e.g. `shum_750` and `zon_moist_750`) may indicate redundancy, though the LAND model's Conv2D branch can learn to handle this.

**Step 6 (lines 133–140): Sample DEM patches**
Visualises 8 sample DEM patches (local + regional) to verify they capture reasonable topography around each station. Local patches should show fine-scale ridge/valley structure; regional patches should show the broader island shape.

**Step 7 (lines 142–153): Sample reanalysis patches**
For 3 representative stations, shows all 15 channels of a single reanalysis patch as small heatmaps. Verifies that the 3×3 spatial structure is preserved and that values look physically reasonable.

**Step 8 (lines 155–161): Per-station DEM summary**
Centre-pixel elevation for each station, comparing local vs regional DEM. Coastal stations should have low elevations; mountain stations should be high.

**Step 9 (lines 163–204): Normalisation verification**
The most critical audit section:
1. Loads tensors and computes splits (same as training scripts).
2. Keeps raw copies of climate and DEM arrays.
3. Applies `normalize_tensors()` using train-only statistics.
4. Plots raw vs normalised distributions side-by-side.
5. Also produces per-channel reanalysis distributions and correlations post-normalisation.

After normalisation, all climate channels should be roughly mean=0, std=1 on the training set. DEM should similarly be centred. If any channel has wildly different statistics, it indicates a normalisation bug.

**Step 10 (lines 206–214): Full normalisation report**
Calls `print_normalization_report()` which prints a formatted table of per-channel statistics. Also saves to `normalization_report.txt` for reference.

## Data Manipulation
- **No data is modified** — this script is read-only. The normalisation is applied to copies for verification purposes only.
- The `compute_station_year_ranges` and `assign_station_groups` calls reproduce the same splits as training, ensuring the normalisation verification uses the correct train indices.

## Architecture Decisions
- **Two-pass loading**: First loads the raw NPZ with numpy (for raw statistics), then loads again via `load_tensors_from_npz` (for normalisation verification). This is deliberate — the raw pass avoids any PyTorch transformations.
- **Comprehensive but non-blocking**: All visualisation calls are wrapped so a failure in one doesn't abort the rest.

## Areas of Improvement
- **Interactive HTML report**: A Plotly/Bokeh dashboard would be more navigable than 10+ static PNGs.
- **Automated anomaly flags**: Could automatically flag channels with >5% NaN, or normalised means deviating from 0 by more than 0.1.
- **DEM spatial plots on a map**: Overlaying DEM patches on an actual geographic map would be more informative than standalone heatmaps.
