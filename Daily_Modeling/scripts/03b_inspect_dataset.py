"""
Step 3b: Comprehensive inspection of the assembled daily_dataset.npz.

Produces a full visual and statistical audit so anyone can verify the data
being fed into models is correct.  Outputs include:

  1. NPZ structure summary (shapes, dtypes, value ranges, NaN counts)
  2. NaN audit bar chart
  3. Raw feature distribution histograms (climate, DEM, rainfall)
  4. Per-channel reanalysis distributions (all 15 variables)
  5. Reanalysis channel correlation heatmap
  6. Sample DEM patches (local + regional) for representative stations
  7. Sample reanalysis patches (all channels for a single sample)
  8. Per-station DEM centre-pixel elevation summary
  9. Normalisation verification: raw vs normalised side-by-side
 10. Full normalisation report (text) - same as what training scripts print

Usage:
    python -m Daily_Modeling.scripts.03b_inspect_dataset
"""

import contextlib
import io
import sys
import argparse

import numpy as np
import pandas as pd

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz, normalize_tensors, print_normalization_report
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, compute_station_year_ranges,
    compute_year_boundaries,
)
from Daily_Modeling.utils.device import select_device
from Daily_Modeling.utils.visualization import (
    plot_sample_dem_patches,
    plot_sample_reanalysis_patches,
    plot_nan_audit,
    plot_feature_distributions,
    plot_reanalysis_channel_distributions,
    plot_reanalysis_correlation,
    plot_normalization_comparison,
    plot_per_station_dem_summary,
    plot_stations_on_dem_raster,
)


def _build_dem_coverage_audit(local_dem_raw, regional_dem_raw, stations) -> pd.DataFrame:
    rows = []
    unique_st = np.unique(stations)
    for st in unique_st:
        idx = np.where(stations == st)[0][0]
        ld_full = local_dem_raw[idx]
        rd_full = regional_dem_raw[idx]
        # Support both (H,W) single-band and (n_bands,H,W) multi-band arrays
        ld = ld_full[0] if ld_full.ndim == 3 else ld_full
        rd = rd_full[0] if rd_full.ndim == 3 else rd_full
        ld_valid = ld > -1
        rd_valid = rd > -1
        rows.append({
            "station": str(st),
            "local_valid_cells": int(ld_valid.sum()),
            "local_total_cells": int(ld.size),
            "local_valid_frac": float(ld_valid.mean()),
            "local_center_m": float(ld[ld.shape[0] // 2, ld.shape[1] // 2]),
            "regional_valid_cells": int(rd_valid.sum()),
            "regional_total_cells": int(rd.size),
            "regional_valid_frac": float(rd_valid.mean()),
            "regional_center_m": float(rd[rd.shape[0] // 2, rd.shape[1] // 2]),
        })
    return pd.DataFrame(rows).sort_values(["local_valid_frac", "regional_valid_frac", "station"])


def main():
    out = config.EDA_DIR / "dataset_inspection"
    out.mkdir(parents=True, exist_ok=True)

    npz_path = config.ASSEMBLED_DIR / "daily_dataset_station_centered.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run steps 01 + 02 first.")
        sys.exit(1)

    # === 1. Raw NPZ structure summary ===
    print("=" * 70)
    print("  RAW NPZ STRUCTURE - daily_dataset.npz")
    print("=" * 70)
    z = np.load(str(npz_path), allow_pickle=True)
    summary_lines = []
    for key in z.files:
        arr = z[key]
        flat = arr.ravel()
        is_numeric = np.issubdtype(arr.dtype, np.number)
        if is_numeric:
            n_nan = int(np.isnan(arr.astype(np.float64)).sum())
            finite = flat[np.isfinite(flat.astype(np.float64))]
            rng = f"[{finite.min():.4f}, {finite.max():.4f}]" if len(finite) > 0 else "[all NaN]"
        else:
            n_nan = 0
            rng = f"[{arr[0]}..{arr[-1]}]" if len(arr) > 0 else "[]"
        line = (f"  {key:<25s}  dtype={str(arr.dtype):<10s}  shape={str(arr.shape):<25s}  "
                f"NaN={n_nan:>8,}  range={rng}")
        print(line)
        summary_lines.append(line)

    # Save text summary
    (out / "npz_structure.txt").write_text("\n".join(summary_lines))
    print(f"\n  Saved text summary to {out / 'npz_structure.txt'}")

    # === Prepare raw numpy arrays for plotting ===
    reanalysis_raw = z["reanalysis_patches"]
    local_dem_raw = z["dem_local_raw"]
    regional_dem_raw = z["dem_regional_raw"]
    month_oh = z["month_onehot"]
    rainfall_raw = z["rainfall_mm_raw"]
    stations = z["stations"]
    years = z["years"]
    months = z["months"]
    variables = z["variables"] if "variables" in z.files else np.array([f"ch{i}" for i in range(reanalysis_raw.shape[1])])
    var_names = list(variables)
    z.close()

    # === 2. NaN audit ===
    print("\n--- NaN Audit ---")
    plot_nan_audit(
        {"reanalysis": reanalysis_raw, "local_dem": local_dem_raw,
         "regional_dem": regional_dem_raw, "month_onehot": month_oh,
         "rainfall_mm": rainfall_raw},
        save_path=out / "nan_audit.png",
    )
    print("  Saved nan_audit.png")

    # === 3. Raw feature distributions ===
    print("\n--- Raw Feature Distributions ---")
    plot_feature_distributions(
        {"reanalysis (all ch)": reanalysis_raw, "local_dem": local_dem_raw,
         "regional_dem": regional_dem_raw, "rainfall_mm": rainfall_raw},
        save_path=out / "feature_distributions_raw.png",
        tag="raw",
    )
    print("  Saved feature_distributions_raw.png")

    # === 4. Per-channel reanalysis distributions ===
    print("\n--- Reanalysis Channel Distributions ---")
    plot_reanalysis_channel_distributions(
        reanalysis_raw, var_names,
        save_path=out / "reanalysis_channel_distributions_raw.png",
        tag="raw",
    )
    print("  Saved reanalysis_channel_distributions_raw.png")

    # === 5. Reanalysis correlation ===
    print("\n--- Reanalysis Channel Correlation ---")
    plot_reanalysis_correlation(
        reanalysis_raw, var_names,
        save_path=out / "reanalysis_correlation_raw.png",
        tag="raw",
    )
    print("  Saved reanalysis_correlation_raw.png")

    # === 6. Sample DEM patches ===
    print("\n--- Sample DEM Patches ---")
    plot_sample_dem_patches(
        local_dem_raw, regional_dem_raw, stations,
        n_samples=8,
        save_path=out / "sample_dem_patches.png",
    )
    print("  Saved sample_dem_patches.png")

    # === DEM land-coverage audit ===
    dem_coverage_df = _build_dem_coverage_audit(local_dem_raw, regional_dem_raw, stations)
    dem_coverage_df.to_csv(out / "dem_patch_coverage_by_station.csv", index=False)
    worst_local = dem_coverage_df.nsmallest(5, "local_valid_frac")
    worst_regional = dem_coverage_df.nsmallest(5, "regional_valid_frac")
    dem_summary_lines = [
        "DEM patch land-coverage audit (cells with elevation > -1)",
        "",
        f"Stations audited: {len(dem_coverage_df)}",
        f"Local valid fraction: mean={dem_coverage_df['local_valid_frac'].mean():.4f}, median={dem_coverage_df['local_valid_frac'].median():.4f}, min={dem_coverage_df['local_valid_frac'].min():.4f}, max={dem_coverage_df['local_valid_frac'].max():.4f}",
        f"Regional valid fraction: mean={dem_coverage_df['regional_valid_frac'].mean():.4f}, median={dem_coverage_df['regional_valid_frac'].median():.4f}, min={dem_coverage_df['regional_valid_frac'].min():.4f}, max={dem_coverage_df['regional_valid_frac'].max():.4f}",
        "",
        "Worst 5 stations by local valid fraction:",
    ]
    for _, row in worst_local.iterrows():
        dem_summary_lines.append(
            f"  {row['station']}: local={row['local_valid_cells']}/{row['local_total_cells']} ({row['local_valid_frac']:.4f}), regional={row['regional_valid_cells']}/{row['regional_total_cells']} ({row['regional_valid_frac']:.4f}), local_center_m={row['local_center_m']:.3f}, regional_center_m={row['regional_center_m']:.3f}"
        )
    dem_summary_lines.append("")
    dem_summary_lines.append("Worst 5 stations by regional valid fraction:")
    for _, row in worst_regional.iterrows():
        dem_summary_lines.append(
            f"  {row['station']}: local={row['local_valid_cells']}/{row['local_total_cells']} ({row['local_valid_frac']:.4f}), regional={row['regional_valid_cells']}/{row['regional_total_cells']} ({row['regional_valid_frac']:.4f}), local_center_m={row['local_center_m']:.3f}, regional_center_m={row['regional_center_m']:.3f}"
        )
    (out / "dem_patch_coverage_summary.txt").write_text("\n".join(dem_summary_lines))
    print("  Saved dem_patch_coverage_by_station.csv")
    print("  Saved dem_patch_coverage_summary.txt")

    # === 7. Sample reanalysis patches (3 different samples) ===
    print("\n--- Sample Reanalysis Patches ---")
    unique_st = np.unique(stations)
    pick_3 = unique_st[np.linspace(0, len(unique_st) - 1, 3, dtype=int)]
    for st_name in pick_3:
        idx = int(np.where(stations == st_name)[0][0])
        safe = str(st_name).replace(" ", "_")
        plot_sample_reanalysis_patches(
            reanalysis_raw, var_names, stations, sample_idx=idx,
            save_path=out / f"reanalysis_patch_{safe}.png",
        )
        print(f"  Saved reanalysis_patch_{safe}.png  (sample #{idx})")

    # === 8. Per-station DEM summary ===
    print("\n--- Per-Station DEM Elevation ---")
    plot_per_station_dem_summary(
        local_dem_raw, regional_dem_raw, stations,
        save_path=out / "station_dem_summary.png",
    )
    print("  Saved station_dem_summary.png")

    # === 9. Normalisation verification ===
    print("\n--- Normalisation Verification ---")
    tensors, meta = load_tensors_from_npz(device=select_device())
    train_yr, val_yr, test_yr = compute_year_boundaries(meta["years"])
    yr_ranges = compute_station_year_ranges(meta["stations"], meta["years"])
    unique = sorted(set(str(s) for s in meta["stations"]))
    groups = assign_station_groups(unique, station_year_ranges=yr_ranges,
                                   val_years=val_yr, test_years=test_yr)
    splits = spatiotemporal_split(meta["stations"], meta["years"], groups,
                                  train_years=train_yr, val_years=val_yr, test_years=test_yr)

    # Keep raw copies before normalisation
    raw_climate = tensors["climate"].cpu().numpy().copy()
    raw_local_dem = tensors["local_dem"].cpu().numpy().copy()
    raw_regional_dem = tensors["regional_dem"].cpu().numpy().copy()

    tensors, stats = normalize_tensors(tensors, splits["train"])

    # Side-by-side raw vs normalised
    plot_normalization_comparison(
        raw_arrays={"climate (all ch)": raw_climate, "local_dem": raw_local_dem, "regional_dem": raw_regional_dem},
        norm_arrays={
            "climate (all ch)": tensors["climate"].cpu().numpy(),
            "local_dem": tensors["local_dem"].cpu().numpy(),
            "regional_dem": tensors["regional_dem"].cpu().numpy(),
        },
        save_path=out / "normalization_comparison.png",
    )
    print("  Saved normalization_comparison.png")

    # Per-channel reanalysis distributions AFTER normalisation
    plot_reanalysis_channel_distributions(
        tensors["climate"].cpu().numpy(), var_names,
        save_path=out / "reanalysis_channel_distributions_normalised.png",
        tag="normalised",
    )
    print("  Saved reanalysis_channel_distributions_normalised.png")

    # Reanalysis correlation AFTER normalisation
    plot_reanalysis_correlation(
        tensors["climate"].cpu().numpy(), var_names,
        save_path=out / "reanalysis_correlation_normalised.png",
        tag="normalised",
    )
    print("  Saved reanalysis_correlation_normalised.png")

    # === 10. Full normalisation report (text) ===
    print_normalization_report(tensors, stats, splits, variable_names=var_names)

    # Save report to text file too
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_normalization_report(tensors, stats, splits, variable_names=var_names)
    (out / "normalization_report.txt").write_text(buf.getvalue())
    print(f"  Saved normalization_report.txt")

    # === 11. DEM patches on geographic map ===
    print("\n--- DEM Patches on Geographic Map ---")
    try:
        from Daily_Modeling.data_utils.load_raw import load_station_metadata
        from Daily_Modeling.utils.visualization import plot_dem_on_map
        station_meta = load_station_metadata()
        plot_dem_on_map(
            local_dem_raw, regional_dem_raw, stations, station_meta,
            n_samples=8, save_path=out / "dem_on_map.png",
        )
        print("  Saved dem_on_map.png")
    except Exception as e:
        print(f"  WARNING: Could not generate DEM-on-map: {e}")

    # === 11b. Stations over full DEM raster (CRS-correct sanity check) ===
    print("\n--- Stations Over Full DEM Raster (CRS sanity check) ---")
    try:
        from Daily_Modeling.data_utils.load_raw import load_station_metadata
        station_meta = load_station_metadata()
        plot_stations_on_dem_raster(
            config.DEM_PATH,
            station_meta,
            save_path=out / "stations_on_dem_full.png",
            title="Stations over full DEM (CRS-correct)",
        )
        print("  Saved stations_on_dem_full.png")
    except Exception as e:
        print(f"  WARNING: Could not generate stations_on_dem_full.png: {e}")

    # === 12. Interactive HTML report (Plotly) ===
    print("\n--- Interactive HTML Report ---")
    try:
        _build_interactive_report(
            out, reanalysis_raw, local_dem_raw, regional_dem_raw,
            rainfall_raw, stations, years, months, var_names,
            tensors, stats, splits,
        )
        print(f"  Saved interactive_report.html")
    except Exception as e:
        print(f"  WARNING: Could not generate interactive HTML report: {e}")
        import traceback; traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"  All dataset inspection outputs saved to {out}")
    print(f"{'=' * 70}")


def _build_interactive_report(
    out_dir, reanalysis_raw, local_dem_raw, regional_dem_raw,
    rainfall_raw, stations, years, months, var_names,
    tensors, stats, splits,
):
    """Build a single interactive HTML dashboard using Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Dataset Inspection Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f8f9fa; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #34495e; margin-top: 40px; }
  .section { background: white; padding: 20px; margin: 15px 0; border-radius: 8px;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap: 10px; margin: 10px 0; }
  .stat-card { background: #ecf0f1; padding: 12px; border-radius: 6px; text-align: center; }
  .stat-card .value { font-size: 1.4em; font-weight: bold; color: #2c3e50; }
  .stat-card .label { font-size: 0.85em; color: #7f8c8d; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; }
  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
  th { background: #3498db; color: white; }
  tr:nth-child(even) { background: #f9f9f9; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }
</style>
</head><body>
<h1>Dataset Inspection Report</h1>
""")

    # === Overview stats ===
    n_samples = len(rainfall_raw)
    n_stations = len(np.unique(stations))
    unique_years = np.unique(years)
    n_years = len(unique_years)
    pct_zero = 100.0 * np.mean(rainfall_raw == 0)
    html_parts.append(f"""
<div class="section">
<h2>Overview</h2>
<div class="stat-grid">
  <div class="stat-card"><div class="value">{n_samples:,}</div><div class="label">Total Samples</div></div>
  <div class="stat-card"><div class="value">{n_stations}</div><div class="label">Stations</div></div>
  <div class="stat-card"><div class="value">{unique_years.min()}-{unique_years.max()}</div><div class="label">Year Range</div></div>
  <div class="stat-card"><div class="value">{n_years}</div><div class="label">Unique Years</div></div>
  <div class="stat-card"><div class="value">{reanalysis_raw.shape[1]}</div><div class="label">Climate Channels</div></div>
  <div class="stat-card"><div class="value">{rainfall_raw.mean():.1f} mm</div><div class="label">Mean Rainfall</div></div>
  <div class="stat-card"><div class="value">{pct_zero:.1f}%</div><div class="label">Zero-Rain Days</div></div>
  <div class="stat-card"><div class="value">{rainfall_raw.max():.1f} mm</div><div class="label">Max Rainfall</div></div>
</div>
</div>
""")

    # === Split breakdown ===
    html_parts.append('<div class="section"><h2>Train/Val/Test Splits</h2><table>')
    html_parts.append('<tr><th>Split</th><th>Samples</th><th>%</th><th>Mean Rain (mm)</th><th>% Zero</th></tr>')
    for split_name in ["train", "val_spatial", "test_spatial", "val_temporal", "test_temporal"]:
        if split_name in splits:
            idx = splits[split_name]
            n = len(idx)
            pct = 100.0 * n / n_samples
            rain_split = rainfall_raw[idx]
            mean_r = rain_split.mean()
            pct_z = 100.0 * np.mean(rain_split == 0)
            html_parts.append(f'<tr><td>{split_name}</td><td>{n:,}</td><td>{pct:.1f}%</td><td>{mean_r:.2f}</td><td>{pct_z:.1f}%</td></tr>')
    html_parts.append('</table></div>')

    # === Rainfall distribution (use lists, not numpy arrays for JSON) ===
    html_parts.append('<div class="section"><h2>Rainfall Distribution</h2>')
    rainy = rainfall_raw[rainfall_raw > 0]
    # Subsample for performance
    max_pts = 10000
    if len(rainy) > max_pts:
        rainy_sub = np.random.RandomState(42).choice(rainy, max_pts, replace=False)
    else:
        rainy_sub = rainy
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Rainy Days (mm)", "Log(1 + Rainfall)"])
    fig.add_trace(go.Histogram(x=rainy_sub.tolist(), nbinsx=50, marker_color="#3498db", name="Rainy"), row=1, col=1)
    log_rain = np.log1p(rainfall_raw)
    if len(log_rain) > max_pts:
        log_rain_sub = np.random.RandomState(42).choice(log_rain, max_pts, replace=False)
    else:
        log_rain_sub = log_rain
    fig.add_trace(go.Histogram(x=log_rain_sub.tolist(), nbinsx=50, marker_color="#e74c3c", name="Log"), row=1, col=2)
    fig.update_layout(height=300, showlegend=False, margin=dict(t=40, b=30, l=50, r=30))
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append('</div>')

    # === Monthly seasonality ===
    html_parts.append('<div class="section"><h2>Monthly Seasonality</h2>')
    df_m = pd.DataFrame({"month": months.astype(int), "rain": rainfall_raw})
    monthly = df_m.groupby("month")["rain"].agg(["mean", "std", "count"]).reset_index()
    se = (monthly["std"] / np.sqrt(monthly["count"])).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month"].tolist(),
        y=monthly["mean"].tolist(),
        error_y=dict(type="data", array=se),
        marker_color="#2ecc71",
        name="Mean ± SE"
    ))
    fig.update_layout(
        xaxis_title="Month", yaxis_title="Mean Rainfall (mm)",
        height=300, margin=dict(t=30, b=40, l=50, r=30)
    )
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append('</div>')

    # === Per-station sample counts ===
    html_parts.append('<div class="section"><h2>Samples per Station</h2>')
    st_counts = pd.Series([str(s) for s in stations]).value_counts().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        y=st_counts.index.tolist(),
        x=st_counts.values.tolist(),
        orientation="h",
        marker_color="#9b59b6"
    ))
    fig.update_layout(
        xaxis_title="Sample Count",
        height=max(300, 22 * n_stations),
        margin=dict(l=140, t=30, b=40, r=30)
    )
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append('</div>')

    # === Reanalysis channel statistics table ===
    html_parts.append('<div class="section"><h2>Reanalysis Channel Statistics (Raw)</h2><table>')
    html_parts.append('<tr><th>Channel</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>')
    for i, vn in enumerate(var_names):
        ch = reanalysis_raw[:, i].ravel()
        html_parts.append(f'<tr><td>{vn}</td><td>{ch.mean():.4g}</td><td>{ch.std():.4g}</td><td>{ch.min():.4g}</td><td>{ch.max():.4g}</td></tr>')
    html_parts.append('</table></div>')

    # === Reanalysis correlation heatmap ===
    html_parts.append('<div class="section"><h2>Reanalysis Channel Correlation</h2>')
    n_samp = min(5000, reanalysis_raw.shape[0])
    idx_samp = np.random.RandomState(42).choice(reanalysis_raw.shape[0], n_samp, replace=False)
    flat = reanalysis_raw[idx_samp].reshape(n_samp, reanalysis_raw.shape[1], -1).mean(axis=-1)
    corr = np.corrcoef(flat.T)
    fig = go.Figure(go.Heatmap(
        z=corr.tolist(),
        x=var_names,
        y=var_names,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr, 2).tolist(),
        texttemplate="%{text}",
        textfont={"size": 9}
    ))
    fig.update_layout(height=450, margin=dict(t=30, b=30, l=120, r=30))
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append('</div>')

    # === DEM statistics ===
    html_parts.append('<div class="section"><h2>DEM Elevation Statistics</h2>')
    html_parts.append('<div class="two-col"><div>')
    # For display: use elevation band only (band 0) if multi-band
    ld_elev = local_dem_raw[:, 0] if local_dem_raw.ndim == 4 else local_dem_raw
    rd_elev = regional_dem_raw[:, 0] if regional_dem_raw.ndim == 4 else regional_dem_raw
    ld_shape_str = f'{local_dem_raw.shape[2]}x{local_dem_raw.shape[3]}' if local_dem_raw.ndim == 4 else f'{local_dem_raw.shape[1]}x{local_dem_raw.shape[2]}'
    rd_shape_str = f'{regional_dem_raw.shape[2]}x{regional_dem_raw.shape[3]}' if regional_dem_raw.ndim == 4 else f'{regional_dem_raw.shape[1]}x{regional_dem_raw.shape[2]}'
    n_bands_str = f' ({local_dem_raw.shape[1]} bands)' if local_dem_raw.ndim == 4 else ''
    html_parts.append('<h3>Local DEM</h3><table>')
    html_parts.append(f'<tr><th>Shape</th><td>{ld_shape_str}{n_bands_str}</td></tr>')
    html_parts.append(f'<tr><th>Min (elev)</th><td>{ld_elev.min():.1f} m</td></tr>')
    html_parts.append(f'<tr><th>Max (elev)</th><td>{ld_elev.max():.1f} m</td></tr>')
    html_parts.append(f'<tr><th>Mean (elev)</th><td>{ld_elev.mean():.1f} m</td></tr>')
    html_parts.append('</table></div><div>')
    html_parts.append('<h3>Regional DEM</h3><table>')
    html_parts.append(f'<tr><th>Shape</th><td>{rd_shape_str}{n_bands_str}</td></tr>')
    html_parts.append(f'<tr><th>Min (elev)</th><td>{rd_elev.min():.1f} m</td></tr>')
    html_parts.append(f'<tr><th>Max (elev)</th><td>{rd_elev.max():.1f} m</td></tr>')
    html_parts.append(f'<tr><th>Mean (elev)</th><td>{rd_elev.mean():.1f} m</td></tr>')
    html_parts.append('</table></div></div></div>')

    # === Normalisation stats table ===
    if stats:
        html_parts.append('<div class="section"><h2>Normalisation Parameters (Train Set)</h2>')
        if "climate_mean" in stats and "climate_std" in stats:
            html_parts.append('<h3>Climate Channels</h3><table>')
            html_parts.append('<tr><th>Channel</th><th>Train Mean</th><th>Train Std</th></tr>')
            cm = stats["climate_mean"]
            cs = stats["climate_std"]
            for i, vn in enumerate(var_names):
                m_val = float(cm[i]) if hasattr(cm, '__len__') and i < len(cm) else float(cm)
                s_val = float(cs[i]) if hasattr(cs, '__len__') and i < len(cs) else float(cs)
                html_parts.append(f'<tr><td>{vn}</td><td>{m_val:.6g}</td><td>{s_val:.6g}</td></tr>')
            html_parts.append('</table>')
        if "dem_mean" in stats and "dem_std" in stats:
            html_parts.append('<h3>DEM</h3><table>')
            html_parts.append('<tr><th>Parameter</th><th>Value</th></tr>')
            html_parts.append(f'<tr><td>Mean</td><td>{float(stats["dem_mean"]):.4f}</td></tr>')
            html_parts.append(f'<tr><td>Std</td><td>{float(stats["dem_std"]):.4f}</td></tr>')
            html_parts.append('</table>')
        if "target_std" in stats:
            html_parts.append('<h3>Target (Rainfall)</h3><table>')
            html_parts.append('<tr><th>Parameter</th><th>Value</th></tr>')
            html_parts.append(f'<tr><td>Train Std (divisor)</td><td>{float(stats["target_std"]):.4f} mm</td></tr>')
            html_parts.append('</table>')
        html_parts.append('</div>')

    # === NaN audit ===
    html_parts.append('<div class="section"><h2>NaN Audit</h2><table>')
    html_parts.append('<tr><th>Array</th><th>Shape</th><th>NaN Count</th><th>NaN %</th></tr>')
    arrays_audit = [
        ("reanalysis", reanalysis_raw),
        ("local_dem", local_dem_raw),
        ("regional_dem", regional_dem_raw),
        ("rainfall", rainfall_raw),
    ]
    for name, arr in arrays_audit:
        flat_a = arr.astype(np.float32).ravel()
        nan_count = int(np.isnan(flat_a).sum())
        nan_pct = 100.0 * nan_count / len(flat_a)
        html_parts.append(f'<tr><td>{name}</td><td>{arr.shape}</td><td>{nan_count:,}</td><td>{nan_pct:.4f}%</td></tr>')
    html_parts.append('</table></div>')

    html_parts.append('<p style="color:#7f8c8d; text-align:center; margin-top:40px;">Generated by 03b_inspect_dataset.py</p>')
    html_parts.append("</body></html>")
    (out_dir / "interactive_report.html").write_text("\n".join(html_parts), encoding="utf-8")


if __name__ == "__main__":

    main()
