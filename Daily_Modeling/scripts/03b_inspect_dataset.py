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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz, normalize_tensors, print_normalization_report
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, compute_station_year_ranges,
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
)


def main():
    out = config.EDA_DIR / "dataset_inspection"
    out.mkdir(parents=True, exist_ok=True)

    npz_path = config.ASSEMBLED_DIR / "daily_dataset.npz"
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
    yr_ranges = compute_station_year_ranges(meta["stations"], meta["years"])
    unique = sorted(set(str(s) for s in meta["stations"]))
    groups = assign_station_groups(unique, station_year_ranges=yr_ranges)
    splits = spatiotemporal_split(meta["stations"], meta["years"], groups)

    # Keep raw copies before normalisation
    raw_climate = tensors["climate"].numpy().copy()
    raw_local_dem = tensors["local_dem"].numpy().copy()
    raw_regional_dem = tensors["regional_dem"].numpy().copy()

    tensors, stats = normalize_tensors(tensors, splits["train"])

    # Side-by-side raw vs normalised
    plot_normalization_comparison(
        raw_arrays={"climate (all ch)": raw_climate, "local_dem": raw_local_dem, "regional_dem": raw_regional_dem},
        norm_arrays={
            "climate (all ch)": tensors["climate"].numpy(),
            "local_dem": tensors["local_dem"].numpy(),
            "regional_dem": tensors["regional_dem"].numpy(),
        },
        save_path=out / "normalization_comparison.png",
    )
    print("  Saved normalization_comparison.png")

    # Per-channel reanalysis distributions AFTER normalisation
    plot_reanalysis_channel_distributions(
        tensors["climate"].numpy(), var_names,
        save_path=out / "reanalysis_channel_distributions_normalised.png",
        tag="normalised",
    )
    print("  Saved reanalysis_channel_distributions_normalised.png")

    # Reanalysis correlation AFTER normalisation
    plot_reanalysis_correlation(
        tensors["climate"].numpy(), var_names,
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
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
               gap: 10px; margin: 10px 0; }
  .stat-card { background: #ecf0f1; padding: 12px; border-radius: 6px; text-align: center; }
  .stat-card .value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
  .stat-card .label { font-size: 0.85em; color: #7f8c8d; }
</style>
</head><body>
<h1>Dataset Inspection Report</h1>
""")

    # Summary stats
    n_samples = len(rainfall_raw)
    n_stations = len(np.unique(stations))
    n_years = len(np.unique(years))
    pct_zero = 100.0 * np.mean(rainfall_raw == 0)
    html_parts.append(f"""
<div class="section">
<h2>Overview</h2>
<div class="stat-grid">
  <div class="stat-card"><div class="value">{n_samples:,}</div><div class="label">Total Samples</div></div>
  <div class="stat-card"><div class="value">{n_stations}</div><div class="label">Stations</div></div>
  <div class="stat-card"><div class="value">{n_years}</div><div class="label">Years</div></div>
  <div class="stat-card"><div class="value">{reanalysis_raw.shape[1]}</div><div class="label">Climate Channels</div></div>
  <div class="stat-card"><div class="value">{rainfall_raw.mean():.1f} mm</div><div class="label">Mean Rainfall</div></div>
  <div class="stat-card"><div class="value">{pct_zero:.1f}%</div><div class="label">Zero-Rain Days</div></div>
  <div class="stat-card"><div class="value">{rainfall_raw.max():.1f} mm</div><div class="label">Max Rainfall</div></div>
  <div class="stat-card"><div class="value">{local_dem_raw.shape[1]}×{local_dem_raw.shape[2]}</div><div class="label">Local DEM Size</div></div>
</div></div>
""")

    # Rainfall distribution (interactive)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Rainfall Distribution (mm)", "Log-Rainfall Distribution"])
    rainy = rainfall_raw[rainfall_raw > 0]
    fig.add_trace(go.Histogram(x=rainy, nbinsx=80, marker_color="#3498db", name="Rainy days"), row=1, col=1)
    fig.add_trace(go.Histogram(x=np.log1p(rainfall_raw), nbinsx=80, marker_color="#e74c3c", name="log(1+rain)"), row=1, col=2)
    fig.update_layout(height=350, showlegend=False, margin=dict(t=40, b=30))
    html_parts.append(f'<div class="section"><h2>Rainfall Distribution</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # Monthly seasonality (interactive)
    df_m = pd.DataFrame({"month": months.astype(int), "rain": rainfall_raw})
    monthly = df_m.groupby("month")["rain"].agg(["mean", "std", "count"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["mean"],
                         error_y=dict(type="data", array=monthly["std"] / np.sqrt(monthly["count"])),
                         marker_color="#2ecc71", name="Mean ± SE"))
    fig.update_layout(title="Mean Daily Rainfall by Month", xaxis_title="Month",
                      yaxis_title="Mean (mm)", height=350, margin=dict(t=40, b=30))
    html_parts.append(f'<div class="section"><h2>Monthly Seasonality</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # Per-station sample counts (interactive)
    st_counts = pd.Series([str(s) for s in stations]).value_counts().sort_values(ascending=True)
    fig = go.Figure(go.Bar(y=st_counts.index, x=st_counts.values, orientation="h",
                           marker_color="#9b59b6"))
    fig.update_layout(title="Samples per Station", xaxis_title="Count",
                      height=max(350, 20 * n_stations), margin=dict(l=120, t=40, b=30))
    html_parts.append(f'<div class="section"><h2>Station Sample Counts</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # Reanalysis channel correlation heatmap (interactive)
    n_samp = min(5000, reanalysis_raw.shape[0])
    idx_samp = np.random.RandomState(42).choice(reanalysis_raw.shape[0], n_samp, replace=False)
    flat = reanalysis_raw[idx_samp].reshape(n_samp, reanalysis_raw.shape[1], -1).mean(axis=-1)
    corr = np.corrcoef(flat.T)
    fig = go.Figure(go.Heatmap(z=corr, x=var_names, y=var_names, colorscale="RdBu_r",
                                zmin=-1, zmax=1, text=np.round(corr, 2), texttemplate="%{text}"))
    fig.update_layout(title="Reanalysis Channel Correlation", height=500, margin=dict(t=40, b=30))
    html_parts.append(f'<div class="section"><h2>Reanalysis Correlation</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # NaN audit (interactive)
    arrays_audit = {"reanalysis": reanalysis_raw, "local_dem": local_dem_raw,
                    "regional_dem": regional_dem_raw, "rainfall": rainfall_raw}
    nan_names, nan_pcts = [], []
    for name, arr in arrays_audit.items():
        flat_a = arr.astype(np.float32).ravel()
        pct = 100.0 * np.isnan(flat_a).sum() / len(flat_a)
        nan_names.append(name)
        nan_pcts.append(pct)
    fig = go.Figure(go.Bar(x=nan_pcts, y=nan_names, orientation="h", marker_color="#e74c3c"))
    fig.update_layout(title="NaN Fraction (%)", height=250, margin=dict(l=100, t=40, b=30))
    html_parts.append(f'<div class="section"><h2>NaN Audit</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>')

    # Normalisation stats table
    if stats:
        norm_rows = []
        if "climate_mean" in stats:
            cm = stats["climate_mean"]
            cs = stats["climate_std"]
            for i, vn in enumerate(var_names):
                m_val = cm[i] if hasattr(cm, '__len__') and i < len(cm) else cm
                s_val = cs[i] if hasattr(cs, '__len__') and i < len(cs) else cs
                norm_rows.append({"Variable": vn, "Train Mean": f"{float(m_val):.4f}",
                                  "Train Std": f"{float(s_val):.4f}"})
        if norm_rows:
            df_norm = pd.DataFrame(norm_rows)
            fig = go.Figure(go.Table(
                header=dict(values=list(df_norm.columns), fill_color="#3498db",
                            font=dict(color="white", size=12)),
                cells=dict(values=[df_norm[c] for c in df_norm.columns],
                           fill_color="#ecf0f1", font=dict(size=11)),
            ))
            fig.update_layout(title="Normalisation Statistics", height=max(200, 30 * len(norm_rows)),
                              margin=dict(t=40, b=10))
            html_parts.append(f'<div class="section"><h2>Normalisation Stats</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>')

    html_parts.append("</body></html>")
    (out_dir / "interactive_report.html").write_text("\n".join(html_parts), encoding="utf-8")


if __name__ == "__main__":
    main()
