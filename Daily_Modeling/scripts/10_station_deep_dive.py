"""
Step 10: Deep-dive diagnostic for per-station model performance.

Investigates WHY certain stations are consistently hard to model by examining:
  1. Data quantity  - n_train, n_val, n_test, % zero-rain days
  2. Target statistics - mean, std, skewness, kurtosis, max event
  3. Cross-run stability - R2/RMSE variance across multiple result JSONs
  4. Climate signal strength - Spearman correlation of each climate channel
     with station rainfall (using train indices only)
  5. Temporal coverage - year range, gaps in record
  6. Spatial context - DEM elevation at station location (local patch mean)
  7. Residual bias - MBE sign and magnitude (systematic over/under-prediction)
  8. Rank-correlation vs R2 gap - high Spearman but low R2 → variance mismatch

Outputs (all in --out-dir):
  - station_diagnostics.csv   (all numeric features per station)
  - performance_summary.csv   (metrics from all runs, mean ± std)
  - climate_signal.csv        (per-station per-channel Spearman r)
  - deep_dive_overview.png    (4-panel summary figure)
  - climate_signal_heatmap.png
  - data_quantity_vs_r2.png
  - temporal_coverage.png

Usage:
    python -m Daily_Modeling.scripts.10_station_deep_dive
    python -m Daily_Modeling.scripts.10_station_deep_dive \\
        --results-dirs output/results/site_mlp_final output/results_1/site_mlp_final \\
        --out-dir output/diagnostics/station_deep_dive
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz, normalize_tensors
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, compute_station_year_ranges, compute_year_boundaries,
    spatiotemporal_split, station_proportional_split,
)
from Daily_Modeling.utils.device import select_device


# ── helpers ────────────────────────────────────────────────────────────────

def _load_metrics(results_dirs):
    """Load metrics_per_station.json from each results dir. Returns list of dicts."""
    records = []
    for d in results_dirs:
        base = Path(d)
        candidates = []
        # 1) As provided
        candidates.append(base)
        # 2) Common invocation mistake: user passes output/... from repo root,
        #    but files live under Daily_Modeling/output/...
        if not base.is_absolute():
            candidates.append(Path("Daily_Modeling") / base)

        metrics_path = None
        checked = []
        for c in candidates:
            mp = c / "metrics_per_station.json"
            checked.append(str(mp))
            if mp.exists():
                metrics_path = mp
                break

        if metrics_path is None:
            print("  WARNING: metrics_per_station.json not found; checked:")
            for cp in checked:
                print(f"    - {cp}")
            continue

        data = json.loads(metrics_path.read_text())
        for station, m in data.items():
            records.append({"run": str(metrics_path.parent), "station": station, **m})
    return pd.DataFrame(records)


def _performance_summary(metrics_df):
    """Aggregate metrics across runs: mean ± std per station."""
    grp = metrics_df.groupby("station")
    rows = []
    for station, g in grp:
        row = {"station": station, "n_runs": len(g)}
        for col in ("rmse", "mae", "mbe", "r2", "spearman_r"):
            if col in g.columns:
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"] = g[col].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("r2_mean")


def _data_diagnostics(tensors, meta, unique_stations):
    """Compute per-station data quantity and target distribution stats."""
    stations = meta["stations"]
    years = meta["years"]
    months = meta["months"]
    days = meta["days"]
    targets_raw = tensors["targets"].cpu().numpy()

    rows = []
    for st in unique_stations:
        sp = station_proportional_split(stations, years, months, days, st)
        n_train = len(sp["train"])
        n_val = len(sp["val"])
        n_test = len(sp["test"])
        n_total = n_train + n_val + n_test

        if n_total == 0:
            rows.append({"station": st, "n_total": 0, "n_train": 0,
                         "n_val": 0, "n_test": 0})
            continue

        all_idx = np.concatenate([sp["train"], sp["val"], sp["test"]])
        y_all = targets_raw[all_idx]
        y_train = targets_raw[sp["train"]] if n_train > 0 else np.array([])

        # Year range
        st_mask = np.array([str(s) == st for s in stations])
        st_years = years[st_mask].astype(int)
        yr_min, yr_max = int(st_years.min()), int(st_years.max())
        yr_span = yr_max - yr_min + 1
        yr_actual = len(np.unique(st_years))
        yr_gaps = yr_span - yr_actual  # years with no data

        # DEM elevation (local patch centre pixel, before normalisation)
        local_dem_raw = tensors["local_dem"].cpu().numpy()
        st_dem = local_dem_raw[all_idx]
        centre = st_dem.shape[-1] // 2
        elev_mean = float(st_dem[:, centre, centre].mean()) if st_dem.ndim == 3 else float(st_dem.mean())

        rows.append({
            "station": st,
            "n_total": n_total,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "yr_min": yr_min,
            "yr_max": yr_max,
            "yr_span": yr_span,
            "yr_gaps": yr_gaps,
            "pct_zero": float(100 * (y_all == 0).mean()),
            "mean_mm": float(y_all.mean()),
            "std_mm": float(y_all.std()),
            "max_mm": float(y_all.max()),
            "skewness": float(scipy_stats.skew(y_all)),
            "kurtosis": float(scipy_stats.kurtosis(y_all)),
            "train_std_mm": float(y_train.std()) if len(y_train) > 0 else np.nan,
            "elev_m": elev_mean,
        })
    return pd.DataFrame(rows)


def _climate_signal(tensors, meta, unique_stations, variable_names):
    """Compute Spearman r between each climate channel and rainfall for each station."""
    stations = meta["stations"]
    years = meta["years"]
    months = meta["months"]
    days = meta["days"]
    climate = tensors["climate"].cpu().numpy()  # (N, C, H, W)
    targets_raw = tensors["targets"].cpu().numpy()

    n_ch = climate.shape[1]
    rows = []
    for st in unique_stations:
        sp = station_proportional_split(stations, years, months, days, st)
        idx = sp["train"]
        if len(idx) < 20:
            continue
        y = targets_raw[idx]
        row = {"station": st}
        for c in range(n_ch):
            ch_vals = climate[idx, c].reshape(len(idx), -1).mean(axis=1)
            r, _ = scipy_stats.spearmanr(ch_vals, y)
            vname = variable_names[c] if c < len(variable_names) else f"ch{c}"
            row[vname] = float(r)
        rows.append(row)
    return pd.DataFrame(rows)


# ── plotting ───────────────────────────────────────────────────────────────

def _plot_overview(perf_df, data_df, out_dir):
    """4-panel overview: R2 bar, RMSE bar, R2 vs n_train scatter, MBE bar."""
    merged = perf_df.merge(data_df, on="station", how="left").sort_values("r2_mean")
    stations = merged["station"].tolist()
    x = np.arange(len(stations))

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # Panel 1: R2 per station (mean ± std across runs)
    ax1 = fig.add_subplot(gs[0, 0])
    colours = ["#c44e52" if v < 0 else "#4c72b0" if v < 0.3 else "#55a868"
               for v in merged["r2_mean"]]
    bars = ax1.barh(x, merged["r2_mean"], color=colours, alpha=0.85)
    if "r2_std" in merged.columns:
        ax1.barh(x, merged["r2_std"], left=merged["r2_mean"] - merged["r2_std"] / 2,
                 color="grey", alpha=0.3, height=0.4)
    ax1.axvline(0, color="black", lw=0.8, ls="--")
    ax1.set_yticks(x)
    ax1.set_yticklabels(stations, fontsize=7)
    ax1.set_xlabel("R²")
    ax1.set_title("R² per Station (mean across runs)\nred<0, blue<0.3, green≥0.3")

    # Panel 2: RMSE per station
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(x, merged["rmse_mean"], color="#dd8452", alpha=0.85)
    if "rmse_std" in merged.columns:
        ax2.barh(x, merged["rmse_std"], left=merged["rmse_mean"] - merged["rmse_std"] / 2,
                 color="grey", alpha=0.3, height=0.4)
    ax2.set_yticks(x)
    ax2.set_yticklabels(stations, fontsize=7)
    ax2.set_xlabel("RMSE (mm)")
    ax2.set_title("RMSE per Station (mean across runs)")

    # Panel 3: R2 vs n_train scatter
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(merged["n_train"], merged["r2_mean"],
                     c=merged["pct_zero"], cmap="RdYlGn_r",
                     s=80, alpha=0.85, edgecolors="grey", linewidths=0.5)
    plt.colorbar(sc, ax=ax3, label="% zero-rain days")
    for _, row in merged.iterrows():
        ax3.annotate(row["station"], (row["n_train"], row["r2_mean"]),
                     fontsize=5.5, ha="left", va="bottom",
                     xytext=(3, 2), textcoords="offset points")
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.set_xlabel("n_train samples")
    ax3.set_ylabel("R² (mean)")
    ax3.set_title("R² vs Training Size\n(colour = % zero-rain days)")

    # Panel 4: MBE (systematic bias)
    ax4 = fig.add_subplot(gs[1, 1])
    mbe_colours = ["#c44e52" if v > 0 else "#4c72b0" for v in merged["mbe_mean"]]
    ax4.barh(x, merged["mbe_mean"], color=mbe_colours, alpha=0.85)
    ax4.axvline(0, color="black", lw=0.8, ls="--")
    ax4.set_yticks(x)
    ax4.set_yticklabels(stations, fontsize=7)
    ax4.set_xlabel("MBE (mm)  [red=over-predict, blue=under-predict]")
    ax4.set_title("Mean Bias Error per Station")

    fig.suptitle("Station Performance Deep Dive — Overview", fontsize=13, y=1.01)
    out = out_dir / "deep_dive_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_climate_signal(clim_df, out_dir):
    """Heatmap of Spearman r between climate channels and rainfall per station."""
    if clim_df.empty:
        return
    clim_df = clim_df.set_index("station")
    fig, ax = plt.subplots(figsize=(max(10, len(clim_df.columns) * 0.7),
                                    max(6, len(clim_df) * 0.35)))
    vmax = max(0.5, clim_df.abs().max().max())
    im = ax.imshow(clim_df.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Spearman r (train set)")
    ax.set_xticks(range(len(clim_df.columns)))
    ax.set_xticklabels(clim_df.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(clim_df)))
    ax.set_yticklabels(clim_df.index, fontsize=7)
    ax.set_title("Climate Channel → Rainfall Spearman Correlation (per station, train only)")
    plt.tight_layout()
    out = out_dir / "climate_signal_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_data_quantity(data_df, perf_df, out_dir):
    """Scatter matrix: n_train, pct_zero, std_mm, elev_m vs R2."""
    merged = data_df.merge(perf_df[["station", "r2_mean", "spearman_r_mean"]], on="station")
    predictors = [
        ("n_train", "Training samples"),
        ("pct_zero", "% zero-rain days"),
        ("std_mm", "Rainfall std (mm)"),
        ("elev_m", "DEM elevation (m, local patch centre)"),
        ("skewness", "Rainfall skewness"),
        ("yr_span", "Record length (years)"),
    ]
    n = len(predictors)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for i, (col, label) in enumerate(predictors):
        ax = axes[i]
        if col not in merged.columns:
            ax.set_visible(False)
            continue
        ax.scatter(merged[col], merged["r2_mean"], alpha=0.8,
                   c=merged["n_train"], cmap="viridis", s=70,
                   edgecolors="grey", linewidths=0.5)
        for _, row in merged.iterrows():
            ax.annotate(row["station"], (row[col], row["r2_mean"]),
                        fontsize=5, ha="left", xytext=(2, 2),
                        textcoords="offset points")
        # Fit line
        valid = merged[[col, "r2_mean"]].dropna()
        if len(valid) > 3:
            slope, intercept, r, p, _ = scipy_stats.linregress(valid[col], valid["r2_mean"])
            xr = np.linspace(valid[col].min(), valid[col].max(), 50)
            ax.plot(xr, slope * xr + intercept, "r--", lw=1.2,
                    label=f"r={r:.2f}  p={p:.3f}")
            ax.legend(fontsize=7)
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("R² (mean)", fontsize=8)
        ax.set_title(f"R² vs {label}", fontsize=9)
    fig.suptitle("What Predicts Station R²?", fontsize=12)
    plt.tight_layout()
    out = out_dir / "data_quantity_vs_r2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_temporal_coverage(meta, unique_stations, perf_df, out_dir):
    """Station × year availability heatmap, coloured by R2."""
    stations_arr = meta["stations"]
    years_arr = meta["years"].astype(int)
    unique_years = sorted(set(years_arr))
    r2_map = dict(zip(perf_df["station"], perf_df["r2_mean"]))

    # Sort stations by R2 (worst first)
    sorted_stations = sorted(unique_stations, key=lambda s: r2_map.get(s, 0))
    s2i = {s: i for i, s in enumerate(sorted_stations)}
    y2j = {y: j for j, y in enumerate(unique_years)}

    grid = np.full((len(sorted_stations), len(unique_years)), np.nan)
    for k in range(len(stations_arr)):
        st = str(stations_arr[k])
        yr = int(years_arr[k])
        if st in s2i and yr in y2j:
            grid[s2i[st], y2j[yr]] = 1.0

    fig, ax = plt.subplots(figsize=(max(14, len(unique_years) * 0.22),
                                    max(5, len(sorted_stations) * 0.35)))
    ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1,
              interpolation="nearest")

    # Colour station labels by R2
    ax.set_yticks(range(len(sorted_stations)))
    yticklabels = ax.set_yticklabels(sorted_stations, fontsize=7)
    for label, st in zip(yticklabels, sorted_stations):
        r2 = r2_map.get(st, 0)
        label.set_color("#c44e52" if r2 < 0 else "#dd8452" if r2 < 0.25 else "#4c72b0")

    step = max(1, len(unique_years) // 15)
    ax.set_xticks(range(0, len(unique_years), step))
    ax.set_xticklabels([unique_years[i] for i in range(0, len(unique_years), step)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_title("Temporal Coverage by Station\n"
                 "(sorted by R², worst first; label colour: red<0, orange<0.25, blue≥0.25)")
    plt.tight_layout()
    out = out_dir / "temporal_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _print_diagnosis(perf_df, data_df):
    """Print a human-readable diagnosis for each bad station."""
    merged = perf_df.merge(data_df, on="station", how="left").sort_values("r2_mean")
    print("\n" + "=" * 70)
    print("  PER-STATION DIAGNOSIS")
    print("=" * 70)
    for _, row in merged.iterrows():
        r2 = row.get("r2_mean", np.nan)
        reasons = []

        n_train = row.get("n_train", 0)
        if n_train < 200:
            reasons.append(f"very few training samples (n_train={int(n_train)})")
        elif n_train < 500:
            reasons.append(f"limited training samples (n_train={int(n_train)})")

        pct_zero = row.get("pct_zero", np.nan)
        if not np.isnan(pct_zero) and pct_zero > 55:
            reasons.append(f"high zero-rain fraction ({pct_zero:.0f}%)")

        mbe = row.get("mbe_mean", np.nan)
        if not np.isnan(mbe) and abs(mbe) > 3:
            direction = "over" if mbe > 0 else "under"
            reasons.append(f"systematic {direction}-prediction (MBE={mbe:+.1f} mm)")

        spearman = row.get("spearman_r_mean", np.nan)
        if not np.isnan(spearman) and not np.isnan(r2):
            if spearman > 0.5 and r2 < 0.1:
                reasons.append(
                    f"rank-correlation OK (ρ={spearman:.2f}) but R²={r2:.2f} → "
                    "model captures direction but not variance magnitude"
                )

        yr_gaps = row.get("yr_gaps", 0)
        if yr_gaps > 3:
            reasons.append(f"record has {int(yr_gaps)} year-gaps")

        r2_std = row.get("r2_std", np.nan)
        if not np.isnan(r2_std) and r2_std > 0.15:
            reasons.append(f"unstable across runs (R² std={r2_std:.2f})")

        flag = "🔴" if r2 < 0 else "🟠" if r2 < 0.2 else "🟡" if r2 < 0.35 else "🟢"
        print(f"\n  {flag} {row['station']:<22s}  R²={r2:+.3f}  RMSE={row.get('rmse_mean', np.nan):.1f}mm"
              f"  n_train={int(n_train)}  pct_zero={pct_zero:.0f}%")
        if reasons:
            for r in reasons:
                print(f"      → {r}")
        else:
            print("      → No obvious single cause; may need per-station HP tuning")
    print("\n" + "=" * 70)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dirs", nargs="+",
        default=[
            str(config.RESULTS_DIR / "site_mlp_final"),
            str(config.RESULTS_DIR.parent / "results_1" / "site_mlp_final"),
        ],
        help="One or more results directories containing metrics_per_station.json",
    )
    parser.add_argument(
        "--out-dir",
        default=str(config.RESULTS_DIR.parent / "diagnostics" / "station_deep_dive"),
        help="Output directory for diagnostic plots and CSVs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metrics from all runs ─────────────────────────────────────────
    print(f"Loading metrics from {len(args.results_dirs)} result dir(s)...")
    metrics_df = _load_metrics(args.results_dirs)
    if metrics_df.empty:
        print("ERROR: No metrics found. Check --results-dirs.")
        return
    print(f"  {len(metrics_df)} station-run records loaded.")

    perf_df = _performance_summary(metrics_df)
    perf_df.to_csv(out_dir / "performance_summary.csv", index=False)
    print(f"  Saved performance_summary.csv  ({len(perf_df)} stations)")

    # ── Load dataset ───────────────────────────────────────────────────────
    print("\nLoading dataset tensors (CPU)...")
    tensors, meta = load_tensors_from_npz(device=select_device())
    unique_stations = sorted(set(str(s) for s in meta["stations"]))
    variable_names = list(meta["variables"]) if len(meta["variables"]) > 0 else []
    print(f"  {len(unique_stations)} unique stations, {len(tensors['targets'])} samples")

    # ── Data diagnostics ──────────────────────────────────────────────────
    print("\nComputing per-station data diagnostics...")
    data_df = _data_diagnostics(tensors, meta, unique_stations)
    data_df.to_csv(out_dir / "station_diagnostics.csv", index=False)
    print(f"  Saved station_diagnostics.csv")

    # ── Climate signal ────────────────────────────────────────────────────
    print("\nComputing climate-rainfall Spearman correlations (train set)...")
    # Normalise first so climate values are on a common scale
    train_yr, val_yr, test_yr = compute_year_boundaries(meta["years"])
    yr_ranges = compute_station_year_ranges(meta["stations"], meta["years"])
    groups = assign_station_groups(
        unique_stations, station_year_ranges=yr_ranges,
        val_years=val_yr, test_years=test_yr,
    )
    splits = spatiotemporal_split(meta["stations"], meta["years"], groups,
                                  train_years=train_yr, val_years=val_yr, test_years=test_yr)
    tensors, _ = normalize_tensors(tensors, splits["train"])

    clim_df = _climate_signal(tensors, meta, unique_stations, variable_names)
    clim_df.to_csv(out_dir / "climate_signal.csv", index=False)
    print(f"  Saved climate_signal.csv")

    # ── Diagnosis printout ────────────────────────────────────────────────
    _print_diagnosis(perf_df, data_df)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    _plot_overview(perf_df, data_df, out_dir)
    _plot_climate_signal(clim_df, out_dir)
    _plot_data_quantity(data_df, perf_df, out_dir)
    _plot_temporal_coverage(meta, unique_stations, perf_df, out_dir)

    print(f"\nAll outputs saved to {out_dir}")
    print("Key files:")
    print(f"  station_diagnostics.csv  — data quantity, DEM, temporal coverage")
    print(f"  performance_summary.csv  — R2/RMSE mean±std across runs")
    print(f"  climate_signal.csv       — Spearman r per station per climate channel")
    print(f"  deep_dive_overview.png   — 4-panel summary")
    print(f"  climate_signal_heatmap.png")
    print(f"  data_quantity_vs_r2.png  — what predicts R2?")
    print(f"  temporal_coverage.png    — record gaps by station")


if __name__ == "__main__":
    main()
