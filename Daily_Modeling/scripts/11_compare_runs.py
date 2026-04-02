"""
Step 11: Compare training runs across all land_daily_* result folders.

Loads hyperparameters, CV fold metrics, CV summary, and test-set metrics
from every run in output/results/ and produces:

  - run_comparison.csv           flat table of all runs (HPs + metrics)
  - cv_fold_metrics.png          per-fold CV metrics grouped by run
  - test_metric_comparison.png   bar chart of test MAE / RMSE / R2 across runs
  - spatial_vs_temporal.png      scatter of spatial vs temporal R2
  - hp_vs_metric.png             HP impact scatter grid (key HPs vs CV MAE)
  - radar_overview.png           radar / parallel-coordinates overview

Usage:
    python -m Daily_Modeling.scripts.11_compare_runs [--results-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Daily_Modeling import config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict if missing."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _short_name(run_name: str) -> str:
    """Shorten run folder name for plot labels."""
    # land_daily_bernoulli_gamma_mae_cv1both_100_4 -> bg_mae_cv1both_100_4
    name = run_name.replace("land_daily_", "")
    name = name.replace("bernoulli_gamma", "bg")
    return name


def load_all_runs(results_dir: Path) -> pd.DataFrame:
    """Crawl *results_dir* and build a flat DataFrame with one row per run."""
    rows: List[dict] = []

    for run_path in sorted(results_dir.iterdir()):
        if not run_path.is_dir():
            continue
        hp_path = run_path / "hyperparameters.json"
        if not hp_path.exists():
            continue  # not a valid run folder

        row: Dict = {"run": run_path.name, "short": _short_name(run_path.name)}

        # Hyperparameters
        hp = _load_json(hp_path)
        for key in (
            "loss_type", "output_head", "climate_units", "dem_units",
            "na", "nb", "dropout_rate", "learning_rate", "weight_decay",
            "batch_size", "tweedie_p", "use_batch_norm",
            "local_dem_patch", "local_dem_km",
            "regional_dem_patch", "regional_dem_km",
            "dem_patch_size", "temporal_units", "climate_processing",
        ):
            row[f"hp_{key}"] = hp.get(key)

        # CV summary
        cv = _load_json(run_path / "cv_summary.json")
        row["cv_mae_mean"] = cv.get("mae_mean")
        row["cv_mae_std"] = cv.get("mae_std")
        row["cv_rmse_mean"] = cv.get("rmse_mean")
        row["cv_rmse_std"] = cv.get("rmse_std")
        row["cv_n_folds"] = cv.get("n_completed_folds")

        # Test metrics (all / spatial / temporal)
        for split in ("all", "spatial", "temporal"):
            m = _load_json(run_path / "inference" / f"metrics_test_{split}.json")
            for k, v in m.items():
                row[f"test_{split}_{k}"] = v

        rows.append(row)

    if not rows:
        print(f"No valid runs found in {results_dir}")
        sys.exit(1)

    return pd.DataFrame(rows)


def load_fold_metrics(results_dir: Path) -> pd.DataFrame:
    """Load per-fold CV validation metrics from every run."""
    rows: List[dict] = []
    for run_path in sorted(results_dir.iterdir()):
        if not run_path.is_dir() or not (run_path / "hyperparameters.json").exists():
            continue
        for fold_dir in sorted(run_path.glob("fold_*")):
            m = _load_json(fold_dir / "metrics_cv_val.json")
            if not m:
                continue
            m["run"] = run_path.name
            m["short"] = _short_name(run_path.name)
            m["fold"] = fold_dir.name
            rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path, dpi: int = 150):
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_cv_fold_metrics(fold_df: pd.DataFrame, out_dir: Path):
    """Box / strip plots of per-fold CV metrics grouped by run."""
    metrics = ["mae", "rmse", "r2", "spearman_r"]
    available = [m for m in metrics if m in fold_df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        sns.stripplot(data=fold_df, x=metric, y="short", hue="fold",
                      dodge=True, size=7, alpha=0.8, ax=ax)
        # overlay per-run mean as a diamond
        means = fold_df.groupby("short")[metric].mean()
        for i, (name, val) in enumerate(means.items()):
            ax.plot(val, i, marker="D", color="black", markersize=8, zorder=5)
        ax.set_ylabel("")
        ax.set_title(metric.upper())
        ax.legend(title="fold", fontsize=7, title_fontsize=8)
    fig.suptitle("Per-Fold CV Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "cv_fold_metrics.png")


def plot_test_comparison(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart of test MAE, RMSE, R2 across runs."""
    test_metrics = {
        "test_all_mae": "MAE (all)",
        "test_spatial_mae": "MAE (spatial)",
        "test_temporal_mae": "MAE (temporal)",
    }
    available = {k: v for k, v in test_metrics.items() if k in df.columns and df[k].notna().any()}
    if not available:
        print("  Skipping test comparison (no test metrics found)")
        return

    melted = df.melt(id_vars=["short"], value_vars=list(available.keys()),
                     var_name="metric", value_name="value")
    melted["metric"] = melted["metric"].map(available)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
    sns.barplot(data=melted, x="value", y="short", hue="metric", ax=ax, palette="Set2")
    ax.set_xlabel("MAE (mm)")
    ax.set_ylabel("")
    ax.set_title("Test MAE by Split Type")
    ax.legend(title="Split")
    fig.tight_layout()
    _save(fig, out_dir / "test_metric_comparison.png")

    # Also plot R2 and RMSE side by side
    r2_cols = {
        "test_all_r2": "R2 (all)", "test_spatial_r2": "R2 (spatial)",
        "test_temporal_r2": "R2 (temporal)",
    }
    rmse_cols = {
        "test_all_rmse": "RMSE (all)", "test_spatial_rmse": "RMSE (spatial)",
        "test_temporal_rmse": "RMSE (temporal)",
    }
    for cols, ylabel, fname in [
        (r2_cols, "R2", "test_r2_comparison.png"),
        (rmse_cols, "RMSE (mm)", "test_rmse_comparison.png"),
    ]:
        avail = {k: v for k, v in cols.items() if k in df.columns and df[k].notna().any()}
        if not avail:
            continue
        m = df.melt(id_vars=["short"], value_vars=list(avail.keys()),
                    var_name="metric", value_name="value")
        m["metric"] = m["metric"].map(avail)
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
        sns.barplot(data=m, x="value", y="short", hue="metric", ax=ax, palette="Set2")
        ax.set_xlabel(ylabel)
        ax.set_ylabel("")
        ax.set_title(f"Test {ylabel} by Split Type")
        ax.legend(title="Split")
        fig.tight_layout()
        _save(fig, out_dir / fname)


def plot_spatial_vs_temporal(df: pd.DataFrame, out_dir: Path):
    """Scatter: spatial R2 vs temporal R2 per run."""
    x_col, y_col = "test_spatial_r2", "test_temporal_r2"
    if x_col not in df.columns or y_col not in df.columns:
        return
    sub = df.dropna(subset=[x_col, y_col])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in sub.iterrows():
        ax.scatter(row[x_col], row[y_col], s=120, zorder=3)
        ax.annotate(row["short"], (row[x_col], row[y_col]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    lo = min(sub[x_col].min(), sub[y_col].min()) - 0.05
    hi = max(sub[x_col].max(), sub[y_col].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.4, label="1:1 line")
    ax.set_xlabel("Spatial R2 (held-out stations)")
    ax.set_ylabel("Temporal R2 (held-out years)")
    ax.set_title("Generalization: Spatial vs Temporal")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "spatial_vs_temporal.png")


def plot_hp_impact(df: pd.DataFrame, out_dir: Path):
    """Scatter grid: key HPs vs CV MAE mean."""
    y_col = "cv_mae_mean"
    if y_col not in df.columns or df[y_col].isna().all():
        return

    hp_cols = [
        "hp_learning_rate", "hp_weight_decay", "hp_climate_units",
        "hp_na", "hp_dropout_rate", "hp_batch_size",
    ]
    available = [c for c in hp_cols if c in df.columns and df[c].notna().any()]
    if not available:
        return

    n = len(available)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, hp_col in enumerate(available):
        ax = axes_flat[i]
        sub = df.dropna(subset=[hp_col, y_col])
        ax.scatter(sub[hp_col], sub[y_col], s=100, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["short"], (row[hp_col], row[y_col]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        label = hp_col.replace("hp_", "")
        ax.set_xlabel(label)
        ax.set_ylabel("CV MAE (mm)")
        ax.set_title(f"{label} vs CV MAE")
        if "learning_rate" in hp_col or "weight_decay" in hp_col:
            ax.set_xscale("log")

    # hide unused axes
    for j in range(len(available), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Hyperparameter Impact on CV MAE", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "hp_vs_metric.png")


def plot_radar_overview(df: pd.DataFrame, out_dir: Path):
    """Radar chart comparing runs on key normalised metrics."""
    cols = {
        "cv_mae_mean": ("CV MAE", True),       # lower is better -> invert
        "test_all_mae": ("Test MAE", True),
        "test_all_r2": ("Test R2", False),
        "test_all_spearman_r": ("Spearman", False),
        "test_spatial_r2": ("Spatial R2", False),
        "test_temporal_r2": ("Temporal R2", False),
    }
    available = {k: v for k, v in cols.items() if k in df.columns and df[k].notna().any()}
    if len(available) < 3:
        return

    keys = list(available.keys())
    labels = [available[k][0] for k in keys]
    invert = [available[k][1] for k in keys]

    # Normalise each metric to [0, 1] across runs
    vals = df[keys].copy()
    for i, col in enumerate(keys):
        lo, hi = vals[col].min(), vals[col].max()
        if hi - lo > 0:
            vals[col] = (vals[col] - lo) / (hi - lo)
        else:
            vals[col] = 0.5
        if invert[i]:
            vals[col] = 1.0 - vals[col]  # flip so higher = better

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for idx, row in df.iterrows():
        values = vals.loc[idx, keys].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=row["short"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Run Overview (higher = better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "radar_overview.png")


def plot_metric_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap of all numeric metrics across runs for quick visual scan."""
    metric_cols = [c for c in df.columns
                   if (c.startswith("cv_") or c.startswith("test_")) and df[c].dtype in ("float64", "float32")]
    if not metric_cols:
        return

    heat = df.set_index("short")[metric_cols].T
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), max(6, len(metric_cols) * 0.35)))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdYlGn_r", linewidths=0.5,
                ax=ax, cbar_kws={"label": "value"})
    ax.set_title("All Metrics Heatmap")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, out_dir / "metric_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare LAND training runs")
    parser.add_argument("--results-dir", type=Path,
                        default=config.RESULTS_DIR,
                        help="Directory containing land_daily_* run folders")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: results_dir/run_comparison)")
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "run_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {out_dir}")

    # Load data
    df = load_all_runs(results_dir)
    fold_df = load_fold_metrics(results_dir)
    print(f"Found {len(df)} runs, {len(fold_df)} fold records\n")

    # Save flat CSV
    csv_path = out_dir / "run_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.name}")

    # Print summary table to console
    summary_cols = ["short", "hp_loss_type", "cv_mae_mean", "cv_rmse_mean",
                    "test_all_mae", "test_all_r2", "test_spatial_r2", "test_temporal_r2"]
    summary_cols = [c for c in summary_cols if c in df.columns]
    print("\n" + df[summary_cols].to_string(index=False))
    print()

    # Generate plots
    sns.set_theme(style="whitegrid", font_scale=0.95)

    plot_cv_fold_metrics(fold_df, out_dir)
    plot_test_comparison(df, out_dir)
    plot_spatial_vs_temporal(df, out_dir)
    plot_hp_impact(df, out_dir)
    plot_radar_overview(df, out_dir)
    plot_metric_heatmap(df, out_dir)

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
