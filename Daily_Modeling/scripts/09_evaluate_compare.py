"""
Step 9: Evaluate and compare all three models side-by-side.

Loads saved predictions from each model's output directory and produces:
  - Combined metrics table (CSV + figure)
  - Per-station comparison bar charts
  - Scatter plots overlaid
  - Seasonal breakdown

Usage:
    python -m Daily_Modeling.scripts.09_evaluate_compare
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Daily_Modeling import config
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, compute_station_year_ranges, compute_year_boundaries,
    spatiotemporal_split,
)
from Daily_Modeling.utils.io_utils import save_json
from Daily_Modeling.utils.metrics import (
    baseline_mean_metrics, compute_metrics, compute_wasserstein, per_station_metrics,
)
from Daily_Modeling.utils.visualization import (
    plot_model_comparison_table, plot_per_station_comparison, plot_scatter,
)


def _load_predictions(run_dir: Path):
    """Load predictions NPZ from a run directory (tries several filenames)."""
    for name in ("predictions_test_spatial.npz", "predictions_test.npz", "predictions.npz"):
        p = run_dir / name
        if p.exists():
            z = np.load(str(p), allow_pickle=True)
            return z["y_true"], z["y_pred"], z.get("stations", np.array([]))
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--land-dir", default=str(config.RESULTS_DIR / "land_final"))
    parser.add_argument("--glm-dir", default=str(config.RESULTS_DIR / "bernoulli_gamma_final"))
    parser.add_argument("--mlp-dir", default=str(config.RESULTS_DIR / "site_mlp_final"))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "comparison"))
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        "LAND": Path(args.land_dir),
        "Bernoulli-Gamma": Path(args.glm_dir),
        "Site MLP": Path(args.mlp_dir),
    }

    # ---- Load all predictions ----
    all_data = {}
    for name, d in model_dirs.items():
        yt, yp, st = _load_predictions(d)
        if yt is not None and len(yt) > 0:
            all_data[name] = {"y_true": yt, "y_pred": yp, "stations": st}
            print(f"Loaded {name}: {len(yt)} predictions from {d}")
        else:
            print(f"WARNING: No predictions found in {d} (skipping)")

    if not all_data:
        print("No model predictions found. Run training scripts first.")
        return

    # ---- Compute metrics ----
    metrics_table = {}
    for name, data in all_data.items():
        m = compute_metrics(data["y_true"], data["y_pred"])
        m["wasserstein"] = compute_wasserstein(data["y_true"], data["y_pred"])
        bl = baseline_mean_metrics(data["y_true"])
        m.update(bl)
        metrics_table[name] = m

    # Add baseline row
    any_yt = next(iter(all_data.values()))["y_true"]
    bl = baseline_mean_metrics(any_yt)
    bl_metrics = {"rmse": bl["baseline_rmse"], "mae": float("nan"),
                  "mbe": 0.0, "r2": 0.0, "spearman_r": float("nan"),
                  "spearman_p": float("nan"), "wasserstein": float("nan")}
    metrics_table["Baseline (mean)"] = bl_metrics

    # ---- Save metrics ----
    df = pd.DataFrame(metrics_table).T
    df.to_csv(out / "comparison_metrics.csv")
    print("\n" + df.round(4).to_string())

    save_json(metrics_table, out / "comparison_metrics.json")
    plot_model_comparison_table(metrics_table, out / "comparison_table.png")

    # ---- Scatter plots (one per model) ----
    for name, data in all_data.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        plot_scatter(
            data["y_true"], data["y_pred"],
            title=f"{name} - Test Predictions",
            save_path=out / f"scatter_{safe_name}.png",
        )

    # ---- Combined scatter (overlay) ----
    fig, ax = plt.subplots(figsize=(7, 7))
    colours = {"LAND": "steelblue", "Bernoulli-Gamma": "coral", "Site MLP": "seagreen"}
    for name, data in all_data.items():
        ax.scatter(data["y_true"], data["y_pred"], s=4, alpha=0.25,
                   label=name, color=colours.get(name, "gray"), rasterized=True)
    lo = 0
    hi = max(np.nanmax(d["y_true"]) for d in all_data.values())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Observed (mm)")
    ax.set_ylabel("Predicted (mm)")
    ax.set_title("All Models - Test Predictions")
    ax.legend(markerscale=4)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "scatter_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Per-station comparison ----
    station_metrics_all = {}
    for name, data in all_data.items():
        if len(data["stations"]) > 0:
            station_metrics_all[name] = per_station_metrics(
                data["y_true"], data["y_pred"], data["stations"]
            )
    if station_metrics_all:
        for metric_name in ("rmse", "mae", "r2"):
            plot_per_station_comparison(
                station_metrics_all, metric_name=metric_name,
                save_path=out / f"station_{metric_name}.png",
            )

    # ---- Seasonal breakdown (if we have month info) ----
    # Try to load months from the assembled dataset
    try:
        npz_path = config.ASSEMBLED_DIR / "daily_dataset.npz"
        z = np.load(str(npz_path), allow_pickle=True)
        all_months = z["months"]
        all_stations = z["stations"]

        if "LAND" in all_data:
            unique = sorted(set(str(s) for s in all_stations))
            train_yr, val_yr, test_yr = compute_year_boundaries(z["years"])
            yr_ranges = compute_station_year_ranges(all_stations, z["years"])
            groups = assign_station_groups(unique, station_year_ranges=yr_ranges,
                                           val_years=val_yr, test_years=test_yr)
            splits = spatiotemporal_split(all_stations, z["years"], groups,
                                          train_years=train_yr, val_years=val_yr, test_years=test_yr)
            test_key = "test_spatial" if "test_spatial" in splits else "test"
            test_months = all_months[splits[test_key]]

            yt = all_data["LAND"]["y_true"]
            yp = all_data["LAND"]["y_pred"]
            if len(test_months) == len(yt):
                seasonal_rows = []
                for season_name, month_set in [("Dry", {5,6,7,8,9,10}), ("Wet", {1,2,3,4,11,12})]:
                    mask = np.isin(test_months, list(month_set))
                    if mask.sum() > 0:
                        m = compute_metrics(yt[mask], yp[mask])
                        m["season"] = season_name
                        m["model"] = "LAND"
                        seasonal_rows.append(m)
                if seasonal_rows:
                    sdf = pd.DataFrame(seasonal_rows)
                    sdf.to_csv(out / "seasonal_land.csv", index=False)
                    print(f"\nSeasonal LAND metrics:\n{sdf.round(4).to_string(index=False)}")
    except Exception as e:
        print(f"Seasonal breakdown skipped: {e}")

    print(f"\nAll comparison outputs saved to {out}")


if __name__ == "__main__":
    main()
