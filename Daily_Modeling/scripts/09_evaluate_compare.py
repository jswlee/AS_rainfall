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

import numpy as np
import pandas as pd

from Daily_Modeling import config
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, compute_station_year_ranges, compute_year_boundaries,
    spatiotemporal_split, station_proportional_split,
)
from Daily_Modeling.utils.io_utils import save_json
from Daily_Modeling.utils.metrics import (
    baseline_mean_metrics, compute_metrics, compute_wasserstein, per_station_metrics,
)
from Daily_Modeling.utils.visualization import (
    plot_model_comparison_table, plot_multi_model_scatter, plot_per_station_comparison, plot_scatter,
)


def _load_predictions(run_dir: Path):
    """Load predictions NPZ from a run directory (tries several filenames)."""
    def _extract(z):
        y_true = z["y_true"]
        y_pred = z["y_pred"] if "y_pred" in z else z["y_pred_mean"]
        stations = z.get("stations", np.array([]))
        return y_true, y_pred, stations

    # Try direct paths first
    for name in ("predictions_test_spatial.npz", "predictions_test.npz", "predictions.npz"):
        p = run_dir / name
        if p.exists():
            z = np.load(str(p), allow_pickle=True)
            return _extract(z)

    # Try inference subdirectory (LAND model structure)
    for name in ("predictions_test_spatial.npz", "predictions_test.npz", "predictions.npz"):
        p = run_dir / "inference" / name
        if p.exists():
            z = np.load(str(p), allow_pickle=True)
            return _extract(z)
    
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--land-dir", default=str(config.RESULTS_DIR / "land_final"))
    parser.add_argument("--mlp-dir", default=str(config.RESULTS_DIR / "site_mlp_final"))
    parser.add_argument("--out-dir", default=str(config.RESULTS_DIR / "comparison"))
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        "LAND": Path(args.land_dir),
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
    plot_multi_model_scatter(
        all_data,
        title="Model Comparison - Test Predictions",
        save_path=out / "scatter_combined.png",
    )

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
        # Accept either naming convention
        for _cand in (config.DATASET_NPZ,
                      config.ASSEMBLED_DIR / "daily_dataset.npz",
                      config.ASSEMBLED_DIR / "daily_dataset_station_centered.npz"):
            npz_path = _cand
            if npz_path.exists():
                break
        z = np.load(str(npz_path), allow_pickle=True)
        all_months = z["months"]
        all_stations = z["stations"]
        all_years = z["years"]

        unique = sorted(set(str(s) for s in all_stations))
        train_yr, val_yr, test_yr = compute_year_boundaries(all_years)
        yr_ranges = compute_station_year_ranges(all_stations, all_years)
        groups = assign_station_groups(unique, station_year_ranges=yr_ranges,
                                       val_years=val_yr, test_years=test_yr)
        splits = spatiotemporal_split(all_stations, all_years, groups,
                                      train_years=train_yr, val_years=val_yr, test_years=test_yr)

        SEASONS = [("Dry (May-Oct)", {5, 6, 7, 8, 9, 10}), ("Wet (Nov-Apr)", {1, 2, 3, 4, 11, 12})]
        all_seasonal_rows = []

        # LAND: spatially held-out test set
        if "LAND" in all_data:
            test_key = "test_spatial" if "test_spatial" in splits else "test"
            test_months = all_months[splits[test_key]]
            yt = all_data["LAND"]["y_true"]
            yp = all_data["LAND"]["y_pred"]
            if len(test_months) == len(yt):
                for season_name, month_set in SEASONS:
                    mask = np.isin(test_months, list(month_set))
                    if mask.sum() > 0:
                        m = compute_metrics(yt[mask], yp[mask])
                        m["season"] = season_name
                        m["model"] = "LAND"
                        m["n"] = int(mask.sum())
                        all_seasonal_rows.append(m)

        # Site MLP: per-station chronological test sets; months come from the
        # per-station split predictions (saved with station labels).  We rebuild
        # month assignments from the full dataset index.
        if "Site MLP" in all_data:
            mlp_st = all_data["Site MLP"]["stations"]
            mlp_yt = all_data["Site MLP"]["y_true"]
            mlp_yp = all_data["Site MLP"]["y_pred"]

            # Build a month array aligned to the Site MLP predictions
            mlp_months = np.zeros(len(mlp_yt), dtype=int)
            cursor = 0
            for stn in unique:
                sp = station_proportional_split(all_stations, all_years,
                                                all_months, z.get("days", np.ones_like(all_months)),
                                                stn)
                n_test = len(sp.get("test", []))
                if n_test == 0:
                    continue
                mlp_months[cursor:cursor + n_test] = all_months[sp["test"]]
                cursor += n_test

            if cursor > 0:
                for season_name, month_set in SEASONS:
                    mask = np.isin(mlp_months[:cursor], list(month_set))
                    if mask.sum() > 0:
                        m = compute_metrics(mlp_yt[mask], mlp_yp[mask])
                        m["season"] = season_name
                        m["model"] = "Site MLP"
                        m["n"] = int(mask.sum())
                        all_seasonal_rows.append(m)

        if all_seasonal_rows:
            sdf = pd.DataFrame(all_seasonal_rows)
            cols = ["model", "season", "n", "rmse", "mae", "r2", "spearman_r"]
            cols = [c for c in cols if c in sdf.columns]
            sdf.to_csv(out / "seasonal_metrics.csv", index=False)
            print(f"\nSeasonal metrics:\n{sdf[cols].round(4).to_string(index=False)}")
    except Exception as e:
        print(f"Seasonal breakdown skipped: {e}")

    print(f"\nAll comparison outputs saved to {out}")


if __name__ == "__main__":
    main()
