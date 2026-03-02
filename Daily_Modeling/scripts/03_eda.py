"""
Step 3: Exploratory data analysis on the assembled dataset.

Produces plots for rainfall distributions, seasonality, station coverage,
reanalysis variable summaries, DEM statistics, spatial maps, correlation
analysis, temporal autocorrelation, exceedance curves, dry/wet spell
distributions, and model architecture diagrams — all broken down by
the spatio-temporal train/val/test splits.

Usage:
    python -m Daily_Modeling.scripts.03_eda
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz
from Daily_Modeling.utils.device import select_device
from Daily_Modeling.data_utils.load_raw import load_station_metadata
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, compute_station_year_ranges,
)
from Daily_Modeling.utils.visualization import (
    plot_rainfall_histograms,
    plot_monthly_seasonality,
    plot_station_sample_counts,
    plot_per_station_histograms,
    plot_station_map,
    plot_reanalysis_rainfall_correlation,
    plot_temporal_autocorrelation,
    plot_rainfall_exceedance,
    plot_dry_wet_spells,
    plot_rainfall_by_station_boxplot,
    plot_annual_rainfall_trends,
    plot_model_architecture,
)


def main():
    out = config.EDA_DIR
    out.mkdir(parents=True, exist_ok=True)

    tensors, meta = load_tensors_from_npz(device=select_device())

    stations = meta["stations"]
    years = meta["years"]
    months = meta["months"]
    rain_mm = tensors["targets"].numpy()
    climate = tensors["climate"].numpy()

    unique_stations = sorted(set(str(s) for s in stations))
    print(f"Unique stations: {len(unique_stations)}")

    # Splits
    yr_ranges = compute_station_year_ranges(stations, years)
    groups = assign_station_groups(unique_stations, station_year_ranges=yr_ranges)
    splits = spatiotemporal_split(stations, years, groups)

    # ---- Rainfall histograms (per split) ----
    core_splits = {k: splits[k] for k in ("train", "val", "test") if k in splits}
    plot_rainfall_histograms(rain_mm, core_splits, out / "rainfall_histograms.png")
    print("Saved rainfall histograms")

    # ---- Monthly seasonality ----
    plot_monthly_seasonality(rain_mm, months, core_splits, out / "monthly_seasonality.png")
    print("Saved monthly seasonality")

    # ---- Station sample counts ----
    plot_station_sample_counts(stations, core_splits, out / "station_sample_counts.png")
    print("Saved station sample counts")

    # ---- Per-station histograms ----
    plot_per_station_histograms(rain_mm, stations, out / "station_histograms")
    print("Saved per-station histograms")

    # ---- Rainfall summary table ----
    rows = []
    for name, idx in core_splits.items():
        y = rain_mm[idx]
        if len(y) == 0:
            rows.append({"split": name, "n": 0})
            continue
        rows.append({
            "split": name, "n": len(y),
            "mean": np.mean(y), "std": np.std(y),
            "min": np.min(y), "p50": np.median(y),
            "p90": np.quantile(y, 0.9), "p95": np.quantile(y, 0.95),
            "p99": np.quantile(y, 0.99), "max": np.max(y),
            "pct_zero": 100.0 * np.mean(y == 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "rainfall_summary.csv", index=False)
    print("Saved rainfall_summary.csv")
    print(df.to_string(index=False))

    # ---- Reanalysis variable means per split ----
    variables = meta.get("variables", np.array([]))
    var_names = list(variables) if len(variables) > 0 else []
    if climate.ndim == 4 and len(variables) > 0:
        re_mean = np.nanmean(climate, axis=(2, 3))  # (N, C)
        rows = []
        for name, idx in core_splits.items():
            for ci, v in enumerate(variables):
                vals = re_mean[idx, ci]
                rows.append({
                    "split": name, "variable": str(v),
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals)),
                })
        re_df = pd.DataFrame(rows)
        re_df.to_csv(out / "reanalysis_summary.csv", index=False)
        print("Saved reanalysis_summary.csv")

        # Train vs test shift
        train_m = re_df[re_df["split"] == "train"].set_index("variable")["mean"]
        test_m = re_df[re_df["split"] == "test"].set_index("variable")["mean"]
        delta = (test_m - train_m).sort_values(key=lambda s: np.abs(s), ascending=False)
        print("\nReanalysis variable shift (test - train mean):")
        print(delta.to_string())

    # ---- Station group summary ----
    print("\nStation assignments:")
    for role in ("train", "val", "test"):
        sts = [s for s, r in groups.items() if r == role]
        print(f"  {role}: {len(sts)} stations - {', '.join(sts)}")

    # ==================================================================
    # NEW: Enhanced visualizations
    # ==================================================================

    # ---- Spatial map of station locations coloured by role ----
    try:
        station_meta = load_station_metadata()
        plot_station_map(station_meta, station_groups=groups,
                         save_path=out / "station_map.png")
        print("Saved station_map.png")
    except Exception as e:
        print(f"WARNING: Could not generate station map: {e}")

    # ---- Reanalysis-rainfall correlation ----
    if climate.ndim == 4 and len(var_names) > 0:
        plot_reanalysis_rainfall_correlation(
            climate, rain_mm, var_names,
            save_path=out / "reanalysis_rainfall_correlation.png",
        )
        print("Saved reanalysis_rainfall_correlation.png")

    # ---- Temporal autocorrelation ----
    plot_temporal_autocorrelation(
        rain_mm, stations, max_lag=14,
        save_path=out / "temporal_autocorrelation.png",
    )
    print("Saved temporal_autocorrelation.png")

    # ---- Rainfall exceedance curve ----
    plot_rainfall_exceedance(
        rain_mm, split_indices=core_splits,
        save_path=out / "rainfall_exceedance.png",
    )
    print("Saved rainfall_exceedance.png")

    # ---- Dry/wet spell distributions ----
    plot_dry_wet_spells(
        rain_mm, stations, threshold_mm=1.0,
        save_path=out / "dry_wet_spells.png",
    )
    print("Saved dry_wet_spells.png")

    # ---- Rainfall box plots by station ----
    plot_rainfall_by_station_boxplot(
        rain_mm, stations,
        save_path=out / "rainfall_boxplot_by_station.png",
    )
    print("Saved rainfall_boxplot_by_station.png")

    # ---- Annual rainfall trends ----
    plot_annual_rainfall_trends(
        rain_mm, years, stations,
        save_path=out / "annual_rainfall_trends.png",
    )
    print("Saved annual_rainfall_trends.png")

    # ---- Model architecture diagrams (dynamic from actual model classes) ----
    try:
        from Daily_Modeling.models.land import create_land_model
        from Daily_Modeling.models.site_mlp import SiteMLP, compute_input_size

        # LAND model with default HP
        land_metadata = {
            "climate_shape": tuple(tensors["climate"].shape[1:]),
            "local_dem_shape": tuple(tensors["local_dem"].shape[1:]),
            "regional_dem_shape": tuple(tensors["regional_dem"].shape[1:]),
            "num_month_features": int(tensors["temporal"].shape[1]),
            "num_climate_vars": int(tensors["climate"].shape[1]),
        }
        land_model = create_land_model(config.LAND_DEFAULT_HP, land_metadata)

        # Build dummy input for torchviz computation graph
        bs = 2
        land_input = {
            "climate": torch.randn(bs, *land_metadata["climate_shape"]),
            "local_dem": torch.randn(bs, *land_metadata["local_dem_shape"]),
            "regional_dem": torch.randn(bs, *land_metadata["regional_dem_shape"]),
            "temporal": torch.randn(bs, land_metadata["num_month_features"]),
        }
        plot_model_architecture(
            land_model, model_name="LAND Model (default HP)",
            input_data=land_input,
            save_path=out / "architecture_land.png",
        )
        print("Saved architecture_land.png")

        # Site MLP with default HP
        input_size = compute_input_size(
            land_metadata["climate_shape"],
            land_metadata["local_dem_shape"],
            land_metadata["regional_dem_shape"],
            land_metadata["num_month_features"],
        )
        mlp_model = SiteMLP(input_size, config.MLP_DEFAULT_HP["hidden_sizes"],
                            config.MLP_DEFAULT_HP["dropout_rate"])
        mlp_input = torch.randn(bs, input_size)
        plot_model_architecture(
            mlp_model, model_name="Site MLP (default HP)",
            input_data=mlp_input,
            save_path=out / "architecture_site_mlp.png",
        )
        print("Saved architecture_site_mlp.png")
    except Exception as e:
        print(f"WARNING: Could not generate architecture diagrams: {e}")

    print(f"\nAll EDA outputs saved to {out}")


if __name__ == "__main__":
    main()
