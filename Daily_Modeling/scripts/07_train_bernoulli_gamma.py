"""
Step 7: Fit Bernoulli-Gamma GLM (site-specific) on each station.

For each station in the training set, fit a two-part model:
  1. Logistic regression: P(rain > 0 | X)
  2. Gamma GLM with log link: E[rain | rain > 0, X]

Then evaluate on val/test splits (temporal only, since this is site-specific).

Usage:
    python -m Daily_Modeling.scripts.07_train_bernoulli_gamma
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz, normalize_tensors, print_normalization_report
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, compute_station_year_ranges, compute_year_boundaries,
    spatiotemporal_split, station_proportional_split,
)
from Daily_Modeling.models.bernoulli_gamma import BernoulliGammaGLM, flatten_features_numpy
from Daily_Modeling.utils.device import select_device
from Daily_Modeling.utils.io_utils import save_json, save_predictions
from Daily_Modeling.utils.metrics import compute_metrics, baseline_mean_metrics
from Daily_Modeling.utils.visualization import plot_scatter, plot_split_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="bernoulli_gamma_final")
    args = parser.parse_args()

    device = select_device()
    tensors, meta = load_tensors_from_npz(device=device)
    stations = meta["stations"]
    years = meta["years"]

    unique = sorted(set(str(s) for s in stations))

    # Data-driven year boundaries
    train_yr, val_yr, test_yr = compute_year_boundaries(years)

    yr_ranges = compute_station_year_ranges(stations, years)
    groups = assign_station_groups(
        unique, station_year_ranges=yr_ranges,
        val_years=val_yr, test_years=test_yr,
    )
    splits = spatiotemporal_split(stations, years, groups,
                                  train_years=train_yr, val_years=val_yr, test_years=test_yr)
    tensors, stats = normalize_tensors(tensors, splits["train"])

    var_names = list(meta["variables"]) if len(meta.get("variables", [])) > 0 else None
    print_normalization_report(tensors, stats, splits, variable_names=var_names)

    # Save split heatmap
    plot_split_heatmap(stations, years, groups, train_yr, val_yr, test_yr,
                       save_path=config.EDA_DIR / "split_heatmap_bernoulli_gamma.png",
                       title="Bernoulli-Gamma Split")

    # Convert to numpy
    climate_np = tensors["climate"].numpy()
    local_dem_np = tensors["local_dem"].numpy()
    regional_dem_np = tensors["regional_dem"].numpy()
    month_np = tensors["temporal"].numpy()
    rain_np = tensors["targets"].numpy()  # raw mm

    out_dir = config.RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # We train site-specific models on ALL stations (temporal split only),
    # but only evaluate on the test/val temporal windows.
    all_results = {}
    models = {}
    all_yt, all_yp, all_st = [], [], []

    for station_name in unique:
        sp = station_proportional_split(stations, years, meta["months"], meta["days"], station_name)
        if len(sp["train"]) < 50:
            print(f"  Skipping {station_name}: only {len(sp['train'])} train samples")
            continue

        X_train = flatten_features_numpy(
            climate_np[sp["train"]], local_dem_np[sp["train"]],
            regional_dem_np[sp["train"]], month_np[sp["train"]],
        )
        y_train = rain_np[sp["train"]]

        # Fit
        try:
            glm = BernoulliGammaGLM()
            glm.fit(X_train, y_train)
            models[station_name] = glm
        except Exception as e:
            print(f"  {station_name}: fitting failed - {e}")
            continue

        # Evaluate on test split (temporal)
        for split_name in ("val", "test"):
            idx = sp.get(split_name, np.array([]))
            if len(idx) < 5:
                continue
            X = flatten_features_numpy(
                climate_np[idx], local_dem_np[idx],
                regional_dem_np[idx], month_np[idx],
            )
            yt = rain_np[idx]
            yp = glm.predict(X)
            m = compute_metrics(yt, yp)
            key = f"{station_name}_{split_name}"
            all_results[key] = m
            all_yt.extend(yt)
            all_yp.extend(yp)
            all_st.extend([station_name] * len(yt))

        print(f"  {station_name}: fitted (train={len(sp['train'])})")

    # Save models
    with open(out_dir / "glm_models.pkl", "wb") as f:
        pickle.dump(models, f)

    # Aggregate metrics
    all_yt = np.array(all_yt)
    all_yp = np.array(all_yp)
    all_st = np.array(all_st)

    agg = compute_metrics(all_yt, all_yp)
    bl = baseline_mean_metrics(all_yt)
    agg.update(bl)
    print(f"\nAggregate: RMSE={agg['rmse']:.2f} mm  MAE={agg['mae']:.2f} mm  "
          f"R2={agg['r2']:.4f}  Baseline RMSE={bl['baseline_rmse']:.2f} mm")

    save_json(agg, out_dir / "metrics_aggregate.json")
    save_json(all_results, out_dir / "metrics_per_station.json")
    save_json(stats, out_dir / "normalization_stats.json")
    save_predictions(all_yt, all_yp, all_st, out_dir / "predictions.npz")

    plot_scatter(all_yt, all_yp, title="Bernoulli-Gamma GLM",
                 save_path=out_dir / "scatter_all.png")

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
