"""Evaluate the LAND_AS pooled baselines on Daily_Modeling's test split.

Diagnostic only: loads the Daily_Modeling weekly NPZ (same schema as
weekly_dataset.npz), reproduces its data-driven 70/20/10 year boundaries and
its station_roles from the run's station_groups.json, then refits the
TEST_BASELINES from s5_evaluate/baselines/models.py on its train split and
scores test_all / test_spatial / test_temporal for comparison with the DM
LAND ensemble's metrics_test_*.json.

Usage: python LAND_AS/scripts/dm_baseline_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from LAND_AS.s2_dataset.data import load_data, normalized_bundle
from LAND_AS.s3_model.metrics import regression_metrics
from LAND_AS.s5_evaluate.baselines.models import TEST_BASELINES

DM_ROOT = ROOT / "Daily_Modeling"
DM_RUN = DM_ROOT / "output/weekly/results/land_weekly_gamma_mse_cv3both_n100_old_BEST"
DM_NPZ = DM_ROOT / "output/weekly/assembled/weekly_dataset_station_centered.npz"


def year_boundaries(years, train_frac=0.70, val_frac=0.20):
    """Daily_Modeling compute_year_boundaries: cutoffs matching sample fractions."""
    yr = np.sort(years.astype(int))
    n = len(yr)
    train_end = int(yr[int(n * train_frac) - 1])
    val_end = int(yr[int(n * (train_frac + val_frac)) - 1])
    return (int(yr[0]), train_end), (train_end + 1, val_end), (val_end + 1, int(yr[-1]))


def main():
    groups = json.loads((DM_RUN / "station_groups.json").read_text())["station_groups"]
    bundle = load_data(path=DM_NPZ)
    stations = bundle.metadata["stations"]
    years = bundle.metadata["years"]
    roles = np.array([groups.get(str(s), "train") for s in stations])
    train_yrs, val_yrs, test_yrs = year_boundaries(years)
    print(f"year ranges: train={train_yrs}  val={val_yrs}  test={test_yrs}")

    in_years = lambda rng: (years >= rng[0]) & (years <= rng[1])
    splits = {
        "train": np.where((roles == "train") & in_years(train_yrs))[0],
        "test_spatial": np.where((roles == "test") & in_years(test_yrs))[0],
        "test_temporal": np.where((roles == "train") & in_years(test_yrs))[0],
    }
    splits["test_all"] = np.sort(np.concatenate([splits["test_spatial"], splits["test_temporal"]]))
    for name, idx in splits.items():
        print(f"  {name:14s}: {len(idx):6d} samples")

    # Renormalize on DM's train rows (bundle stats were built on LAND_AS's split).
    nbundle = normalized_bundle(bundle, splits["train"])

    reference = {}
    for kind in ("all", "spatial", "temporal"):
        path = DM_RUN / f"inference/metrics_test_{kind}.json"
        if path.exists():
            reference[f"test_{kind}"] = json.loads(path.read_text())

    for split_name in ("test_all", "test_spatial", "test_temporal"):
        test_idx = splits[split_name]
        observed = nbundle.arrays["target"][test_idx].numpy()
        ref = reference.get(split_name, {})
        print(f"\n=== {split_name} (n={len(test_idx)})"
              + (f"  | DM LAND: rmse={ref.get('rmse'):.2f} mae={ref.get('mae'):.2f} r2={ref.get('r2'):.3f}" if ref else ""))
        for name, fn in TEST_BASELINES.items():
            pred = np.asarray(fn(nbundle, splits["train"], test_idx), dtype=float)
            m = regression_metrics(observed, pred)
            print(f"  {name:>18s}: rmse={m['rmse']:6.2f}  mae={m['mae']:6.2f}  r2={m['r2']:6.3f}  "
                  f"rho={m['spearman_r']:6.3f}  bias={m['bias']:+6.2f}")


if __name__ == "__main__":
    main()
