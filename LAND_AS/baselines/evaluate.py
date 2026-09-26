"""Evaluate baselines on the same splits as the LAND ensemble.

Writes metrics with the same schema as ``evaluate.py`` output so results can
be compared directly: ``output/baselines/test_metrics.json`` maps each
baseline name to its ``regression_metrics`` dict, per-station breakdowns go
to ``test_metrics_by_station.json`` (baselines plus the ``--run`` model when
its saved predictions exist), and per-fold site-specific comparisons go to
``fold_metrics.json``.

Usage: python -m LAND_AS.baselines.evaluate [--run weekly_land_v3] [--folds]
"""

import argparse
import json

import numpy as np

from LAND_AS import config
from LAND_AS.data import cv_folds, load_data
from LAND_AS.engine import save_json
from LAND_AS.metrics import extreme_metrics, regression_metrics
from LAND_AS.baselines.models import TEST_BASELINES, station_climatology


def per_station(observed, predicted, stations):
    """regression_metrics + dispersion per station."""
    out = {}
    for station in sorted(set(stations.astype(str))):
        mask = stations.astype(str) == station
        obs, pred = np.asarray(observed)[mask], np.asarray(predicted)[mask]
        out[station] = {
            **regression_metrics(obs, pred),
            "n": int(mask.sum()),
            "obs_std": float(obs.std()),
            "pred_std": float(pred.std()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate pooled and site-specific baselines.")
    parser.add_argument("--run", default=None, help="run name whose test metrics to include alongside")
    parser.add_argument("--folds", action="store_true", help="also run per-LOSO-fold station climatology")
    args = parser.parse_args()

    bundle = load_data()
    train_idx, test_idx = bundle.splits["train"], bundle.splits["test"]
    observed = bundle.arrays["target"][test_idx].numpy()
    test_stations = bundle.metadata["stations"][test_idx]
    output = config.OUTPUT_DIR / "baselines"
    output.mkdir(parents=True, exist_ok=True)

    results, predictions = {}, {}
    for name, predict_fn in TEST_BASELINES.items():
        pred = np.asarray(predict_fn(bundle, train_idx, test_idx), dtype=float)
        predictions[name] = pred
        results[name] = {**regression_metrics(observed, pred), **extreme_metrics(observed, pred)}
        print(f"{name:>18s}: MAE={results[name]['mae']:6.2f}  R2={results[name]['r2']:6.3f}  "
              f"rho={results[name]['spearman_r']:6.3f}  bias={results[name]['bias']:+6.2f}")
    save_json(results, output / "test_metrics.json")
    np.savez_compressed(
        output / "test_predictions.npz", observed=observed,
        stations=test_stations, years=bundle.metadata["years"][test_idx],
        **{f"pred_{name}": pred for name, pred in predictions.items()},
    )

    # Per-station breakdown: every baseline plus the --run model's saved
    # ensemble predictions, so dispersion/variance differences are visible.
    by_station = {name: per_station(observed, pred, test_stations) for name, pred in predictions.items()}
    if args.run:
        run_eval = config.RUNS_DIR / args.run / "evaluation"
        run_predictions = run_eval / "test_predictions.npz"
        if run_predictions.exists():
            with np.load(run_predictions) as z:
                by_station[args.run] = per_station(z["observed"], z["predicted"], z["stations"])
            run_metrics = run_eval / "test_metrics.json"
            if run_metrics.exists():
                print(f"\n{args.run} (test): {json.loads(run_metrics.read_text())}")
        else:
            print(f"\nno saved predictions found for run '{args.run}'")
    save_json(by_station, output / "test_metrics_by_station.json")

    # Compact per-station MAE / dispersion table for the interesting models.
    rows = ["gbm", "tweedie_glm", "ridge", args.run] if args.run else ["gbm", "tweedie_glm", "ridge"]
    rows = [r for r in rows if r in by_station]
    stations = sorted(set(test_stations.astype(str)))
    print(f"\n{'station':<12}" + "".join(f"{r:>14}" for r in rows) + f"{'obs_std':>10}")
    for station in stations:
        line = f"{station:<12}" + "".join(
            f"{by_station[r][station]['mae']:>14.1f}" for r in rows)
        line += f"{by_station[rows[0]][station]['obs_std']:>10.1f}"
        print(line)
    print("\nMAE per station above; see test_metrics_by_station.json for pred_std/R2/rho per station")

    if args.folds:
        fold_results = []
        for fold_number, (_train_idx, val_idx) in enumerate(cv_folds(bundle, mode="loso")):
            station = sorted(set(bundle.metadata["stations"][val_idx].astype(str)))[0]
            pred = station_climatology(bundle, val_idx)
            fold_results.append({
                "fold": fold_number, "station": station,
                **regression_metrics(bundle.arrays["target"][val_idx].numpy(), pred),
            })
        save_json(fold_results, output / "fold_metrics.json")
        maes = [f["mae"] for f in fold_results]
        print(f"fold station climatology: mean MAE={np.mean(maes):.2f} mm over {len(maes)} folds")


if __name__ == "__main__":
    main()
