"""Stage 5: evaluate baselines on the same splits as the LAND ensemble.

Fits the baseline definitions in models.py on the s2_dataset train/test split
and aligns saved run predictions (from s5_evaluate/evaluate.py) for direct
comparison.

Writes metrics with the same schema as ``evaluate.py`` output so results can
be compared directly: ``output/baselines/test_metrics.json`` maps each
baseline name to its ``regression_metrics`` dict, per-station breakdowns for
baselines and every selected run go to ``test_metrics_by_station.json``,
and per-fold site-specific comparisons go to ``fold_metrics.json``.

Usage: python -m LAND_AS.s5_evaluate.baselines.evaluate [--all-runs] [--run NAME] [--folds]
"""

import argparse
import csv

import numpy as np

from LAND_AS import config
from LAND_AS.s2_dataset.data import cv_folds, load_data
from LAND_AS.s3_model.engine import save_json
from LAND_AS.s3_model.metrics import extreme_metrics, regression_metrics
from LAND_AS.s5_evaluate.baselines.models import TEST_BASELINES, station_climatology


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


def write_station_tables(by_station, output_dir):
    """Write long-form CSV and model-by-station Markdown comparison tables."""
    fields = [
        "model", "station", "n", "mse", "rmse", "mae", "bias", "r2",
        "spearman_r", "obs_std", "pred_std",
    ]
    rows = []
    for model_name, station_rows in by_station.items():
        for station, values in station_rows.items():
            row = {"model": model_name, "station": station}
            for key, value in values.items():
                if isinstance(value, (int, np.integer)):
                    row[key] = int(value)
                else:
                    value = float(value)
                    row[key] = value if np.isfinite(value) else ""
            rows.append(row)

    csv_path = output_dir / "test_metrics_by_station.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    model_names = list(by_station)
    station_names = sorted({row["station"] for row in rows})
    table_metrics = ["mae", "rmse", "bias", "r2", "spearman_r", "obs_std", "pred_std"]

    def cell(station, model_name, metric):
        value = by_station.get(model_name, {}).get(station, {}).get(metric)
        if value is None:
            return ""
        value = float(value)
        if not np.isfinite(value):
            return ""
        if metric == "n":
            return str(int(value))
        if metric in {"r2", "spearman_r"}:
            return f"{value:.3f}"
        return f"{value:.2f}"

    lines = [
        "# Test metrics by station",
        "",
        "Each table is stations x models. Rainfall units are mm/week.",
    ]
    for metric in table_metrics:
        lines.extend(["", f"## {metric}", ""])
        lines.append("| station | " + " | ".join(model_names) + " |")
        lines.append("|---" * (len(model_names) + 1) + "|")
        for station in station_names:
            lines.append(
                "| " + station + " | "
                + " | ".join(cell(station, model_name, metric) for model_name in model_names)
                + " |"
            )
    md_path = output_dir / "test_metrics_by_station.md"
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def _prediction_sources(include_all, run_names):
    """Yield ``(display_name, kind, prediction_path)`` for evaluated outputs."""
    for name in run_names or []:
        yield name, "run", config.RUNS_DIR / name / "evaluation" / "test_predictions.npz"
    if not include_all:
        return
    for path in sorted(config.RUNS_DIR.glob("*/evaluation/test_predictions.npz")):
        yield path.parents[1].name, "run", path


def _load_model_predictions(observed, stations, years, include_all, run_names):
    """Load aligned run predictions and compute consistent metrics."""
    model_metrics, by_station, predictions = {}, {}, {}
    for name, kind, path in _prediction_sources(include_all, run_names):
        if name in model_metrics:
            continue
        if not path.exists():
            print(f"no saved predictions found for {kind} '{name}'")
            continue
        with np.load(path, allow_pickle=True) as z:
            current_observed = z["observed"]
            current_stations = z["stations"]
            current_years = z["years"] if "years" in z.files else None
            predicted = z["predicted"]
        if not np.allclose(current_observed, observed):
            print(f"skipping {name}: observations do not align with baseline test set")
            continue
        if not np.array_equal(current_stations.astype(str), stations.astype(str)):
            print(f"skipping {name}: stations do not align with baseline test set")
            continue
        if current_years is not None and not np.array_equal(
            current_years.astype(int), np.asarray(years).astype(int)
        ):
            print(f"skipping {name}: years do not align with baseline test set")
            continue
        predictions[name] = predicted
        model_metrics[name] = {
            "kind": kind,
            **regression_metrics(current_observed, predicted),
            **extreme_metrics(current_observed, predicted),
        }
        by_station[name] = per_station(current_observed, predicted, current_stations)
    return model_metrics, by_station, predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate pooled and site-specific baselines.")
    parser.add_argument("--run", action="append", default=[],
                        help="run name whose test metrics to include; repeatable")
    parser.add_argument("--all-runs", action="store_true",
                        help="include every evaluated output under runs/")
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

    # Per-station breakdown: every baseline plus saved run ensemble
    # predictions, so dispersion/variance differences are directly comparable.
    by_station = {name: per_station(observed, pred, test_stations) for name, pred in predictions.items()}
    model_metrics, model_by_station, model_predictions = _load_model_predictions(
        observed, test_stations, bundle.metadata["years"][test_idx], args.all_runs, args.run
    )
    by_station.update(model_by_station)
    save_json(by_station, output / "test_metrics_by_station.json")
    station_csv, station_md = write_station_tables(by_station, output)
    save_json(model_metrics, output / "model_metrics.json")
    if model_predictions:
        np.savez_compressed(
            output / "model_predictions.npz",
            observed=observed, stations=test_stations,
            years=bundle.metadata["years"][test_idx],
            **{f"pred_{name}": pred for name, pred in model_predictions.items()},
        )
    for name, metrics in model_metrics.items():
        print(f"{name:>38s}: MAE={metrics['mae']:6.2f}  R2={metrics['r2']:6.3f}  "
              f"rho={metrics['spearman_r']:6.3f}  bias={metrics['bias']:+6.2f}")

    # Compact per-station MAE / dispersion table for the strongest references.
    preferred = ["gbm", "tweedie_glm", "ridge", "weekly_land_v5",
                 "weekly_land_v5_huber_rw_msemon"]
    rows = [name for name in preferred if name in by_station]
    rows += [name for name in args.run if name in by_station and name not in rows]
    rows = [r for r in rows if r in by_station]
    stations = sorted(set(test_stations.astype(str)))
    col_width = max(len(r) for r in rows) + 2
    print(f"\n{'station':<12}" + "".join(f"{r:>{col_width}}" for r in rows) + f"{'obs_std':>10}")
    for station in stations:
        line = f"{station:<12}" + "".join(
            f"{by_station[r][station]['mae']:>{col_width}.1f}" for r in rows)
        line += f"{by_station[rows[0]][station]['obs_std']:>10.1f}"
        print(line)
    print("\nMAE per station above")
    print(f"station tables written to {station_csv} and {station_md}")

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
