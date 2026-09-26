import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from LAND_AS import config
from LAND_AS.data import crop_from_hp, cv_folds, load_data, loaders, model_metadata
from LAND_AS.engine import device, predict, save_json
from LAND_AS.metrics import extreme_metrics, regression_metrics
from LAND_AS.model import build_model
from LAND_AS.provenance import EVAL_SOURCE_FILES, snapshot_code


BLEND_SOURCE_FILES = (
    *EVAL_SOURCE_FILES,
    "LAND_AS/parallelize.py",
    "LAND_AS/train.py",
    "LAND_AS/train_v5_huber.py",
    "LAND_AS/blend_v5_huber.py",
)


def _run_paths(name):
    run_dir = config.RUNS_DIR / name
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    return run_dir


def _metrics(observed, predicted):
    metrics = regression_metrics(observed, predicted)
    metrics.update(extreme_metrics(observed, predicted))
    return metrics


def _objective_score(metrics, objective, observed_mean):
    if objective == "mse":
        return metrics["mse"]
    if objective == "mae":
        return metrics["mae"]
    if objective == "balanced":
        return (
            0.5 * metrics["mae"] / observed_mean
            + 0.3 * metrics["rmse"] / observed_mean
            + 0.2 * abs(metrics["bias"]) / observed_mean
        )
    raise ValueError(f"Unknown objective: {objective!r}")


def _oof_predictions(run_dir, bundle, folds, batch_size, cache_path):
    fold_dirs = sorted(run_dir.glob("fold_*"), key=lambda path: int(path.name.split("_")[1]))
    checkpoint_names = np.asarray([
        str(path.relative_to(run_dir)) for fold_dir in fold_dirs
        for path in sorted(fold_dir.glob("seed_*.pt"))
    ])
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if "checkpoints" in cached.files and np.array_equal(cached["checkpoints"], checkpoint_names):
            return cached["indices"], cached["observed"], cached["predicted"]
        print(f"checkpoint set changed; rebuilding OOF cache {cache_path.name}")

    hp = json.loads((run_dir / "hyperparameters.json").read_text())
    metadata = model_metadata(bundle, hp.get("climate_patch"), hp.get("climate_lag_weeks"), hp.get("rain_lag_weeks"))
    target_device = device()
    all_indices, all_observed, all_predicted = [], [], []

    if len(fold_dirs) != len(folds):
        raise RuntimeError(f"{run_dir.name}: expected {len(folds)} fold directories, found {len(fold_dirs)}")

    for fold_number, ((train_index, val_index), fold_dir) in enumerate(zip(folds, fold_dirs)):
        expected = fold_dir / f"fold_{fold_number}"
        if fold_dir.name != expected.name:
            raise RuntimeError(f"Unexpected fold directory order at {fold_dir}")
        checkpoints = sorted(fold_dir.glob("seed_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {fold_dir}")
        loader = loaders(
            bundle, {"val": val_index}, batch_size, shuffle_train=False, crop=crop_from_hp(hp)
        )["val"]
        predictions = []
        observed = None
        for checkpoint in checkpoints:
            model = build_model(hp, metadata).to(target_device)
            model.load_state_dict(torch.load(checkpoint, map_location=target_device, weights_only=True)["state_dict"])
            current_observed, prediction = predict(model, loader, bundle.stats["target_scale"], target_device)
            observed = current_observed
            predictions.append(prediction)
        all_indices.append(val_index)
        all_observed.append(observed)
        all_predicted.append(np.mean(np.stack(predictions), axis=0))
        print(f"{run_dir.name} fold {fold_number}: blended {len(checkpoints)} seed predictions")

    indices = np.concatenate(all_indices)
    observed = np.concatenate(all_observed)
    predicted = np.concatenate(all_predicted)
    np.savez_compressed(
        cache_path, indices=indices, observed=observed, predicted=predicted,
        checkpoints=checkpoint_names,
    )
    return indices, observed, predicted


def _test_predictions(run_dir):
    path = run_dir / "evaluation" / "test_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluated test predictions: {path}")
    data = np.load(path, allow_pickle=True)
    return data["observed"], data["predicted"], data["stations"], data["years"]


def main():
    parser = argparse.ArgumentParser(
        description="Blend weekly_land_v5 and its Huber variant using a weight selected only on LOSO predictions."
    )
    parser.add_argument("--gamma-run", default="weekly_land_v5")
    parser.add_argument("--huber-run", default="weekly_land_v5_huber")
    parser.add_argument("--output", default="v5_gamma_huber_cv")
    parser.add_argument("--objective", choices=["mse", "mae", "balanced"], default="mse")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    gamma_dir = _run_paths(args.gamma_run)
    huber_dir = _run_paths(args.huber_run)
    output = config.OUTPUT_DIR / "blends" / args.output
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(output / "code_snapshot", BLEND_SOURCE_FILES)

    bundle = load_data()
    folds = cv_folds(bundle, mode="loso")
    gamma_index, gamma_observed, gamma_oof = _oof_predictions(
        gamma_dir, bundle, folds, args.batch_size, output / f"oof_{args.gamma_run}.npz")
    huber_index, huber_observed, huber_oof = _oof_predictions(
        huber_dir, bundle, folds, args.batch_size, output / f"oof_{args.huber_run}.npz")
    if not np.array_equal(gamma_index, huber_index) or not np.allclose(gamma_observed, huber_observed):
        raise RuntimeError("Gamma and Huber OOF prediction indices/observations do not align")

    weights = np.linspace(0, 1, 1001)
    rows = []
    observed_mean = float(np.mean(gamma_observed))
    for weight in weights:
        predicted = weight * huber_oof + (1 - weight) * gamma_oof
        metrics = _metrics(gamma_observed, predicted)
        row = {"huber_weight": float(weight), **metrics}
        row["mse_objective"] = metrics["mse"]
        row["mae_objective"] = metrics["mae"]
        row["balanced_objective"] = _objective_score(metrics, "balanced", observed_mean)
        rows.append(row)

    weight_by_objective = {
        objective: float(min(rows, key=lambda row: row[f"{objective}_objective"])["huber_weight"])
        for objective in ("mse", "mae", "balanced")
    }
    selected_weight = weight_by_objective[args.objective]
    selected_row = min(rows, key=lambda row: abs(row["huber_weight"] - selected_weight))

    grid_path = output / "oof_weight_grid.csv"
    with open(grid_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    oof_predicted = selected_weight * huber_oof + (1 - selected_weight) * gamma_oof
    oof_metrics = _metrics(gamma_observed, oof_predicted)
    save_json(oof_metrics, output / "oof_metrics.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        indices=gamma_index,
        observed=gamma_observed,
        gamma_predicted=gamma_oof,
        huber_predicted=huber_oof,
        predicted=oof_predicted,
        stations=bundle.metadata["stations"][gamma_index],
        years=bundle.metadata["years"][gamma_index],
    )

    gamma_test_observed, gamma_test, stations, years = _test_predictions(gamma_dir)
    huber_test_observed, huber_test, _, _ = _test_predictions(huber_dir)
    if not np.allclose(gamma_test_observed, huber_test_observed):
        raise RuntimeError("Gamma and Huber test observations do not align")
    test_predicted = selected_weight * huber_test + (1 - selected_weight) * gamma_test
    test_metrics = _metrics(gamma_test_observed, test_predicted)
    save_json(test_metrics, output / "test_metrics.json")
    np.savez_compressed(
        output / "test_predictions.npz",
        observed=gamma_test_observed,
        gamma_predicted=gamma_test,
        huber_predicted=huber_test,
        predicted=test_predicted,
        stations=stations,
        years=years,
    )

    save_json({
        "gamma_run": args.gamma_run,
        "huber_run": args.huber_run,
        "primary_objective": args.objective,
        "selected_huber_weight": selected_weight,
        "weights_by_objective": weight_by_objective,
        "selected_oof_row": selected_row,
        "oof_samples": int(len(gamma_observed)),
        "test_samples": int(len(gamma_test_observed)),
    }, output / "selection.json")
    save_json({
        "experiment": "Gamma/Huber blend selected on LOSO out-of-fold predictions",
        "gamma_run": args.gamma_run,
        "huber_run": args.huber_run,
        "primary_objective": args.objective,
        "leakage_control": "test predictions are blended only after the weight is selected on held-out training-station predictions",
    }, output / "experiment.json")
    print(f"selected Huber weight ({args.objective}): {selected_weight:.3f}")
    print(f"weights by objective: {weight_by_objective}")
    print(f"blend outputs saved to {output}")


if __name__ == "__main__":
    main()
