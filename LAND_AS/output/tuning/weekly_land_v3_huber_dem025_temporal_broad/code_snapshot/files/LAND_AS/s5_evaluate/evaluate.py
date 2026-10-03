"""Stage 5 entry point: evaluate a trained LOSO ensemble on the test set.

Loads a run produced by s4_train/train.py (output/runs/<run>/), rebuilds each
fold's model from its checkpoint, averages seed predictions, and writes
test_metrics.json / test_predictions.npz under <run>/evaluation/ using the
shared metrics from s3_model/metrics.py.

Also hosts trial selection: --study NAME analyzes a tuning study from
s4_train/tune.py and picks a robust trial (fold-normalized rank + worst-fold
checks) instead of the noisy raw best. train.py uses pick_trial() for the same
purpose when --study is given without --trial.

Usage:
  python -m LAND_AS.s5_evaluate.evaluate --run NAME        # evaluate a run
  python -m LAND_AS.s5_evaluate.evaluate --study NAME      # pick a tuning trial
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from LAND_AS import config
from LAND_AS.s2_dataset.data import crop_from_hp, cv_folds, load_data, loaders, model_metadata, normalized_bundle
from LAND_AS.s3_model.engine import device, predict, save_json
from LAND_AS.s3_model.metrics import EXTREME_THRESHOLDS_MM, extreme_metrics, regression_metrics
from LAND_AS.s3_model.model import build_model
from LAND_AS.provenance import EVAL_SOURCE_FILES, snapshot_code


def _ranks(values):
    """Rank values ascending (rank 0 = smallest); ties share the lower rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(len(values))
    return ranks


def pick_trial(study_name, pool_size=10, quiet=False):
    """Select a robust trial from a tuning study rather than the raw best.

    The single-seed objective is noisy, so rank-1 is partly luck. Selection
    instead ranks the top `pool_size` trials by raw objective on three
    criteria and picks the lowest composite rank:
      1. raw objective value (median fold score),
      2. mean per-fold rank across all trials (consistency),
      3. worst fold score (minimax guard against a weak regime).
    Returns the winning trial number; prints the ranking table unless quiet.
    """
    import optuna

    storage = f"sqlite:///{config.TUNING_DIR / study_name / 'study.db'}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and "fold_scores" in t.user_attrs
    ]
    if not trials:
        raise RuntimeError(f"Study '{study_name}' has no completed trials with fold_scores")

    numbers = np.array([t.number for t in trials])
    values = np.array([t.value for t in trials])
    fold_scores = np.array([t.user_attrs["fold_scores"] for t in trials])
    mean_rank = np.stack([_ranks(fold_scores[:, f]) for f in range(fold_scores.shape[1])], axis=1).mean(axis=1)
    worst = fold_scores.max(axis=1)

    pool = np.argsort(values, kind="stable")[:pool_size]
    composite = _ranks(values[pool]) + _ranks(mean_rank[pool]) + _ranks(worst[pool])
    winner = pool[int(np.argmin(composite))]

    if not quiet:
        header = f"{'trial':>6} {'value':>8} {'mean_rank':>9} {'worst':>8} {'composite':>9}"
        arch = [k for k in ("hidden_units", "dropout", "batch_size", "dem_units", "lightweight", "use_lag", "dem_elev_only")
                if any(k in t.params for t in trials)]
        for key in arch:
            header += f" {key[:12]:>12}"
        print(header)
        for idx in pool[np.argsort(composite, kind="stable")]:
            pool_rank = int(np.where(pool == idx)[0][0])
            mark = "*" if idx == winner else " "
            row = f"{mark}{numbers[idx]:>5} {values[idx]:>8.4f} {mean_rank[idx]:>9.1f} {worst[idx]:>8.4f} {composite[pool_rank]:>9}"
            for key in arch:
                value = trials[idx].params.get(key)
                row += f" {value:>12.3g}" if isinstance(value, float) else f" {value:>12}"
            print(row)
        print(f"selected trial {numbers[winner]} (objective={values[winner]:.4f})")
    return int(numbers[winner])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a weekly LAND ensemble or select a tuning trial.")
    parser.add_argument("--run", default="weekly_land")
    parser.add_argument("--study", default=None,
                        help="analyze a tuning study and pick the robust trial (no evaluation performed)")
    parser.add_argument("--pool-size", type=int, default=10,
                        help="top-N raw-objective trials considered by --study selection")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--daily", action="store_true",
                        help="evaluate against daily_dataset.npz (must match the run's dataset_freq)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset NPZ to evaluate on (default: the config path for the freq; "
                             "should match the dataset the run was trained on)")
    args = parser.parse_args()

    if args.study:
        pick_trial(args.study, pool_size=args.pool_size)
        return

    freq = "daily" if args.daily else "weekly"
    run_dir = config.RUNS_DIR / args.run
    hp = json.loads((run_dir / "hyperparameters.json").read_text())
    if hp.get("dataset_freq", freq) != freq:
        raise ValueError(
            f"Run '{args.run}' was trained on {hp['dataset_freq']} data; "
            f"pass {'--daily' if hp['dataset_freq'] == 'daily' else 'no --daily'}"
        )
    checkpoints = sorted(run_dir.glob("fold_*/seed_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    dataset_path = Path(args.dataset) if args.dataset else config.dataset_path_for(freq)
    bundle = load_data(path=dataset_path, freq=freq)
    dem_cell_km = bundle.metadata.get("dem_cell_km", 1.0)
    folds = cv_folds(bundle, mode="loso")
    run_normalization = run_dir / "normalization.json"
    default_bundle = (
        normalized_bundle(bundle, stats=json.loads(run_normalization.read_text()))
        if run_normalization.exists()
        else bundle
    )
    metadata = model_metadata(default_bundle, hp.get("climate_patch"), hp.get("climate_lag_weeks"), hp.get("rain_lag_weeks"), dem_channels=hp.get("dem_channels"))
    test_loaders = loaders(
        default_bundle,
        {name: index for name, index in default_bundle.splits.items() if name.startswith("test")},
        args.batch_size,
        shuffle_train=False,
        crop=crop_from_hp(hp, dem_cell_km),
    )
    fold_test_loaders = {}
    output = run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(output / "code_snapshot", EVAL_SOURCE_FILES, dataset_path=dataset_path)
    target_device = device()

    for split_name, loader in test_loaders.items():
        predictions, observed = [], None
        for checkpoint in checkpoints:
            fold_number = int(checkpoint.parent.name.split("_")[1])
            fold_loader, target_scale = loader, default_bundle.stats["target_scale"]
            if checkpoint.with_name(f"{checkpoint.stem}_normalization.json").exists():
                if fold_number not in fold_test_loaders:
                    fold_bundle = normalized_bundle(bundle, folds[fold_number][0])
                    fold_test_loaders[fold_number] = (
                        loaders(
                            fold_bundle,
                            {split_name: fold_bundle.splits[split_name]},
                            args.batch_size,
                            shuffle_train=False,
                            crop=crop_from_hp(hp, dem_cell_km),
                        )[split_name],
                        fold_bundle.stats["target_scale"],
                    )
                fold_loader, target_scale = fold_test_loaders[fold_number]
            model = build_model(hp, metadata).to(target_device)
            model.load_state_dict(torch.load(checkpoint, map_location=target_device, weights_only=True)["state_dict"])
            current_observed, current_prediction = predict(model, fold_loader, target_scale, target_device)
            observed = current_observed
            predictions.append(current_prediction)
        predictions = np.stack(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        metrics = regression_metrics(observed, mean)
        metrics.update(extreme_metrics(observed, mean, threshold_mm=EXTREME_THRESHOLDS_MM[freq]))
        save_json(metrics, output / f"{split_name}_metrics.json")
        indices = bundle.splits[split_name]
        np.savez_compressed(
            output / f"{split_name}_predictions.npz",
            observed=observed,
            predicted=mean,
            uncertainty=std,
            stations=bundle.metadata["stations"][indices],
            years=bundle.metadata["years"][indices],
        )
    print(f"evaluation saved to {output}")


if __name__ == "__main__":
    main()
