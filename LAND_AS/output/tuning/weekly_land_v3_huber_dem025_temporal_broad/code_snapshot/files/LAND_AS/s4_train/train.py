"""Stage 4 entry point: train a LOSO ensemble for a tuned configuration.

Loads hyperparameters from an Optuna study (--study/--trial, written by
tune.py) or --hyperparameters JSON, then trains one model per
leave-one-station-out fold x seed via parallelize.py. Writes
output/runs/<run>/ with checkpoints, histories, and provenance snapshots;
evaluate it with python -m LAND_AS.s5_evaluate.evaluate --run <run>.

Usage: python -m LAND_AS.s4_train.train --study NAME --trial N --run NAME
"""
import argparse
import json
import os
from pathlib import Path

from LAND_AS import config
from LAND_AS.s2_dataset.data import cv_folds, load_data
from LAND_AS.s3_model.engine import device, save_json
from LAND_AS.s4_train.parallelize import train_folds
from LAND_AS.provenance import TRAIN_SOURCE_FILES, snapshot_code


def _load_hyperparameters(args, bundle):
    """Resolve hyperparameters from --hyperparameters JSON or --study Optuna DB."""
    hp = dict(config.DEFAULTS)
    if args.hyperparameters:
        hp.update(json.loads(open(args.hyperparameters).read()))
    elif args.study:
        best_path = config.TUNING_DIR / args.study / "best.json"
        # Without --trial, select the robust trial (fold-rank + worst-fold
        # composite) instead of trusting the noisy raw objective best.
        if args.trial is None and (config.TUNING_DIR / args.study / "study.db").exists():
            from LAND_AS.s5_evaluate.evaluate import pick_trial
            args.trial = pick_trial(args.study)
        study = None
        if args.trial is not None or not best_path.exists():
            import optuna
            storage = f"sqlite:///{config.TUNING_DIR / args.study / 'study.db'}"
            study = optuna.load_study(study_name=args.study, storage=storage)
        if args.trial is not None:
            matches = [trial for trial in study.trials if trial.number == args.trial]
            if not matches:
                raise RuntimeError(f"Study '{args.study}' has no trial #{args.trial}")
            selected = dict(matches[0].params)
            if best_path.exists():
                fixed = json.loads(best_path.read_text())
                best = {**fixed, **selected}
            else:
                best = selected
            description = f"study '{args.study}' trial #{args.trial}"
        elif best_path.exists():
            best = json.loads(best_path.read_text())
            description = f"study '{args.study}' best.json"
        else:
            if len(study.trials) == 0 or study.best_trial is None:
                raise RuntimeError(f"Study '{args.study}' has no completed trials; cannot load best hyperparameters")
            best = dict(study.best_params)
            description = (
                f"study '{args.study}' trial #{study.best_trial.number} "
                f"(objective={study.best_value:.3f})"
            )
        # Older studies searched climate_multiplier x channel blocks; new
        # studies fix climate_units in config.DEFAULTS.
        if "climate_multiplier" in best:
            lag = best.get("climate_lag_weeks", bundle.metadata["lag_max"])
            best["climate_units"] = (1 + lag) * bundle.metadata["climate_block"] * best.pop("climate_multiplier")
        hp.update(best)
        print(f"Loaded best hyperparameters from {description}")
    return hp


def main():
    parser = argparse.ArgumentParser(description="Train cross-validated weekly LAND ensembles.")
    parser.add_argument("--hyperparameters", type=str)
    parser.add_argument("--study", type=str, help="Optuna study name to load best hyperparameters from")
    parser.add_argument("--trial", type=int,
                        help="use a specific trial number instead of the study's raw-objective best trial")
    parser.add_argument("--run", default="weekly_land")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override batch_size from the study/defaults")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="override learning_rate from the study/defaults")
    parser.add_argument("--workers", type=int, default=None,
                        help="folds trained in parallel processes (default: min(4, cpu_count); 1 = sequential)")
    parser.add_argument("--balanced-stations", action="store_true",
                        help="sample training weeks with equal expected station weight "
                             "(overrides the study's balanced_stations setting)")
    parser.add_argument("--cv-mode", default="loso", choices=["loso", "loso_recent"],
                        help="loso_recent early-stops each fold on the held-out station's "
                             "post-LOSO_RECENT_YEAR_START years only (matches the test protocol)")
    parser.add_argument("--daily", action="store_true",
                        help="train on daily_dataset.npz (must match the study's dataset_freq)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset NPZ to train on (default: the config path for the freq)")
    args = parser.parse_args()

    if not args.hyperparameters and not args.study:
        parser.error("Must provide either --hyperparameters or --study")

    freq = "daily" if args.daily else "weekly"
    dataset_path = Path(args.dataset) if args.dataset else config.dataset_path_for(freq)
    bundle = load_data(path=dataset_path, freq=freq)
    hp = _load_hyperparameters(args, bundle)
    if hp.get("dataset_freq", freq) != freq:
        raise ValueError(
            f"Hyperparameters were tuned on {hp['dataset_freq']} data; rerun with "
            f"{'--daily' if hp['dataset_freq'] == 'daily' else 'no --daily'} or fix dataset_freq"
        )
    dem_cell_km = bundle.metadata.get("dem_cell_km", 1.0)
    if abs(hp.get("dem_cell_km", dem_cell_km) - dem_cell_km) > 1e-9:
        raise ValueError(
            f"Hyperparameters were tuned on {hp['dem_cell_km']} km DEM cells but "
            f"{dataset_path.name} has {dem_cell_km} km cells; local/regional_dem_cfg "
            f"indices are resolution-specific -- retune on this dataset"
        )
    hp["dataset_freq"] = freq
    hp["dem_cell_km"] = dem_cell_km
    hp["dataset_path"] = str(dataset_path)
    hp["train_cv_mode"] = args.cv_mode
    if args.balanced_stations:
        hp["balanced_stations"] = True
    for key in ("batch_size", "learning_rate"):
        value = getattr(args, key)
        if value is not None:
            print(f"overriding {key}: {hp.get(key)} -> {value}")
            hp[key] = value
    folds = cv_folds(bundle, count=args.folds, mode=args.cv_mode)
    output = config.RUNS_DIR / args.run
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(output / "code_snapshot", TRAIN_SOURCE_FILES, dataset_path=dataset_path)
    workers = args.workers if args.workers is not None else min(4, os.cpu_count() or 1)
    workers = max(1, min(workers, len(folds)))
    print(f"using device: {device()}")
    if workers > 1 and device().type == "cuda":
        print("warning: parallel workers share one GPU; consider --workers 1")
    print(f"{args.cv_mode} training folds: {len(folds)} ({workers} worker{'s' if workers > 1 else ''})")
    save_json(hp, output / "hyperparameters.json")
    save_json(bundle.stats, output / "normalization.json")
    save_json({"station_roles": bundle.metadata["roles"]}, output / "split.json")

    train_folds(bundle, folds, hp, args.seeds, args.epochs, args.patience, output, workers)
    print(f"models saved to {output}")


if __name__ == "__main__":
    main()
