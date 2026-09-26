import argparse
import json
import os

from LAND_AS import config
from LAND_AS.data import cv_folds, load_data
from LAND_AS.engine import device, save_json
from LAND_AS.parallelize import train_folds
from LAND_AS.provenance import TRAIN_SOURCE_FILES, snapshot_code


def _load_hyperparameters(args, bundle):
    """Resolve hyperparameters from --hyperparameters JSON or --study Optuna DB."""
    hp = dict(config.DEFAULTS)
    if args.hyperparameters:
        hp.update(json.loads(open(args.hyperparameters).read()))
    elif args.study:
        import optuna
        storage = f"sqlite:///{config.TUNING_DIR / args.study / 'study.db'}"
        study = optuna.load_study(study_name=args.study, storage=storage)
        if len(study.trials) == 0 or study.best_trial is None:
            raise RuntimeError(f"Study '{args.study}' has no completed trials; cannot load best hyperparameters")
        best = dict(study.best_params)
        # Older studies searched climate_multiplier x channel blocks; new
        # studies fix climate_units in config.DEFAULTS.
        if "climate_multiplier" in best:
            lag = best.get("climate_lag_weeks", bundle.metadata["lag_max"])
            best["climate_units"] = (1 + lag) * bundle.metadata["climate_block"] * best.pop("climate_multiplier")
        hp.update(best)
        print(f"Loaded best hyperparameters from study '{args.study}' (trial #{study.best_trial.number}, MAE={study.best_value:.3f} mm)")
    return hp


def main():
    parser = argparse.ArgumentParser(description="Train cross-validated weekly LAND ensembles.")
    parser.add_argument("--hyperparameters", type=str)
    parser.add_argument("--study", type=str, help="Optuna study name to load best hyperparameters from")
    parser.add_argument("--run", default="weekly_land")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None,
                        help="folds trained in parallel processes (default: min(4, cpu_count); 1 = sequential)")
    args = parser.parse_args()

    if not args.hyperparameters and not args.study:
        parser.error("Must provide either --hyperparameters or --study")

    bundle = load_data()
    hp = _load_hyperparameters(args, bundle)
    folds = cv_folds(bundle, count=args.folds, mode="loso")
    output = config.RUNS_DIR / args.run
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(output / "code_snapshot", TRAIN_SOURCE_FILES)
    workers = args.workers if args.workers is not None else min(4, os.cpu_count() or 1)
    workers = max(1, min(workers, len(folds)))
    print(f"using device: {device()}")
    if workers > 1 and device().type == "cuda":
        print("warning: parallel workers share one GPU; consider --workers 1")
    print(f"LOSO training folds: {len(folds)} ({workers} worker{'s' if workers > 1 else ''})")
    save_json(hp, output / "hyperparameters.json")
    save_json(bundle.stats, output / "normalization.json")
    save_json({"station_roles": bundle.metadata["roles"]}, output / "split.json")

    train_folds(bundle, folds, hp, args.seeds, args.epochs, args.patience, output, workers)
    print(f"models saved to {output}")


if __name__ == "__main__":
    main()
