import argparse
import json

import numpy as np
import torch

from LAND_AS import config
from LAND_AS.data import cv_folds, load_data, loaders, model_metadata
from LAND_AS.engine import device, fit, save_json
from LAND_AS.model import LAND


def _load_hyperparameters(args, climate_channels):
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
        best["climate_units"] = climate_channels * best.pop("climate_multiplier")
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
    args = parser.parse_args()

    if not args.hyperparameters and not args.study:
        parser.error("Must provide either --hyperparameters or --study")

    bundle = load_data()
    metadata = model_metadata(bundle)
    hp = _load_hyperparameters(args, metadata["climate_shape"][0])
    folds = cv_folds(bundle, count=args.folds, mode="loso")
    output = config.RUNS_DIR / args.run
    output.mkdir(parents=True, exist_ok=True)
    print(f"using device: {device()}")
    print(f"LOSO training folds: {len(folds)}")
    save_json(hp, output / "hyperparameters.json")
    save_json(bundle.stats, output / "normalization.json")
    save_json({"station_roles": bundle.metadata["roles"]}, output / "split.json")

    for fold_number, (train_index, val_index) in enumerate(folds):
        held_out_station = sorted(set(bundle.metadata["stations"][val_index]))[0]
        fold_output = output / f"fold_{fold_number}"
        fold_output.mkdir(parents=True, exist_ok=True)
        fold_loaders = loaders(bundle, {"train": train_index, "val": val_index}, hp["batch_size"])
        print(f"\n--- Fold {fold_number}: held-out station = {held_out_station} ({len(val_index)} samples) ---")
        for seed_number in range(args.seeds):
            seed = config.SEED + seed_number
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = LAND(**metadata, **{key: hp[key] for key in ("climate_units", "dem_units", "month_units", "hidden_units", "dropout")})
            history, score = fit(
                model, fold_loaders["train"], fold_loaders["val"], args.epochs, args.patience,
                hp["learning_rate"], hp["weight_decay"], bundle.stats["target_scale"], device(),
            )
            torch.save({"state_dict": model.state_dict(), "hyperparameters": hp}, fold_output / f"seed_{seed}.pt")
            save_json({**history, "best_val_mae_mm": score, "held_out_station": held_out_station}, fold_output / f"seed_{seed}_history.json")
    print(f"models saved to {output}")


if __name__ == "__main__":
    main()
