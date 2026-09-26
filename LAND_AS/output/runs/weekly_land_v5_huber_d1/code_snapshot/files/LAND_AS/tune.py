import argparse

import numpy as np
import optuna
import torch

from LAND_AS import config
from LAND_AS.data import (
    DEM_LOCAL_CHOICES,
    DEM_REGIONAL_CHOICES,
    crop_from_hp,
    cv_folds,
    load_data,
    loaders,
    model_metadata,
)
from LAND_AS.engine import device, fit, save_json
from LAND_AS.model import LAND, MODEL_HP_KEYS
from LAND_AS.provenance import TRAIN_SOURCE_FILES, snapshot_code


def main():
    parser = argparse.ArgumentParser(description="Tune the weekly American Samoa LAND model.")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=30,
                        help="early stopping is not allowed before this many epochs")
    parser.add_argument("--opt-metric", default="mae", choices=["mae", "mse"],
                        help="validation metric the study minimizes and early stopping monitors")
    parser.add_argument("--study", default="weekly_land")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    bundle = load_data()
    folds = cv_folds(bundle, count=args.folds, mode="kfold")
    output = config.TUNING_DIR / args.study
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(output / "code_snapshot", TRAIN_SOURCE_FILES)
    print(f"using device: {device()}")
    print(f"Spatial {len(folds)}-fold CV for tuning: {len(folds)} folds")

    def objective(trial):
        # Fixed knobs live in config.DEFAULTS; only params with measured
        # importance are searched (v5: lr >> rain_lag ~ local_dem_cfg >> rest).
        hp = {
            **config.DEFAULTS,
            # Climate lag had zero importance in v5; fixed to current week only
            # (keeps inputs at 30 channels instead of 120).
            "climate_lag_weeks": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "rain_lag_weeks": trial.suggest_int("rain_lag_weeks", 0, bundle.metadata["lag_max"]),
            "local_dem_cfg": trial.suggest_int("local_dem_cfg", 0, len(DEM_LOCAL_CHOICES) - 1),
            "regional_dem_cfg": trial.suggest_int("regional_dem_cfg", 0, len(DEM_REGIONAL_CHOICES) - 1),
        }
        crop = crop_from_hp(hp)
        trial_metadata = model_metadata(bundle, climate_lag=hp["climate_lag_weeks"], rain_lag=hp["rain_lag_weeks"])
        scores = []
        for train_index, val_index in folds:
            fold_loaders = loaders(bundle, {"train": train_index, "val": val_index}, hp["batch_size"], crop=crop)
            model = LAND(**trial_metadata, **{key: hp[key] for key in MODEL_HP_KEYS if key in hp})
            _, score = fit(
                model, fold_loaders["train"], fold_loaders["val"], args.epochs, args.patience,
                hp["learning_rate"], hp["weight_decay"], bundle.stats["target_scale"], device(),
                monitor=args.opt_metric, min_epochs=args.min_epochs,
            )
            scores.append(score)
        return float(np.mean(scores))

    storage = f"sqlite:///{output / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study, storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(objective, n_trials=args.trials)
    best = dict(study.best_params)
    # Persist the tuning objective so trained runs early-stop on the same metric
    best["monitor"] = args.opt_metric
    best["min_epochs"] = args.min_epochs
    save_json(best, output / "best.json")
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    print(f"best {args.opt_metric}={study.best_value:.3f}; parameters saved to {output / 'best.json'}")


if __name__ == "__main__":
    main()
