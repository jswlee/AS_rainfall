import argparse

import numpy as np
import optuna
import torch

from LAND_AS import config
from LAND_AS.data import cv_folds, load_data, loaders, model_metadata
from LAND_AS.engine import device, fit, save_json
from LAND_AS.model import LAND


def main():
    parser = argparse.ArgumentParser(description="Tune the weekly American Samoa LAND model.")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--study", default="weekly_land")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    bundle = load_data()
    metadata = model_metadata(bundle)
    folds = cv_folds(bundle, args.folds)
    output = config.TUNING_DIR / args.study
    output.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        hp = {
            "climate_units": metadata["climate_shape"][0] * trial.suggest_int("climate_multiplier", 2, 8),
            "dem_units": trial.suggest_int("dem_units", 32, 128, step=32),
            "month_units": trial.suggest_int("month_units", 16, 64, step=16),
            "hidden_units": trial.suggest_int("hidden_units", 64, 512, step=64),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
        scores = []
        for train_index, val_index in folds:
            fold_loaders = loaders(bundle, {"train": train_index, "val": val_index}, hp["batch_size"])
            model = LAND(**metadata, **{key: hp[key] for key in ("climate_units", "dem_units", "month_units", "hidden_units", "dropout")})
            _, score = fit(
                model, fold_loaders["train"], fold_loaders["val"], args.epochs, args.patience,
                hp["learning_rate"], hp["weight_decay"], bundle.stats["target_scale"], device(),
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
    best["climate_units"] = metadata["climate_shape"][0] * best.pop("climate_multiplier")
    save_json(best, output / "best.json")
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    print(f"best MAE={study.best_value:.3f} mm; parameters saved to {output / 'best.json'}")


if __name__ == "__main__":
    main()
