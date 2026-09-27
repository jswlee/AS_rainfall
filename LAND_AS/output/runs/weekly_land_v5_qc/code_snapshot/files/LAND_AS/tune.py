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
    normalized_bundle,
)
from LAND_AS.engine import device, fit, save_json
from LAND_AS.model import build_model
from LAND_AS.provenance import TRAIN_SOURCE_FILES, snapshot_code


def main():
    parser = argparse.ArgumentParser(description="Tune the weekly American Samoa LAND model.")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=30,
                        help="early stopping is not allowed before this many epochs")
    parser.add_argument("--opt-metric", default="mae", choices=["mae", "mse", "mse_ratio"],
                        help="objective and checkpoint monitor; mse_ratio divides validation MSE by each fold's train-mean MSE")
    parser.add_argument("--cv-mode", default="kfold", choices=["kfold", "loso", "temporal", "both"],
                        help="validation fold design: spatial k-fold/LOSO, blocked years, or LOSO+temporal")
    parser.add_argument("--fold-agg", default="mean", choices=["mean", "median"],
                        help="aggregate fold scores by mean or median")
    parser.add_argument("--model-type", default="gamma", choices=["gamma", "huber"],
                        help="output head to tune")
    parser.add_argument("--loss-type", default="huber", choices=["huber", "huber_weighted"],
                        help="scalar Huber loss when --model-type huber")
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--rainfall-weight", action="store_true",
                        help="weight Gamma NLL by log1p(target); ignored by scalar Huber")
    parser.add_argument("--balanced-stations", action="store_true",
                        help="sample training weeks with equal expected station weight")
    parser.add_argument("--search-space", default="core", choices=["core", "broad"],
                        help="core retunes v5-sensitive knobs; broad also searches model width, dropout, batch size, and DEM size")
    parser.add_argument("--study", default="weekly_land")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    bundle = load_data()
    folds = cv_folds(bundle, count=args.folds, mode=args.cv_mode)
    if not folds:
        raise ValueError(f"{args.cv_mode} produced no validation folds; use --folds >= 2")
    output = config.TUNING_DIR / args.study
    output.mkdir(parents=True, exist_ok=True)
    snapshot_code(output / "code_snapshot", TRAIN_SOURCE_FILES)
    print(f"using device: {device()}")
    print(f"{args.cv_mode} CV for tuning: {len(folds)} folds; aggregation={args.fold_agg}")

    def objective(trial):
        # Fixed knobs live in config.DEFAULTS; only params with measured
        # importance are searched (v5: lr >> rain_lag ~ local_dem_cfg >> rest).
        hp = {
            **config.DEFAULTS,
            # Climate lag had zero importance in v5; fixed to current week only
            # (keeps inputs at 30 channels instead of 120).
            "climate_lag_weeks": 0,
            "model_type": args.model_type,
            "rainfall_weight": args.rainfall_weight,
            "balanced_stations": args.balanced_stations,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "rain_lag_weeks": trial.suggest_int("rain_lag_weeks", 0, bundle.metadata["lag_max"]),
            "local_dem_cfg": trial.suggest_int("local_dem_cfg", 0, len(DEM_LOCAL_CHOICES) - 1),
            "regional_dem_cfg": trial.suggest_int("regional_dem_cfg", 0, len(DEM_REGIONAL_CHOICES) - 1),
        }
        if args.search_space == "broad":
            channels = int(bundle.metadata["climate_block"]) * (1 + hp["climate_lag_weeks"])
            hp.update({
                "climate_units": channels * trial.suggest_int("climate_multiplier", 4, 16),
                "dem_units": trial.suggest_categorical("dem_units", [16, 32, 64, 96]),
                "month_units": trial.suggest_categorical("month_units", [8, 16, 32, 64]),
                "hidden_units": trial.suggest_categorical("hidden_units", [128, 192, 256, 320, 384]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.6),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024]),
                "dem_size": trial.suggest_int("dem_size", 4, 12),
            })
        if args.model_type == "huber":
            hp["loss_type"] = args.loss_type
            hp["huber_delta"] = args.huber_delta
        crop = crop_from_hp(hp)
        trial_metadata = model_metadata(bundle, climate_lag=hp["climate_lag_weeks"], rain_lag=hp["rain_lag_weeks"])
        scores, raw_scores = [], []
        fit_monitor = "mse" if args.opt_metric in ("mse", "mse_ratio") else "mae"
        for train_index, val_index in folds:
            fold_bundle = normalized_bundle(bundle, train_index)
            fold_loaders = loaders(
                fold_bundle, {"train": train_index, "val": val_index}, hp["batch_size"],
                crop=crop, balance_stations=hp["balanced_stations"],
            )
            model = build_model(hp, trial_metadata)
            _, score = fit(
                model, fold_loaders["train"], fold_loaders["val"], args.epochs, args.patience,
                hp["learning_rate"], hp["weight_decay"], fold_bundle.stats["target_scale"], device(),
                rainfall_weight=hp["rainfall_weight"], monitor=fit_monitor,
                min_epochs=args.min_epochs, huber_delta=hp.get("huber_delta", 0.5),
                loss_type=hp.get("loss_type"),
            )
            raw_scores.append(score)
            if args.opt_metric == "mse_ratio":
                target = fold_bundle.arrays["target"][val_index].numpy()
                baseline = np.mean(fold_bundle.arrays["target"][train_index].numpy())
                denominator = np.mean((target - baseline) ** 2)
                score = score / max(float(denominator), 1e-8)
            scores.append(score)
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("fold_scores_raw", raw_scores)
        aggregate = np.median(scores) if args.fold_agg == "median" else np.mean(scores)
        return float(aggregate)

    storage = f"sqlite:///{output / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study, storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(objective, n_trials=args.trials)
    best = dict(study.best_params)
    # Persist fixed settings and the tuning objective so trained runs reproduce it
    best.update({
        "monitor": "mse" if args.opt_metric == "mse_ratio" else args.opt_metric,
        "opt_metric": args.opt_metric,
        "min_epochs": args.min_epochs,
        "model_type": args.model_type,
        "rainfall_weight": args.rainfall_weight,
        "balanced_stations": args.balanced_stations,
        "cv_mode": args.cv_mode,
        "fold_agg": args.fold_agg,
        "search_space": args.search_space,
    })
    if args.model_type == "huber":
        best["loss_type"] = args.loss_type
        best["huber_delta"] = args.huber_delta
    save_json(best, output / "best.json")
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    print(f"best {args.opt_metric}={study.best_value:.3f}; parameters saved to {output / 'best.json'}")


if __name__ == "__main__":
    main()
