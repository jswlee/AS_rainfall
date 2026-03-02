"""
Step 4: Hyperparameter tuning for the LAND model using Optuna.

Supports four loss / output-head configurations:
  - mse:             softplus head, MSE loss, objective = val MSE
  - gamma:           2-output head (alpha, beta), Gamma NLL on wet days only, objective = val MAE (mm)
  - tweedie:         softplus head, Tweedie deviance loss (p tuned), objective = val MAE (mm)
  - bernoulli_gamma: 3-output head, Bernoulli-Gamma NLL loss, objective = val MAE (mm)

CV strategy (--cv-folds N, default 3):
  Each Optuna trial trains N folds and returns mean val MAE across folds.
  Fold construction is spatio-temporal: each fold rotates which station group
  and which year block are held out for validation, so the objective is never
  evaluated on the same data every trial.

Usage:
    python -m Daily_Modeling.scripts.04_tune_land [--n-trials 50] [--loss-type gamma]
    python -m Daily_Modeling.scripts.04_tune_land --cv-folds 3 --no-early-stopping
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import json
import torch

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz, normalize_tensors, make_dataloaders
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, compute_station_year_ranges,
    compute_year_boundaries, plot_split_heatmap,
)
from Daily_Modeling.models.land import create_land_model
from Daily_Modeling.utils.training import train_model
from Daily_Modeling.utils.metrics import compute_metrics, compute_extreme_metrics
from Daily_Modeling.utils.io_utils import save_json
from Daily_Modeling.utils.device import select_device


def _get_metadata(tensors, dem_crop_config=None):
    c = tensors["climate"]
    meta = {
        "climate_shape": tuple(c.shape[1:]),
        "local_dem_shape": tuple(tensors["local_dem"].shape[1:]),
        "regional_dem_shape": tuple(tensors["regional_dem"].shape[1:]),
        "num_month_features": int(tensors["temporal"].shape[1]),
        "num_climate_vars": int(c.shape[1]),
    }
    # Override DEM shapes if cropping is configured
    if dem_crop_config is not None:
        if "local_patch_size" in dem_crop_config:
            lp = dem_crop_config["local_patch_size"]
            meta["local_dem_shape"] = (lp, lp)
        if "regional_patch_size" in dem_crop_config:
            rp = dem_crop_config["regional_patch_size"]
            meta["regional_dem_shape"] = (rp, rp)
    return meta


# Tuning-specific constants (faster than full training)
_TUNE_MAX_EPOCHS = 75
_TUNE_PATIENCE = 15
_TUNE_SUBSET_FRAC = 1.0  # train on 50% of data per fold for speed
_TUNE_N_CV_FOLDS = 3


# Maps loss_type -> required output_head
_LOSS_TO_HEAD = {
    "mse": "softplus",
    "gamma": "gamma",
    "tweedie": "softplus",
    "bernoulli_gamma": "bernoulli_gamma",
}


def _predict_mm(model, loader, device, target_scale, output_head):
    """Run inference and return (preds_mm, targets_mm) numpy arrays."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for feats, tgt in loader:
            feats = {k: torch.nan_to_num(v.to(device)) for k, v in feats.items()}
            out = model(feats)
            if output_head == "bernoulli_gamma":
                p_rain = torch.sigmoid(out[:, 0])
                alpha = torch.nn.functional.softplus(out[:, 1]).clamp(min=1e-6)
                beta  = torch.nn.functional.softplus(out[:, 2]).clamp(min=1e-6)
                pred = p_rain * alpha * beta  # E[Y] = p * alpha * beta
            elif output_head == "gamma":
                alpha = torch.nn.functional.softplus(out[:, 0]).clamp(min=1e-6)
                beta  = torch.nn.functional.softplus(out[:, 1]).clamp(min=1e-6)
                pred  = alpha * beta  # E[Y] = alpha * beta
            else:
                pred = out.squeeze(-1)
            preds.append(pred.cpu().numpy().ravel())
            targets.append(tgt.cpu().numpy().ravel())
    preds_mm = np.concatenate(preds) * target_scale
    targets_mm = np.concatenate(targets) * target_scale
    return preds_mm, targets_mm


def _build_criterion(loss_type, hp):
    """Build the loss criterion for a given loss_type."""
    if loss_type == "tweedie":
        from Daily_Modeling.models.losses import TweedieDeviance
        return TweedieDeviance(p=hp.get("tweedie_p", 1.5))
    elif loss_type == "bernoulli_gamma":
        from Daily_Modeling.models.losses import BernoulliGammaNLL
        return BernoulliGammaNLL()
    elif loss_type == "gamma":
        from Daily_Modeling.models.losses import GammaNLL
        return GammaNLL()
    else:
        return None  # default MSE inside train_model


def _make_metric_fn(loss_type, output_head, target_scale):
    """Return a per-batch MAE-in-mm metric function, or None for MSE."""
    if loss_type not in ("tweedie", "bernoulli_gamma", "gamma"):
        return None

    def _val_mae_mm(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.view(-1)
        if output_head == "bernoulli_gamma":
            p_rain = torch.sigmoid(outputs[:, 0])
            alpha = torch.nn.functional.softplus(outputs[:, 1]).clamp(min=1e-6)
            beta  = torch.nn.functional.softplus(outputs[:, 2]).clamp(min=1e-6)
            pred  = p_rain * alpha * beta
        elif output_head == "gamma":
            alpha = torch.nn.functional.softplus(outputs[:, 0]).clamp(min=1e-6)
            beta  = torch.nn.functional.softplus(outputs[:, 1]).clamp(min=1e-6)
            pred  = alpha * beta
        else:
            pred = outputs.view(-1)
        return (pred - y).abs().mean() * float(target_scale)

    return _val_mae_mm


def _make_cv_folds(splits, n_folds, cv_mode, rng_seed):
    """Build CV folds based on mode: temporal, spatial, both, or mixed.

    - temporal: folds split val_temporal only (train stations, held-out years)
    - spatial: folds split val_spatial only (held-out stations)
    - both: folds alternate between temporal and spatial validation
    - mixed: pool train+val_temporal+val_spatial, shuffle, and split (legacy)
    """
    rng = np.random.RandomState(rng_seed)
    train_idx = splits.get("train", np.array([], dtype=int))
    val_temporal = splits.get("val_temporal", np.array([], dtype=int))
    val_spatial = splits.get("val_spatial", np.array([], dtype=int))

    if n_folds <= 1:
        if cv_mode == "spatial":
            val_idx = val_spatial if len(val_spatial) > 0 else val_temporal
        else:
            val_idx = val_temporal if len(val_temporal) > 0 else val_spatial
        return [(train_idx.astype(int), val_idx.astype(int))]

    if cv_mode == "temporal":
        if len(val_temporal) == 0:
            raise ValueError("cv_mode=temporal but val_temporal is empty")
        if len(val_temporal) < n_folds:
            print(f"Warning: val_temporal has only {len(val_temporal)} samples for {n_folds} folds; using single fold")
            return [(train_idx.astype(int), val_temporal.astype(int))]
        shuffled = rng.permutation(val_temporal)
        fold_indices = np.array_split(shuffled, n_folds)
        folds = []
        for i in range(n_folds):
            val_fold = fold_indices[i]
            folds.append((train_idx.astype(int), val_fold.astype(int)))
        return folds

    elif cv_mode == "spatial":
        if len(val_spatial) == 0:
            raise ValueError("cv_mode=spatial but val_spatial is empty")
        if len(val_spatial) < n_folds:
            print(f"Warning: val_spatial has only {len(val_spatial)} samples for {n_folds} folds; using single fold")
            return [(train_idx.astype(int), val_spatial.astype(int))]
        shuffled = rng.permutation(val_spatial)
        fold_indices = np.array_split(shuffled, n_folds)
        folds = []
        for i in range(n_folds):
            val_fold = fold_indices[i]
            folds.append((train_idx.astype(int), val_fold.astype(int)))
        return folds

    elif cv_mode == "both":
        if len(val_temporal) == 0 or len(val_spatial) == 0:
            raise ValueError("cv_mode=both requires both val_temporal and val_spatial to be non-empty")
        n_temp = n_folds // 2
        n_spat = n_folds - n_temp
        folds = []
        if n_temp > 0:
            if len(val_temporal) < n_temp:
                n_temp = 1
            temp_shuffled = rng.permutation(val_temporal)
            temp_folds = np.array_split(temp_shuffled, n_temp)
            for val_fold in temp_folds:
                folds.append((train_idx.astype(int), val_fold.astype(int)))
        if n_spat > 0:
            if len(val_spatial) < n_spat:
                n_spat = 1
            spat_shuffled = rng.permutation(val_spatial)
            spat_folds = np.array_split(spat_shuffled, n_spat)
            for val_fold in spat_folds:
                folds.append((train_idx.astype(int), val_fold.astype(int)))
        return folds

    elif cv_mode == "mixed":
        all_non_test = np.concatenate([train_idx, val_temporal, val_spatial]).astype(int)
        all_non_test = np.unique(all_non_test)
        n = len(all_non_test)
        if n < n_folds * 20:
            val_idx = val_temporal if len(val_temporal) > 0 else val_spatial
            return [(train_idx.astype(int), val_idx.astype(int))]
        shuffled = rng.permutation(all_non_test)
        fold_indices = np.array_split(shuffled, n_folds)
        folds = []
        for i in range(n_folds):
            val_fold = fold_indices[i]
            others = [fold_indices[j] for j in range(n_folds) if j != i]
            train_fold = np.concatenate(others) if len(others) > 0 else np.array([], dtype=int)
            folds.append((train_fold.astype(int), val_fold.astype(int)))
        return folds

    else:
        raise ValueError(f"Unknown cv_mode: {cv_mode}")


def objective(
    trial: optuna.Trial,
    tensors,
    splits,
    base_metadata,
    device,
    target_scale,
    loss_type="mse",
    n_cv_folds=3,
    cv_mode="temporal",
    opt_metric: str = "auto",
    extreme_percentile: float = 98.0,
    csi_threshold_mm: float = 50.0,
    no_early_stopping=False,
):
    num_cv = base_metadata["num_climate_vars"]

    # --- DEM patch size HPs ---
    local_candidates = config.DEM_LOCAL_CANDIDATES
    regional_candidates = config.DEM_REGIONAL_CANDIDATES
    local_idx = trial.suggest_int("local_dem_cfg", 0, len(local_candidates) - 1)
    regional_idx = trial.suggest_int("regional_dem_cfg", 0, len(regional_candidates) - 1)
    lp, lk = local_candidates[local_idx]
    rp, rk = regional_candidates[regional_idx]
    dem_crop = {
        "local_patch_size": lp, "local_km": lk,
        "regional_patch_size": rp, "regional_km": rk,
    }
    metadata = _get_metadata(tensors, dem_crop_config=dem_crop)

    output_head = _LOSS_TO_HEAD[loss_type]

    hp = {
        "climate_units": trial.suggest_int("climate_units", num_cv * 34, num_cv * 102, step=num_cv),
        "dem_units":     trial.suggest_int("dem_units", 16, 128, step=16),
        "temporal_units": trial.suggest_int("temporal_units", 4, 32, step=4),
        "na": trial.suggest_int("na", 64, 512, step=64),
        "nb": trial.suggest_int("nb", 32, 128, step=32),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.3, 0.6, step=0.05),
        # Tune LR at a reference batch size, then scale based on chosen batch_size.
        # For AdamW, sqrt scaling is a reasonable default when varying batch size.
        "base_lr": trial.suggest_float("base_lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "climate_processing": "conv2d",
        "output_head": output_head,
        "loss_type": loss_type,
        "local_dem_patch": lp, "local_dem_km": lk,
        "regional_dem_patch": rp, "regional_dem_km": rk,
    }

    _LR_REF_BATCH = 32
    lr_scale = float(hp["batch_size"]) / float(_LR_REF_BATCH)
    hp["learning_rate"] = float(hp["base_lr"]) * (lr_scale ** 0.5)

    # Tweedie-specific: tune the p-value
    if loss_type == "tweedie":
        hp["tweedie_p"] = trial.suggest_float("tweedie_p", 1.1, 1.9, step=0.05)

    if hp["climate_units"] % num_cv != 0:
        hp["climate_units"] = (hp["climate_units"] // num_cv) * num_cv

    criterion = _build_criterion(loss_type, hp)
    metric_fn  = _make_metric_fn(loss_type, output_head, target_scale)

    # --- Build CV folds ---
    cv_folds = _make_cv_folds(splits, n_cv_folds, cv_mode, rng_seed=config.RANDOM_SEED + trial.number)

    print(
        f"\nTrial {trial.number}: params≈{hp['na']}+{hp['nb']}  dem={hp['dem_units']}  "
        f"DEM local={lp}x{lp}@{lk}km  regional={rp}x{rp}@{rk}km  "
        f"loss={loss_type}  folds={len(cv_folds)}"
        + (f"  tweedie_p={hp['tweedie_p']:.2f}" if loss_type == "tweedie" else "")
    )
    print("Tuned HPs:\n" + json.dumps(hp, indent=2, sort_keys=True))

    fold_scores = []
    for fold_i, (train_fold_idx, val_fold_idx) in enumerate(cv_folds):
        # Subsample training data for speed
        rng = np.random.RandomState(config.RANDOM_SEED + trial.number * 100 + fold_i)
        if _TUNE_SUBSET_FRAC < 1.0 and len(train_fold_idx) > 200:
            n_sub = max(100, int(len(train_fold_idx) * _TUNE_SUBSET_FRAC))
            train_fold_idx = rng.choice(train_fold_idx, n_sub, replace=False)

        fold_splits = {"train": train_fold_idx, "val": val_fold_idx}
        loaders = make_dataloaders(tensors, fold_splits, target_scale=target_scale,
                                   batch_size=hp["batch_size"],
                                   dem_crop_config=dem_crop)

        if "val" not in loaders or len(loaders["val"].dataset) == 0:
            print(f"  Fold {fold_i}: empty val, skipping")
            continue

        model = create_land_model(hp, metadata).to(device)

        history = train_model(
            model, loaders["train"], loaders["val"], device,
            epochs=_TUNE_MAX_EPOCHS, patience=_TUNE_PATIENCE,
            min_epochs=40,
            learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
            criterion=criterion,
            metric_fn=metric_fn,
            verbose=1,
            trial=None,  # no Optuna pruning inside CV folds
            no_early_stopping=no_early_stopping,
            monitor=("val_metric" if loss_type != "mse" else "val_loss"),
        )

        if loss_type == "mse":
            # Training objective for MSE remains the loss curve
            fold_score = float(min(history["val_loss"]))
            metric_label = "MSE"
        else:
            preds_mm, targets_mm = _predict_mm(
                model, loaders["val"], device,
                target_scale, output_head
            )
            m = compute_metrics(targets_mm, preds_mm)
            m.update(
                compute_extreme_metrics(
                    targets_mm,
                    preds_mm,
                    percentile=extreme_percentile,
                    csi_threshold_mm=csi_threshold_mm,
                )
            )

            if opt_metric == "auto":
                # Default historical behavior
                fold_score = float(m["mae"])
                metric_label = "MAE_mm"
            elif opt_metric == "mae":
                fold_score = float(m["mae"])
                metric_label = "MAE_mm"
            elif opt_metric == "pctl_abs_rel_bias":
                fold_score = float(m.get("pctl_abs_rel_bias", float("inf")))
                metric_label = f"P{int(extreme_percentile)}_abs_rel_bias"
            elif opt_metric == "csi":
                # Maximise CSI; Optuna minimises so use (1 - CSI)
                csi = float(m.get("csi", float("nan")))
                fold_score = float(1.0 - csi) if np.isfinite(csi) else float("inf")
                metric_label = f"1-CSI@{csi_threshold_mm:g}mm"
            else:
                raise ValueError(f"Unknown opt_metric: {opt_metric}")

        print(f"  Fold {fold_i}: {metric_label}={fold_score:.6f}")
        fold_scores.append(fold_score)

    if len(fold_scores) == 0:
        return float("inf")

    mean_score = float(np.mean(fold_scores))
    print(f"  Trial {trial.number} mean CV score: {mean_score:.4f}")
    return mean_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default=None,
                        help="Study name (default: land_daily_<loss_type>)")
    parser.add_argument("--loss-type", default="gamma",
                        choices=["mse", "gamma", "tweedie", "bernoulli_gamma"],
                        help="Loss / output-head configuration (default: gamma)")
    parser.add_argument("--cv-folds", type=int, default=_TUNE_N_CV_FOLDS,
                        help="Number of CV folds per trial (default: 3)")
    parser.add_argument("--cv-mode", default="temporal",
                        choices=["temporal", "spatial", "both", "mixed"],
                        help="CV fold construction mode: temporal (held-out years), spatial (held-out stations), "
                             "both (mix of temporal and spatial folds), mixed (shuffled pool, legacy)")
    parser.add_argument("--opt-metric", default="auto",
                        choices=["auto", "mae", "pctl_abs_rel_bias", "csi"],
                        help="Optuna objective metric for non-MSE losses: auto (MAE), mae, pctl_abs_rel_bias, or csi")
    parser.add_argument("--extreme-percentile", type=float, default=98.0,
                        help="Percentile for extreme bias metric (default: 98)")
    parser.add_argument("--csi-threshold-mm", type=float, default=50.0,
                        help="Threshold in mm for CSI metric (default: 50)")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping; train all epochs per fold")
    args = parser.parse_args()
    if args.study_name is None:
        args.study_name = f"land_daily_{args.loss_type}"

    device = select_device()
    print(f"Device: {device}")

    tensors, meta = load_tensors_from_npz(device=device)
    stations = meta["stations"]
    years = meta["years"]

    # Data-driven year boundaries
    train_yr, val_yr, test_yr = compute_year_boundaries(years)

    yr_ranges = compute_station_year_ranges(stations, years)
    groups = assign_station_groups(
        sorted(set(str(s) for s in stations)),
        station_year_ranges=yr_ranges,
        val_years=val_yr, test_years=test_yr,
    )
    splits = spatiotemporal_split(stations, years, groups,
                                  train_years=train_yr, val_years=val_yr, test_years=test_yr)
    tensors, stats = normalize_tensors(tensors, splits["train"])
    target_scale = stats["target_std_mm"]
    base_metadata = _get_metadata(tensors)

    # Save split heatmap
    plot_split_heatmap(stations, years, groups, train_yr, val_yr, test_yr,
                       save_path=config.EDA_DIR / "split_heatmap_land_tuning.png",
                       title="LAND Tuning Split")

    out_dir = config.TUNING_DIR / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
    )

    if args.loss_type == "mse":
        metric_name = "val_MSE"
    else:
        if args.opt_metric in ("auto", "mae"):
            metric_name = "val_MAE_mm"
        elif args.opt_metric == "pctl_abs_rel_bias":
            metric_name = f"P{int(args.extreme_percentile)}_abs_rel_bias"
        else:
            metric_name = f"1-CSI@{args.csi_threshold_mm:g}mm"

    study.optimize(
        lambda trial: objective(
            trial, tensors, splits, base_metadata, device,
            target_scale,
            loss_type=args.loss_type,
            n_cv_folds=args.cv_folds,
            cv_mode=args.cv_mode,
            opt_metric=args.opt_metric,
            extreme_percentile=args.extreme_percentile,
            csi_threshold_mm=args.csi_threshold_mm,
            no_early_stopping=args.no_early_stopping,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest trial: {study.best_trial.number}  {metric_name}={study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    # Enrich best params with resolved DEM config + static defaults
    best_hp = dict(study.best_params)

    # Reconstruct derived hyperparameters that were not directly tuned.
    if "base_lr" in best_hp and "batch_size" in best_hp and "learning_rate" not in best_hp:
        _LR_REF_BATCH = 32
        lr_scale = float(best_hp["batch_size"]) / float(_LR_REF_BATCH)
        best_hp["learning_rate"] = float(best_hp["base_lr"]) * (lr_scale ** 0.5)

    dem_crop = config.resolve_dem_crop(best_hp)
    if dem_crop is not None:
        best_hp["local_dem_patch"] = dem_crop["local_patch_size"]
        best_hp["local_dem_km"] = dem_crop["local_km"]
        best_hp["regional_dem_patch"] = dem_crop["regional_patch_size"]
        best_hp["regional_dem_km"] = dem_crop["regional_km"]
    best_hp.setdefault("climate_processing", "conv2d")
    best_hp["output_head"] = _LOSS_TO_HEAD[args.loss_type]
    best_hp["loss_type"] = args.loss_type
    save_json(best_hp, out_dir / "best_hyperparameters.json")
    save_json({"target_std_mm": target_scale, **stats}, out_dir / "normalization_stats.json")

    # Save all trials
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append({"trial": t.number, "value": t.value, **t.params})
    pd.DataFrame(rows).to_csv(out_dir / "all_trials.csv", index=False)

    # --- HP importance & tuning visualisations ---
    _save_tuning_visuals(study, out_dir)

    print(f"Results saved to {out_dir}")


def _save_tuning_visuals(study, out_dir):
    """Save Optuna HP importance, optimization history, and parallel coordinate plots."""
    try:
        from optuna.visualization.matplotlib import (
            plot_param_importances,
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_slice,
        )

        # HP importance (fANOVA-based)
        fig = plot_param_importances(study)
        fig.figure.savefig(str(out_dir / "hp_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig.figure)
        print(f"  Saved hp_importance.png")

        # Optimization history
        fig = plot_optimization_history(study)
        fig.figure.savefig(str(out_dir / "optimization_history.png"), dpi=150, bbox_inches="tight")
        plt.close(fig.figure)
        print(f"  Saved optimization_history.png")

        # Parallel coordinate
        fig = plot_parallel_coordinate(study)
        fig.figure.savefig(str(out_dir / "parallel_coordinate.png"), dpi=150, bbox_inches="tight")
        plt.close(fig.figure)
        print(f"  Saved parallel_coordinate.png")

        # Slice plots for each HP
        fig = plot_slice(study)
        if hasattr(fig, 'figure'):
            fig.figure.savefig(str(out_dir / "slice_plots.png"), dpi=150, bbox_inches="tight")
            plt.close(fig.figure)
        else:
            fig.savefig(str(out_dir / "slice_plots.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"  Saved slice_plots.png")

    except Exception as e:
        print(f"  WARNING: Could not generate tuning visuals: {e}")


if __name__ == "__main__":
    main()
