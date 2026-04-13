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

import numpy as np
import optuna
import pandas as pd
import json
import torch

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz, normalize_tensors, make_dataloaders, get_dataset_metadata, precompute_dem_crops
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, compute_station_year_ranges,
    compute_year_boundaries, make_cv_folds,
)
from Daily_Modeling.models.land import create_land_model
from Daily_Modeling.utils.training import train_model
from Daily_Modeling.models.losses import get_criterion
from Daily_Modeling.utils.inference import predict_mm, make_metric_fn
from Daily_Modeling.utils.metrics import compute_metrics, compute_extreme_metrics
from Daily_Modeling.utils.io_utils import save_json
from Daily_Modeling.utils.device import select_device
from Daily_Modeling.utils.visualization import plot_split_heatmap, save_optuna_visualizations


def _dist_to_dict(d):
    try:
        from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    except Exception:
        return {"type": type(d).__name__, "repr": repr(d)}

    if isinstance(d, CategoricalDistribution):
        return {"type": "categorical", "choices": list(d.choices)}
    if isinstance(d, FloatDistribution):
        return {
            "type": "float",
            "low": float(d.low),
            "high": float(d.high),
            "log": bool(d.log),
            "step": None if d.step is None else float(d.step),
        }
    if isinstance(d, IntDistribution):
        return {
            "type": "int",
            "low": int(d.low),
            "high": int(d.high),
            "log": bool(d.log),
            "step": int(d.step),
        }
    return {"type": type(d).__name__, "repr": repr(d)}


def objective(
    trial: optuna.Trial,
    tensors,
    splits,
    base_metadata,
    device,
    target_scale,
    args,
    can_compile=False,
):
    num_cv = base_metadata["num_climate_vars"]

    # Select DEM configuration
    local_idx = trial.suggest_int("local_dem_cfg", 0, len(config.DEM_LOCAL_CANDIDATES) - 1)
    regional_idx = trial.suggest_int("regional_dem_cfg", 0, len(config.DEM_REGIONAL_CANDIDATES) - 1)
    lp, lk = config.DEM_LOCAL_CANDIDATES[local_idx]
    rp, rk = config.DEM_REGIONAL_CANDIDATES[regional_idx]
    dem_crop = {
        "local_patch_size": lp, "local_km": lk,
        "regional_patch_size": rp, "regional_km": rk,
    }

    # Batch-crop all DEM samples once (eliminates per-sample cropping in DataLoader)
    cropped_tensors = precompute_dem_crops(tensors, dem_crop)
    metadata = get_dataset_metadata(cropped_tensors)

    output_head = config.LOSS_TO_HEAD[args.loss_type]

    hp = {
        "climate_units": trial.suggest_int("climate_units", num_cv * 140, num_cv * 250, step=num_cv),
        "dem_units": trial.suggest_int("dem_units", 16, 256, step=16),
        "dem_patch_size": trial.suggest_int("dem_patch_size", 3, 12),
        "temporal_units": trial.suggest_int("temporal_units", 4, 64, step=4),
        "na": trial.suggest_int("na", 16, 512, step=16),
        "nb": trial.suggest_int("nb", 16, 128, step=16),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        # "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "climate_processing": "conv2d",
        "output_head": output_head,
        "loss_type": args.loss_type,
        "use_batch_norm": args.use_batch_norm,
        "local_dem_patch": lp, "local_dem_km": lk,
        "regional_dem_patch": rp, "regional_dem_km": rk,
    }

    # Tweedie-specific: tune the p-value
    if args.loss_type == "tweedie":
        hp["tweedie_p"] = trial.suggest_float("tweedie_p", 1.6, 1.9, step=0.05)

    if args.loss_type == "tweedie":
        hp["tweedie_mu_max"] = args.tweedie_mu_max
        hp["tweedie_loss_cap"] = args.tweedie_loss_cap

    if hp["climate_units"] % num_cv != 0:
        hp["climate_units"] = (hp["climate_units"] // num_cv) * num_cv

    if args.loss_type == "mse":
        criterion = None  # default MSE inside train_model
    elif args.loss_type == "tweedie":
        criterion = get_criterion("tweedie", p=hp.get("tweedie_p", 1.5),
                                  mu_max=hp.get("tweedie_mu_max"),
                                  loss_cap=hp.get("tweedie_loss_cap"))
    else:
        criterion = get_criterion(args.loss_type)
    metric_fn  = make_metric_fn(args.loss_type, output_head, target_scale, opt_metric=args.opt_metric)

    # --- Build CV folds ---
    cv_folds = make_cv_folds(splits, args.cv_folds, args.cv_mode, rng_seed=config.RANDOM_SEED + trial.number)

    print(
        f"\nTrial {trial.number}: params≈{hp['na']}+{hp['nb']}  dem={hp['dem_units']}  "
        f"DEM local={lp}x{lp}@{lk}km  regional={rp}x{rp}@{rk}km  "
        f"loss={args.loss_type}  folds={len(cv_folds)}"
        + (f"  tweedie_p={hp['tweedie_p']:.2f}" if args.loss_type == "tweedie" else "")
    )
    print("Tuned HPs:\n" + json.dumps(hp, indent=2, sort_keys=True))

    # Build metric label once (used for fold-level printing)
    if args.opt_metric == "pctl_abs_rel_bias":
        metric_label = f"P{int(args.extreme_percentile)}_abs_rel_bias"
    elif args.opt_metric == "csi":
        metric_label = f"1-CSI@{args.csi_threshold_mm:g}mm"
    else:
        metric_label = args.opt_metric.upper()

    fold_scores = []
    for fold_i, (train_fold_idx, val_fold_idx) in enumerate(cv_folds):
        # Subsample training data for speed
        rng = np.random.RandomState(config.RANDOM_SEED + trial.number * 100 + fold_i)
        if args.subset_frac < 1.0 and len(train_fold_idx) > 200:
            n_sub = max(100, int(len(train_fold_idx) * args.subset_frac))
            train_fold_idx = rng.choice(train_fold_idx, n_sub, replace=False)

        fold_splits = {"train": train_fold_idx, "val": val_fold_idx}
        loaders = make_dataloaders(cropped_tensors, fold_splits, target_scale=target_scale,
                                   batch_size=hp["batch_size"],
                                   num_workers=args.num_workers,
                                   pin_memory=args.pin_memory,
                                   persistent_workers=args.persistent_workers,
                                   prefetch_factor=args.prefetch_factor,
                                   dem_crop_config=None,
                                   device=(device if args.small_batch_processing else None))

        if "val" not in loaders or len(loaders["val"].dataset) == 0:
            print(f"  Fold {fold_i}: empty val, skipping")
            continue

        # Create or reset model
        if fold_i == 0:
            model = create_land_model(hp, metadata).to(device)
            if can_compile:
                model = torch.compile(model)
            initial_state_dict = model.state_dict()
        else:
            # Reset model to initial weights for new fold
            model = create_land_model(hp, metadata).to(device)
            if can_compile:
                model = torch.compile(model)
            model.load_state_dict(initial_state_dict)

        history = train_model(
            model, loaders["train"], loaders["val"], device,
            epochs=args.max_epochs, patience=args.patience,
            min_epochs=40,
            learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
            criterion=criterion,
            metric_fn=metric_fn,
            verbose=5,
            trial=None,  # no Optuna pruning inside CV folds
            no_early_stopping=args.no_early_stopping,
            monitor=("val_metric" if args.loss_type != "mse" else "val_loss"),
            scheduler_type=args.scheduler,
            use_amp=args.amp,
            debug_early_stopping=False,
        )

        # --- Score the fold ---
        if args.loss_type == "mse" and args.opt_metric == "mse":
            # Shortcut: use normalised validation loss directly
            fold_score = float(min(history["val_loss"]))
        else:
            preds_mm, targets_mm = predict_mm(
                model, loaders["val"], device, target_scale, output_head
            )
            m = compute_metrics(targets_mm, preds_mm)
            if args.opt_metric in ("pctl_abs_rel_bias", "csi"):
                m.update(compute_extreme_metrics(
                    targets_mm, preds_mm,
                    percentile=args.extreme_percentile,
                    csi_threshold_mm=args.csi_threshold_mm,
                ))
            if args.opt_metric == "csi":
                csi = float(m.get("csi", float("nan")))
                fold_score = float(1.0 - csi) if np.isfinite(csi) else float("inf")
            else:
                fold_score = float(m.get(args.opt_metric, float("inf")))

        print(f"  Fold {fold_i}: {metric_label}={fold_score:.6f}")
        fold_scores.append(fold_score)

    if len(fold_scores) == 0:
        return float("inf")

    mean_score = float(np.mean(fold_scores))
    print(f"  Trial {trial.number} mean CV score: {mean_score:.4f}")
    return mean_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--study-name", default=None,
                        help="Study name (default: land_daily_<loss_type>_<opt_metric>_<cv_folds><cv_mode>_<n_trials>)")
    parser.add_argument("--loss-type", default="bernoulli_gamma",
                        choices=["mse", "gamma", "tweedie", "bernoulli_gamma"],
                        help="Loss / output-head configuration (default: gamma)")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Number of CV folds per trial (default: 3)")
    parser.add_argument("--cv-mode", default="both",
                        choices=["temporal", "spatial", "both"],
                        help="CV fold construction mode: temporal (held-out years), spatial (held-out stations), "
                             "both (mix of temporal and spatial folds)")
    parser.add_argument("--opt-metric", default="mse",
                        choices=["mae", "mse", "pctl_abs_rel_bias", "csi"],
                        help="Optuna objective metric (default: mae)")
    parser.add_argument("--extreme-percentile", type=float, default=98.0,
                        help="Percentile for extreme bias metric (default: 98)")
    parser.add_argument("--csi-threshold-mm", type=float, default=50.0,
                        help="Threshold in mm for CSI metric (default: 50)")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping; train all epochs per fold")
    parser.add_argument("--num-workers", type=int, default=config.DATALOADER_NUM_WORKERS,
                        help=f"DataLoader num_workers (default: {config.DATALOADER_NUM_WORKERS}).")
    parser.add_argument("--pin-memory", action="store_true", default=config.DATALOADER_PIN_MEMORY,
                        help=f"Enable pinned memory (default: {config.DATALOADER_PIN_MEMORY}).")
    parser.add_argument("--persistent-workers", action="store_true", default=config.DATALOADER_PERSISTENT_WORKERS,
                        help=f"Keep DataLoader workers alive (default: {config.DATALOADER_PERSISTENT_WORKERS}).")
    parser.add_argument("--prefetch-factor", type=int, default=config.DATALOADER_PREFETCH_FACTOR,
                        help=f"DataLoader prefetch_factor (default: {config.DATALOADER_PREFETCH_FACTOR}).")
    parser.add_argument("--tweedie-mu-max", type=float, default=None,
                        help="Optional cap on Tweedie mean (mu) in normalized units (default: None).")
    parser.add_argument("--tweedie-loss-cap", type=float, default=None,
                        help="Optional cap on per-sample Tweedie deviance (default: None).")
    parser.add_argument("--scheduler", default="cosine",
                        choices=["cosine", "none"],
                        help="LR scheduler type")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable mixed-precision (AMP) training. Off by default for Gamma/Tweedie stability.")
    parser.add_argument("--use-batch-norm", action="store_true", default=False,
                        help="Enable BatchNorm layers in the LAND model for all tuning trials.")
    parser.add_argument("--max-epochs", type=int, default=300,
                        help="Maximum epochs per trial (default: 300)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (default: 60)")
    parser.add_argument("--subset-frac", type=float, default=1.0,
                        help="Training subset fraction per fold (default: 1.0)")
    parser.add_argument("--small-batch-processing", action="store_true", default=False,
                        help="Optimise for small batch sizes: pre-stage tensors on GPU and "
                             "use torch.compile to reduce per-step overhead.")
    args = parser.parse_args()
    if args.study_name is None:
        args.study_name = (
            f"land_daily_{args.loss_type}"
            f"_{args.opt_metric}"
            f"_cv{int(args.cv_folds)}{args.cv_mode}"
            f"_n{int(args.n_trials)}"
        )

    device = select_device()
    print(f"Device: {device}")

    # Small-batch optimisations: TF32 precision + check torch.compile availability
    _can_compile = False
    if args.small_batch_processing:
        torch.set_float32_matmul_precision("high")
        if hasattr(torch, "compile"):
            try:
                import triton  # noqa: F401
                _can_compile = True
                print("small-batch-processing: GPU pre-staging + torch.compile enabled")
            except ImportError:
                print("small-batch-processing: GPU pre-staging enabled (torch.compile skipped — Triton not available on Windows)")
        else:
            print("small-batch-processing: GPU pre-staging enabled (torch.compile requires PyTorch 2.0+)")

    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Load tensors on CPU so DataLoader workers can prefetch efficiently.
    # Features/targets are moved to GPU inside the training loop.
    tensors, meta = load_tensors_from_npz(device=torch.device("cpu"))
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
    base_metadata = get_dataset_metadata(tensors)
    print(f"Target scale (train target std): {target_scale:.6f} mm")

    # Save split heatmap
    plot_split_heatmap(stations, years, groups, train_yr, val_yr, test_yr,
                       save_path=config.EDA_DIR / "split_heatmap_land_tuning.png",
                       title="LAND Tuning Split")

    out_dir = config.TUNING_DIR / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # SQLite storage for persistence - allows resuming interrupted runs
    db_path = out_dir / "optuna_study.db"
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    
    # load_if_exists=True allows resuming an interrupted study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
    )
    print(f"Optuna study stored at: {db_path}")
    if len(study.trials) > 0:
        print(f"  Resuming from {len(study.trials)} existing trials")

    hp_space_path = out_dir / "hp_space.json"
    wrote_hp_space = hp_space_path.exists()

    def _hp_space_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        nonlocal wrote_hp_space
        if wrote_hp_space:
            return
        space = {k: _dist_to_dict(v) for k, v in trial.distributions.items()}
        payload = {
            "study_name": args.study_name,
            "loss_type": args.loss_type,
            "opt_metric": args.opt_metric,
            "cv_folds": int(args.cv_folds),
            "cv_mode": args.cv_mode,
            "search_space": space,
            "dem_local_candidates": list(getattr(config, "DEM_LOCAL_CANDIDATES", [])),
            "dem_regional_candidates": list(getattr(config, "DEM_REGIONAL_CANDIDATES", [])),
        }
        hp_space_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        wrote_hp_space = True

    if args.opt_metric == "pctl_abs_rel_bias":
        metric_name = f"P{int(args.extreme_percentile)}_abs_rel_bias"
    elif args.opt_metric == "csi":
        metric_name = f"1-CSI@{args.csi_threshold_mm:g}mm"
    else:
        metric_name = f"val_{args.opt_metric.upper()}"

    study.optimize(
        lambda trial: objective(
            trial, tensors, splits, base_metadata, device,
            target_scale, args, _can_compile
        ),
        n_trials=args.n_trials,
        callbacks=[_hp_space_callback],
    )

    print(f"\nOptimization complete!")

    print(f"\nBest trial: {study.best_trial.number}  {metric_name}={study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    # Enrich best params with resolved DEM config + static defaults
    best_hp = dict(study.best_params)

    dem_crop = config.resolve_dem_crop(best_hp)
    if dem_crop is not None:
        best_hp["local_dem_patch"] = dem_crop["local_patch_size"]
        best_hp["local_dem_km"] = dem_crop["local_km"]
        best_hp["regional_dem_patch"] = dem_crop["regional_patch_size"]
        best_hp["regional_dem_km"] = dem_crop["regional_km"]
    best_hp.setdefault("climate_processing", "conv2d")
    best_hp["output_head"] = config.LOSS_TO_HEAD[args.loss_type]
    best_hp["loss_type"] = args.loss_type
    best_hp["use_batch_norm"] = bool(args.use_batch_norm)
    save_json(best_hp, out_dir / "best_hyperparameters.json")
    save_json({"target_std_mm": target_scale, **stats}, out_dir / "normalization_stats.json")

    # Save all trials
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append({"trial": t.number, "value": t.value, **t.params})
    pd.DataFrame(rows).to_csv(out_dir / "all_trials.csv", index=False)

    # --- HP importance & tuning visualisations ---
    save_optuna_visualizations(study, out_dir)

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
