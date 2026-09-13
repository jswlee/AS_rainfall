"""Step 6: Train the LAND model with the best hyperparameters.

Supports four loss / output-head configurations:
  - mse:             softplus head, MSE loss
  - gamma:           2-output head (alpha, beta), Gamma NLL on wet days only (paper default)
  - tweedie:         softplus head, Tweedie deviance loss
  - bernoulli_gamma: 3-output head (logit_p, log_alpha, log_beta), BG NLL loss

Early stopping uses val_temporal (same stations, held-out years) to avoid the double
spatial+temporal shift of val_spatial.  val_spatial is used for final evaluation only.

Evaluates on val_spatial, test_spatial, val_temporal, test_temporal.

Usage:
    python -m Daily_Modeling.scripts.06_train_land [--hp-dir ...] [--loss-type gamma]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import (
    load_tensors_from_npz, normalize_tensors, make_dataloaders,
    print_normalization_report, get_dataset_metadata, precompute_dem_crops,
)
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, compute_station_year_ranges,
    compute_year_boundaries, make_cv_folds, validate_test_separation,
)
from Daily_Modeling.models.land import create_land_model
from Daily_Modeling.utils.training import train_model
from Daily_Modeling.models.losses import get_criterion
from Daily_Modeling.utils.inference import predict, predict_mm, make_metric_fn, run_ensemble_inference_from_dir
from Daily_Modeling.utils.metrics import compute_metrics, compute_extreme_metrics, baseline_mean_metrics, per_station_metrics
from Daily_Modeling.utils.visualization import (
    plot_model_comparison_table, plot_scatter, plot_split_heatmap, plot_training_history,
)
from Daily_Modeling.utils.io_utils import save_json, save_model, save_predictions
from Daily_Modeling.utils.device import select_device


def _load_hp_from_dir(hp_dir: Path) -> dict:
    hp_json = hp_dir / "best_hyperparameters.json"
    if hp_json.exists():
        return json.loads(hp_json.read_text())

    db_path = hp_dir / "optuna_study.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Could not find best_hyperparameters.json or optuna_study.db under: {hp_dir}"
        )

    try:
        import optuna
    except Exception as e:
        raise RuntimeError(
            "optuna is required to load hyperparameters from optuna_study.db, but could not be imported. "
            f"Original error: {e}"
        )

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )

    # Prefer study name inferred from directory name (matches 04_tune_land.py default)
    study = None
    try:
        study = optuna.load_study(study_name=hp_dir.name, storage=storage)
    except Exception:
        summaries = optuna.study.get_all_study_summaries(storage)
        if len(summaries) == 0:
            raise RuntimeError(f"No studies found in Optuna DB: {db_path}")
        study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)

    hp = dict(study.best_params)

    # Reconstruct derived hyperparameters that may not be present in best_params.
    if "base_lr" in hp and "batch_size" in hp and "learning_rate" not in hp:
        _LR_REF_BATCH = 32
        lr_scale = float(hp["batch_size"]) / float(_LR_REF_BATCH)
        hp["learning_rate"] = float(hp["base_lr"]) * (lr_scale ** 0.5)

    return hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp-dir", default=None,
                        help="Dir with best_hyperparameters.json (default: LAND defaults)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Training epochs per ensemble member")
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--run-name", default=None,
                        help="Run name (default: derived from hp-dir directory name, or 'land_final' if no hp-dir)")
    parser.add_argument("--loss-type", default=None,
                        choices=["mse", "gamma", "tweedie", "bernoulli_gamma"],
                        help="Override loss type (default: read from HP file, else "
                             "auto — gamma for weekly, bernoulli_gamma for daily)")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "none"],
                        help="LR scheduler: cosine (default, matches tuning) or none (flat LR)")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping; always train all --epochs")
    parser.add_argument("--ensemble-seeds", type=int, default=5,
                        help="Number of ensemble members (different random seeds, default: 5)")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Number of CV folds. If >1, trains an ensemble per fold and reports mean/STD metrics.")
    parser.add_argument("--cv-mode", default="both",
                        choices=["temporal", "spatial", "both"],
                        help="CV fold construction mode: temporal (held-out years), spatial (held-out stations), "
                             "both (mix of temporal and spatial folds)")
    parser.add_argument("--extreme-percentile", type=float, default=98.0,
                        help="Percentile for extreme bias metric (default: 98)")
    parser.add_argument("--csi-threshold-mm", type=float, default=50.0,
                        help="Threshold in mm for CSI metric (default: 50)")
    parser.add_argument(
        "--monitor",
        default=None,
        choices=["val_loss", "mse", "rmse", "mae", "pctl_abs_rel_bias", "csi"],
        help="Metric used to select the best epoch / early stopping. "
             "'val_loss' uses the raw NLL/MSE loss directly (most consistent with --opt-metric val_loss). "
             "Default: existing behavior (loss for MSE runs, MAE for non-MSE runs).",
    )
    parser.add_argument("--num-workers", type=int, default=config.DATALOADER_NUM_WORKERS,
                        help=f"DataLoader num_workers (default: {config.DATALOADER_NUM_WORKERS}).")
    parser.add_argument("--pin-memory", action="store_true", default=config.DATALOADER_PIN_MEMORY,
                        help=f"Enable pinned memory (default: {config.DATALOADER_PIN_MEMORY}).")
    parser.add_argument("--persistent-workers", action="store_true", default=config.DATALOADER_PERSISTENT_WORKERS,
                        help=f"Keep DataLoader workers alive (default: {config.DATALOADER_PERSISTENT_WORKERS}).")
    parser.add_argument("--prefetch-factor", type=int, default=config.DATALOADER_PREFETCH_FACTOR,
                        help=f"DataLoader prefetch_factor (default: {config.DATALOADER_PREFETCH_FACTOR}).")
    parser.add_argument("--grad-clip-norm", type=float, default=0,
                        help="Gradient clipping max norm (default: 0 = disabled). Set >0 to enable.")
    parser.add_argument("--tweedie-mu-max", type=float, default=None,
                        help="Optional cap on Tweedie mean (mu) in normalized units (default: None).")
    parser.add_argument("--tweedie-loss-cap", type=float, default=None,
                        help="Optional cap on per-sample Tweedie deviance (default: None).")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable mixed-precision (AMP) training. Off by default for Gamma/Tweedie stability.")
    parser.add_argument("--batch-norm", type=str, choices=["true", "false"], default=None,
                        help="Override BatchNorm setting (true/false). Omit to inherit from HP file.")
    parser.add_argument("--lightweight", action="store_true", default=None,
                        help="Use simplified architecture (single-layer branches). "
                             "Default: True for small datasets (inherit from config or HP file).")
    parser.add_argument("--no-lightweight", dest="lightweight", action="store_false",
                        help="Disable lightweight mode - use full 2-layer architecture.")
    parser.add_argument("--small-batch-processing", action="store_true", default=False,
                        help="Optimise for small batch sizes: pre-stage tensors on GPU and "
                             "use torch.compile to reduce per-step overhead.")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from latest checkpoint if available.")
    # HP overrides
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from tuned HPs")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate from tuned HPs")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="Multiply the tuned LR by this factor (default: 1.0). "
                             "E.g. --lr-scale 0.1 trains at LR/10 for finer convergence.")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override weight decay from tuned HPs")
    parser.add_argument("--dropout-rate", type=float, default=None,
                        help="Override dropout rate from tuned HPs")
    parser.add_argument("--no-post-inference", action="store_true", default=False,
                        help="Skip automatic ensemble inference on test splits after training.")
    parser.add_argument("--inference-splits", default="both",
                        choices=["temporal", "spatial", "both"],
                        help="Test split(s) to evaluate during post-training inference (default: both).")
    parser.add_argument("--inference-batch-size", type=int, default=512,
                        help="Batch size for post-training inference dataloaders (default: 512).")
    parser.add_argument("--wet-dry-threshold", type=float, default=1.0,
                        help="Wet-day threshold in mm for wet/dry evaluation (default: 1.0).")
    parser.add_argument("--rainfall-weight", action="store_true", default=False,
                        help="Fix I: up-weight heavy-rain samples by log1p(y) in Gamma/BG loss.")
    args = parser.parse_args()

    # Derive run-name from hp-dir if not explicitly provided
    if args.run_name is None:
        if args.hp_dir:
            args.run_name = Path(args.hp_dir).name
        else:
            args.run_name = "land_final"

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

    # --- Load data ---
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
    metadata = get_dataset_metadata(tensors)

    var_names = list(meta["variables"]) if len(meta["variables"]) > 0 else None
    print_normalization_report(tensors, stats, splits, variable_names=var_names)

    # Save split heatmap
    plot_split_heatmap(stations, years, groups, train_yr, val_yr, test_yr,
                       save_path=config.EDA_DIR / "split_heatmap_land.png",
                       title="LAND Spatiotemporal Split")

    # --- Load HP ---
    if args.hp_dir:
        hp = _load_hp_from_dir(Path(args.hp_dir))
    else:
        hp = dict(config.LAND_DEFAULT_HP)
    # Apply CLI overrides
    if args.loss_type is not None:
        hp["loss_type"] = args.loss_type
    hp.setdefault("loss_type", config.DEFAULT_LOSS_TYPE)
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
        print(f"Overriding batch_size: {args.batch_size}")
    if args.learning_rate is not None:
        hp["learning_rate"] = args.learning_rate
        print(f"Overriding learning_rate: {args.learning_rate}")
    if args.lr_scale != 1.0:
        hp["learning_rate"] = hp["learning_rate"] * args.lr_scale
        print(f"Scaling learning_rate by {args.lr_scale}: {hp['learning_rate']:.2e}")
    if args.weight_decay is not None:
        hp["weight_decay"] = args.weight_decay
        print(f"Overriding weight_decay: {args.weight_decay}")
    if args.dropout_rate is not None:
        hp["dropout_rate"] = args.dropout_rate
        print(f"Overriding dropout_rate: {args.dropout_rate}")
    if args.batch_norm is not None:
        hp["use_batch_norm"] = (args.batch_norm == "true")
        print(f"Overriding use_batch_norm: {hp['use_batch_norm']}")
    if args.lightweight is not None:
        hp["lightweight"] = args.lightweight
        print(f"Overriding lightweight: {hp['lightweight']}")

    hp["output_head"] = config.LOSS_TO_HEAD[hp["loss_type"]]
    hp.setdefault("climate_processing", "conv2d")
    hp.setdefault("tweedie_p", 1.5)
    hp.setdefault("use_batch_norm", False)
    hp.setdefault("lightweight", True)  # Default to True for small datasets
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}")

    # Build DEM crop config from HPs (handles both index and explicit keys)
    dem_crop = config.resolve_dem_crop(hp) or {}
    # Fix F: apply climate centre-crop if tuned
    if "reanalysis_patch_size" in hp:
        dem_crop["climate_patch_size"] = int(hp["reanalysis_patch_size"])
    if dem_crop.get("local_patch_size"):
        lp = dem_crop["local_patch_size"]
        rp = dem_crop["regional_patch_size"]
        print(f"DEM crop: local={lp}x{lp}@{dem_crop['local_km']}km  "
              f"regional={rp}x{rp}@{dem_crop['regional_km']}km")
    if "climate_patch_size" in dem_crop:
        print(f"Climate crop: {dem_crop['climate_patch_size']}x{dem_crop['climate_patch_size']}")
    dem_crop_arg = dem_crop if dem_crop else None

    # Batch-crop all DEM samples once (eliminates per-sample cropping in DataLoader)
    cropped_tensors = precompute_dem_crops(tensors, dem_crop_arg)
    metadata = get_dataset_metadata(cropped_tensors)

    out_dir = config.RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Validate test set separation ---
    validate_test_separation(splits, stations, years, train_yr, val_yr, test_yr)

    # --- CV folds ---
    cv_folds = make_cv_folds(splits, args.cv_folds, args.cv_mode, rng_seed=config.RANDOM_SEED)
    print(f"CV mode: {args.cv_mode}  |  {len(cv_folds)} fold(s)")

    # Build criterion
    loss_type = hp["loss_type"]
    if loss_type == "mse":
        criterion = None  # default MSE inside train_model
    elif loss_type == "tweedie":
        criterion = get_criterion("tweedie", p=hp["tweedie_p"],
                                  mu_max=args.tweedie_mu_max,
                                  loss_cap=args.tweedie_loss_cap)
    elif loss_type == "bernoulli_gamma":
        # Fix A: compute dry/wet ratio from training targets for pos_weight
        train_targets_mm = tensors["targets"][splits["train"]].numpy()
        n_wet = float(np.sum(train_targets_mm >= 1.0))
        n_dry = float(np.sum(train_targets_mm < 1.0))
        dry_wet_ratio = n_dry / max(n_wet, 1.0)
        lambda_bce = float(hp.get("lambda_bce", 1.0))
        print(f"  dry_wet_ratio={dry_wet_ratio:.3f}  lambda_bce={lambda_bce:.2f}")
        criterion = get_criterion(
            "bernoulli_gamma",
            dry_wet_ratio=dry_wet_ratio,
            lambda_bce=lambda_bce,
            rainfall_weight=args.rainfall_weight,
        )
    elif loss_type == "gamma":
        criterion = get_criterion("gamma", rainfall_weight=args.rainfall_weight)
    else:
        criterion = get_criterion(loss_type)

    metric_fn = make_metric_fn(loss_type, hp["output_head"], target_scale)

    grad_clip = None if float(args.grad_clip_norm) <= 0 else float(args.grad_clip_norm)

    monitor = "val_metric" if metric_fn is not None else "val_loss"
    monitor_fn = None
    monitor_name = "monitor"

    def _make_monitor(reducer):
        """Build a monitor_fn that runs predict_mm then applies *reducer(yp_mm, yt_mm)*."""
        def _fn(_model, _val_loader, _device):
            yp_mm, yt_mm = predict_mm(
                _model, _val_loader, _device, target_scale, hp["output_head"],
            )
            return reducer(yp_mm, yt_mm)
        return _fn

    def _extreme(yp, yt):
        return compute_extreme_metrics(
            yt, yp,
            percentile=args.extreme_percentile,
            csi_threshold_mm=args.csi_threshold_mm,
        )

    if args.monitor is not None:
        m = args.monitor
        if m == "val_loss":
            monitor = "val_loss"
        elif m == "mse":
            if metric_fn is None:
                # For MSE training runs, validation loss is already MSE (in normalized units).
                monitor = "val_loss"
            else:
                # For non-MSE losses, val_loss is deviance/NLL — compute true MSE in mm².
                monitor_name = "mse_mm2"
                monitor_fn = _make_monitor(
                    lambda yp, yt: float(np.mean((np.float64(yp) - np.float64(yt)) ** 2)))
        elif m == "mae":
            if metric_fn is not None:
                # Non-MSE losses already have a metric_fn that returns MAE in mm.
                monitor = "val_metric"
            else:
                # MSE loss: compute MAE in mm.
                monitor_name = "mae_mm"
                monitor_fn = _make_monitor(
                    lambda yp, yt: float(np.mean(np.abs(np.float64(yp) - np.float64(yt)))))
        elif m == "rmse":
            monitor_name = "rmse_mm"
            monitor_fn = _make_monitor(
                lambda yp, yt: float(np.sqrt(np.mean((yp - yt) ** 2))))
        elif m == "pctl_abs_rel_bias":
            monitor_name = f"p{int(args.extreme_percentile)}_abs_rel_bias"
            monitor_fn = _make_monitor(
                lambda yp, yt: float(_extreme(yp, yt).get("pctl_abs_rel_bias", float("inf"))))
        elif m == "csi":
            monitor_name = f"1-csi@{args.csi_threshold_mm:g}mm"
            def _csi_reducer(yp, yt):
                csi = float(_extreme(yp, yt).get("csi", float("nan")))
                return float(1.0 - csi) if np.isfinite(csi) else float("inf")
            monitor_fn = _make_monitor(_csi_reducer)
        else:
            raise ValueError(f"Unknown --monitor: {m}")

    n_seeds = args.ensemble_seeds
    print(f"\nTraining with cv_folds={args.cv_folds} and ensemble_seeds={n_seeds}  "
          f"({args.epochs} epochs, early_stopping={not args.no_early_stopping})")

    fold_summaries = []
    for fold_i, (train_idx, val_idx) in enumerate(cv_folds):
        fold_dir = out_dir / (f"fold_{fold_i}" if args.cv_folds > 1 else "")
        if args.cv_folds > 1:
            fold_dir.mkdir(parents=True, exist_ok=True)
        else:
            fold_dir = out_dir

        fold_splits = dict(splits)
        fold_splits["train"] = train_idx
        # Use a dedicated key so we don't collide with the canonical split names
        fold_splits["cv_val"] = val_idx

        loaders = make_dataloaders(
            cropped_tensors, fold_splits, target_scale=target_scale,
            batch_size=hp.get("batch_size", 256),
            num_workers=args.num_workers,
            pin_memory=(args.pin_memory and device.type == "cuda"),
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            dem_crop_config=None,
            device=(device if args.small_batch_processing else None),
        )

        if "cv_val" not in loaders or len(loaders["cv_val"].dataset) == 0:
            print(f"Fold {fold_i}: empty validation loader, skipping")
            continue

        ensemble_models = []
        all_histories = []
        for seed_i in range(n_seeds):
            seed = config.RANDOM_SEED + seed_i
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = create_land_model(hp, metadata).to(device)
            if _can_compile:
                model = torch.compile(model)
            if fold_i == 0 and seed_i == 0:
                print(f"Model parameters: {model.count_parameters():,}")
                # Save architecture source once
                arch_src = Path(__file__).resolve().parent.parent / "models" / "land.py"
                if arch_src.exists():
                    shutil.copy2(str(arch_src), str(out_dir / "model_architecture.py"))
                try:
                    from Daily_Modeling.utils.visualization import plot_model_architecture  # noqa: PLC0415
                    dummy_input = {
                        "climate":     torch.randn(2, *metadata["climate_shape"], device=device),
                        "local_dem":   torch.randn(2, *metadata["local_dem_shape"], device=device),
                        "regional_dem": torch.randn(2, *metadata["regional_dem_shape"], device=device),
                        "temporal":    torch.randn(2, metadata["num_month_features"], device=device),
                    }
                    plot_model_architecture(
                        model, model_name="LAND Model",
                        input_data=dummy_input,
                        save_path=out_dir / "architecture_land.png",
                    )
                except Exception as e:
                    print(f"WARNING: architecture diagram failed: {e}")

            print(f"\n--- Fold {fold_i+1}/{len(cv_folds)}  Seed {seed_i+1}/{n_seeds} (seed={seed}) ---")
            # Checkpoint directory for this fold/seed
            ckpt_dir = fold_dir / f"checkpoints_seed{seed_i}"
            resume_path = None
            if args.resume:
                ckpt_files = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
                if ckpt_files:
                    resume_path = str(ckpt_files[-1])
                    print(f"  Resuming from: {resume_path}")
                else:
                    print(f"  --resume: no checkpoint found in {ckpt_dir}, starting fresh")
            history = train_model(
                model, loaders["train"], loaders["cv_val"], device,
                epochs=args.epochs, patience=args.patience,
                learning_rate=hp.get("learning_rate", 5e-5),
                weight_decay=hp.get("weight_decay", 1e-5),
                criterion=criterion,
                metric_fn=metric_fn,
                verbose=1,
                scheduler_type=args.scheduler,
                no_early_stopping=args.no_early_stopping,
                monitor=monitor,
                monitor_fn=monitor_fn,
                monitor_name=monitor_name,
                grad_clip_norm=grad_clip,
                use_amp=args.amp,
                checkpoint_dir=str(ckpt_dir),
                checkpoint_every=20,
                resume_from=resume_path,
            )
            ensemble_models.append(model)
            all_histories.append(history)

            # Save individual model
            save_model(model, fold_dir / f"model_seed{seed_i}.pth", hyperparams=hp)
            plot_training_history(
                history,
                title=f"LAND Fold {fold_i} Seed {seed_i} Training History",
                save_path=fold_dir / f"training_history_seed{seed_i}.png",
            )

        # --- Fold evaluation: average predictions across seeds on the fold val set ---
        seed_preds = []
        yp_ref, yt_ref = None, None
        for model in ensemble_models:
            yp, yt = predict(model, loaders["cv_val"], device, output_head=hp["output_head"])
            seed_preds.append(yp)
            if yt_ref is None:
                yt_ref = yt
        yp_ens = np.mean(seed_preds, axis=0)
        yp_mm = yp_ens * target_scale
        yt_mm = yt_ref * target_scale

        m = compute_metrics(yt_mm, yp_mm)
        m.update(
            compute_extreme_metrics(
                yt_mm,
                yp_mm,
                percentile=args.extreme_percentile,
                csi_threshold_mm=args.csi_threshold_mm,
            )
        )
        fold_summaries.append(m)
        save_json(m, fold_dir / "metrics_cv_val.json")
        p_bias = m.get("pctl_rel_bias", float("nan"))
        csi = m.get("csi", float("nan"))
        print(
            f"\nFold {fold_i} CV-val: RMSE={m['rmse']:.2f} mm  MAE={m['mae']:.2f} mm  R2={m['r2']:.4f}  "
            f"P{m.get('pctl', args.extreme_percentile):.0f}_rel_bias={p_bias:.3f}  "
            f"CSI@{m.get('csi_threshold_mm', args.csi_threshold_mm):g}mm={csi:.3f}"
        )

        plot_scatter(yt_mm, yp_mm, title=f"LAND Fold {fold_i} CV-val",
                     save_path=fold_dir / "scatter_cv_val.png")
        save_predictions(yt_mm, yp_mm,
                         stations[val_idx] if len(val_idx) > 0 else np.array([]),
                         fold_dir / "predictions_cv_val.npz")

    # --- Save shared artefacts ---
    save_json(hp, out_dir / "hyperparameters.json")
    save_json(stats, out_dir / "normalization_stats.json")
    save_json({"station_groups": groups}, out_dir / "station_groups.json")

    if len(fold_summaries) == 0:
        raise RuntimeError("No CV folds completed; cannot report CV results")

    # Summarize CV results
    mae_vals = np.array([m["mae"] for m in fold_summaries], dtype=float)
    rmse_vals = np.array([m["rmse"] for m in fold_summaries], dtype=float)
    summary = {
        "cv_folds": int(args.cv_folds),
        "n_completed_folds": int(len(fold_summaries)),
        "mae_mean": float(np.mean(mae_vals)),
        "mae_std": float(np.std(mae_vals)),
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_std": float(np.std(rmse_vals)),
    }
    save_json(summary, out_dir / "cv_summary.json")
    print(f"\nCV summary: MAE={summary['mae_mean']:.3f}±{summary['mae_std']:.3f}  "
          f"RMSE={summary['rmse_mean']:.3f}±{summary['rmse_std']:.3f}")

    print(f"\nAll results saved to {out_dir}")

    # --- Post-training ensemble inference on held-out test splits ---
    if not args.no_post_inference:
        print("\n" + "=" * 60)
        print("Post-training ensemble inference on test splits")
        print("=" * 60)
        try:
            run_ensemble_inference_from_dir(
                run_dir=out_dir,
                splits=args.inference_splits,
                batch_size=args.inference_batch_size,
                wet_dry_threshold_mm=args.wet_dry_threshold,
            )
        except Exception as e:
            print(f"WARNING: post-training inference failed: {e}")
    else:
        print("\nSkipping post-training inference (--no-post-inference set).")


if __name__ == "__main__":
    main()
