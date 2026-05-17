"""
Step 8: Train site-specific MLPs (one per station) with best hyperparameters.

Workflow:
  1. Pretrain a shared backbone on ALL stations' training data combined.
  2. For each station, initialise from the pretrained backbone and train
     with a reduced learning rate.
  3. Network size is adapted per station (smaller for stations with fewer samples).
  4. Loss is configurable: MSE, log-MSE, or Tweedie.

Usage:
    python -m Daily_Modeling.scripts.08_train_site_mlp [--hp-dir ...] [--loss-type log_mse]
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import torch

from Daily_Modeling import config
from torch.utils.data import DataLoader

from Daily_Modeling.data_utils.dataset import (
    load_tensors_from_npz, normalize_tensors, RainfallDataset,
    print_normalization_report, FlatDataset,
)
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, station_proportional_split,
    compute_station_year_ranges, compute_year_boundaries,
    sorted_sample_indices, expanding_time_folds,
)
from Daily_Modeling.models.site_mlp import (
    SiteMLP, SiteGLU, build_model, compute_input_size, adaptive_hidden_sizes,
)
from Daily_Modeling.utils.io_utils import load_json, save_json, save_model, save_predictions
from Daily_Modeling.utils.metrics import compute_metrics, compute_extreme_metrics, baseline_mean_metrics
from Daily_Modeling.utils.training import train_model
from Daily_Modeling.models.losses import get_criterion
from Daily_Modeling.utils.device import select_device
from Daily_Modeling.utils.visualization import (
    plot_model_architecture,
    plot_scatter,
    plot_split_heatmap,
    plot_station_proportional_cv_folds_heatmap,
    plot_station_proportional_split_daily_raster,
    plot_station_proportional_split_heatmap,
    plot_training_history,
)


_ARCH_VIZ_LOCK = Lock()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



@torch.no_grad()
def _predict_mm(model: torch.nn.Module, loader: DataLoader, device: torch.device, target_scale: float):
    model.eval()
    preds, targets = [], []
    for x, t in loader:
        x = torch.nan_to_num(x.to(device))
        out = model(x)
        preds.append(out.detach().cpu().numpy().ravel())
        targets.append(t.detach().cpu().numpy().ravel())
    yp = np.concatenate(preds) * target_scale
    yt = np.concatenate(targets) * target_scale
    return yt, yp



def _pretrain_backbone(
    tensors, all_train_indices, target_scale, input_size,
    hidden, dropout, lr, wd, bs, loss_type, device,
    epochs=80, patience=20, dem_crop_config=None, tweedie_p=1.5,
    num_workers: int = 0, pin_memory: bool = False,
    persistent_workers: bool = False, prefetch_factor: int = 2,
):
    """Pretrain a shared MLP on all stations' training data combined."""
    n = len(all_train_indices)
    if n == 0:
        return None
    print(f"\n=== Pretraining shared backbone on {n:,d} samples ===")
    ds = FlatDataset(RainfallDataset(tensors, all_train_indices, target_scale,
                                      dem_crop_config=dem_crop_config))
    # 90/10 split for pretrain val
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )
    use_persistent = bool(persistent_workers) and int(num_workers) > 0
    tl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=(prefetch_factor if int(num_workers) > 0 else None),
    )
    vl = DataLoader(
        val_ds,
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=(prefetch_factor if int(num_workers) > 0 else None),
    )

    model = SiteMLP(input_size, hidden, dropout).to(device)
    criterion = get_criterion(loss_type, p=tweedie_p) if loss_type == "tweedie" else get_criterion(loss_type)
    history = train_model(
        model, tl, vl, device, epochs=epochs, patience=patience,
        learning_rate=lr, weight_decay=wd, criterion=criterion, verbose=10,
    )
    print(f"  Pretrain done: best val loss = {min(history['val_loss']):.6f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp-dir", default=None)
    parser.add_argument("--per-station-hp-dir", default=None,
                        help="Root of per-station tuning output (e.g. output/tuning/site_mlp_daily). "
                             "Loads <root>/per_station/<loss_type>/<station>/best_hyperparameters.json "
                             "for each station if it exists, falling back to --hp-dir or defaults.")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--run-name", default="site_mlp_final")
    parser.add_argument("--loss-type", default="mse",
                        choices=["mse", "log_mse", "tweedie"],
                        help="Loss function for training (default: mse, per paper)")
    parser.add_argument("--tweedie-p", type=float, default=1.7,
                        help="Tweedie power p in (1, 2) (default: 1.7)")
    parser.add_argument("--pretrain", action="store_true",
                        help="Enable shared pretraining phase (off by default, per paper)")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Number of backbone layers to freeze during fine-tuning (0=none)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of stations to train concurrently (1=sequential)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stations whose model file already exists in the output directory")

    parser.add_argument("--num-workers", type=int, default=config.DATALOADER_NUM_WORKERS,
                        help=f"DataLoader num_workers (default: {config.DATALOADER_NUM_WORKERS}).")
    parser.add_argument("--pin-memory", action="store_true", default=config.DATALOADER_PIN_MEMORY,
                        help=f"Enable pinned memory (default: {config.DATALOADER_PIN_MEMORY}).")
    parser.add_argument("--persistent-workers", action="store_true", default=config.DATALOADER_PERSISTENT_WORKERS,
                        help=f"Keep DataLoader workers alive (default: {config.DATALOADER_PERSISTENT_WORKERS}).")
    parser.add_argument("--prefetch-factor", type=int, default=config.DATALOADER_PREFETCH_FACTOR,
                        help=f"DataLoader prefetch_factor (default: {config.DATALOADER_PREFETCH_FACTOR}).")

    # Optional: time-based CV and seed ensemble
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Time-based expanding-window CV folds per station (1 disables CV; default: 1)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Train an ensemble of 3 models per station using seeds 42/43/44 and average predictions")
    args = parser.parse_args()

    if (args.cv_folds and args.cv_folds > 1) or args.ensemble:
        # CV/ensemble training does not currently support the pretrain/freeze workflow.
        # (pretraining a shared backbone is not well-defined across multiple folds/seeds).
        args.pretrain = False
        args.freeze_layers = 0

    device = select_device()
    print(f"Device: {device}")

    pin_memory = bool(args.pin_memory) and device.type == "cuda"
    use_persistent = bool(args.persistent_workers) and int(args.num_workers) > 0

    tensors, meta = load_tensors_from_npz(device=torch.device("cpu"))
    stations = meta["stations"]
    years = meta["years"]

    unique = sorted(set(str(s) for s in stations))

    # Data-driven year boundaries
    train_yr, val_yr, test_yr = compute_year_boundaries(years)

    yr_ranges = compute_station_year_ranges(stations, years)
    groups = assign_station_groups(
        unique, station_year_ranges=yr_ranges,
        val_years=val_yr, test_years=test_yr,
    )
    splits = spatiotemporal_split(stations, years, groups,
                                  train_years=train_yr, val_years=val_yr, test_years=test_yr)
    tensors, stats = normalize_tensors(tensors, splits["train"])
    target_scale = stats["target_std_mm"]

    var_names = list(meta["variables"]) if len(meta.get("variables", [])) > 0 else None
    print_normalization_report(tensors, stats, splits, variable_names=var_names)

    # Save split heatmaps
    # 1) Spatiotemporal grouping heatmap (same scheme as LAND; useful for context)
    plot_split_heatmap(stations, years, groups, train_yr, val_yr, test_yr,
                       save_path=config.EDA_DIR / "split_heatmap_site_mlp_spatiotemporal.png",
                       title="Site MLP/GLU: spatiotemporal grouping (context)")
    # 2) Actual site-model split: per-station chronological 70/20/10
    plot_station_proportional_split_heatmap(
        stations, years, meta["months"], meta["days"],
        save_path=config.EDA_DIR / "split_heatmap_site_mlp.png",
        title="Site MLP/GLU: per-station chronological split",
    )

    # 3) Exact per-day raster (one mark per sample-day; no year binning)
    plot_station_proportional_split_daily_raster(
        stations, years, meta["months"], meta["days"],
        save_path=config.EDA_DIR / "split_raster_site_mlp_daily.png",
        title="Site MLP/GLU: per-day split raster (exact)",
    )

    if args.cv_folds and args.cv_folds > 1:
        plot_station_proportional_cv_folds_heatmap(
            stations, years, meta["months"], meta["days"],
            cv_folds=args.cv_folds,
            save_path=config.EDA_DIR / f"split_heatmap_site_mlp_cv{args.cv_folds}.png",
            title=f"Site MLP/GLU: expanding-window CV folds (k={args.cv_folds})",
        )

    metadata = {
        "climate_shape": tuple(tensors["climate"].shape[1:]),
        "local_dem_shape": tuple(tensors["local_dem"].shape[1:]),
        "regional_dem_shape": tuple(tensors["regional_dem"].shape[1:]),
        "num_month_features": int(tensors["temporal"].shape[1]),
    }

    # Validate HP arguments based on usage
    if args.pretrain and not args.hp_dir:
        raise ValueError("--hp-dir is required when using --pretrain (shared backbone needs global HPs)")
    if not args.hp_dir and not args.per_station_hp_dir:
        raise ValueError("Must provide either --hp-dir or --per-station-hp-dir (or both)")

    # Load global HP
    hp: dict = {}
    if args.hp_dir:
        hp = json.loads((Path(args.hp_dir) / "best_hyperparameters.json").read_text())

    # Prefer hidden_sizes list; fall back to legacy h1/h2/h3 keys
    if "hidden_sizes" in hp:
        hidden = list(hp["hidden_sizes"])
    else:
        hidden = [v for v in [hp.get("h1"), hp.get("h2"), hp.get("h3")] if v is not None] or [512, 512, 512]
    dropout = hp.get("dropout", 0.3)
    lr = hp.get("lr", 1e-4)
    wd = hp.get("wd", 1e-5)
    bs = hp.get("bs", 256)
    loss_type = hp.get("loss_type", args.loss_type)
    tweedie_p = hp.get("tweedie_p", args.tweedie_p)
    arch_type = hp.get("arch_type", "mlp")

    per_station_hp_root = Path(args.per_station_hp_dir) if args.per_station_hp_dir else None
    if per_station_hp_root:
        print(f"Per-station HP root: {per_station_hp_root}")

    # Build DEM crop config from HPs (handles both index and explicit keys)
    dem_crop = config.resolve_dem_crop(hp)
    ld_shape = metadata["local_dem_shape"]
    rd_shape = metadata["regional_dem_shape"]
    if dem_crop is not None:
        ld_shape = (dem_crop["local_patch_size"], dem_crop["local_patch_size"])
        rd_shape = (dem_crop["regional_patch_size"], dem_crop["regional_patch_size"])
        print(f"DEM crop: local={ld_shape[0]}x{ld_shape[1]}@{dem_crop['local_km']}km  "
              f"regional={rd_shape[0]}x{rd_shape[1]}@{dem_crop['regional_km']}km")

    input_size = compute_input_size(
        climate_shape=metadata["climate_shape"],
        local_dem_shape=ld_shape,
        regional_dem_shape=rd_shape,
        num_month=metadata["num_month_features"],
    )

    out_dir = config.RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save dynamic architecture diagram from the selected global model HPs
    try:
        _viz_model = build_model(arch_type, input_size, hidden, dropout)
        _viz_input = torch.randn(2, input_size)
        _viz_name = f"Site {arch_type.upper()} (global architecture)"
        plot_model_architecture(
            _viz_model, model_name=_viz_name,
            input_data=_viz_input,
            save_path=out_dir / f"architecture_site_{arch_type}.png",
        )
        print(f"Saved architecture_site_{arch_type}.png")
        del _viz_model
    except Exception as e:
        print(f"WARNING: Could not generate architecture diagram: {e}")

    criterion = get_criterion(loss_type, p=tweedie_p) if loss_type == "tweedie" else get_criterion(loss_type)
    p_str = f"  (p={tweedie_p})" if loss_type == "tweedie" else ""
    print(f"Loss function: {loss_type}{p_str}")

    # --- Phase 1: Pretrain shared backbone on all stations ---
    pretrained_backbone = None
    if args.pretrain:
        # Collect all per-station train indices
        all_train_idx = []
        for sn in unique:
            sp = station_proportional_split(stations, years, meta["months"], meta["days"], sn)
            if len(sp["train"]) >= 50:
                all_train_idx.append(sp["train"])
        if all_train_idx:
            combined_train = np.concatenate(all_train_idx)
            pretrained_backbone = _pretrain_backbone(
                tensors, combined_train, target_scale, input_size,
                hidden, dropout, lr, wd, bs, loss_type, device,
                epochs=args.epochs, patience=args.patience,
                dem_crop_config=dem_crop, tweedie_p=tweedie_p,
            )
            save_model(pretrained_backbone, out_dir / "pretrained_backbone.pth")

    # --- Phase 2: Train per station ---
    station_lr = lr  # use the configured learning rate for per-station training
    print(f"\n=== Training per station (LR={station_lr:.2e}) ===")

    all_yt, all_yp, all_st = [], [], []
    station_results = {}

    def _train_station(station_name: str):
        """Train and evaluate a single station. Returns (name, metrics, yt, yp) or None."""
        station_model_path = out_dir / f"model_{station_name}.pth"
        if args.skip_existing and station_model_path.exists():
            print(f"  [{station_name}] SKIP: model already exists at {station_model_path}")
            return None

        sp = station_proportional_split(stations, years, meta["months"], meta["days"], station_name)
        n_train = len(sp["train"])
        if n_train < 50 or len(sp["val"]) < 10:
            print(f"  [{station_name}] SKIP: insufficient data (n_train={n_train}, n_val={len(sp['val'])})")
            return None

        # Load HPs: use per-station if provided, else global (per-station-hp-dir requires all stations to have HPs)
        stn_hidden, stn_dropout, stn_lr, stn_wd, stn_bs = hidden, dropout, lr, wd, bs
        stn_loss_type, stn_tweedie_p = loss_type, tweedie_p
        stn_arch_type = arch_type
        stn_dem_crop = dem_crop
        stn_input_size = input_size
        if per_station_hp_root is not None:
            # Try each loss type subdir (or the specific one if loss_type is set)
            candidate = per_station_hp_root / "per_station" / loss_type / station_name / "best_hyperparameters.json"
            if not candidate.exists():
                # Try any available loss type
                for lt in ("mse", "log_mse", "tweedie"):
                    candidate = per_station_hp_root / "per_station" / lt / station_name / "best_hyperparameters.json"
                    if candidate.exists():
                        break
            if not candidate.exists():
                raise ValueError(f"Per-station HPs requested but not found for station '{station_name}' "
                                 f"(tried loss types: mse, log_mse, tweedie). "
                                 f"Expected file at: {per_station_hp_root}/per_station/<loss_type>/{station_name}/best_hyperparameters.json")
            stn_hp = json.loads(candidate.read_text())
            # Prefer hidden_sizes list; fall back to legacy h1/h2/h3 keys
            if "hidden_sizes" in stn_hp:
                stn_hidden = list(stn_hp["hidden_sizes"])
            else:
                stn_hidden = [v for v in [stn_hp.get("h1"), stn_hp.get("h2"), stn_hp.get("h3")] if v is not None]
            stn_dropout = stn_hp.get("dropout", dropout)
            stn_lr = stn_hp.get("lr", lr)
            stn_wd = stn_hp.get("wd", wd)
            stn_bs = stn_hp.get("bs", bs)
            stn_loss_type = stn_hp.get("loss_type", loss_type)
            stn_tweedie_p = stn_hp.get("tweedie_p", tweedie_p)
            stn_arch_type = stn_hp.get("arch_type", arch_type)
            stn_dem_crop_cfg = config.resolve_dem_crop(stn_hp)
            if stn_dem_crop_cfg is not None:
                stn_dem_crop = stn_dem_crop_cfg
                ld = (stn_dem_crop["local_patch_size"], stn_dem_crop["local_patch_size"])
                rd = (stn_dem_crop["regional_patch_size"], stn_dem_crop["regional_patch_size"])
                stn_input_size = compute_input_size(
                    climate_shape=metadata["climate_shape"],
                    local_dem_shape=ld, regional_dem_shape=rd,
                    num_month=metadata["num_month_features"],
                )
            print(f"  [{station_name}] loaded per-station HPs from {candidate.parent}")

        # Adaptive sizing
        stn_hidden = adaptive_hidden_sizes(n_train, stn_hidden)

        # Fixed chronological test set (latest chunk) from station_proportional_split
        test_idx = sp.get("test", [])
        if len(test_idx) == 0:
            print(f"  [{station_name}] SKIP: no test indices")
            return None
        test_ds = FlatDataset(RainfallDataset(tensors, test_idx, target_scale, dem_crop_config=stn_dem_crop))
        test_loader = DataLoader(
            test_ds,
            batch_size=min(stn_bs, len(test_idx)),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=use_persistent,
            prefetch_factor=(args.prefetch_factor if int(args.num_workers) > 0 else None),
        )

        # Build time-based CV folds over train+val (forward-chaining)
        trainval_idx = list(sp["train"]) + list(sp["val"])
        trainval_sorted = sorted_sample_indices(trainval_idx, years, meta["months"], meta["days"])
        folds = expanding_time_folds(trainval_sorted, args.cv_folds)
        if args.cv_folds > 1 and len(folds) == 0:
            print(f"  [{station_name}] SKIP: cv-folds requested but could not build folds")
            return None

        seeds = [42, 43, 44] if args.ensemble else [42]

        def _train_one_model(train_idx, val_idx, seed: int):
            _set_seed(seed)
            train_ds = FlatDataset(RainfallDataset(tensors, train_idx, target_scale, dem_crop_config=stn_dem_crop))
            val_ds = FlatDataset(RainfallDataset(tensors, val_idx, target_scale, dem_crop_config=stn_dem_crop))
            tl = DataLoader(
                train_ds,
                batch_size=min(stn_bs, len(train_idx)),
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=use_persistent,
                prefetch_factor=(args.prefetch_factor if int(args.num_workers) > 0 else None),
            )
            vl = DataLoader(
                val_ds,
                batch_size=min(stn_bs, len(val_idx)),
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=use_persistent,
                prefetch_factor=(args.prefetch_factor if int(args.num_workers) > 0 else None),
            )

            model = build_model(stn_arch_type, stn_input_size, stn_hidden, stn_dropout).to(device)
            stn_criterion = get_criterion(stn_loss_type, p=stn_tweedie_p) if stn_loss_type == "tweedie" else get_criterion(stn_loss_type)
            train_model(
                model, tl, vl, device, epochs=args.epochs, patience=args.patience,
                learning_rate=stn_lr, weight_decay=stn_wd, criterion=stn_criterion, verbose=0,
            )

            yt_val, yp_val = _predict_mm(model, vl, device, target_scale)
            yt_test, yp_test = _predict_mm(model, test_loader, device, target_scale)
            return model, (yt_val, yp_val), (yt_test, yp_test)

        # If CV disabled, use the existing single holdout (train/val) split
        if args.cv_folds <= 1:
            val_idx = sp["val"] if len(sp["val"]) > 0 else sp["train"][-max(len(sp["train"]) // 5, 1):]
            train_idx = sp["train"]
            folds = [(train_idx, val_idx)]

        fold_summaries = []
        test_pred_ensemble_accum = []
        test_target_ref = None

        # Train folds sequentially within a station (keeps GPU usage sane)
        for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
            if len(tr_idx) < 50 or len(va_idx) < 10:
                continue

            seed_val_preds = []
            seed_test_preds = []
            seed_test_targets = None
            for seed in seeds:
                model, (yt_val, yp_val), (yt_test, yp_test) = _train_one_model(tr_idx, va_idx, seed)
                seed_val_preds.append(yp_val)
                seed_test_preds.append(yp_test)
                seed_test_targets = yt_test
                if test_target_ref is None:
                    test_target_ref = yt_test

            # Ensemble predictions by averaging across seeds
            yp_val_ens = np.mean(np.stack(seed_val_preds, axis=0), axis=0)
            yp_test_ens = np.mean(np.stack(seed_test_preds, axis=0), axis=0)

            m_val = compute_metrics(yt_val, yp_val_ens)
            m_test = compute_metrics(seed_test_targets, yp_test_ens)
            fold_summaries.append({"fold": fold_i, "val": m_val, "test": m_test, "n_train": len(tr_idx), "n_val": len(va_idx)})
            test_pred_ensemble_accum.append(yp_test_ens)

        if len(fold_summaries) == 0:
            print(f"  [{station_name}] SKIP: no valid CV folds")
            return None

        # Aggregate test predictions across folds by averaging (reduces variance)
        yp_test_final = np.mean(np.stack(test_pred_ensemble_accum, axis=0), axis=0)
        yt_test_final = test_target_ref
        m = compute_metrics(yt_test_final, yp_test_final)

        # Save per-station fold summary
        try:
            save_json({"station": station_name, "folds": fold_summaries, "ensemble": bool(args.ensemble)},
                      out_dir / f"cv_summary_{station_name}.json")
        except Exception:
            pass

        # Per-station architecture diagram (only when using per-station HPs)
        if args.per_station_hp_dir is not None:
            try:
                viz_path = out_dir / f"architecture_{station_name}_{stn_arch_type}.png"
                if not viz_path.exists():
                    with _ARCH_VIZ_LOCK:
                        _viz_model = build_model(stn_arch_type, stn_input_size, stn_hidden, stn_dropout)
                        _viz_input = torch.randn(2, stn_input_size)
                        _viz_name = f"{station_name} | {stn_arch_type.upper()} {stn_hidden}"
                        plot_model_architecture(
                            _viz_model, model_name=_viz_name,
                            input_data=_viz_input,
                            save_path=viz_path,
                        )
                        del _viz_model
            except Exception as e:
                print(f"  [{station_name}] WARNING: could not generate architecture diagram: {e}")

        save_model(model, out_dir / f"model_{station_name}.pth")
        save_predictions(yt_test_final, yp_test_final,
                         np.array([station_name] * len(yt_test_final)),
                         out_dir / f"predictions_{station_name}.npz")
        plot_scatter(yt_test_final, yp_test_final,
                     title=f"{station_name} (test)",
                     save_path=out_dir / f"scatter_{station_name}.png")
        print(f"  {station_name}: RMSE={m['rmse']:.2f}  R2={m['r2']:.4f}  "
              f"n_test={len(yt_test_final)}  arch={stn_arch_type}{stn_hidden}  "
              f"cv_folds={len(fold_summaries)}  ensemble={int(args.ensemble)}")
        return (station_name, m, yt_test_final, yp_test_final)

    # Execute per-station training (parallel or sequential)
    if args.parallel > 1:
        print(f"  Using {args.parallel} parallel workers")
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_train_station, sn): sn for sn in unique}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    sn, m, yt, yp = result
                    station_results[sn] = m
                    all_yt.extend(yt)
                    all_yp.extend(yp)
                    all_st.extend([sn] * len(yt))
    else:
        for station_name in unique:
            result = _train_station(station_name)
            if result is not None:
                sn, m, yt, yp = result
                station_results[sn] = m
                all_yt.extend(yt)
                all_yp.extend(yp)
                all_st.extend([sn] * len(yt))

    # Aggregate
    all_yt = np.array(all_yt)
    all_yp = np.array(all_yp)
    all_st = np.array(all_st)

    agg = compute_metrics(all_yt, all_yp)
    bl = baseline_mean_metrics(all_yt)
    agg.update(bl)
    print(f"\nAggregate: RMSE={agg['rmse']:.2f}  MAE={agg['mae']:.2f}  "
          f"R2={agg['r2']:.4f}  Baseline RMSE={bl['baseline_rmse']:.2f}")

    save_json(agg, out_dir / "metrics_aggregate.json")
    save_json(station_results, out_dir / "metrics_per_station.json")
    save_json(stats, out_dir / "normalization_stats.json")
    save_predictions(all_yt, all_yp, all_st, out_dir / "predictions.npz")
    plot_scatter(all_yt, all_yp, title="Site-specific MLP",
                 save_path=out_dir / "scatter_all.png")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
