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
import torch.nn as nn

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import (
    load_tensors_from_npz, normalize_tensors, make_dataloaders,
    print_normalization_report,
)
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split, compute_station_year_ranges,
    compute_year_boundaries, plot_split_heatmap,
)
from Daily_Modeling.models.land import create_land_model
from Daily_Modeling.utils.training import train_model, get_criterion
from Daily_Modeling.utils.metrics import compute_metrics, compute_extreme_metrics, baseline_mean_metrics, per_station_metrics
from Daily_Modeling.utils.visualization import (
    plot_training_history, plot_scatter, plot_model_comparison_table,
)
from Daily_Modeling.utils.io_utils import save_json, save_model, save_predictions
from Daily_Modeling.utils.device import select_device

def _get_metadata(tensors):
    c = tensors["climate"]
    return {
        "climate_shape": tuple(c.shape[1:]),
        "local_dem_shape": tuple(tensors["local_dem"].shape[1:]),
        "regional_dem_shape": tuple(tensors["regional_dem"].shape[1:]),
        "num_month_features": int(tensors["temporal"].shape[1]),
        "num_climate_vars": int(c.shape[1]),
    }


@torch.no_grad()
def predict(model: nn.Module, loader, device, output_head: str = "softplus") -> tuple:
    """Run inference.  Returns (preds, targets) in normalised units.

    For bernoulli_gamma head, predictions are E[Y] = p_rain * alpha * beta.
    """
    model.eval()
    preds, targets = [], []
    for features, tgt in loader:
        features = {k: torch.nan_to_num(v.to(device)) for k, v in features.items()}
        out = model(features)
        if output_head == "bernoulli_gamma":
            p_rain = torch.sigmoid(out[:, 0])
            alpha = torch.nn.functional.softplus(out[:, 1]).clamp(min=1e-6)
            beta  = torch.nn.functional.softplus(out[:, 2]).clamp(min=1e-6)
            pred = p_rain * alpha * beta
        elif output_head == "gamma":
            alpha = torch.nn.functional.softplus(out[:, 0]).clamp(min=1e-6)
            beta  = torch.nn.functional.softplus(out[:, 1]).clamp(min=1e-6)
            pred  = alpha * beta  # E[Y] = alpha * beta
        else:
            pred = out.squeeze(-1)
        preds.append(pred.cpu().numpy().ravel())
        targets.append(tgt.cpu().numpy().ravel())
    return np.concatenate(preds), np.concatenate(targets)


def _make_metric_fn(loss_type: str, output_head: str, target_scale: float):
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


def _validate_test_separation(splits, stations, years, train_yr, val_yr, test_yr):
    """Validate that test sets are temporally and spatially distinct from train/val."""
    train_idx = splits.get("train", np.array([], dtype=int))
    val_temporal = splits.get("val_temporal", np.array([], dtype=int))
    val_spatial = splits.get("val_spatial", np.array([], dtype=int))
    test_temporal = splits.get("test_temporal", np.array([], dtype=int))
    test_spatial = splits.get("test_spatial", np.array([], dtype=int))

    # Check temporal test: should only contain years >= test_yr[0]
    if len(test_temporal) > 0:
        test_temp_years = years[test_temporal]
        if np.any(test_temp_years < test_yr[0]):
            raise ValueError(f"test_temporal contains years before {test_yr[0]}")

    # Check spatial test: should only contain stations not in train
    if len(test_spatial) > 0 and len(train_idx) > 0:
        test_spatial_stations = set(stations[test_spatial])
        train_stations = set(stations[train_idx])
        overlap = test_spatial_stations & train_stations
        if overlap:
            raise ValueError(f"test_spatial shares {len(overlap)} stations with train: {overlap}")

    print("✓ Test sets are temporally and spatially distinct from train/val")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp-dir", default=None,
                        help="Dir with best_hyperparameters.json (default: LAND defaults)")
    parser.add_argument("--epochs", type=int, default=400,
                        help="Training epochs per ensemble member (default: 400)")
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--run-name", default="land_final")
    parser.add_argument("--loss-type", default=None,
                        choices=["mse", "gamma", "tweedie", "bernoulli_gamma"],
                        help="Override loss type (default: read from HP file, else mse)")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "onecycle"],
                        help="LR scheduler: cosine (default) or onecycle")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping; always train all --epochs")
    parser.add_argument("--ensemble-seeds", type=int, default=5,
                        help="Number of ensemble members (different random seeds, default: 5)")
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Number of CV folds. If >1, trains an ensemble per fold and reports mean/STD metrics.")
    parser.add_argument("--cv-mode", default="temporal",
                        choices=["temporal", "spatial", "both", "mixed"],
                        help="CV fold construction mode: temporal (held-out years), spatial (held-out stations), "
                             "both (mix of temporal and spatial folds), mixed (shuffled pool, legacy)")
    parser.add_argument("--extreme-percentile", type=float, default=98.0,
                        help="Percentile for extreme bias metric (default: 98)")
    parser.add_argument("--csi-threshold-mm", type=float, default=50.0,
                        help="Threshold in mm for CSI metric (default: 50)")
    # HP overrides
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from tuned HPs")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate from tuned HPs")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override weight decay from tuned HPs")
    parser.add_argument("--dropout-rate", type=float, default=None,
                        help="Override dropout rate from tuned HPs")
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device}")

    # --- Load data ---
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
    metadata = _get_metadata(tensors)

    var_names = list(meta["variables"]) if len(meta["variables"]) > 0 else None
    print_normalization_report(tensors, stats, splits, variable_names=var_names)

    # Save split heatmap
    plot_split_heatmap(stations, years, groups, train_yr, val_yr, test_yr,
                       save_path=config.EDA_DIR / "split_heatmap_land.png",
                       title="LAND Spatiotemporal Split")

    # --- Load HP ---
    if args.hp_dir:
        hp = json.loads((Path(args.hp_dir) / "best_hyperparameters.json").read_text())
    else:
        hp = dict(config.LAND_DEFAULT_HP)
    # Apply CLI overrides
    if args.loss_type is not None:
        hp["loss_type"] = args.loss_type
    hp.setdefault("loss_type", "mse")
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
        print(f"Overriding batch_size: {args.batch_size}")
    if args.learning_rate is not None:
        hp["learning_rate"] = args.learning_rate
        print(f"Overriding learning_rate: {args.learning_rate}")
    if args.weight_decay is not None:
        hp["weight_decay"] = args.weight_decay
        print(f"Overriding weight_decay: {args.weight_decay}")
    if args.dropout_rate is not None:
        hp["dropout_rate"] = args.dropout_rate
        print(f"Overriding dropout_rate: {args.dropout_rate}")

    # Map loss_type -> output_head
    _LOSS_TO_HEAD = {
        "mse": "softplus",
        "gamma": "gamma",
        "tweedie": "softplus",
        "bernoulli_gamma": "bernoulli_gamma",
    }
    hp["output_head"] = _LOSS_TO_HEAD[hp["loss_type"]]
    hp.setdefault("climate_processing", "conv2d")
    hp.setdefault("tweedie_p", 1.5)
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}")

    # Build DEM crop config from HPs (handles both index and explicit keys)
    dem_crop = config.resolve_dem_crop(hp)
    if dem_crop is not None:
        lp = dem_crop["local_patch_size"]
        rp = dem_crop["regional_patch_size"]
        metadata["local_dem_shape"] = (lp, lp)
        metadata["regional_dem_shape"] = (rp, rp)
        print(f"DEM crop: local={lp}x{lp}@{dem_crop['local_km']}km  "
              f"regional={rp}x{rp}@{dem_crop['regional_km']}km")

    out_dir = config.RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Validate test set separation ---
    _validate_test_separation(splits, stations, years, train_yr, val_yr, test_yr)

    # --- CV folds ---
    cv_folds = _make_cv_folds(splits, args.cv_folds, args.cv_mode, rng_seed=config.RANDOM_SEED)
    print(f"CV mode: {args.cv_mode}  |  {len(cv_folds)} fold(s)")

    # Build criterion
    loss_type = hp["loss_type"]
    if loss_type == "tweedie":
        criterion = get_criterion("tweedie", p=hp["tweedie_p"])
    elif loss_type == "bernoulli_gamma":
        criterion = get_criterion("bernoulli_gamma")
    elif loss_type == "gamma":
        from Daily_Modeling.models.losses import GammaNLL
        criterion = GammaNLL()
    else:
        criterion = None  # default MSE

    metric_fn = _make_metric_fn(loss_type, hp["output_head"], target_scale)
    monitor = "val_metric" if metric_fn is not None else "val_loss"

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
            tensors, fold_splits, target_scale=target_scale,
            batch_size=hp.get("batch_size", 256),
            dem_crop_config=dem_crop,
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
        pctl_key = "pctl" if "pctl" in m else None
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


if __name__ == "__main__":
    main()
