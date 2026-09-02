"""
Shared inference utilities for LAND model prediction and metric construction.

Used by tuning (04_tune_land.py), training (06_train_land.py), and inference
(07_infer_land_ensemble.py) scripts.  run_ensemble_inference_from_dir() is the
primary entry point for post-training evaluation on held-out test splits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def decode_head_output(outputs: torch.Tensor, output_head: str) -> torch.Tensor:
    """Convert raw model outputs to predicted mean rainfall (in normalised units).

    Args:
        outputs: raw model output tensor.
        output_head: one of 'softplus', 'bernoulli_gamma', 'gamma'.

    Returns:
        1-D tensor of predicted mean values.
    """
    if output_head == "bernoulli_gamma":
        p_rain = torch.sigmoid(outputs[:, 0])
        alpha = torch.nn.functional.softplus(outputs[:, 1]).clamp(min=1e-6)
        beta = torch.nn.functional.softplus(outputs[:, 2]).clamp(min=1e-6)
        return p_rain * alpha * beta
    elif output_head == "gamma":
        alpha = torch.nn.functional.softplus(outputs[:, 0]).clamp(min=1e-6)
        beta = torch.nn.functional.softplus(outputs[:, 1]).clamp(min=1e-6)
        return alpha * beta
    else:
        return outputs.squeeze(-1)


@torch.no_grad()
def collect_bg_logits(
    model: nn.Module, loader, device
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect raw logit_p values and targets from a bernoulli_gamma model.

    Returns (logit_p_np, targets_np) in normalised units.
    """
    model.eval()
    logits, targets = [], []
    for features, tgt in loader:
        features = {k: torch.nan_to_num(v.to(device)) for k, v in features.items()}
        out = model(features).float()
        logits.append(out[:, 0].cpu().numpy().ravel())
        targets.append(tgt.cpu().numpy().ravel())
    return np.concatenate(logits), np.concatenate(targets)


def calibrate_threshold(
    logit_p: np.ndarray,
    targets_mm: np.ndarray,
    threshold_mm: float = 1.0,
    n_steps: int = 200,
) -> Tuple[float, float]:
    """Find the logit_p threshold that maximises ETS on a validation set.

    Sweeps ``n_steps`` candidate probability thresholds in (0.01, 0.99),
    converts each to the corresponding logit, and picks the one with the
    highest Equitable Threat Score (ETS) for wet-day detection.

    Args:
        logit_p:      raw logit values from the bernoulli_gamma head (N,).
        targets_mm:   observed rainfall in mm (N,).
        threshold_mm: wet-day observation threshold in mm (default 1.0).
        n_steps:      number of probability thresholds to evaluate.

    Returns:
        (best_prob_threshold, best_ets) — the optimal probability threshold
        and its ETS score on the calibration set.
    """
    obs_wet = targets_mm >= threshold_mm
    n = float(len(obs_wet))
    n_obs_wet = float(obs_wet.sum())
    if n_obs_wet == 0 or n_obs_wet == n:
        return 0.5, float("nan")

    probs = torch.sigmoid(torch.tensor(logit_p, dtype=torch.float32)).numpy()

    best_ets = -np.inf
    best_prob = 0.5
    for p_thr in np.linspace(0.01, 0.99, n_steps):
        pred_wet = probs >= p_thr
        tp = float(np.sum(obs_wet & pred_wet))
        fp = float(np.sum(~obs_wet & pred_wet))
        fn = float(np.sum(obs_wet & ~pred_wet))
        tc = (tp + fp) * n_obs_wet / n
        denom = tp + fp + fn - tc
        ets = (tp - tc) / denom if denom > 0 else -np.inf
        if ets > best_ets:
            best_ets = ets
            best_prob = float(p_thr)

    return best_prob, float(best_ets)


@torch.no_grad()
def predict(model: nn.Module, loader, device, output_head: str = "softplus") -> tuple:
    """Run inference.  Returns (preds, targets) as numpy arrays in normalised units.

    For bernoulli_gamma head, predictions are E[Y] = p_rain * alpha * beta.
    """
    model.eval()
    preds, targets = [], []
    for features, tgt in loader:
        features = {k: torch.nan_to_num(v.to(device)) for k, v in features.items()}
        out = model(features)
        pred = decode_head_output(out, output_head)
        preds.append(pred.cpu().numpy().ravel())
        targets.append(tgt.cpu().numpy().ravel())
    return np.concatenate(preds), np.concatenate(targets)


@torch.no_grad()
def predict_bg_calibrated(
    model: nn.Module,
    loader,
    device,
    prob_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inference for bernoulli_gamma head using a calibrated probability threshold.

    Returns (preds, targets) in normalised units.  On days where
    sigmoid(logit_p) < prob_threshold, the predicted amount is set to 0
    (model predicts dry).  On wet days the prediction is alpha * beta.

    Args:
        prob_threshold: calibrated wet-day probability threshold (from
            ``calibrate_threshold`` on the validation set).
    """
    model.eval()
    preds, targets = [], []
    for features, tgt in loader:
        features = {k: torch.nan_to_num(v.to(device)) for k, v in features.items()}
        out = model(features).float()
        p_rain = torch.sigmoid(out[:, 0])                               # (B,)
        alpha = torch.nn.functional.softplus(out[:, 1]).clamp(min=1e-6) # (B,)
        beta = torch.nn.functional.softplus(out[:, 2]).clamp(min=1e-6)  # (B,)
        amount = alpha * beta                                            # (B,)
        # Zero out predictions where p_rain < calibrated threshold
        pred = torch.where(p_rain >= prob_threshold, amount, torch.zeros_like(amount))
        preds.append(pred.cpu().numpy().ravel())
        targets.append(tgt.cpu().numpy().ravel())
    return np.concatenate(preds), np.concatenate(targets)


@torch.no_grad()
def predict_mm(model: nn.Module, loader, device, target_scale: float, output_head: str) -> tuple:
    """Return (preds_mm, targets_mm) in physical units (mm)."""
    yp, yt = predict(model, loader, device, output_head=output_head)
    return yp * float(target_scale), yt * float(target_scale)


def make_metric_fn(loss_type: str, output_head: str, target_scale: float,
                   opt_metric: str = "auto"):
    """Return a per-batch metric function for ValMetric logging.

    Args:
        loss_type: one of 'mse', 'tweedie', 'gamma', 'bernoulli_gamma'.
        output_head: one of 'softplus', 'gamma', 'bernoulli_gamma'.
        target_scale: target standard deviation in mm (for unit conversion).
        opt_metric: 'auto' (MAE for non-MSE, None for MSE), 'mae', or 'mse'.

    Returns:
        A callable ``(outputs, targets) -> scalar_tensor`` or ``None``.
    """
    if loss_type == "mse":
        if opt_metric != "mse":
            return None
        # MSE loss + explicit --opt-metric=mse: report MSE in mm²
        def _val_mse_mm2_mse_head(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            pred = outputs.view(-1)
            y = targets.view(-1)
            return (pred - y).pow(2).mean() * (float(target_scale) ** 2)
        return _val_mse_mm2_mse_head

    if loss_type not in ("tweedie", "bernoulli_gamma", "gamma"):
        return None

    if opt_metric == "mse":
        def _val_mse_mm2(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            y = targets.view(-1)
            pred = decode_head_output(outputs, output_head)
            return (pred - y).pow(2).mean() * (float(target_scale) ** 2)
        return _val_mse_mm2

    # Default: MAE in mm
    def _val_mae_mm(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = targets.view(-1)
        pred = decode_head_output(outputs, output_head)
        return (pred - y).abs().mean() * float(target_scale)

    return _val_mae_mm


# ---------------------------------------------------------------------------
# Wet/dry day evaluation helper
# ---------------------------------------------------------------------------

def run_wetdry_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    split_name: str,
    threshold_mm: float = 1.0,
    period_noun: str = "day",
) -> dict:
    """Compute and save wet/dry metrics + visualization for one split.

    Saves:
        ``<out_dir>/wetdry_metrics_<split_name>.json``
        ``<out_dir>/wetdry_eval_<split_name>.png``

    Args:
        y_true: observed rainfall in mm.
        y_pred: predicted (ensemble mean) rainfall in mm.
        out_dir: directory to write outputs into.
        split_name: label used in filenames and titles (e.g. 'test_temporal').
        threshold_mm: wet-period threshold in mm (default 1.0).
        period_noun: "day" or "week" — controls labels in plot titles/legends.

    Returns:
        Metrics dict from ``compute_wetdry_metrics``.
    """
    from Daily_Modeling.utils.metrics import compute_wetdry_metrics
    from Daily_Modeling.utils.visualization import plot_wetdry_evaluation
    from Daily_Modeling.utils.io_utils import save_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = compute_wetdry_metrics(y_true, y_pred, threshold_mm=threshold_mm)
    save_json(m, out_dir / f"wetdry_metrics_{split_name}.json")

    n_wet = int(m.get("n_obs_wet", 0) or 0)
    print(
        f"  [{split_name}] wet/dry (thr={threshold_mm:.1f}mm):"
        f"  n_wet={n_wet}  POD={m.get('pod', float('nan')):.3f}"
        f"  FAR={m.get('far', float('nan')):.3f}  CSI={m.get('csi', float('nan')):.3f}"
        f"  ETS={m.get('ets', float('nan')):.3f}  HSS={m.get('hss', float('nan')):.3f}"
        f"  wet_RMSE={m.get('wet_rmse', float('nan')):.2f} mm"
        f"  wet_R2={m.get('wet_r2', float('nan')):.4f}"
    )

    plot_wetdry_evaluation(
        y_true, y_pred,
        threshold_mm=threshold_mm,
        title=f"Wet/Dry Evaluation — {split_name}",
        save_path=out_dir / f"wetdry_eval_{split_name}.png",
        period_noun=period_noun,
    )
    return m


# ---------------------------------------------------------------------------
# Ensemble inference helpers
# ---------------------------------------------------------------------------

def _apply_saved_normalization(
    tensors: Dict[str, torch.Tensor], stats: dict
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Apply saved normalization parameters in-place; returns (tensors, target_scale)."""
    device = tensors["climate"].device

    cm = torch.tensor(stats["climate_mean"], device=device, dtype=tensors["climate"].dtype)
    cs = torch.tensor(stats["climate_std"], device=device, dtype=tensors["climate"].dtype)
    tensors["climate"] = (tensors["climate"] - cm[None, :, None, None]) / cs[None, :, None, None]

    for key in ("local_dem", "regional_dem"):
        m = torch.tensor(stats[f"{key}_mean"], device=device, dtype=tensors[key].dtype)
        s = torch.tensor(stats[f"{key}_std"], device=device, dtype=tensors[key].dtype)
        t = tensors[key]
        # Multi-band (N,C,H,W): broadcast per-channel stats. Single-band scalar: no reshape needed.
        if m.dim() == 1 and t.dim() == 4:
            m = m[None, :, None, None]
            s = s[None, :, None, None]
        tensors[key] = (t - m) / s

    target_scale = float(stats["target_std_mm"])
    return tensors, target_scale


def _discover_checkpoints(run_dir: Path) -> List[Path]:
    ckpts = sorted(run_dir.glob("fold_*/model_seed*.pth"))
    if len(ckpts) == 0:
        ckpts = sorted(run_dir.glob("model_seed*.pth"))
    return ckpts


def _save_ensemble_npz(
    path: Path,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    stations: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        y_true=y_true,
        y_pred_mean=y_mean,
        y_pred_std=y_std,
        stations=stations,
    )


def _concat_arrays(arrays: List[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(a) for a in arrays if a is not None and len(a) > 0]
    if not valid:
        return np.array([])
    return np.concatenate(valid, axis=0)


def run_ensemble_inference_from_dir(
    run_dir,
    out_dir=None,
    splits: str = "both",
    batch_size: int = 512,
    wet_dry_threshold_mm: float = 1.0,
) -> dict:
    """Load all trained models from *run_dir* and evaluate on held-out test splits.

    Produces metrics JSON files and compressed prediction arrays under *out_dir*
    (default: ``<run_dir>/inference``).

    Args:
        run_dir: Path to a training output directory containing
            ``hyperparameters.json``, ``normalization_stats.json``,
            ``station_groups.json``, and ``fold_*/model_seed*.pth`` checkpoints.
        out_dir: Directory for inference outputs (default: ``<run_dir>/inference``).
        splits: Which test split(s) to evaluate — ``"temporal"``, ``"spatial"``,
            or ``"both"`` (default).
        batch_size: Batch size for inference dataloaders.
        wet_dry_threshold_mm: Wet-day threshold in mm for wet/dry classification
            metrics (POD, FAR, CSI, ETS, HSS) and conditional intensity metrics
            (default: 1.0 mm, standard WMO/hydrology convention).

    Returns:
        dict mapping split name → metrics dict (keys: rmse, mae, r2, …).
    """
    # Local imports keep module-level load fast and avoid circular imports.
    from Daily_Modeling import config
    from Daily_Modeling.data_utils.dataset import (
        load_tensors_from_npz, make_dataloaders, get_dataset_metadata,
    )
    from Daily_Modeling.data_utils.splits import (
        assign_station_groups, compute_station_year_ranges,
        compute_year_boundaries, spatiotemporal_split,
    )
    from Daily_Modeling.models.land import create_land_model
    from Daily_Modeling.utils.io_utils import load_json, load_model_state, save_json
    from Daily_Modeling.utils.metrics import compute_metrics
    from Daily_Modeling.utils.device import select_device

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = Path(out_dir) if out_dir else (run_dir / "inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()

    hp = load_json(run_dir / "hyperparameters.json")
    stats = load_json(run_dir / "normalization_stats.json")
    station_groups_payload = load_json(run_dir / "station_groups.json")
    groups = station_groups_payload.get("station_groups", station_groups_payload)

    tensors, meta = load_tensors_from_npz(device=device)
    stations = meta["stations"]
    years = meta["years"]

    train_yr, val_yr, test_yr = compute_year_boundaries(years)
    yr_ranges = compute_station_year_ranges(stations, years)

    if not groups:
        groups = assign_station_groups(
            sorted(set(str(s) for s in stations)),
            station_year_ranges=yr_ranges,
            val_years=val_yr,
            test_years=test_yr,
        )

    sp = spatiotemporal_split(
        stations, years, groups,
        train_years=train_yr, val_years=val_yr, test_years=test_yr,
    )

    tensors, target_scale = _apply_saved_normalization(tensors, stats)
    metadata = get_dataset_metadata(tensors)

    dem_crop = config.resolve_dem_crop(hp)
    if dem_crop is not None:
        lp = dem_crop["local_patch_size"]
        rp = dem_crop["regional_patch_size"]
        # Preserve channel dim (e.g. (C, H, W) for multi-band DEM)
        local_shape = tuple(metadata.get("local_dem_shape", (lp, lp)))
        regional_shape = tuple(metadata.get("regional_dem_shape", (rp, rp)))
        if len(local_shape) == 3:
            metadata["local_dem_shape"] = (local_shape[0], lp, lp)
            metadata["regional_dem_shape"] = (regional_shape[0], rp, rp)
        else:
            metadata["local_dem_shape"] = (lp, lp)
            metadata["regional_dem_shape"] = (rp, rp)
        print(
            f"DEM crop: local={lp}x{lp}@{dem_crop['local_km']}km  "
            f"regional={rp}x{rp}@{dem_crop['regional_km']}km"
        )

    # Fix F: apply the same climate centre-crop used during training so the
    # reconstructed model's climate branch matches the saved checkpoint shape.
    if "reanalysis_patch_size" in hp:
        if dem_crop is None:
            dem_crop = {}
        cps = int(hp["reanalysis_patch_size"])
        dem_crop["climate_patch_size"] = cps
        climate_shape = tuple(metadata.get("climate_shape", (15, 3, 3)))
        if len(climate_shape) == 3:
            c, h, w = climate_shape
            # Cannot crop larger than source (mirrors crop_climate_patch guard)
            eff = min(cps, h, w)
            metadata["climate_shape"] = (c, eff, eff)
            print(f"Climate crop: {eff}x{eff}")

    split_indices: Dict[str, np.ndarray] = {}
    if splits in ("temporal", "both"):
        split_indices["test_temporal"] = sp.get("test_temporal", np.array([], dtype=int))
    if splits in ("spatial", "both"):
        split_indices["test_spatial"] = sp.get("test_spatial", np.array([], dtype=int))

    if all(len(v) == 0 for v in split_indices.values()):
        raise RuntimeError(f"No test indices found for splits={splits!r}")

    loaders = make_dataloaders(
        tensors, split_indices,
        target_scale=target_scale,
        batch_size=batch_size,
        dem_crop_config=dem_crop,
    )

    ckpts = _discover_checkpoints(run_dir)
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoints found under {run_dir}")
    print(f"Found {len(ckpts)} checkpoint(s)")

    output_head = hp.get("output_head", "softplus")

    # Fix B: calibrate wet-day probability threshold for bernoulli_gamma head
    calibrated_prob_threshold: Optional[float] = None
    if output_head == "bernoulli_gamma":
        val_cal_idx = sp.get("val_temporal", np.array([], dtype=int))
        if len(val_cal_idx) > 0:
            cal_loaders = make_dataloaders(
                tensors, {"val_cal": val_cal_idx},
                target_scale=target_scale,
                batch_size=batch_size,
                dem_crop_config=dem_crop,
            )
            cal_loader = cal_loaders.get("val_cal")
            if cal_loader is not None and len(cal_loader.dataset) > 0:
                print("Calibrating wet-day probability threshold on val_temporal ...")
                all_logits: List[np.ndarray] = []
                cal_targets_mm: Optional[np.ndarray] = None
                for ckpt_path in ckpts:
                    cal_model = create_land_model(hp, metadata).to(device)
                    cal_model = load_model_state(ckpt_path, cal_model)
                    lp_arr, tgt_arr = collect_bg_logits(cal_model, cal_loader, device)
                    all_logits.append(lp_arr)
                    if cal_targets_mm is None:
                        cal_targets_mm = tgt_arr * target_scale
                # Average logits across ensemble before calibrating
                mean_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
                calibrated_prob_threshold, cal_ets = calibrate_threshold(
                    mean_logits, cal_targets_mm, threshold_mm=wet_dry_threshold_mm
                )
                print(
                    f"  Calibrated threshold: p_rain >= {calibrated_prob_threshold:.3f}"
                    f"  (val ETS={cal_ets:.4f})"
                )
                save_json(
                    {
                        "prob_threshold": calibrated_prob_threshold,
                        "val_ets": cal_ets,
                        "wet_dry_threshold_mm": wet_dry_threshold_mm,
                        "n_val_samples": int(len(cal_targets_mm)),
                    },
                    out_dir / "calibrated_threshold.json",
                )

    all_metrics: dict = {}
    split_outputs: dict = {}

    for split_name, loader in loaders.items():
        model_preds_mm: List[np.ndarray] = []
        yt_mm_ref: Optional[np.ndarray] = None

        for ckpt_path in ckpts:
            model = create_land_model(hp, metadata).to(device)
            model = load_model_state(ckpt_path, model)
            if calibrated_prob_threshold is not None:
                yp, yt = predict_bg_calibrated(
                    model, loader, device, prob_threshold=calibrated_prob_threshold
                )
            else:
                yp, yt = predict(model, loader, device, output_head=output_head)
            model_preds_mm.append(yp * target_scale)
            if yt_mm_ref is None:
                yt_mm_ref = yt * target_scale

        preds_stack = np.stack(model_preds_mm, axis=0)
        yp_mean = preds_stack.mean(axis=0)
        yp_std = preds_stack.std(axis=0)

        m = compute_metrics(yt_mm_ref, yp_mean)
        save_json(m, out_dir / f"metrics_{split_name}.json")
        print(
            f"{split_name}: RMSE={m['rmse']:.2f} mm  MAE={m['mae']:.2f} mm  "
            f"R2={m['r2']:.4f}  (models={len(ckpts)})"
        )

        idx = split_indices[split_name]
        _save_ensemble_npz(
            out_dir / f"predictions_{split_name}.npz",
            y_true=yt_mm_ref,
            y_mean=yp_mean,
            y_std=yp_std,
            stations=stations[idx] if len(idx) > 0 else np.array([]),
        )
        all_metrics[split_name] = m
        split_outputs[split_name] = {
            "y_true": yt_mm_ref,
            "y_pred_mean": yp_mean,
            "y_pred_std": yp_std,
            "stations": stations[idx] if len(idx) > 0 else np.array([]),
        }

    if {"test_temporal", "test_spatial"}.issubset(split_outputs.keys()):
        yt_all = _concat_arrays([
            split_outputs["test_temporal"]["y_true"],
            split_outputs["test_spatial"]["y_true"],
        ])
        yp_all = _concat_arrays([
            split_outputs["test_temporal"]["y_pred_mean"],
            split_outputs["test_spatial"]["y_pred_mean"],
        ])
        yp_std_all = _concat_arrays([
            split_outputs["test_temporal"]["y_pred_std"],
            split_outputs["test_spatial"]["y_pred_std"],
        ])
        stations_all = _concat_arrays([
            split_outputs["test_temporal"]["stations"],
            split_outputs["test_spatial"]["stations"],
        ])

        if len(yt_all) > 0:
            m_all = compute_metrics(yt_all, yp_all)
            save_json(m_all, out_dir / "metrics_test_all.json")
            print(
                f"test_all: RMSE={m_all['rmse']:.2f} mm  MAE={m_all['mae']:.2f} mm  "
                f"R2={m_all['r2']:.4f}  (n={len(yt_all)})"
            )
            _save_ensemble_npz(
                out_dir / "predictions_test_all.npz",
                y_true=yt_all,
                y_mean=yp_all,
                y_std=yp_std_all,
                stations=stations_all,
            )
            all_metrics["test_all"] = m_all

    # --- Wet/dry evaluation ---
    from Daily_Modeling import config as _cfg
    period_noun = "week" if _cfg.FREQ == "weekly" else "day"
    print(f"\n--- Wet/dry {period_noun} evaluation ---")
    wetdry_all_metrics: dict = {}
    for split_name, data in split_outputs.items():
        wd = run_wetdry_evaluation(
            data["y_true"],
            data["y_pred_mean"],
            out_dir=out_dir,
            split_name=split_name,
            threshold_mm=wet_dry_threshold_mm,
            period_noun=period_noun,
        )
        wetdry_all_metrics[split_name] = wd

    if "test_all" in all_metrics and {"test_temporal", "test_spatial"}.issubset(split_outputs.keys()):
        wd_all = run_wetdry_evaluation(
            _concat_arrays([split_outputs["test_temporal"]["y_true"],
                            split_outputs["test_spatial"]["y_true"]]),
            _concat_arrays([split_outputs["test_temporal"]["y_pred_mean"],
                            split_outputs["test_spatial"]["y_pred_mean"]]),
            out_dir=out_dir,
            split_name="test_all",
            threshold_mm=wet_dry_threshold_mm,
            period_noun=period_noun,
        )
        wetdry_all_metrics["test_all"] = wd_all

    manifest = {
        "run_dir": str(run_dir),
        "n_models": int(len(ckpts)),
        "splits": splits,
        "output_head": output_head,
        "target_scale_mm": float(target_scale),
        "wet_dry_threshold_mm": float(wet_dry_threshold_mm),
        "checkpoints": [str(p.relative_to(run_dir)) for p in ckpts],
    }
    save_json(manifest, out_dir / "inference_manifest.json")
    print(f"\nSaved inference outputs to: {out_dir}")
    return all_metrics
