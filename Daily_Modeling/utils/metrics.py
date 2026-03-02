"""
Evaluation metrics for daily rainfall downscaling.

Includes: RMSE, MAE, MBE, Spearman correlation, R2, Wasserstein-1 distance.
"""

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import wasserstein_distance
from typing import Dict, Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: observed rainfall (mm).
        y_pred: predicted rainfall (mm).

    Returns:
        dict with rmse, mae, mbe, r2, spearman_r, spearman_p.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    if len(yt) < 2:
        return {k: float("nan") for k in
                ("rmse", "mae", "mbe", "r2", "spearman_r", "spearman_p")}

    residual = yp - yt
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    mbe = float(np.mean(residual))

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    sr, sp = sp_stats.spearmanr(yt, yp)

    return {
        "rmse": rmse,
        "mae": mae,
        "mbe": mbe,
        "r2": r2,
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def compute_wasserstein(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Wasserstein-1 distance between observed and predicted distributions."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 2:
        return float("nan")
    return float(wasserstein_distance(yt[mask], yp[mask]))


def baseline_mean_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """Metrics for a constant-mean baseline predictor."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yt = yt[np.isfinite(yt)]
    if len(yt) < 2:
        return {"baseline_rmse": float("nan"), "baseline_r2": 0.0}
    mean_pred = np.full_like(yt, yt.mean())
    mse = float(np.mean((yt - mean_pred) ** 2))
    return {
        "baseline_rmse": float(np.sqrt(mse)),
        "baseline_r2": 0.0,  # by definition
    }


def per_station_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stations: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics grouped by station."""
    result = {}
    for st in np.unique(stations):
        mask = stations == st
        if mask.sum() < 2:
            continue
        result[str(st)] = compute_metrics(y_true[mask], y_pred[mask])
    return result


def percentile_relative_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float = 98.0,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {
            "pctl": float(percentile),
            "pctl_true": float("nan"),
            "pctl_pred": float("nan"),
            "pctl_rel_bias": float("nan"),
            "pctl_abs_rel_bias": float("nan"),
        }

    p_true = float(np.percentile(yt, percentile))
    p_pred = float(np.percentile(yp, percentile))
    if p_true == 0:
        rel_bias = float("nan")
    else:
        rel_bias = float((p_pred - p_true) / p_true)
    return {
        "pctl": float(percentile),
        "pctl_true": p_true,
        "pctl_pred": p_pred,
        "pctl_rel_bias": rel_bias,
        "pctl_abs_rel_bias": float(abs(rel_bias)) if np.isfinite(rel_bias) else float("nan"),
    }


def critical_success_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_mm: float = 50.0,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {
            "csi_threshold_mm": float(threshold_mm),
            "csi": float("nan"),
            "hits": 0.0,
            "misses": 0.0,
            "false_alarms": 0.0,
        }

    obs_event = yt >= float(threshold_mm)
    pred_event = yp >= float(threshold_mm)

    hits = float(np.sum(obs_event & pred_event))
    misses = float(np.sum(obs_event & ~pred_event))
    false_alarms = float(np.sum(~obs_event & pred_event))
    denom = hits + misses + false_alarms
    csi = float(hits / denom) if denom > 0 else float("nan")

    return {
        "csi_threshold_mm": float(threshold_mm),
        "csi": csi,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
    }


def compute_extreme_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float = 98.0,
    csi_threshold_mm: float = 50.0,
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    p = percentile_relative_bias(y_true, y_pred, percentile=percentile)
    c = critical_success_index(y_true, y_pred, threshold_mm=csi_threshold_mm)
    out = {**p, **c}
    if prefix:
        return {f"{prefix}{k}": v for k, v in out.items()}
    return out
