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
        "mse": mse,
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


def compute_wetdry_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_mm: float = 1.0,
) -> Dict[str, float]:
    """Compute wet/dry day classification skill scores and conditional intensity metrics.

    Follows standard verification practice in precipitation downscaling literature
    (e.g. Wilks 2011, WMO guidelines).  A day is classified as *wet* when rainfall
    ≥ *threshold_mm*.

    Classification metrics (2×2 contingency table):
        - pod          Probability of Detection  = TP / (TP + FN)
        - far          False Alarm Ratio          = FP / (TP + FP)
        - freq_bias    Frequency Bias             = (TP + FP) / (TP + FN)
        - csi          Critical Success Index     = TP / (TP + FP + FN)
        - ets          Equitable Threat Score     = (TP - Tc) / (TP + FP + FN - Tc)
        - hss          Heidke Skill Score         = 2(TP·TN - FP·FN) / denom
        - n_obs_wet    Number of observed wet days
        - n_pred_wet   Number of predicted wet days

    Conditional intensity metrics (observed wet days, y_true ≥ threshold):
        - wet_rmse, wet_mae, wet_mbe, wet_r2, wet_spearman_r
        - wet_mean_obs, wet_mean_pred

    Args:
        y_true: observed rainfall (mm).
        y_pred: predicted rainfall (mm).
        threshold_mm: wet-day threshold in mm (default 1.0).

    Returns:
        dict of scalar float values.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    nan_scalar: float = float("nan")

    if len(yt) < 2:
        return {
            "threshold_mm": float(threshold_mm),
            "n_obs_wet": nan_scalar, "n_pred_wet": nan_scalar,
            "pod": nan_scalar, "far": nan_scalar, "freq_bias": nan_scalar,
            "csi": nan_scalar, "ets": nan_scalar, "hss": nan_scalar,
            "wet_rmse": nan_scalar, "wet_mae": nan_scalar, "wet_mbe": nan_scalar,
            "wet_r2": nan_scalar, "wet_spearman_r": nan_scalar,
            "wet_mean_obs": nan_scalar, "wet_mean_pred": nan_scalar,
        }

    obs_wet = yt >= threshold_mm
    pred_wet = yp >= threshold_mm

    tp = float(np.sum(obs_wet & pred_wet))
    fn = float(np.sum(obs_wet & ~pred_wet))
    fp = float(np.sum(~obs_wet & pred_wet))
    tn = float(np.sum(~obs_wet & ~pred_wet))
    n = float(len(yt))

    pod = tp / (tp + fn) if (tp + fn) > 0 else nan_scalar
    far = fp / (tp + fp) if (tp + fp) > 0 else nan_scalar
    freq_bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else nan_scalar
    csi_denom = tp + fp + fn
    csi = tp / csi_denom if csi_denom > 0 else nan_scalar

    tc = (tp + fp) * (tp + fn) / n if n > 0 else 0.0
    ets_denom = tp + fp + fn - tc
    ets = (tp - tc) / ets_denom if ets_denom > 0 else nan_scalar

    hss_num = 2.0 * (tp * tn - fp * fn)
    hss_denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = hss_num / hss_denom if hss_denom > 0 else nan_scalar

    out: Dict[str, float] = {
        "threshold_mm": float(threshold_mm),
        "n_obs_wet": float(np.sum(obs_wet)),
        "n_pred_wet": float(np.sum(pred_wet)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pod": pod,
        "far": far,
        "freq_bias": freq_bias,
        "csi": csi,
        "ets": ets,
        "hss": hss,
    }

    wet_mask = obs_wet
    if wet_mask.sum() >= 2:
        yt_w, yp_w = yt[wet_mask], yp[wet_mask]
        res_w = yp_w - yt_w
        ss_res = float(np.sum(res_w ** 2))
        ss_tot = float(np.sum((yt_w - yt_w.mean()) ** 2))
        r2_wet = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else nan_scalar
        sr_wet, _ = sp_stats.spearmanr(yt_w, yp_w)
        out.update({
            "wet_rmse": float(np.sqrt(np.mean(res_w ** 2))),
            "wet_mae": float(np.mean(np.abs(res_w))),
            "wet_mbe": float(np.mean(res_w)),
            "wet_r2": r2_wet,
            "wet_spearman_r": float(sr_wet),
            "wet_mean_obs": float(yt_w.mean()),
            "wet_mean_pred": float(yp_w.mean()),
        })
    else:
        out.update({k: nan_scalar for k in (
            "wet_rmse", "wet_mae", "wet_mbe", "wet_r2",
            "wet_spearman_r", "wet_mean_obs", "wet_mean_pred",
        )})

    return out
