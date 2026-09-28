"""Evaluation metrics shared by runs and baselines.

regression_metrics (MSE/RMSE/MAE/bias/R2/Spearman) and extreme_metrics are the
common scoring schema written by s5_evaluate/evaluate.py and
s5_evaluate/baselines/evaluate.py so model and baseline outputs are directly
comparable.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    error = predicted - observed
    rho, _ = spearmanr(observed, predicted)
    return {
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(observed, predicted)),
        "spearman_r": float(rho),
    }


def extreme_metrics(observed, predicted, percentile=98.0, threshold_mm=50.0):
    """High-end skill metrics: percentile relative bias and CSI above a
    rainfall threshold (same convention as Daily_Modeling's eval)."""
    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    obs_p = float(np.percentile(observed, percentile))
    pred_p = float(np.percentile(predicted, percentile))
    hits = np.sum((observed >= threshold_mm) & (predicted >= threshold_mm))
    misses = np.sum((observed >= threshold_mm) & (predicted < threshold_mm))
    false_alarms = np.sum((observed < threshold_mm) & (predicted >= threshold_mm))
    denom = hits + misses + false_alarms
    return {
        "pctl_obs_mm": obs_p,
        "pctl_pred_mm": pred_p,
        "pctl_rel_bias": float((pred_p - obs_p) / obs_p) if obs_p > 0 else float("nan"),
        f"csi_{threshold_mm:g}mm": float(hits / denom) if denom else float("nan"),
    }
