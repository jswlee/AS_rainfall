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
