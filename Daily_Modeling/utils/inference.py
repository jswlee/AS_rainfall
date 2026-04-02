"""
Shared inference utilities for LAND model prediction and metric construction.

Used by both tuning (04_tune_land.py) and training (06_train_land.py) scripts
to avoid duplicating output-head unpacking logic.
"""

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
