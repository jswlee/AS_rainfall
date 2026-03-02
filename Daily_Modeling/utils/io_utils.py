"""
I/O helpers: save/load hyperparameters, model checkpoints, results.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def save_json(data: dict, path: Path):
    """Save a JSON-serialisable dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_model(model: nn.Module, path: Path, hyperparams: Optional[dict] = None):
    """Save full model checkpoint (state_dict + optional HP)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    if hyperparams:
        payload["hyperparams"] = hyperparams
    torch.save(payload, path)


def load_model_state(path: Path, model: nn.Module) -> nn.Module:
    """Load state_dict into an existing model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stations: np.ndarray,
    path: Path,
):
    """Save predictions as compressed NPZ for later analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        y_true=y_true,
        y_pred=y_pred,
        stations=stations,
    )
