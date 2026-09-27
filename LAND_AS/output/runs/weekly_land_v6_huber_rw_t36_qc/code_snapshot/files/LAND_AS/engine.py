import copy
import json
from pathlib import Path

import numpy as np
import torch

from LAND_AS.model import prediction_mean, training_loss


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit(model, train_loader, val_loader, epochs, patience, learning_rate, weight_decay, target_scale,
        target_device=None, use_cosine_scheduler=True, rainfall_weight=False, monitor="mae",
        min_epochs=0, huber_delta=0.5, loss_type=None):
    target_device = target_device or device()
    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if use_cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
    best_state, best_score, best_epoch, stale = None, float("inf"), 0, 0
    history = {"train_loss": [], "val_mae_mm": [], "val_mse_mm2": [], "learning_rate": []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for features, target in train_loader:
            features = {key: value.to(target_device) for key, value in features.items()}
            target = target.to(target_device)
            optimizer.zero_grad()
            loss = training_loss(model, model(features), target, rainfall_weight, huber_delta, loss_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        observed, predicted = predict(model, val_loader, target_scale, target_device)
        val_mae = float(np.mean(np.abs(predicted - observed)))
        val_mse = float(np.mean((predicted - observed) ** 2))
        score = val_mae if monitor == "mae" else val_mse
        history["train_loss"].append(float(np.mean(losses)))
        history["val_mae_mm"].append(val_mae)
        history["val_mse_mm2"].append(val_mse)
        history["learning_rate"].append(current_lr)
        if score < best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1} loss={history['train_loss'][-1]:.4f} val_mae={val_mae:.3f} mm val_mse={val_mse:.1f} mm² lr={current_lr:.2e}")
        if stale >= patience and epoch + 1 >= min_epochs:
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_score"] = best_score
    history["best_val_mae_mm"] = history["val_mae_mm"][best_epoch - 1]
    history["best_val_mse_mm2"] = history["val_mse_mm2"][best_epoch - 1]
    history["monitor"] = monitor
    return history, best_score


@torch.no_grad()
def predict(model, loader, target_scale, target_device=None):
    target_device = target_device or device()
    model.eval()
    observed, predicted = [], []
    for features, target in loader:
        features = {key: value.to(target_device) for key, value in features.items()}
        predicted.append((prediction_mean(model, model(features)) * target_scale).cpu().numpy())
        observed.append((target * target_scale).numpy())
    return np.concatenate(observed), np.concatenate(predicted)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False))
