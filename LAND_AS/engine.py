import copy
import json
from pathlib import Path

import numpy as np
import torch

from LAND_AS.model import gamma_mean, gamma_nll


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit(model, train_loader, val_loader, epochs, patience, learning_rate, weight_decay, target_scale, target_device=None):
    target_device = target_device or device()
    model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state, best_mae, stale = None, float("inf"), 0
    history = {"train_loss": [], "val_mae_mm": []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for features, target in train_loader:
            features = {key: value.to(target_device) for key, value in features.items()}
            target = target.to(target_device)
            optimizer.zero_grad()
            loss = gamma_nll(model(features), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        observed, predicted = predict(model, val_loader, target_scale, target_device)
        val_mae = float(np.mean(np.abs(predicted - observed)))
        history["train_loss"].append(float(np.mean(losses)))
        history["val_mae_mm"].append(val_mae)
        if val_mae < best_mae:
            best_mae, stale = val_mae, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} loss={history['train_loss'][-1]:.4f} val_mae={val_mae:.3f} mm")
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return history, best_mae


@torch.no_grad()
def predict(model, loader, target_scale, target_device=None):
    target_device = target_device or device()
    model.eval()
    observed, predicted = [], []
    for features, target in loader:
        features = {key: value.to(target_device) for key, value in features.items()}
        predicted.append((gamma_mean(model(features)) * target_scale).cpu().numpy())
        observed.append((target * target_scale).numpy())
    return np.concatenate(observed), np.concatenate(predicted)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
