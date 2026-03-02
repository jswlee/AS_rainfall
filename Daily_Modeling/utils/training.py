"""
Shared training utilities: training loop, early stopping, LR scheduling.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class LogMSELoss(nn.Module):
    """MSE in log-space: MSE(log(1 + pred), log(1 + true)).

    Down-weights extreme events relative to raw MSE, giving the model a
    better signal for the bulk of the (zero-inflated) distribution.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(
            torch.log1p(pred.clamp(min=0)),
            torch.log1p(target.clamp(min=0)),
        )


class TweedieLoss(nn.Module):
    """Tweedie deviance loss with power *p* in (1, 2).

    The Tweedie distribution is a natural fit for zero-inflated continuous
    data like daily rainfall.  p=1.5 is a common default.

    D(y, mu) = 2 * [ y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) ]
    """
    def __init__(self, p: float = 1.5):
        super().__init__()
        assert 1.0 < p < 2.0, "Tweedie power p must be in (1, 2)"
        self.p = p

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = pred.clamp(min=1e-6)
        y = target.clamp(min=0)
        p = self.p
        dev = 2 * (
            torch.pow(y, 2 - p) / ((1 - p) * (2 - p))
            - y * torch.pow(mu, 1 - p) / (1 - p)
            + torch.pow(mu, 2 - p) / (2 - p)
        )
        return dev.mean()


def get_criterion(loss_type: str = "mse", **kwargs) -> nn.Module:
    """Factory for loss functions.

    Args:
        loss_type: one of 'mse', 'log_mse', 'tweedie', 'bernoulli_gamma'.
        **kwargs: passed to the loss constructor (e.g. p=1.5 for Tweedie).
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "log_mse":
        return LogMSELoss()
    elif loss_type == "tweedie":
        return TweedieLoss(**kwargs)
    elif loss_type == "bernoulli_gamma":
        from Daily_Modeling.models.losses import BernoulliGammaNLL
        return BernoulliGammaNLL()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")


# ---------------------------------------------------------------------------
# Early stopping & LR scheduling
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights: Optional[dict] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_weights is not None:
            dev = next(model.parameters()).device
            model.load_state_dict({k: v.to(dev) for k, v in self.best_weights.items()})


class CosineWarmup:
    """Cosine-annealing LR with linear warmup."""

    def __init__(self, optimizer: optim.Optimizer, warmup: int = 5,
                 total: int = 200, min_lr: float = 1e-6):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch: int):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = lr


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device,
                scaler=None, flatten_fn=None) -> float:
    """Run one training epoch.  Returns average loss."""
    model.train()
    total, n = 0.0, 0
    use_amp = scaler is not None

    for features, targets in loader:
        if isinstance(features, dict):
            features = {k: torch.nan_to_num(v.to(device)) for k, v in features.items()}
            if flatten_fn is not None:
                x = flatten_fn(features)
            else:
                x = features
        else:
            x = torch.nan_to_num(features.to(device))
        targets = torch.nan_to_num(targets.to(device)).unsqueeze(1)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                out = model(x)
                loss = criterion(out, targets)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, targets)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    flatten_fn=None,
    metric_fn=None,
):
    """Run one evaluation epoch.

    Returns:
        - If metric_fn is None: float average loss
        - Else: (avg_loss, avg_metric)
    """
    model.eval()
    total_loss, total_metric, n = 0.0, 0.0, 0
    for features, targets in loader:
        if isinstance(features, dict):
            features = {k: torch.nan_to_num(v.to(device)) for k, v in features.items()}
            if flatten_fn is not None:
                x = flatten_fn(features)
            else:
                x = features
        else:
            x = torch.nan_to_num(features.to(device))
        targets = torch.nan_to_num(targets.to(device)).unsqueeze(1)

        out = model(x)
        loss = criterion(out, targets)
        if torch.isfinite(loss):
            total_loss += loss.item()
            if metric_fn is not None:
                m = metric_fn(out, targets)
                if torch.isfinite(m):
                    total_metric += float(m.item())
                else:
                    total_metric += float("nan")
            n += 1

    avg_loss = total_loss / max(n, 1)
    if metric_fn is None:
        return avg_loss
    avg_metric = total_metric / max(n, 1)
    return avg_loss, avg_metric


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    min_epochs: int = 0,
    patience: int = 30,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-5,
    criterion: Optional[nn.Module] = None,
    flatten_fn=None,
    metric_fn=None,
    verbose: int = 5,
    trial=None,
    scheduler_type: str = "cosine",
    no_early_stopping: bool = False,
    monitor: str = "val_loss",
) -> Dict[str, list]:
    """Full training loop with early stopping, LR scheduling, and optional AMP.

    Args:
        scheduler_type: 'cosine' (CosineWarmup, default) or 'onecycle' (OneCycleLR).
        no_early_stopping: if True, always train for all *epochs* (best weights are
            still tracked and restored at the end, but training is never cut short).

    If *trial* is an ``optuna.Trial`` instance, reports val_loss each epoch
    and raises ``TrialPruned`` when the pruner decides to stop early.

    Returns history dict with train_loss, val_loss lists.
    If metric_fn is not None, also includes val_metric.
    """
    if monitor not in ("val_loss", "val_metric"):
        raise ValueError(f"Unknown monitor: {monitor!r} (expected 'val_loss' or 'val_metric')")
    if criterion is None:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler_type == "onecycle":
        steps_per_epoch = max(len(train_loader), 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate * 10,
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            pct_start=0.1, anneal_strategy="cos",
        )
        _use_onecycle = True
    else:
        scheduler = CosineWarmup(optimizer, warmup=5, total=epochs)
        _use_onecycle = False
    es = EarlyStopping(patience=patience)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision (AMP) training")

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    if metric_fn is not None:
        history["val_metric"] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        if not _use_onecycle:
            scheduler.step(epoch - 1)
        tl = train_epoch(model, train_loader, optimizer, criterion, device,
                         scaler=scaler, flatten_fn=flatten_fn)
        if _use_onecycle:
            # OneCycleLR steps per batch internally, but we call step() per epoch
            # if steps_per_epoch was set correctly it auto-advances
            pass

        if metric_fn is None:
            vl = eval_epoch(model, val_loader, criterion, device, flatten_fn=flatten_fn)
            vm = None
        else:
            vl, vm = eval_epoch(
                model, val_loader, criterion, device,
                flatten_fn=flatten_fn,
                metric_fn=metric_fn,
            )
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if metric_fn is not None:
            history["val_metric"].append(vm)

        if monitor == "val_metric":
            if metric_fn is None:
                raise ValueError("monitor='val_metric' requires metric_fn to be provided")
            monitor_value = vm
        else:
            monitor_value = vl

        if verbose and epoch % verbose == 0:
            lr = optimizer.param_groups[0]["lr"]
            if metric_fn is None:
                print(f"  Epoch {epoch:>4d}/{epochs} - Train: {tl:.6f}  Val: {vl:.6f}  LR: {lr:.2e}")
            else:
                print(
                    f"  Epoch {epoch:>4d}/{epochs} - Train: {tl:.6f}  Val: {vl:.6f}  "
                    f"ValMetric: {vm:.4f}  LR: {lr:.2e}"
                )

        # Optuna epoch-level pruning (only when early stopping is active)
        if trial is not None and not no_early_stopping and epoch >= min_epochs:
            trial.report(monitor_value, epoch)
            if trial.should_prune():
                es.restore(model)
                import optuna
                raise optuna.TrialPruned(f"Pruned at epoch {epoch}")

        # Always call es() so it tracks best weights; only break if ES is enabled
        stopped = es(monitor_value, model)
        if not no_early_stopping and epoch >= min_epochs and stopped:
            print(f"  Early stopping at epoch {epoch}")
            break

    es.restore(model)
    elapsed = time.time() - t0
    best_val = es.best_loss
    label = "val loss" if monitor == "val_loss" else "val metric"
    print(f"  Training done in {elapsed:.1f}s - best {label}: {best_val:.6f}")
    return history
