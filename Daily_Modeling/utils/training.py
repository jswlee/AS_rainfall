"""
Shared training utilities: training loop, early stopping, LR scheduling.
"""

import pathlib
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


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
                scaler=None, flatten_fn=None, grad_clip_norm: float | None = 1.0) -> float:
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
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, targets)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
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
    monitor_fn=None,
    monitor_name: str = "monitor",
    grad_clip_norm: float | None = 1.0,
    use_amp: bool = False,
    debug_early_stopping: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 10,
    resume_from: Optional[str] = None,
) -> Dict[str, list]:
    """Full training loop with early stopping, LR scheduling, and optional AMP.

    Args:
        scheduler_type: 'cosine' (CosineWarmup, default) or 'none' (flat LR).
        no_early_stopping: if True, always train for all *epochs* (best weights are
            still tracked and restored at the end, but training is never cut short).

    If *trial* is an ``optuna.Trial`` instance, reports val_loss each epoch
    and raises ``TrialPruned`` when the pruner decides to stop early.

    Returns history dict with train_loss, val_loss lists.
    If metric_fn is not None, also includes val_metric.
    """
    if monitor_fn is None and monitor not in ("val_loss", "val_metric"):
        raise ValueError(f"Unknown monitor: {monitor!r} (expected 'val_loss' or 'val_metric')")
    if criterion is None:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler_type == "none":
        scheduler = None
    else:
        scheduler = CosineWarmup(optimizer, warmup=5, total=epochs)
    es = EarlyStopping(patience=patience)

    use_amp = bool(use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision (AMP) training")

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    if metric_fn is not None:
        history["val_metric"] = []
    if monitor_fn is not None:
        history[monitor_name] = []
    t0 = time.time()

    start_epoch = 1
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        es.best_loss = ckpt.get("best_val", float("inf"))
        es.counter = ckpt.get("es_counter", 0)
        if ckpt.get("es_best_weights") is not None:
            es.best_weights = ckpt["es_best_weights"]
        resumed_history = ckpt.get("history", {})
        for k in history:
            if k in resumed_history:
                history[k] = resumed_history[k]
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {ckpt['epoch']} "
              f"(best monitor: {es.best_loss:.6f}, patience: {es.counter}/{patience})")

    for epoch in range(start_epoch, epochs + 1):
        if scheduler is not None:
            scheduler.step(epoch - 1)
        tl = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            flatten_fn=flatten_fn,
            grad_clip_norm=grad_clip_norm,
        )
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

        if monitor_fn is not None:
            monitor_value = float(monitor_fn(model, val_loader, device))
            history[monitor_name].append(monitor_value)
        else:
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
            if monitor_fn is not None:
                print(f"           {monitor_name}: {monitor_value:.6f}")

        # Optuna epoch-level pruning (only when early stopping is active)
        if trial is not None and not no_early_stopping and epoch >= min_epochs:
            trial.report(monitor_value, epoch)
            if trial.should_prune():
                es.restore(model)
                import optuna
                raise optuna.TrialPruned(f"Pruned at epoch {epoch}")

        prev_best = es.best_loss
        prev_counter = es.counter
        stopped = es(monitor_value, model)
        if debug_early_stopping:
            improved = monitor_value < prev_best - es.min_delta
            print(
                f"           ES debug - monitor={monitor_value:.6f}  prev_best={prev_best:.6f}  "
                f"best={es.best_loss:.6f}  improved={improved}  counter={es.counter}/{es.patience}  "
                f"prev_counter={prev_counter}  min_epochs={min_epochs}  stop={stopped and epoch >= min_epochs and not no_early_stopping}"
            )
        if not no_early_stopping and epoch >= min_epochs and stopped:
            print(f"  Early stopping at epoch {epoch}")
            break

        # Save checkpoint periodically
        if checkpoint_dir is not None and epoch % checkpoint_every == 0:
            ckpt_path = pathlib.Path(checkpoint_dir) / f"checkpoint_epoch{epoch}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_val": es.best_loss,
                "es_counter": es.counter,
                "es_best_weights": es.best_weights,
                "history": history,
            }, ckpt_path)
            # Keep only last 3 checkpoints to save space
            ckpts = sorted(pathlib.Path(checkpoint_dir).glob("checkpoint_epoch*.pt"))
            for old_ckpt in ckpts[:-3]:
                old_ckpt.unlink()

    es.restore(model)

    # Clean up checkpoints after successful training completion
    if checkpoint_dir is not None:
        ckpt_p = pathlib.Path(checkpoint_dir)
        if ckpt_p.exists():
            shutil.rmtree(ckpt_p)
            print(f"  Checkpoints cleaned up: {ckpt_p}")

    elapsed = time.time() - t0
    best_val = es.best_loss
    if monitor_fn is not None:
        label = monitor_name
    else:
        label = "val loss" if monitor == "val_loss" else "val metric"
    print(f"  Training done in {elapsed:.1f}s - best {label}: {best_val:.6f}")
    return history
