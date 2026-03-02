#!/usr/bin/env python3
"""
PyTorch training utilities for the LAND rainfall prediction model.

This module provides:
- Training and validation loops with early stopping
- Custom loss functions (WeightedMSE)
- Model evaluation and metrics calculation
- Training visualization and result saving
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json


# ================================================================
# Training Utilities and Custom Components
# ================================================================

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should be stopped
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
            return True
        return False


class CosineAnnealingWarmup:
    """Cosine annealing learning rate scheduler with warmup."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int = 5, 
                 total_epochs: int = 100, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch: int):
        """Update learning rate based on current epoch."""
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# ================================================================
# Core Training and Validation Functions
# ================================================================

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device, scaler=None) -> Tuple[float, float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        scaler: GradScaler for mixed precision training (optional)
        
    Returns:
        Tuple of (average_loss, average_mae, average_unweighted_mse)
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_mse_unweighted = 0.0
    num_batches = 0
    use_amp = scaler is not None
    
    for features, targets in dataloader:
        # Move data to device
        features = {k: torch.nan_to_num(v.to(device=device, non_blocking=True)) for k, v in features.items()}
        targets = torch.nan_to_num(targets.to(device=device, non_blocking=True)).unsqueeze(dim=1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(features)
                loss = criterion(outputs, targets)
            # Skip non-finite batches
            if not torch.isfinite(loss) or not torch.isfinite(outputs).all():
                # Do not update optimizer; continue to next batch
                continue
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Unscale before clipping, then clip gradients to avoid explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(features)
            loss = criterion(outputs, targets)
            # Skip non-finite batches
            if not torch.isfinite(loss) or not torch.isfinite(outputs).all():
                continue
            loss.backward()
            # Clip gradients to stabilize updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        with torch.no_grad():
            mae = torch.mean(input=torch.abs(input=outputs - targets)).item()
            total_mae += mae
            # Plain (unweighted) MSE regardless of the selected criterion
            mse_unweighted = torch.mean((outputs - targets) ** 2).item()
            total_mse_unweighted += mse_unweighted
        
        num_batches += 1
    
    if num_batches == 0:
        return float('inf'), float('inf'), float('inf')
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_mse_unweighted = total_mse_unweighted / num_batches
    return avg_loss, avg_mae, avg_mse_unweighted


# ================================================================
# Global Threshold Computation
# ================================================================

def compute_global_thresholds(train_loader, device, percentiles=(0.95, 0.99)) -> dict:
    """
    Compute global percentile thresholds from the entire training set.
    
    Args:
        train_loader: Training DataLoader
        device: Device to compute on
        percentiles: Tuple of percentiles to compute (e.g., (0.95, 0.99))
        
    Returns:
        Dictionary mapping percentiles to threshold tensors
    """
    vals = []
    with torch.no_grad():
        for _, targets in train_loader:
            vals.append(targets.view(-1))
    
    all_targets = torch.cat(vals).to(device=device, dtype=torch.float32)
    thresholds = {}
    
    for q in percentiles:
        thresholds[q] = torch.quantile(all_targets, torch.tensor(q, device=all_targets.device))
    
    return thresholds  # e.g. {0.95: tensor(...), 0.99: tensor(...)}


# ================================================================
# Custom Loss Functions
# ================================================================

class WeightedMSELoss(nn.Module):
    """Weighted MSE that emphasizes errors at the high end of the target.

    Uses a precomputed global threshold (from the full training set) to define
    the heavy-rain region. For targets above the threshold, the weight increases
    as (target - threshold)^power.

    Args:
        alpha: Strength of upweighting (>=0). 0 means plain MSE.
        power: Exponent for how fast weights grow with exceedance.
        global_threshold: Required precomputed threshold tensor (on any device).
    """

    def __init__(self, alpha: float = 2.0, power: float = 1.0, global_threshold: torch.Tensor = None):
        super().__init__()
        self.alpha = float(alpha)
        self.power = float(power)
        self.global_threshold = global_threshold

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Require global threshold to be set (no per-batch fallback)
        if self.global_threshold is None:
            raise ValueError("WeightedMSELoss requires a precomputed global_threshold; per-batch mode is disabled.")
        threshold = self.global_threshold.to(targets.device)

        # Compute how much each target exceeds threshold
        excess = (targets - threshold).clamp(min=0.0)
        # Compute weights for each target
        raw_weights = 1.0 + self.alpha * (excess ** self.power)
        # Normalize weights to keep the expected scale of the loss stable
        normalized_weights = raw_weights / (raw_weights.mean() + 1e-8)

        loss = torch.mean(normalized_weights * (preds - targets) ** 2)
        return loss


class TweedieLoss(nn.Module):
    """Tweedie loss for modeling non-negative continuous data with point mass at zero.
    
    Commonly used for rainfall prediction. The Tweedie distribution is a member of the
    exponential dispersion family that includes special cases:
    - p = 0: Normal distribution
    - p = 1: Poisson distribution
    - 1 < p < 2: Compound Poisson-Gamma (typical for rainfall)
    - p = 2: Gamma distribution
    - p = 3: Inverse Gaussian
    
    Args:
        p: Power parameter (1 < p < 2 recommended for rainfall, default 1.5)
    """
    
    def __init__(self, p: float = 1.5):
        super().__init__()
        if p < 1.0 or p >= 2.0:
            raise ValueError(f"Tweedie power parameter p should be in range [1, 2) for rainfall, got {p}")
        self.p = float(p)
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Tweedie deviance loss.
        
        The deviance is: 2 * [y^(2-p) / ((1-p)(2-p)) - y*mu^(1-p) / (1-p) + mu^(2-p) / (2-p)]
        
        For numerical stability, we add a small epsilon to predictions.
        """
        eps = 1e-8
        preds = torch.clamp(preds, min=eps)  # Ensure positive predictions
        targets = torch.clamp(targets, min=0.0)  # Ensure non-negative targets
        
        p = self.p
        
        # Compute Tweedie deviance components
        # Component 1: y^(2-p) / ((1-p)(2-p))
        a = torch.pow(targets + eps, 2 - p) / ((1 - p) * (2 - p))
        
        # Component 2: -y * mu^(1-p) / (1-p)
        b = -targets * torch.pow(preds, 1 - p) / (1 - p)
        
        # Component 3: mu^(2-p) / (2-p)
        c = torch.pow(preds, 2 - p) / (2 - p)
        
        # Tweedie deviance
        deviance = 2 * (a + b + c)
        
        # Return mean deviance
        return torch.mean(deviance)


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device, use_amp: bool = False) -> Tuple[float, float, float]:
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        use_amp: Whether to use mixed precision for inference
        
    Returns:
        Tuple of (average_loss, average_mae)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse_unweighted = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for features, targets in dataloader:
            # Move data to device
            features = {k: torch.nan_to_num(v.to(device=device)) for k, v in features.items()}
            targets = torch.nan_to_num(targets.to(device=device)).unsqueeze(dim=1)
            
            # Forward pass with optional mixed precision
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(features)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(features)
                loss = criterion(outputs, targets)
            # Skip non-finite batches to avoid poisoning the epoch metrics
            if (not torch.isfinite(loss)) or (not torch.isfinite(outputs).all()) or (not torch.isfinite(targets).all()):
                continue
            
            # Accumulate metrics
            total_loss += loss.item()
            mae = torch.mean(input=torch.abs(input=outputs - targets)).item()
            total_mae += mae
            # Plain (unweighted) MSE regardless of criterion
            mse_unweighted = torch.mean((outputs - targets) ** 2).item()
            total_mse_unweighted += mse_unweighted
            num_batches += 1
    
    # Return inf if no valid batches (consistent with train_epoch)
    if num_batches == 0:
        return float('inf'), float('inf'), float('inf')
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_mse_unweighted = total_mse_unweighted / num_batches
    
    return avg_loss, avg_mae, avg_mse_unweighted


def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], 
                epochs: int = 100, learning_rate: float = 0.001, 
                weight_decay: float = 0.001, patience: int = 10,
                device: Optional[torch.device] = None,
                save_path: Optional[str] = None,
                verbose: bool = True,
                loss_name: str = 'mse',
                loss_params: Optional[Dict[str, Any]] = None,
                use_amp: bool = True) -> Dict[str, List[float]]:
    """
    Train a PyTorch model with early stopping and learning rate scheduling.
    
    Args:
        model: PyTorch model to train
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        epochs: Maximum number of epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        patience: Early stopping patience
        device: Device to run on (auto-detected if None)
        save_path: Path to save the best model (optional)
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing training history
    """
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        )
    
    model = model.to(device=device)
    
    # ================================================================
    # Training Configuration Setup
    # ================================================================
    # Setup optimizer and loss function
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Mixed precision training (AMP) for faster GPU training
    scaler = None
    if use_amp and device.type == 'cuda':
        scaler = torch.amp.GradScaler(device.type)
        if verbose:
            print("Using mixed precision (AMP) training")

    if loss_name == 'weighted_mse':
        params = (loss_params or {}).copy()

        # Compute global threshold from training data for stability
        if verbose:
            print("Computing global thresholds from training data...")

        percentile = params.pop('percentile', 0.95)  # used only for computing the threshold
        global_thresholds = compute_global_thresholds(
            dataloaders['train'],
            device,
            percentiles=(percentile,)
        )

        # Build criterion with required args only
        alpha = float(params.get('alpha', 2.0))
        power = float(params.get('power', 1.0))
        criterion = WeightedMSELoss(alpha=alpha, power=power, global_threshold=global_thresholds[percentile])

        if verbose:
            print(f"Global threshold ({percentile*100:.1f}%): {global_thresholds[percentile]:.6f}")
    elif loss_name == 'tweedie':
        params = loss_params or {}
        p = float(params.get('p', 1.5))
        criterion = TweedieLoss(p=p)
        if verbose:
            print(f"Using Tweedie loss with p={p}")
    else:
        criterion = nn.MSELoss(reduction='mean')
    
    # Setup learning rate scheduler and early stopping
    scheduler = CosineAnnealingWarmup(optimizer, warmup_epochs=5, total_epochs=epochs)
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    
    # ================================================================
    # Training Loop Execution
    # ================================================================
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_mse_unweighted': [],
        'val_mse_unweighted': [],
        'lr': []
    }
    
    if verbose:
        print(f"Training on {device}")
        print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
        print(f"Loss function: {loss_name}")
        if loss_name == 'weighted_mse':
            print(f"Loss params: {loss_params}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # ----------------------------------------------------------------
        # Epoch Training and Validation
        # ----------------------------------------------------------------
        # Update learning rate
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Train for one epoch
        train_loss, train_mae, train_mse_unw = train_epoch(model, dataloaders['train'], optimizer, criterion, device, scaler)
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_mse_unweighted'].append(train_mse_unw)
        
        # Validate
        val_loss, val_mae, val_mse_unw = validate_epoch(model, dataloaders['val'], criterion, device, use_amp=(scaler is not None))
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_mse_unweighted'].append(val_mse_unw)
        
        # Always show progress for hyperparameter tuning (more frequent updates)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} - "
                  f"Val Loss: {val_loss:.6f}, Train Loss: {train_loss:.6f}, "
                  f"LR: {current_lr:.2e}", flush=True)  # Force immediate output
        
        # Check early stopping
        if early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {training_time:.2f} seconds")
        # Clarify which criterion the validation loss refers to, and report the plain MSE too
        try:
            import numpy as _np
            best_idx = int(_np.argmin(history['val_loss']))
            mse_unw_at_best = history.get('val_mse_unweighted', [None])[best_idx]
        except Exception:
            best_idx = None
            mse_unw_at_best = None
        print(f"Best validation loss (criterion={loss_name}): {early_stopping.best_loss:.6f}")
        if mse_unw_at_best is not None:
            print(f"Validation MSE (unweighted) at best epoch{f' {best_idx+1}' if best_idx is not None else ''}: {mse_unw_at_best:.6f}")
    
    # ================================================================
    # Model Saving and Cleanup
    # ================================================================
    # Save model if path provided
    if save_path:
        torch.save(obj={
            'model_state_dict': model.state_dict(),
            'history': history,
            'hyperparams': {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'epochs': epoch + 1,
                'best_val_loss': early_stopping.best_loss
            }
        }, f=save_path)
        if verbose:
            print(f"Model saved to {save_path}")
    
    return history


# ================================================================
# Model Evaluation and Metrics
# ================================================================

def evaluate_model(model: nn.Module, dataloader: DataLoader, 
                  device: Optional[torch.device] = None,
                  rainfall_std: Optional[float] = None) -> Dict[str, float]:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained PyTorch model
        dataloader: Data loader for evaluation
        device: Device to run on
        rainfall_std: Standard deviation for denormalizing predictions (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        )
    
    model = model.to(device=device)
    model.eval()
    
    # ----------------------------------------------------------------
    # Inference Loop
    # ----------------------------------------------------------------
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = {k: torch.nan_to_num(v.to(device=device)) for k, v in features.items()}
            targets = torch.nan_to_num(targets.to(device=device))
            
            outputs = model(features)
            
            all_predictions.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # ----------------------------------------------------------------
    # Metrics Calculation
    # ----------------------------------------------------------------
    # Calculate metrics in normalized space
    mse = mean_squared_error(y_true=targets, y_pred=predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true=targets, y_pred=predictions)
    r2 = r2_score(y_true=targets, y_pred=predictions)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    # Add denormalized metrics if rainfall_std provided
    if rainfall_std is not None and rainfall_std > 0:
        denorm_predictions = predictions * rainfall_std
        denorm_targets = targets * rainfall_std
        
        denorm_mse = mean_squared_error(y_true=denorm_targets, y_pred=denorm_predictions)
        denorm_rmse = np.sqrt(denorm_mse)
        denorm_mae = mean_absolute_error(y_true=denorm_targets, y_pred=denorm_predictions)
        
        metrics.update({
            'denorm_mse_mm': float(denorm_mse),
            'denorm_rmse_mm': float(denorm_rmse),
            'denorm_mae_mm': float(denorm_mae)
        })
    
    return metrics


# ================================================================
# Visualization and Results Saving
# ================================================================

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    
    # Loss plot (criterion loss) with optional overlay of plain MSE
    axes[0, 0].plot(history['train_loss'], label='Train Loss (criterion)', alpha=0.9)
    axes[0, 0].plot(history['val_loss'], label='Val Loss (criterion)', alpha=0.9)
    if 'train_mse_unweighted' in history and 'val_mse_unweighted' in history:
        axes[0, 0].plot(history['train_mse_unweighted'], '--', label='Train MSE (unweighted)', alpha=0.7)
        axes[0, 0].plot(history['val_mse_unweighted'], '--', label='Val MSE (unweighted)', alpha=0.7)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[0, 1].plot(history['train_mae'], label='Train MAE', alpha=0.8)
    axes[0, 1].plot(history['val_mae'], label='Val MAE', alpha=0.8)
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(history['lr'], alpha=0.8)
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss zoom (last 50% of training)
    start_idx = len(history['train_loss']) // 2
    axes[1, 1].plot(history['train_loss'][start_idx:], label='Train Loss (criterion)', alpha=0.9)
    axes[1, 1].plot(history['val_loss'][start_idx:], label='Val Loss (criterion)', alpha=0.9)
    if 'train_mse_unweighted' in history and 'val_mse_unweighted' in history:
        axes[1, 1].plot(history['train_mse_unweighted'][start_idx:], '--', label='Train MSE (unweighted)', alpha=0.7)
        axes[1, 1].plot(history['val_mse_unweighted'][start_idx:], '--', label='Val MSE (unweighted)', alpha=0.7)
    axes[1, 1].set_title('Loss (Last 50% of Training)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(fname=save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_predictions(model: nn.Module, dataloader: DataLoader, save_path: str,
                    device: Optional[torch.device] = None,
                    rainfall_std: Optional[float] = None):
    """
    Save model predictions to a file.
    
    Args:
        model: Trained model
        dataloader: Data loader
        save_path: Path to save predictions
        device: Device to run on
        rainfall_std: Standard deviation for denormalizing
    """
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        )
    
    model = model.to(device=device)
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for features, batch_targets in dataloader:
            features = {k: torch.nan_to_num(v.to(device=device)) for k, v in features.items()}
            outputs = model(features)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(batch_targets.numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Create results dictionary
    results = {
        'predictions_normalized': predictions.tolist(),
        'targets_normalized': targets.tolist()
    }
    
    # Add denormalized values if rainfall_std provided
    if rainfall_std is not None and rainfall_std > 0:
        results['predictions_mm'] = (predictions * rainfall_std).tolist()
        results['targets_mm'] = (targets * rainfall_std).tolist()
        results['rainfall_std'] = rainfall_std
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    print("PyTorch training utilities module loaded successfully.")
    print("Import this module to use training functions like train_model(), train_epoch(), etc.")
