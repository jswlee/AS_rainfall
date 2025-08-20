#!/usr/bin/env python3
"""
PyTorch training utilities for the LAND rainfall prediction model.
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


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, average_mae)
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    for features, targets in dataloader:
        # Move data to device
        features = {k: v.to(device=device) for k, v in features.items()}
        targets = targets.to(device=device).unsqueeze(dim=1)  # Add dimension for output
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        with torch.no_grad():
            mae = torch.mean(input=torch.abs(input=outputs - targets)).item()
            total_mae += mae
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> Tuple[float, float]:
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, average_mae)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for features, targets in dataloader:
            # Move data to device
            features = {k: v.to(device=device) for k, v in features.items()}
            targets = targets.to(device=device).unsqueeze(dim=1)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Accumulate metrics
            total_loss += loss.item()
            mae = torch.mean(input=torch.abs(input=outputs - targets)).item()
            total_mae += mae
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae


def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], 
                epochs: int = 100, learning_rate: float = 0.001, 
                weight_decay: float = 0.001, patience: int = 10,
                device: Optional[torch.device] = None,
                save_path: Optional[str] = None,
                verbose: bool = True) -> Dict[str, List[float]]:
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device=device)
    
    # Setup optimizer and loss function
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='mean')
    
    # Setup learning rate scheduler and early stopping
    scheduler = CosineAnnealingWarmup(optimizer, warmup_epochs=5, total_epochs=epochs)
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'lr': []
    }
    
    if verbose:
        print(f"Training on {device}")
        print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Update learning rate
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Train for one epoch
        train_loss, train_mae = train_epoch(model, dataloaders['train'], optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        
        # Validate
        val_loss, val_mae = validate_epoch(model, dataloaders['val'], criterion, device)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
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
        print(f"Best validation loss: {early_stopping.best_loss:.6f}")
    
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device=device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = {k: v.to(device=device) for k, v in features.items()}
            targets = targets.to(device=device)
            
            outputs = model(features)
            
            all_predictions.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
    
    predictions = np.concatenate(arrays=all_predictions)
    targets = np.concatenate(arrays=all_targets)
    
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


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.8)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
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
    axes[1, 1].plot(history['train_loss'][start_idx:], label='Train Loss', alpha=0.8)
    axes[1, 1].plot(history['val_loss'][start_idx:], label='Val Loss', alpha=0.8)
    axes[1, 1].set_title('Loss (Last 50% of Training)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device=device)
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for features, batch_targets in dataloader:
            features = {k: v.to(device=device) for k, v in features.items()}
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
