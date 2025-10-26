#!/usr/bin/env python3
"""
Simplified PyTorch training script for the best LAND model using optimized hyperparameters.

This script has been updated to use the simplified data utilities from the Hyperparameter_Tuning directory:
- DataManager for streamlined data loading and splitting
- RainfallDataset for memory-efficient tensor indexing
- Simplified hyperparameter loading from JSON
- Consistent cross-validation approach with hp_tuning_simplified.py
"""

import os
import json
import time
import torch
import numpy as np
import random
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Ensure Hyperparameter_Tuning is in path for imports
import sys
if os.path.join(os.path.dirname(__file__), '..', 'Hyperparameter_Tuning') not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Hyperparameter_Tuning'))

# Ensure deterministic cuBLAS behavior on CUDA when deterministic algorithms are enabled
# Must be set before first CUDA matmul; safe to set here near the top of the script
if torch.cuda.is_available():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from Hyperparameter_Tuning.data_utils_simplified import DataManager, create_pytorch_dataloaders, RainfallDataset
from Hyperparameter_Tuning.model import create_model_from_hyperparams
from Hyperparameter_Tuning.model_training import train_model, evaluate_model, plot_training_history, save_predictions


def create_scatter_plot(y_true, y_pred, title, save_path, rainfall_std=None):
    """Create scatter plot of predictions vs actual values."""
    plt.figure(figsize=(8, 8))
    
    # Use denormalized values if rainfall_std provided
    if rainfall_std is not None and rainfall_std > 0:
        y_true_plot = y_true * rainfall_std
        y_pred_plot = y_pred * rainfall_std
        unit_label = "mm"
    else:
        y_true_plot = y_true * 100  # Convert to inches
        y_pred_plot = y_pred * 100
        unit_label = "inches"
    
    plt.scatter(x=y_true_plot, y=y_pred_plot, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel(f'Actual Rainfall ({unit_label})')
    plt.ylabel(f'Predicted Rainfall ({unit_label})')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    plt.text(x=0.05, y=0.95, s=f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(fname=save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_evaluation_metrics(metrics, save_path, rainfall_std=None):
    """Save evaluation metrics to CSV."""
    metrics_data = []
    
    # Add normalized metrics
    metrics_data.append({
        'Metric': 'R²',
        'Value': metrics['r2'],
        'Unit': 'dimensionless'
    })
    metrics_data.append({
        'Metric': 'MSE',
        'Value': metrics['mse'],
        'Unit': 'normalized²'
    })
    metrics_data.append({
        'Metric': 'RMSE',
        'Value': metrics['rmse'],
        'Unit': 'normalized'
    })
    metrics_data.append({
        'Metric': 'MAE',
        'Value': metrics['mae'],
        'Unit': 'normalized'
    })
    
    # Add denormalized metrics if available
    if rainfall_std is not None and rainfall_std > 0:
        metrics_data.extend([
            {
                'Metric': 'RMSE',
                'Value': metrics['denorm_rmse_mm'],
                'Unit': 'mm'
            },
            {
                'Metric': 'MAE',
                'Value': metrics['denorm_mae_mm'],
                'Unit': 'mm'
            }
        ])
    else:
        # Convert to inches
        metrics_data.extend([
            {
                'Metric': 'RMSE',
                'Value': metrics['rmse'] * 100,
                'Unit': 'inches'
            },
            {
                'Metric': 'MAE',
                'Value': metrics['mae'] * 100,
                'Unit': 'inches'
            }
        ])
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(path_or_buf=save_path, index=False)


def save_training_summary(output_dir, hyperparams, history, test_metrics, training_time, rainfall_std=None, loss_name: str = 'mse', 
                         npz_path: str = None, epochs: int = None, patience: int = None, n_folds: int = None, val_size: float = None, seed: int = None,
                         learning_rate_used: float = None, cv_results: dict = None):
    """Save training summary to text file."""
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("PyTorch LAND Model Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Training Configuration
        f.write("Training Configuration:\n")
        if npz_path:
            f.write(f"  Data path: {npz_path}\n")
        if epochs:
            f.write(f"  Max epochs: {epochs}\n")
        if patience:
            f.write(f"  Patience: {patience}\n")
        if n_folds:
            f.write(f"  Cross-validation folds: {n_folds}\n")
        if val_size is not None:
            f.write(f"  Validation size: {val_size:.1%} of train+val\n")
        if seed is not None:
            f.write(f"  Random seed: {seed}\n")
        f.write(f"  Loss function: {loss_name}\n")
        if learning_rate_used is not None:
            f.write(f"  Learning rate (actual): {learning_rate_used:.6e}\n")
        f.write("\n")
        
        f.write("Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Cross-Validation Results (if available)
        if cv_results is not None:
            f.write("Cross-Validation Results:\n")
            f.write(f"  Fold val losses: {[f'{v:.6f}' for v in cv_results['fold_val_losses']]}\n")
            f.write(f"  Avg val loss: {cv_results['avg_val_loss']:.6f} +/- {cv_results['std_val_loss']:.6f}\n")
            if cv_results.get('fold_val_mse_unweighted'):
                f.write(f"  Fold val MSE (unweighted): {[f'{v:.6f}' for v in cv_results['fold_val_mse_unweighted']]}\n")
                f.write(f"  Avg val MSE (unweighted): {cv_results['avg_val_mse_unweighted']:.6f} +/- {cv_results['std_val_mse_unweighted']:.6f}\n")
            f.write(f"  Best fold: {cv_results['best_fold_index'] + 1}\n")
            f.write("\n")
        
        f.write("Training Results (Best Fold):\n")
        f.write(f"  Criterion (loss_name): {loss_name}\n")
        f.write(f"  Training time: {training_time:.2f} seconds\n")
        f.write(f"  Final train loss (criterion): {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Final val loss (criterion): {history['val_loss'][-1]:.6f}\n")
        best_idx = int(np.argmin(history['val_loss']))
        f.write(f"  Best val loss (criterion): {history['val_loss'][best_idx]:.6f} at epoch {best_idx+1}\n")
        # Also report plain MSE at the best criterion epoch
        if 'val_mse_unweighted' in history and len(history['val_mse_unweighted']) > best_idx:
            f.write(f"  Val MSE (unweighted) at best epoch: {history['val_mse_unweighted'][best_idx]:.6f}\n")
        f.write(f"  Total epochs: {len(history['train_loss'])}\n")
        f.write("\n")
        
        f.write("Test Set Evaluation:\n")
        f.write(f"  R²: {test_metrics['r2']:.4f}\n")
        f.write(f"  MSE: {test_metrics['mse']:.6f}\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.6f}\n")
        
        if rainfall_std is not None and rainfall_std > 0:
            f.write(f"\nDenormalized Metrics (mm):\n")
            f.write(f"  RMSE: {test_metrics['denorm_rmse_mm']:.4f} mm\n")
            f.write(f"  MAE: {test_metrics['denorm_mae_mm']:.4f} mm\n")
        else:
            f.write(f"\nMetrics in inches:\n")
            f.write(f"  RMSE: {test_metrics['rmse'] * 100:.4f} inches\n")
            f.write(f"  MAE: {test_metrics['mae'] * 100:.4f} inches\n")
    
    return summary_path


def train_best_model_pytorch(
    npz_path: str = None,
    hyperparams_dir: str = None,
    output_dir: str = None,
    test_indices_path: str = None,
    epochs: int = 300,
    save_model: bool = True,
    loss_name: str = 'mse',
    loss_params: dict | None = None,
    n_folds: int = 5,
    val_size: float = 0.1,
    seed: int = 42,
    patience: int = 60,
):
    """
    Train the best LAND model using PyTorch with cross-validation.
    
    Args:
        npz_path: Path to assembled NPZ data (required)
        hyperparams_dir: Directory containing best hyperparameters (required)
        output_dir: Output directory for results (required)
        test_indices_path: Path to test indices (required)
        epochs: Maximum training epochs
        save_model: Whether to save the trained model
        loss_name: Loss function name ('mse' or 'weighted_mse')
        loss_params: Parameters for the loss function
        n_folds: Number of cross-validation folds (minimum 2)
        val_size: Validation set size as fraction of train+val (default 0.1 to match HP tuning)
        seed: Random seed for reproducibility
        patience: Early stopping patience
    """
    # ================================================================
    # Input Validation and Setup
    # ================================================================
    # Validate required paths (no internal defaults)
    if npz_path is None or hyperparams_dir is None or output_dir is None or test_indices_path is None:
        raise ValueError("npz_path, hyperparams_dir, output_dir, and test_indices_path must be provided. Use CLI arguments in __main__.")
    
    # Enforce minimum number of folds for cross-validation
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2 for cross-validation training")
    
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Reproducibility: set seeds and deterministic flags where possible
    # ------------------------------------------------------------------
    try:
        # Python & NumPy
        random.seed(seed)
        np.random.seed(seed)
        # PyTorch (CPU/CUDA/MPS)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # cuDNN determinism (may impact performance)
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        # Global deterministic algorithms (may not be fully supported on MPS)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception as _seed_e:
        print(f"Warning: Failed to fully set deterministic seeds: {_seed_e}")
    
    print("PyTorch LAND Model Training")
    print("=" * 50)
    
    # ================================================================
    # Data Loading and Preprocessing
    # ================================================================
    print(f"Loading data from {npz_path}...")
    # Use DataManager for simplified data loading (deterministic given random_state)
    data_manager = DataManager(
        npz_path=npz_path,
        test_indices_path=test_indices_path,
        random_state=seed
    )
    # Note: If test_indices_path does not exist, DataManager will generate and save it deterministically
    # using the provided random_state. This matches the tuner behavior.
    
    # Get datasets and metadata
    datasets = data_manager.get_datasets()
    metadata = data_manager.metadata
    
    # Package into data dict for backward compatibility
    data = {
        'datasets': datasets,
        'metadata': metadata
    }
    
    # ================================================================
    # Hyperparameter Loading and Configuration
    # ================================================================
    print(f"Loading hyperparameters from {hyperparams_dir}...")
    hyperparams = None
    trial_number = None
    
    # Try loading from JSON first
    json_path = os.path.join(hyperparams_dir, 'best_hyperparameters.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            hp_data = json.load(f)
            hyperparams = hp_data.get('best_params', hp_data)
            print("Loaded hyperparameters from JSON")
    
    # Require hyperparameters to be found
    if hyperparams is None:
        raise RuntimeError(
            f"No hyperparameters found at {json_path}. "
            "Please run hyperparameter tuning first."
        )
    
    print("Hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # ================================================================
    # Extract Loss Parameters from Hyperparameter Tuning Results
    # ================================================================
    # If loss parameters were tuned, extract them and override command-line params
    tuned_loss_params = None
    if loss_name == 'weighted_mse':
        # Check if loss parameters were tuned as part of hyperparameter search
        if any(key.startswith('loss_') for key in hyperparams):
            tuned_loss_params = {
                'alpha': hyperparams.get('loss_alpha'),
                'power': hyperparams.get('loss_power'), 
                'percentile': hyperparams.get('loss_percentile')
            }
            print(f"Found tuned loss parameters: {tuned_loss_params}")
            
            # Override command-line loss_params with tuned parameters
            if loss_params is None:
                loss_params = tuned_loss_params
                print("Using tuned loss parameters from hyperparameter optimization")
            else:
                print("Warning: Both command-line and tuned loss parameters found.")
                print(f"Command-line params: {loss_params}")
                print(f"Tuned params: {tuned_loss_params}")
                print("Using command-line parameters (override tuned parameters)")
        elif loss_params is None:
            print("Warning: Using weighted_mse but no loss parameters provided and none found in tuning results")
            print("Consider providing --loss-params or re-running hyperparameter tuning with loss parameter optimization")
    
    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    # ================================================================
    # Cross-Validation Training Setup
    # ================================================================
    batch_size = hyperparams.get('batch_size', 32)
    cv_results = None
    model_save_path = os.path.join(output_dir, 'best_model.pth') if save_model else None
    
    # ----------------------------------------------------------------
    # Cross-Validation Training Path
    # ----------------------------------------------------------------
    print(f"\nRunning cross-validation with {n_folds} folds on train+val...")
    
    # Get CV tensors and indices from DataManager
    cv_tensors, cv_indices = data_manager.get_cv_tensors()
    
    # ----------------------------------------------------------------
    # Stratified Cross-Validation Setup with Custom val_size
    # ----------------------------------------------------------------
    # Create stratification bins (same approach as tuner)
    y = cv_tensors['targets'][cv_indices].cpu().numpy().ravel()
    n_bins = 5
    try:
        edges = np.quantile(y[y > 0], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            raise ValueError("Not enough unique quantile edges.")
        y_bins = np.digitize(y, edges[1:-1])
    except Exception as e:
        print(f"Warning: stratification binning failed ({e}); falling back to single bin.")
        y_bins = np.zeros_like(y, dtype=int)

    # Proper cross-validation: use StratifiedKFold for non-overlapping folds
    # Calculate n_folds needed to achieve desired val_size
    # For val_size=0.1, we need n_folds=10 (each fold is 10% validation)
    # For val_size=0.2, we need n_folds=5 (each fold is 20% validation)
    
    # If user specified n_folds, use it directly (standard CV)
    # If user wants specific val_size, calculate required n_folds
    if val_size is not None and val_size != (1.0 / n_folds):
        # Calculate n_folds needed for desired val_size
        calculated_n_folds = int(round(1.0 / val_size))
        print(f"Note: To achieve val_size={val_size:.1%}, using {calculated_n_folds} folds instead of {n_folds}")
        print(f"      (Each fold will have {1/calculated_n_folds:.1%} validation)")
        n_folds = calculated_n_folds
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_splits = []
    
    for train_fold_idx, val_fold_idx in skf.split(np.zeros_like(y_bins), y_bins):
        train_original_indices = cv_indices[train_fold_idx]
        val_original_indices = cv_indices[val_fold_idx]
        cv_splits.append((train_original_indices, val_original_indices))
    
    actual_val_size = 1.0 / n_folds
    print(f"Using {n_folds}-fold cross-validation (val_size={actual_val_size:.1%} per fold)")

    # ----------------------------------------------------------------
    # Cross-Validation Execution Loop
    # ----------------------------------------------------------------
    fold_histories = []
    fold_val_losses = []
    fold_val_metrics = []
    fold_models = []

    start_time = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"Fold {fold_idx+1}/{n_folds}: train={len(train_idx)} val={len(val_idx)}")

        # Create datasets using the shared tensors and indices
        fold_train_ds = RainfallDataset(cv_tensors, train_idx)
        fold_val_ds = RainfallDataset(cv_tensors, val_idx)

        fold_loaders = create_pytorch_dataloaders(
            {'train': fold_train_ds, 'val': fold_val_ds},
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )

        model = create_model_from_hyperparams(hyperparams, metadata)
        
        # Apply learning rate scaling (same as HP tuning)
        batch_ref = 1024
        alpha = 0.5
        base_lr = float(hyperparams.get('learning_rate', 0.001))
        lr_scale = (batch_size / batch_ref) ** alpha
        scaled_lr = base_lr * lr_scale
        
        hist = train_model(
            model=model,
            dataloaders=fold_loaders,
            epochs=epochs,
            learning_rate=scaled_lr,
            weight_decay=hyperparams.get('weight_decay', 0.001),
            patience=patience,
            save_path=None,
            verbose=True,
            loss_name=loss_name,
            loss_params=loss_params,
        )

        best_idx_fold = int(np.argmin(hist['val_loss']))
        best_val = float(hist['val_loss'][best_idx_fold])
        fold_val_losses.append(best_val)
        fold_histories.append(hist)
        # Evaluate on the fold's validation set for R² and other metrics
        rainfall_std = metadata.get('rainfall_mm_std', None)
        val_metrics = evaluate_model(model, fold_loaders['val'], rainfall_std=rainfall_std)
        fold_val_metrics.append(val_metrics)
        fold_models.append(model)

        # Also capture unweighted MSE at the best criterion epoch for clarity
        val_mse_unw_best = float(hist['val_mse_unweighted'][best_idx_fold]) if 'val_mse_unweighted' in hist else float('nan')
        print(f"Best val loss (criterion={loss_name}): {best_val:.6f} | Val MSE (unweighted) at best epoch: {val_mse_unw_best:.6f}")
        # Stash for CV summary
        if 'fold_val_mse_unw' not in locals():
            fold_val_mse_unw = []
        fold_val_mse_unw.append(val_mse_unw_best)

    # ----------------------------------------------------------------
    # Cross-Validation Results Analysis
    # ----------------------------------------------------------------
    training_time = time.time() - start_time
    best_fold = int(np.argmin(fold_val_losses))
    print(f"Best fold: {best_fold+1} with val_loss={fold_val_losses[best_fold]:.6f}")

    # Select best fold model and history for downstream saving/plots
    model = fold_models[best_fold]
    history = fold_histories[best_fold]
    
    # Calculate best epoch and best val loss from history
    best_epoch = int(np.argmin(history['val_loss']))
    best_val_loss = float(np.min(history['val_loss']))
    
    # Optionally save the selected best-fold model with full checkpoint
    if model_save_path is not None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'hyperparameters': hyperparams,
            'metadata': data_manager.metadata,
            'best_fold': best_fold,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'val_loss': fold_val_losses[best_fold],
            'loss_name': loss_name,
            'loss_params': loss_params,
        }
        torch.save(checkpoint, model_save_path)
        print(f"Saved full model checkpoint to {model_save_path}")
        
        # Save a copy of the model architecture code for reproducibility
        import shutil
        model_py_source = os.path.join('Hyperparameter_Tuning', 'model.py')
        model_py_backup = os.path.join(output_dir, 'model_architecture.py')
        if os.path.exists(model_py_source):
            shutil.copy2(model_py_source, model_py_backup)
            print(f"Saved model architecture to {model_py_backup}")

    # Summarize CV
    cv_results = {
        'fold_val_losses': fold_val_losses,
        'avg_val_loss': float(np.mean(fold_val_losses)),
        'std_val_loss': float(np.std(fold_val_losses)),
        'fold_val_mse_unweighted': fold_val_mse_unw if 'fold_val_mse_unw' in locals() else [],
        'avg_val_mse_unweighted': float(np.mean(fold_val_mse_unw)) if 'fold_val_mse_unw' in locals() else float('nan'),
        'std_val_mse_unweighted': float(np.std(fold_val_mse_unw)) if 'fold_val_mse_unw' in locals() else float('nan'),
        'fold_val_r2': [m['r2'] for m in fold_val_metrics],
        'avg_val_r2': float(np.mean([m['r2'] for m in fold_val_metrics])),
        'std_val_r2': float(np.std([m['r2'] for m in fold_val_metrics])),
        'best_fold_index': best_fold,
    }
    
    # Print model information
    print(f"Model has {model.get_num_parameters():,} trainable parameters")
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # ================================================================
    # Final Model Evaluation on Test Set
    # ================================================================
    print(f"\nEvaluating on test set...")
    rainfall_std = metadata.get('rainfall_mm_std', None)
    # Build a test loader
    test_loaders = create_pytorch_dataloaders(
        {'test': datasets['test']},
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = test_loaders['test']

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        rainfall_std=rainfall_std
    )
    
    print(f"Test Results:")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    
    if rainfall_std is not None and rainfall_std > 0:
        print(f"  RMSE (mm): {test_metrics['denorm_rmse_mm']:.4f}")
        print(f"  MAE (mm): {test_metrics['denorm_mae_mm']:.4f}")
    else:
        print(f"  RMSE (inches): {test_metrics['rmse'] * 100:.4f}")
        print(f"  MAE (inches): {test_metrics['mae'] * 100:.4f}")
    
    # ================================================================
    # Results Saving and Visualization
    # ================================================================
    print(f"\nSaving results to {output_dir}...")
    
    # Plot training history
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Save evaluation metrics
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    save_evaluation_metrics(test_metrics, metrics_path, rainfall_std)

    # ----------------------------------------------------------------
    # Cross-Validation Summary Reports
    # ----------------------------------------------------------------
    if cv_results is not None:
        cv_txt = os.path.join(output_dir, 'cv_summary.txt')
        with open(cv_txt, 'w') as f:
            f.write("Cross-Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"n_folds: {n_folds}\n")
            f.write(f"Criterion (loss_name): {loss_name}\n")
            f.write(f"Fold best val losses (criterion): {[f'{v:.6f}' for v in cv_results['fold_val_losses']]}\n")
            f.write(f"Avg val loss (criterion): {cv_results['avg_val_loss']:.6f} +/- {cv_results['std_val_loss']:.6f}\n")
            if cv_results['fold_val_mse_unweighted']:
                f.write(f"Fold val MSE (unweighted) at best epoch: {[f'{v:.6f}' for v in cv_results['fold_val_mse_unweighted']]}\n")
                f.write(f"Avg val MSE (unweighted): {cv_results['avg_val_mse_unweighted']:.6f} +/- {cv_results['std_val_mse_unweighted']:.6f}\n")
            f.write(f"Fold val R2: {[f'{v:.4f}' for v in cv_results['fold_val_r2']]}\n")
            f.write(f"Avg val R2: {cv_results['avg_val_r2']:.4f} +/- {cv_results['std_val_r2']:.4f}\n")
            f.write(f"Best fold index (0-based): {cv_results['best_fold_index']}\n")

        # Also save a CSV for programmatic use
        cv_csv = os.path.join(output_dir, 'cv_metrics.csv')
        df = pd.DataFrame({
            'fold': list(range(1, n_folds + 1)),
            'val_loss_criterion': cv_results['fold_val_losses'],
            'val_mse_unweighted': cv_results['fold_val_mse_unweighted'] if cv_results['fold_val_mse_unweighted'] else [np.nan]*n_folds,
            'val_r2': cv_results['fold_val_r2'],
        })
        df.to_csv(cv_csv, index=False)
    
    # ----------------------------------------------------------------
    # Generate Prediction Visualizations
    # ----------------------------------------------------------------
    # Create scatter plots of predictions vs actual values
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_targets = []
        
        # Ensure inference runs on the same device as the model (cuda/mps/cpu)
        _infer_device = next(model.parameters()).device
        for features, targets in test_loader:
            features = {k: torch.nan_to_num(v.to(device=_infer_device)) for k, v in features.items()}
            outputs = model(features)
            test_predictions.extend(outputs.cpu().numpy().flatten())
            test_targets.extend(targets.numpy().flatten())
        
        test_predictions = np.array(object=test_predictions)
        test_targets = np.array(object=test_targets)
    
    # Test set scatter plot
    scatter_path = os.path.join(output_dir, 'test_predictions_scatter.png')
    create_scatter_plot(
        test_targets, test_predictions,
        'Test Set: Predicted vs Actual Rainfall',
        scatter_path, rainfall_std
    )
    
    # Save predictions (JSON makes it easy to inspect later)
    pred_path = os.path.join(output_dir, 'test_predictions.json')
    save_predictions(model, test_loader, pred_path, rainfall_std=rainfall_std)
    
    # Save training summary (use actual_val_size from CV setup)
    actual_val_size = 1.0 / n_folds if n_folds > 0 else val_size
    summary_path = save_training_summary(
        output_dir, hyperparams, history, test_metrics, training_time, rainfall_std, loss_name=loss_name,
        npz_path=npz_path, epochs=epochs, patience=patience, n_folds=n_folds, val_size=actual_val_size, seed=seed,
        learning_rate_used=scaled_lr, cv_results=cv_results
    )


    print(f"\nResults saved:")
    print(f"  Training history plot: {plot_path}")
    print(f"  Evaluation metrics: {metrics_path}")
    print(f"  Test predictions scatter: {scatter_path}")
    print(f"  Test predictions: {pred_path}")
    print(f"  Training summary: {summary_path}")
    if save_model:
        print(f"  Saved model: {model_save_path}")
    
    return {
        'model': model,
        'history': history,
        'test_metrics': test_metrics,
        'hyperparams': hyperparams,
        'training_time': training_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train best LAND model (PyTorch) with tuned hyperparameters")
    parser.add_argument("--npz-path", default="ML_Data_Preprocessing/output/assembled_npz/full_training_data_daily_3x3_2km8km_cyclical.npz", help="Path to assembled NPZ data file")
    parser.add_argument("--hyperparams-dir", default="Hyperparameter_Tuning/output/daily_3x3_2km8km_cyclical_attention_deeptemp_1980-1999_2", help="Directory containing best_hyperparameters.json or Optuna DB")
    parser.add_argument("--output-dir", default="Train_Best_Model/output/daily_3x3_2km8km_cyclical_attention_deeptemp_1980-1999_2", help="Directory to write training outputs")
    parser.add_argument("--test-indices-path", default="Hyperparameter_Tuning/output/daily_3x3_2km8km_cyclical_attention_deeptemp_1980-1999_2 /test_indices.pkl", help="Path to test indices file for reproducibility")

    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=40, help="Patience for early stopping")
    parser.add_argument("--save-model", action="store_true", help="Save trained model state_dict to output dir")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false", help="Do not save model")
    parser.set_defaults(save_model=True)

    parser.add_argument("--loss-name", type=str, default="mse", choices=["mse", "weighted_mse"], help="Training loss name")
    parser.add_argument("--loss-params", type=str, default=None)
    # parser.add_argument("--loss-params", type=str, default='{"alpha": 2.0, "power": 1.5, "percentile": 0.90}', help="JSON string of loss params, e.g. '{\"alpha\": 5, \"power\": 4, \"percentile\": 0.9}'")

    parser.add_argument("--n-folds", type=int, default=5, help="If >1, perform CV on train+val and select best fold")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation set size as fraction of train+val (default 0.1 to match HP tuning)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Parse loss params JSON if provided
    loss_params = None
    if args.loss_params:
        try:
            loss_params = json.loads(args.loss_params)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --loss-params JSON: {e}")

    results = train_best_model_pytorch(
        npz_path=args.npz_path,
        hyperparams_dir=args.hyperparams_dir,
        output_dir=args.output_dir,
        test_indices_path=args.test_indices_path,
        epochs=args.epochs,
        patience=args.patience,
        save_model=args.save_model,
        loss_name=args.loss_name,
        loss_params=loss_params,
        val_size=args.val_size,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    print("\nTraining completed successfully!")
    print(f"Final test R²: {results['test_metrics']['r2']:.4f}")
