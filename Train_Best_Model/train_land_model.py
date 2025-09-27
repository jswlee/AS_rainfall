#!/usr/bin/env python3
"""
PyTorch training script for the best LAND model using optimized hyperparameters.
"""

import os
import json
import time
import torch
import numpy as np
import random
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import optuna

# Import PyTorch utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Hyperparameter_Tuning'))

# Import robust MLflow utilities for experiment tracking
from Hyperparameter_Tuning.mlflow_utils import (
    create_mlflow_logger, log_hyperparameters, log_model_summary,
    log_evaluation_results, start_pretraining_preview_run, MLFLOW_AVAILABLE
)

from Hyperparameter_Tuning.data_utils import load_assembled_npz_data_pytorch, create_pytorch_dataloaders
from Hyperparameter_Tuning.model import create_model_from_hyperparams
from Hyperparameter_Tuning.model_training import train_model, evaluate_model, plot_training_history, save_predictions
from Hyperparameter_Tuning.hp_tuning import load_best_hyperparameters_pytorch


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


def save_training_summary(output_dir, hyperparams, history, test_metrics, training_time, rainfall_std=None, loss_name: str = 'mse'):
    """Save training summary to text file."""
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("PyTorch LAND Model Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Training Results:\n")
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
    seed: int = 42,
    # MLflow experiment tracking options
    enable_mlflow: bool = False,
    mlflow_experiment: str = "AS_Rainfall_Production_Training", 
    mlflow_run_name: str | None = None,
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
        seed: Random seed for reproducibility
        enable_mlflow: Whether to enable MLflow experiment tracking
        mlflow_experiment: MLflow experiment name
        mlflow_run_name: Optional MLflow run name
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
    data = load_assembled_npz_data_pytorch(
        npz_path=npz_path,
        test_indices_path=test_indices_path,
        random_state=seed
    )
    
    # ================================================================
    # Hyperparameter Loading and Configuration
    # ================================================================
    print(f"Loading hyperparameters from {hyperparams_dir}...")
    # Prefer loading directly from the Optuna SQLite database
    # Load hyperparameters from JSON first (includes trial metadata), fallback to Optuna DB
    hyperparams_data = None
    trial_number = None
    
    try:
        hyperparams_data = load_best_hyperparameters_pytorch(hyperparams_dir)
        if 'hyperparameters' in hyperparams_data:
            hyperparams = hyperparams_data['hyperparameters']
            trial_number = hyperparams_data.get('trial_number')
            print("Loaded hyperparameters from JSON:")
            for k, v in hyperparams.items():
                print(f"  {k}: {v}")
            if trial_number is not None:
                print(f"  Source trial: {trial_number}")
        else:
            hyperparams = hyperparams_data
            print("Loaded hyperparameters from JSON (legacy format)")
    except FileNotFoundError:
        # Fallback to Optuna DB
        db_path = os.path.join(hyperparams_dir, 'land_model_tuning.db')
        if os.path.exists(db_path):
            try:
                storage = f"sqlite:///{db_path}"
                study = optuna.load_study(study_name="land_model_tuning", storage=storage)
                hyperparams = dict(study.best_trial.params)
                trial_number = study.best_trial.number
                print("Loaded hyperparameters from Optuna DB:")
                for k, v in hyperparams.items():
                    print(f"  {k}: {v}")
                print(f"  Source trial: {trial_number}")
            except Exception as e:
                print(f"Warning: Failed to load hyperparameters from DB: {e}")
                hyperparams = None
        else:
            hyperparams = None
    
    # Require at least one of the above to succeed
    if hyperparams is None:
        raise RuntimeError(
            "No hyperparameters found in Optuna DB. Please run hyperparameter tuning first "
            "(run_complete_pytorch_pipeline.py --only-tuning) so that Hyperparameter_Tuning/output/land_model_tuning.db exists."
        )
    
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
    # Kick off a short pre-training MLflow run so the experiment appears
    # immediately in the UI (useful for long trainings/CV). Main logging later.
    # ------------------------------------------------------------------
    if enable_mlflow and MLFLOW_AVAILABLE:
        try:
            training_config_preview = {
                "loss_criterion": loss_name,
                "loss_params": loss_params,
                "epochs_requested": epochs,
                "n_folds": n_folds,
                "device": str(torch.device('cuda' if torch.cuda.is_available()
                                         else 'mps' if torch.backends.mps.is_available()
                                         else 'cpu')),
                "training_mode": "cross_validation",
                "model_type": "LAND_rainfall_prediction",
            }
            _ = start_pretraining_preview_run(
                experiment_name=mlflow_experiment,
                run_name=mlflow_run_name or f"pre_training_{int(time.time())}",
                hyperparams=hyperparams,
                training_config_preview=training_config_preview,
                enabled=True,
            )
        except Exception as _e:
            print(f"Warning: Pre-training MLflow kick-off failed: {_e}")
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
    # Combine train and val like the tuner
    train_dataset = data['datasets']['train']
    val_dataset = data['datasets']['val']
    cv_climate = torch.cat([train_dataset.climate_data, val_dataset.climate_data])
    cv_local_dem = torch.cat([train_dataset.local_dem_data, val_dataset.local_dem_data])
    cv_regional_dem = torch.cat([train_dataset.regional_dem_data, val_dataset.regional_dem_data])
    cv_month = torch.cat([train_dataset.month_data, val_dataset.month_data])
    cv_targets = torch.cat([train_dataset.targets, val_dataset.targets])

    # ----------------------------------------------------------------
    # Stratified Cross-Validation Setup
    # ----------------------------------------------------------------
    # Create stratification bins (same approach as tuner)
    y = cv_targets.numpy().ravel()
    n_bins = 5
    try:
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y, q)
        uniq = np.unique(edges)
        if uniq.size < edges.size:
            edges = np.linspace(y.min(), y.max(), n_bins + 1)
        y_bins = np.digitize(y, edges[1:-1], right=True)
    except Exception as e:
        print(f"Warning: stratification binning failed ({e}); falling back to single bin.")
        y_bins = np.zeros_like(y, dtype=int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros_like(y_bins), y_bins))

    from Hyperparameter_Tuning.data_utils import RainfallDataset

    # ----------------------------------------------------------------
    # Cross-Validation Execution Loop
    # ----------------------------------------------------------------
    fold_histories = []
    fold_val_losses = []
    fold_val_metrics = []
    fold_models = []

    start_time = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        print(f"Fold {fold_idx+1}/{n_folds}: train={len(tr_idx)} val={len(va_idx)}")

        fold_train_ds = RainfallDataset(
            cv_climate[tr_idx].numpy(),
            cv_local_dem[tr_idx].numpy(),
            cv_regional_dem[tr_idx].numpy(),
            cv_month[tr_idx].numpy(),
            cv_targets[tr_idx].numpy(),
        )
        fold_val_ds = RainfallDataset(
            cv_climate[va_idx].numpy(),
            cv_local_dem[va_idx].numpy(),
            cv_regional_dem[va_idx].numpy(),
            cv_month[va_idx].numpy(),
            cv_targets[va_idx].numpy(),
        )

        fold_loaders = create_pytorch_dataloaders(
            {'train': fold_train_ds, 'val': fold_val_ds},
            batch_size=batch_size,
            num_workers=0,
        )

        model = create_model_from_hyperparams(hyperparams, data['metadata'])
        hist = train_model(
            model=model,
            dataloaders=fold_loaders,
            epochs=epochs,
            learning_rate=hyperparams.get('learning_rate', 0.001),
            weight_decay=hyperparams.get('weight_decay', 0.001),
            patience=30,
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
        rainfall_std = data['metadata'].get('rainfall_mm_std', None)
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
    # Optionally save the selected best-fold model
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)

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
    rainfall_std = data['metadata'].get('rainfall_mm_std', None)
    # Build a test loader
    test_loaders = create_pytorch_dataloaders(
        {'test': data['datasets']['test']},
        batch_size=batch_size,
        num_workers=0,
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
    # Create scatter plots (we also log this image as an artifact when MLflow is enabled)
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_targets = []
        
        # Ensure inference runs on the same device as the model (cuda/mps/cpu)
        _infer_device = next(model.parameters()).device
        for features, targets in test_loader:
            features = {k: v.to(device=_infer_device) for k, v in features.items()}
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
    
    # Save predictions (JSON makes it easy to inspect later; we'll also log it to MLflow when enabled)
    pred_path = os.path.join(output_dir, 'test_predictions.json')
    save_predictions(model, test_loader, pred_path, rainfall_std=rainfall_std)
    
    # Save training summary
    summary_path = save_training_summary(
        output_dir, hyperparams, history, test_metrics, training_time, rainfall_std, loss_name=loss_name
    )

    # ============================================================================
    # MLflow Experiment Tracking - Production Model Training
    # ============================================================================
    
    if enable_mlflow and MLFLOW_AVAILABLE:
        # Create MLflow logger with comprehensive error handling
        mlflow_logger = create_mlflow_logger(
            experiment_name=mlflow_experiment,
            run_name=mlflow_run_name or f"best_model_training_{int(time.time())}",
            enabled=True
        )
        
        # Use context manager for automatic run lifecycle management
        with mlflow_logger.start_run():
            ### 1. Log Configuration and Hyperparameters

            log_hyperparameters(mlflow_logger, hyperparams, prefix="hp")
            
            # Log training configuration
            training_config = {
                "epochs_requested": epochs,
                "loss_name": loss_name,
                "n_folds": n_folds if n_folds and n_folds > 1 else 1,
                "device": str(torch.device('cuda' if torch.cuda.is_available() 
                                         else 'mps' if torch.backends.mps.is_available() 
                                         else 'cpu')),
                "source_trial_number": trial_number,
                "model_saved": save_model,
                "training_mode": "cross_validation" if n_folds and n_folds > 1 else "single_split"
            }
            
            mlflow_logger.log_params(training_config)
            
            # Log loss function parameters if provided
            if loss_params:
                loss_config = {f"loss_{k}": v for k, v in loss_params.items()}
                mlflow_logger.log_params(loss_config)
            else:
                mlflow_logger.log_params({"loss_name": "unweighted_mse"})
            # Set descriptive tags for easy filtering and organization
            mlflow_logger.set_tags({
                "model_type": "LAND_rainfall_prediction",
                "experiment_phase": "production_training",
                "data_version": "full_training_data",
                "framework": "pytorch"
            })
            
            ### 2. Log Model Architecture

            # Create and log detailed model summary
            if n_folds and n_folds > 1:
                # For CV, use the selected best model
                log_model_summary(mlflow_logger, model, "best_fold_model_architecture.txt")
            else:
                # For single training, log the trained model
                log_model_summary(mlflow_logger, model, "trained_model_architecture.txt")
            
            ### 3. Log Training Curves and Progress

            # Log detailed training history for analysis and debugging
            mlflow_logger.log_training_curves(history, start_epoch=1)
            
            # Log training summary metrics
            training_summary = {
                "final_train_loss": float(history['train_loss'][-1]),
                "final_val_loss": float(history['val_loss'][-1]),
                "best_val_loss": float(min(history['val_loss'])),
                "best_epoch": int(np.argmin(history['val_loss'])) + 1,
                "total_epochs_trained": len(history['train_loss']),
                "training_time_seconds": training_time
            }
            
            # Add unweighted MSE metrics if available
            if 'val_mse_unweighted' in history:
                best_epoch_idx = int(np.argmin(history['val_loss']))
                training_summary.update({
                    "best_val_mse_unweighted": float(history['val_mse_unweighted'][best_epoch_idx]),
                    "final_val_mse_unweighted": float(history['val_mse_unweighted'][-1])
                })
            
            # Use standardized logging with training prefix
            log_evaluation_results(mlflow_logger, training_summary, prefix="training")
            
            ### 4. Log Cross-Validation Results (if applicable)
            if cv_results is not None:
                cv_metrics = {
                    "avg_val_loss": cv_results['avg_val_loss'],
                    "std_val_loss": cv_results['std_val_loss'],
                    "avg_val_r2": cv_results['avg_val_r2'],
                    "std_val_r2": cv_results['std_val_r2'],
                    "best_fold": cv_results['best_fold_index'] + 1
                }
                
                if cv_results['avg_val_mse_unweighted'] != float('nan'):
                    cv_metrics.update({
                        "avg_val_mse_unweighted": cv_results['avg_val_mse_unweighted'],
                        "std_val_mse_unweighted": cv_results['std_val_mse_unweighted']
                    })
                
                # Use standardized logging with cv prefix
                log_evaluation_results(mlflow_logger, cv_metrics, prefix="cv")
                mlflow_logger.set_tag("cv_enabled", "true")
            else:
                mlflow_logger.set_tag("cv_enabled", "false")
            
            ### 5. Log Test Set Evaluation Results

            # Log comprehensive test metrics using standardized function
            test_metrics_base = {
                "r2": test_metrics["r2"],
                "mse": test_metrics["mse"],
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"]
            }
            
            # Add unit-specific metrics (mm or inches)
            if rainfall_std is not None and rainfall_std > 0:
                test_metrics_base.update({
                    "rmse_mm": test_metrics.get("denorm_rmse_mm", 0.0),
                    "mae_mm": test_metrics.get("denorm_mae_mm", 0.0)
                })
                mlflow_logger.set_tag("units", "mm")
            else:
                test_metrics_base.update({
                    "rmse_inches": test_metrics["rmse"] * 100.0,
                    "mae_inches": test_metrics["mae"] * 100.0
                })
                mlflow_logger.set_tag("units", "inches")
            
            # Use standardized logging with test prefix
            log_evaluation_results(mlflow_logger, test_metrics_base, prefix="test")
            
            ### 6. Log Artifacts (Files, Plots, Models)
  
            # Training and evaluation artifacts
            artifact_paths = {
                "training_history_plot": plot_path,
                "evaluation_metrics_csv": metrics_path,
                "test_predictions_scatter": scatter_path,
                "test_predictions_json": pred_path,
                "training_summary_txt": summary_path
            }
            
            # Log each artifact with descriptive names
            for artifact_name, path in artifact_paths.items():
                if os.path.exists(path):
                    mlflow_logger.log_artifact(path)
                    mlflow_logger.set_tag(f"has_{artifact_name}", "true")
            
            # Log cross-validation artifacts if available
            if cv_results is not None:
                cv_summary_path = os.path.join(output_dir, 'cv_summary.txt')
                cv_metrics_path = os.path.join(output_dir, 'cv_metrics.csv')
                
                if os.path.exists(cv_summary_path):
                    mlflow_logger.log_artifact(cv_summary_path)
                if os.path.exists(cv_metrics_path):
                    mlflow_logger.log_artifact(cv_metrics_path)
            
            ### 7. Log Trained Model for Deployment

            # Log the final trained model for deployment and inference
            if save_model and model_save_path and os.path.exists(model_save_path):
                # Log model state dict as artifact
                mlflow_logger.log_artifact(model_save_path)
                
                # Also log as MLflow model for easy deployment
                # Note: We use the model object directly, not the state dict file
                mlflow_logger.log_model(
                    model,
                    artifact_path="trained_model",
                    # Additional metadata for model serving
                )
                
                mlflow_logger.set_tag("model_saved", "true")
                mlflow_logger.set_tag("model_format", "pytorch")
            
            ### 8. Log Success Status and Summary

            mlflow_logger.set_tag("training_status", "completed")
            mlflow_logger.set_tag("final_test_r2", f"{test_metrics['r2']:.4f}")
            
            print(f"\n✓ MLflow tracking completed successfully!")
            print(f"  Experiment: {mlflow_experiment}")
            print(f"  Run ID: {mlflow_logger.get_run_id()}")
            print(f"  View results: mlflow ui")
    
    elif enable_mlflow and not MLFLOW_AVAILABLE:
        print("\nMLflow logging requested but MLflow is not available.")
        print("   Install MLflow with: pip install mlflow")
    
    else:
        print("\nMLflow logging disabled. Enable with enable_mlflow=True for experiment tracking.")

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
    parser.add_argument("--npz-path", default=os.path.join("ML_Data_Preprocessing", "output", "assembled_npz", "full_training_data.npz"), help="Path to assembled NPZ data file")
    parser.add_argument("--hyperparams-dir", default=os.path.join("Hyperparameter_Tuning", "output_highRainfall"), help="Directory containing best_hyperparameters.json or Optuna DB")
    parser.add_argument("--output-dir", default=os.path.join("Train_Best_Model", "output_highRainfall"), help="Directory to write training outputs")
    parser.add_argument("--test-indices-path", default=os.path.join("Hyperparameter_Tuning", "output_highRainfall", "test_indices.pkl"), help="Path to test indices file for reproducibility")

    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs")
    parser.add_argument("--save-model", action="store_true", help="Save trained model state_dict to output dir")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false", help="Do not save model")
    parser.set_defaults(save_model=True)

    parser.add_argument("--loss-name", type=str, default="mse", choices=["mse", "weighted_mse"], help="Training loss name")
    parser.add_argument("--loss-params", type=str, default=None, help="JSON string of loss params, e.g. '{\"alpha\": 5, \"power\": 4, \"percentile\": 0.9}'")

    parser.add_argument("--n-folds", type=int, default=5, help="If >1, perform CV on train+val and select best fold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--enable-mlflow", action="store_true", help="Enable MLflow experiment tracking")
    parser.add_argument("--mlflow-experiment", type=str, default="AS_Rainfall_Production_Training", help="MLflow experiment name")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Optional MLflow run name")

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
        save_model=args.save_model,
        loss_name=args.loss_name,
        loss_params=loss_params,
        n_folds=args.n_folds,
        seed=args.seed,
        enable_mlflow=args.enable_mlflow,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )

    print("\nTraining completed successfully!")
    print(f"Final test R²: {results['test_metrics']['r2']:.4f}")
