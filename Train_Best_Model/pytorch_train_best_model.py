#!/usr/bin/env python3
"""
PyTorch training script for the best LAND model using optimized hyperparameters.
"""

import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd

# Import PyTorch utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Hyperparameter_Tuning'))

from pytorch_data_utils import load_assembled_npz_data_pytorch, create_pytorch_dataloaders
from pytorch_model import create_model_from_hyperparams
from pytorch_training import train_model, evaluate_model, plot_training_history, save_predictions
from pytorch_hyperparameter_tuning import load_best_hyperparameters_pytorch


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


def save_training_summary(output_dir, hyperparams, history, test_metrics, training_time, rainfall_std=None):
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
        f.write(f"  Training time: {training_time:.2f} seconds\n")
        f.write(f"  Final train loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Final val loss: {history['val_loss'][-1]:.6f}\n")
        f.write(f"  Best val loss: {min(history['val_loss']):.6f}\n")
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
    epochs: int = 150,
    save_model: bool = True
):
    """
    Train the best LAND model using PyTorch.
    
    Args:
        npz_path: Path to assembled NPZ data
        hyperparams_dir: Directory containing best hyperparameters
        output_dir: Output directory for results
        test_indices_path: Path to test indices
        epochs: Maximum training epochs
        save_model: Whether to save the trained model
    """
    # Set default paths
    if npz_path is None:
        npz_path = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz')
    if hyperparams_dir is None:
        hyperparams_dir = os.path.join('Hyperparameter_Tuning', 'output')
    if output_dir is None:
        output_dir = os.path.join('Train_Best_Model', 'output', 'pytorch_best_model')
    if test_indices_path is None:
        test_indices_path = os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("PyTorch LAND Model Training")
    print("=" * 50)
    
    # Load data
    print(f"Loading data from {npz_path}...")
    data = load_assembled_npz_data_pytorch(
        npz_path=npz_path,
        test_indices_path=test_indices_path,
        random_state=42
    )
    
    # Load best hyperparameters
    print(f"Loading hyperparameters from {hyperparams_dir}...")
    try:
        hyperparams = load_best_hyperparameters_pytorch(hyperparams_dir)
        print("Loaded hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
    except FileNotFoundError:
        print("No hyperparameters found, using defaults...")
        hyperparams = {
            'climate_units': 128,
            'local_dem_units': 64,
            'regional_dem_units': 64,
            'month_units': 32,
            'na': 256,
            'nb': 128,
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'batch_size': 32,
            'use_residual': True,
            'activation': 'relu',
            'output_activation': 'relu'
        }
    
    # Create dataloaders
    batch_size = hyperparams.get('batch_size', 32)
    dataloaders = create_pytorch_dataloaders(
        data['datasets'],
        batch_size=batch_size,
        num_workers=0
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(data['datasets']['train'])} samples")
    print(f"  Val: {len(data['datasets']['val'])} samples")
    print(f"  Test: {len(data['datasets']['test'])} samples")
    
    # Create model
    print(f"\nCreating model...")
    model = create_model_from_hyperparams(hyperparams, data['metadata'])
    print(f"Model has {model.get_num_parameters():,} trainable parameters")
    
    # Train model
    print(f"\nTraining model for up to {epochs} epochs...")
    start_time = time.time()
    
    model_save_path = os.path.join(output_dir, 'best_model.pth') if save_model else None
    
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        epochs=epochs,
        learning_rate=hyperparams.get('learning_rate', 0.001),
        weight_decay=hyperparams.get('weight_decay', 0.001),
        patience=15,
        save_path=model_save_path,
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    rainfall_std = data['metadata'].get('rainfall_mm_std', None)
    test_metrics = evaluate_model(
        model=model,
        dataloader=dataloaders['test'],
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
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    
    # Plot training history
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Save evaluation metrics
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    save_evaluation_metrics(test_metrics, metrics_path, rainfall_std)
    
    # Create scatter plots
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_targets = []
        
        for features, targets in dataloaders['test']:
            outputs = model(features)
            test_predictions.extend(outputs.numpy().flatten())
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
    
    # Save predictions
    pred_path = os.path.join(output_dir, 'test_predictions.json')
    save_predictions(model, dataloaders['test'], pred_path, rainfall_std=rainfall_std)
    
    # Save training summary
    summary_path = save_training_summary(
        output_dir, hyperparams, history, test_metrics, training_time, rainfall_std
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
        'training_time': training_time
    }


if __name__ == "__main__":
    # Train the best model
    results = train_best_model_pytorch()
    
    print(f"\nTraining completed successfully!")
    print(f"Final test R²: {results['test_metrics']['r2']:.4f}")
