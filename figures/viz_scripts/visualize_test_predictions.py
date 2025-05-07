#!/usr/bin/env python3
"""
Visualize test predictions vs actual values for the rainfall prediction model.
This script can be run after training to generate visualizations of model performance.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_predictions(predictions_path):
    """Load test predictions from a pickle file."""
    with open(predictions_path, 'rb') as f:
        return pickle.load(f)

def create_visualizations(predictions, output_dir, model_name):
    """Create and save visualizations for test predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data - handle different prediction formats
    if isinstance(predictions, dict):
        if 'y_true' in predictions and 'y_pred' in predictions:
            # Direct format with true and predicted values
            y_true = predictions['y_true']
            y_pred = predictions['y_pred']
        elif 'fold_results' in predictions:
            # Ensemble results format
            print("Using ensemble results format...")
            
            # Extract test metrics from the ensemble results
            test_r2 = predictions.get('test_r2', None)
            test_rmse = predictions.get('test_rmse', None)
            test_mae = predictions.get('test_mae', None)
            
            print(f"Found test metrics in ensemble results: R²={test_r2:.4f}, RMSE={test_rmse:.2f}, MAE={test_mae:.2f}")
            
            # Create a summary report without visualizations
            metrics_path = os.path.join(output_dir, f'{model_name}_test_metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Test Set Metrics (from ensemble results):\n")
                f.write(f"R² = {test_r2:.4f}\n")
                f.write(f"RMSE = {test_rmse:.2f} mm\n")
                f.write(f"MAE = {test_mae:.2f} mm\n")
                
                # Add hyperparameters if available
                if 'hyperparams' in predictions:
                    f.write("\nHyperparameters:\n")
                    for k, v in predictions['hyperparams'].items():
                        f.write(f"{k}: {v}\n")
                
                # Add training details if available
                if 'training_time' in predictions:
                    hours, remainder = divmod(predictions['training_time'], 3600)
                    minutes, seconds = divmod(remainder, 60)
                    f.write(f"\nTraining time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}\n")
                
                if 'n_folds' in predictions and 'n_models' in predictions:
                    f.write(f"Number of folds: {predictions['n_folds']}\n")
                    f.write(f"Models per fold: {predictions['n_models']}\n")
                    f.write(f"Total models: {predictions['n_folds'] * predictions['n_models']}\n")
            
            print(f"Saved metrics to {metrics_path}")
            
            return {
                'r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            }
        else:
            raise ValueError("Unsupported prediction format. Expected 'y_true'/'y_pred' or ensemble results format.")
    else:
        raise ValueError("Predictions must be a dictionary.")
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.1
    
    # Plot points
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Plot perfect prediction line
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    # Add text with metrics
    plt.text(0.05 * max_val, 0.9 * max_val, 
             f'R² = {r2:.4f}\nRMSE = {rmse:.2f} mm\nMAE = {mae:.2f} mm',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Actual Rainfall (mm)')
    plt.ylabel('Predicted Rainfall (mm)')
    plt.title(f'Test Set: Actual vs Predicted Rainfall - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    scatter_path = os.path.join(output_dir, f'{model_name}_test_scatter.png')
    plt.savefig(scatter_path, dpi=300)
    print(f"Saved scatter plot to {scatter_path}")
    
    # Create residual plot
    plt.figure(figsize=(10, 8))
    residuals = y_pred - y_true
    
    plt.scatter(y_true, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('Actual Rainfall (mm)')
    plt.ylabel('Residuals (Predicted - Actual) (mm)')
    plt.title(f'Test Set: Residual Plot - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    residual_path = os.path.join(output_dir, f'{model_name}_test_residuals.png')
    plt.savefig(residual_path, dpi=300)
    print(f"Saved residual plot to {residual_path}")
    
    # Create histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    
    plt.xlabel('Residuals (Predicted - Actual) (mm)')
    plt.ylabel('Frequency')
    plt.title(f'Test Set: Distribution of Residuals - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    hist_path = os.path.join(output_dir, f'{model_name}_test_residual_hist.png')
    plt.savefig(hist_path, dpi=300)
    print(f"Saved residual histogram to {hist_path}")
    
    # Save metrics to text file
    metrics_path = os.path.join(output_dir, f'{model_name}_test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Set Metrics:\n")
        f.write(f"R² = {r2:.4f}\n")
        f.write(f"RMSE = {rmse:.2f} mm\n")
        f.write(f"MAE = {mae:.2f} mm\n")
    print(f"Saved metrics to {metrics_path}")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

def main():
    parser = argparse.ArgumentParser(description='Visualize test predictions')
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to the pickle file containing test predictions')
    parser.add_argument('--output_dir', type=str, default='figures/test_results',
                        help='Directory to save visualization results')
    parser.add_argument('--model_name', type=str, default='ensemble',
                        help='Name of the model for plot titles and filenames')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions_path}...")
    predictions = load_predictions(args.predictions_path)
    
    # Create visualizations
    print(f"Creating visualizations in {args.output_dir}...")
    metrics = create_visualizations(predictions, args.output_dir, args.model_name)
    
    print("\nTest Set Metrics:")
    print(f"R² = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.2f} mm")
    print(f"MAE = {metrics['mae']:.2f} mm")

if __name__ == '__main__':
    main()
