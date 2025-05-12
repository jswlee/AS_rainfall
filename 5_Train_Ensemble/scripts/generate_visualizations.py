#!/usr/bin/env python3
"""
Generate visualizations for the ensemble model.
This script loads the trained models and creates visualizations for the ensemble predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import pickle
import argparse

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

# Import data_utils from the correct path
from importlib.machinery import SourceFileLoader
data_utils = SourceFileLoader("data_utils", os.path.join(parent_dir, "4_Train_Best_Model/scripts/data_utils.py")).load_module()

# Set project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_fold_metrics(output_dir):
    """
    Load metrics from all fold summaries.
    
    Parameters
    ----------
    output_dir : str
        Path to the output directory
        
    Returns
    -------
    dict
        Dictionary containing fold metrics
    """
    fold_metrics = {
        'r2': [],
        'rmse': [],
        'mae': []
    }
    
    # Iterate through folds
    for fold_idx in range(1, 6):
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx}')
        fold_summary_path = os.path.join(fold_dir, 'fold_summary.txt')
        
        if os.path.exists(fold_summary_path):
            with open(fold_summary_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'R²:' in line:
                        fold_metrics['r2'].append(float(line.split(':')[1].strip()))
                    elif 'RMSE:' in line:
                        fold_metrics['rmse'].append(float(line.split(':')[1].split()[0].strip()))
                    elif 'MAE:' in line:
                        fold_metrics['mae'].append(float(line.split(':')[1].split()[0].strip()))
    
    return fold_metrics

def load_test_predictions(output_dir):
    """
    Load test predictions from all models.
    
    Parameters
    ----------
    output_dir : str
        Path to the output directory
        
    Returns
    -------
    tuple
        Tuple containing (actual, predictions, ensemble_prediction)
    """
    # Always load individual model predictions to create a true ensemble
    all_predictions = []
    actual = None
    model_count = 0
    
    print("\nLoading individual model predictions...")
    for fold_idx in range(1, 6):
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx}')
        fold_predictions = []
        
        for model_idx in range(1, 6):
            model_dir = os.path.join(fold_dir, f'model_{model_idx}')
            model_pred_path = os.path.join(model_dir, 'test_predictions.csv')
            
            if os.path.exists(model_pred_path):
                model_pred_df = pd.read_csv(model_pred_path, comment='#')
                # Check which column names are present
                if 'actual_inches' in model_pred_df.columns:
                    actual_col = 'actual_inches'
                    pred_col = 'predicted_inches'
                elif 'actual' in model_pred_df.columns:
                    actual_col = 'actual'
                    pred_col = 'predicted'
                else:
                    print(f"Warning: Could not determine column names in {model_pred_path}")
                    print(f"Available columns: {model_pred_df.columns.tolist()}")
                    continue
                
                if actual is None:
                    actual = model_pred_df[actual_col].values
                fold_predictions.append(model_pred_df[pred_col].values)
                model_count += 1
        
        # Add all fold predictions to the overall list
        all_predictions.extend(fold_predictions)
    
    if len(all_predictions) > 0:
        print(f"Successfully loaded predictions from {model_count} individual models")
        # Calculate ensemble prediction by averaging all individual model predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        # Save the true ensemble predictions
        ensemble_df = pd.DataFrame({
            'actual': actual,
            'predicted': ensemble_pred
        })
        ensemble_df.to_csv(os.path.join(output_dir, 'true_ensemble_predictions.csv'), index=False)
        
        return actual, ensemble_pred
    else:
        print("No model predictions found.")
        return None, None

def generate_visualizations(output_dir):
    """
    Generate visualizations for the ensemble model.
    
    Parameters
    ----------
    output_dir : str
        Path to the output directory
    """
    print(f"Generating visualizations for ensemble model in {output_dir}...")
    
    # Load fold metrics
    fold_metrics = load_fold_metrics(output_dir)
    
    if len(fold_metrics['r2']) == 0:
        print("No fold metrics found. Make sure the ensemble model has been trained.")
        return
    
    # Calculate average metrics
    avg_r2 = np.mean(fold_metrics['r2'])
    avg_rmse = np.mean(fold_metrics['rmse'])
    avg_mae = np.mean(fold_metrics['mae'])
    
    print("\nIndividual Fold R² values:")
    for i, r2 in enumerate(fold_metrics['r2']):
        print(f"Fold {i+1}: {r2:.4f}")
    
    print("\nCross-Validation Results:")
    print(f"Average CV R²: {avg_r2:.4f}")
    print(f"Average CV RMSE: {avg_rmse:.4f} in")
    print(f"Average CV MAE: {avg_mae:.4f} in")
    
    # Load test predictions
    actual, ensemble_pred = load_test_predictions(output_dir)
    
    if actual is None or ensemble_pred is None:
        print("No test predictions found. Make sure the ensemble model has been trained.")
        return
    
    # Calculate test metrics
    test_r2 = r2_score(actual, ensemble_pred)
    test_rmse = np.sqrt(mean_squared_error(actual, ensemble_pred))
    test_mae = mean_absolute_error(actual, ensemble_pred)
    
    print("\nFinal Ensemble Results:")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} in")
    print(f"Test MAE: {test_mae:.4f} in")
    
    # Create final ensemble predictions plot
    plt.figure(figsize=(10, 8))
    plt.scatter(actual, ensemble_pred, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel('Actual Rainfall (inches)')
    plt.ylabel('Predicted Rainfall (inches)')
    plt.title('Final Ensemble: Actual vs Predicted Rainfall')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'final_ensemble_predictions.png'), dpi=300)
    plt.close()
    
    # Create residual plot
    residuals = actual - ensemble_pred
    plt.figure(figsize=(10, 8))
    plt.scatter(ensemble_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Rainfall (inches)')
    plt.ylabel('Residuals (inches)')
    plt.title('Final Ensemble: Residual Plot')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'final_ensemble_residuals.png'), dpi=300)
    plt.close()
    
    # Create histogram of residuals
    plt.figure(figsize=(10, 8))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residual Value (inches)')
    plt.ylabel('Frequency')
    plt.title('Final Ensemble: Histogram of Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'final_ensemble_residuals_hist.png'), dpi=300)
    plt.close()
    
    # Create updated ensemble summary
    summary_path = os.path.join(output_dir, 'ensemble_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Ensemble Model Summary\n")
        f.write(f"Number of Folds: 5\n")
        f.write(f"Models per Fold: 5\n")
        f.write(f"Total Models: 25\n\n")
        
        f.write("Individual Fold Results:\n")
        for i, r2 in enumerate(fold_metrics['r2']):
            f.write(f"  Fold {i+1} R²: {r2:.4f}, RMSE: {fold_metrics['rmse'][i]:.4f} in, MAE: {fold_metrics['mae'][i]:.4f} in\n")
        f.write("\n")
        
        f.write("Cross-Validation Results:\n")
        f.write(f"  Average CV R²: {avg_r2:.4f}\n")
        f.write(f"  Average CV RMSE: {avg_rmse:.4f} in\n")
        f.write(f"  Average CV MAE: {avg_mae:.4f} in\n\n")
        
        f.write("Final Ensemble Results:\n")
        f.write(f"  Test R²: {test_r2:.4f}\n")
        f.write(f"  Test RMSE: {test_rmse:.4f} in\n")
        f.write(f"  Test MAE: {test_mae:.4f} in\n")
    
    print(f"\nVisualizations and summary saved to {output_dir}")

def main():
    """Main function to generate visualizations."""
    parser = argparse.ArgumentParser(description='Generate visualizations for ensemble model')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(PIPELINE_DIR, 'output', 'simple_ensemble'),
                        help='Directory with model weights and results')
    args = parser.parse_args()
    
    # Generate visualizations
    generate_visualizations(args.output_dir)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
