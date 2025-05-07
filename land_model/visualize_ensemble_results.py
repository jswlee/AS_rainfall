#!/usr/bin/env python3
"""
Visualize Ensemble Model Results

This script loads the ensemble model results from a pickle file and creates
visualizations with correct units (inches).
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_results(results_path):
    """Load ensemble results from pickle file."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Correct the scaling of RMSE and MAE values (divide by 100)
    # These values were incorrectly scaled in the original results
    results['test_rmse'] /= 100
    results['test_mae'] /= 100
    results['avg_cv_rmse'] /= 100
    results['avg_cv_mae'] /= 100
    
    # Correct fold results
    for fold in results['fold_results']:
        fold['rmse'] /= 100
        fold['mae'] /= 100
    
    return results

def load_predictions(predictions_path):
    """Load test predictions and actual values from numpy files."""
    try:
        test_pred = np.load(predictions_path + '_predictions.npy')
        test_actual = np.load(predictions_path + '_actual.npy')
        return test_pred, test_actual
    except FileNotFoundError:
        print(f"Warning: Could not find prediction files at {predictions_path}")
        return None, None

def create_scatter_plot(y_true, y_pred, output_path, title="Actual vs Predicted Rainfall"):
    """Create scatter plot of actual vs predicted values with correct units (inches)."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics (on the original scale)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Add metrics to plot
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f} inches\nMAE = {mae:.4f} inches',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                verticalalignment='top')
    
    # Set labels and title
    plt.xlabel('Actual Rainfall (inches)')
    plt.ylabel('Predicted Rainfall (inches)')
    plt.title(title)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Scatter plot saved to {output_path}")
    return r2, rmse, mae

def create_residual_plot(y_true, y_pred, output_path, title="Residual Plot"):
    """Create residual plot with correct units (inches)."""
    plt.figure(figsize=(10, 8))
    
    # Calculate residuals
    residuals = y_pred - y_true
    
    # Create scatter plot
    plt.scatter(y_true, residuals, alpha=0.5)
    
    # Add zero line
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Calculate metrics on residuals
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Add metrics to plot
    plt.annotate(f'Mean = {mean_residual:.4f} inches\nStd Dev = {std_residual:.4f} inches',
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                verticalalignment='top')
    
    # Set labels and title
    plt.xlabel('Actual Rainfall (inches)')
    plt.ylabel('Residual (Predicted - Actual) (inches)')
    plt.title(title)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Residual plot saved to {output_path}")

def create_fold_comparison_plot(results, output_path):
    """Create bar plot comparing performance across folds with correct units (inches)."""
    plt.figure(figsize=(12, 8))
    
    # Extract fold results
    fold_results = results['fold_results']
    n_folds = len(fold_results)
    
    # Prepare data
    fold_names = [f'Fold {i+1}' for i in range(n_folds)]
    r2_values = [fold['r2'] for fold in fold_results]
    rmse_values = [fold['rmse'] for fold in fold_results]
    mae_values = [fold['mae'] for fold in fold_results]
    
    # Add test results
    fold_names.append('Test')
    r2_values.append(results['test_r2'])
    rmse_values.append(results['test_rmse'])
    mae_values.append(results['test_mae'])
    
    # Create bar plot
    x = np.arange(len(fold_names))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # Plot R² on left axis
    bars1 = ax1.bar(x - width, r2_values, width, label='R²', color='blue', alpha=0.7)
    ax1.set_ylabel('R²', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot RMSE and MAE on right axis
    bars2 = ax2.bar(x, rmse_values, width, label='RMSE (inches)', color='red', alpha=0.7)
    bars3 = ax2.bar(x + width, mae_values, width, label='MAE (inches)', color='green', alpha=0.7)
    ax2.set_ylabel('Error (inches)', color='black')
    
    # Set x-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_names)
    
    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Add title
    plt.title('Model Performance Across Folds')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Fold comparison plot saved to {output_path}")

def create_histogram_plot(y_true, y_pred, output_path, title="Distribution of Actual vs Predicted Rainfall"):
    """Create histogram of actual and predicted values with correct units (inches)."""
    plt.figure(figsize=(10, 8))
    
    # Create histograms
    plt.hist(y_true, bins=30, alpha=0.5, label='Actual')
    plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted')
    
    # Set labels and title
    plt.xlabel('Rainfall (inches)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Histogram plot saved to {output_path}")

def main():
    """Main function to create visualizations."""
    # Define paths
    results_path = '/Users/jlee/Desktop/github/RainfallPredictionWithClimateData/land_model_ensemble_raw/ensemble_results.pkl'
    predictions_path = '/Users/jlee/Desktop/github/RainfallPredictionWithClimateData/land_model_ensemble_raw/test'
    output_dir = '/Users/jlee/Desktop/github/RainfallPredictionWithClimateData/figures/ensemble_results'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(results_path)
    print("Loaded ensemble results")
    
    # Load predictions
    test_pred, test_actual = load_predictions(predictions_path)
    
    # Create visualizations
    if test_pred is not None and test_actual is not None:
        print("Creating visualizations with test predictions...")
        
        # Create scatter plot
        scatter_path = os.path.join(output_dir, 'actual_vs_predicted.png')
        r2, rmse, mae = create_scatter_plot(test_actual, test_pred, scatter_path)
        
        # Create residual plot
        residual_path = os.path.join(output_dir, 'residual_plot.png')
        create_residual_plot(test_actual, test_pred, residual_path)
        
        # Create histogram plot
        hist_path = os.path.join(output_dir, 'distribution_plot.png')
        create_histogram_plot(test_actual, test_pred, hist_path)
        
        # Print metrics
        print("\nTest Set Metrics (calculated from predictions):")
        print(f"R² = {r2:.4f}")
        print(f"RMSE = {rmse:.4f} inches")
        print(f"MAE = {mae:.4f} inches")
    else:
        print("No test predictions found. Using metrics from results file...")
        
        # Print metrics from results file
        print("\nTest Set Metrics (from results file):")
        print(f"R² = {results['test_r2']:.4f}")
        print(f"RMSE = {results['test_rmse']:.4f} inches")
        print(f"MAE = {results['test_mae']:.4f} inches")
    
    # Create fold comparison plot
    fold_path = os.path.join(output_dir, 'fold_comparison.png')
    create_fold_comparison_plot(results, fold_path)
    
    print("\nAll visualizations created successfully!")

if __name__ == "__main__":
    main()
