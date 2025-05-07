#!/usr/bin/env python3
"""
Generate rainfall predictions using the trained model.

This script loads a trained model and generates rainfall predictions for specified dates,
creating visualizations of the predicted rainfall patterns.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the model
from src.deep_learning.model import RainfallModel

def visualize_predictions(predictions, grid_size, date_str, output_dir, dem_data=None):
    """
    Visualize rainfall predictions on a grid.
    
    Parameters
    ----------
    predictions : array
        Predicted rainfall values
    grid_size : int
        Size of the grid (e.g., 5 for a 5x5 grid)
    date_str : str
        Date string for the title
    output_dir : str
        Directory to save the visualization
    dem_data : array, optional
        DEM data for contour overlay
    """
    # Reshape predictions to grid
    rainfall_grid = predictions.reshape(grid_size, grid_size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot rainfall as a filled contour
    im = ax.imshow(rainfall_grid, cmap='Blues', interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rainfall (mm)')
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    # Add grid point values
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f"{rainfall_grid[i, j]:.1f}", 
                    ha="center", va="center", color="black", fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    # Add title and labels
    ax.set_title(f'Predicted Rainfall for {date_str}', fontsize=16)
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'predicted_rainfall_{date_str}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate rainfall predictions using the trained model')
    parser.add_argument('--data', type=str, default='output/rainfall_prediction_data.h5',
                        help='Path to H5 file with processed data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions and visualizations')
    parser.add_argument('--date_indices', type=int, nargs='+',
                        help='Indices of dates to predict (if not specified, all dates will be used)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data first to get input shapes
    print(f"Loading data from {args.data}...")
    with h5py.File(args.data, 'r') as h5_file:
        # Get all date keys
        date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
        
        # Filter date keys if date_indices is specified
        if args.date_indices:
            date_keys = [date_keys[i] for i in args.date_indices if i < len(date_keys)]
        
        # Get metadata
        grid_size = 5  # Default
        if 'metadata' in h5_file and 'grid_size' in h5_file['metadata'].attrs:
            grid_size = h5_file['metadata'].attrs['grid_size']
        
        # Get sample data to determine input shapes
        sample_date_key = date_keys[0]
        sample_date_group = h5_file[sample_date_key]
        
        # Extract climate variables
        climate_data = []
        for var_name in sample_date_group['climate_vars']:
            var_data = sample_date_group['climate_vars'][var_name][:]
            climate_data.append(var_data)
        
        # Stack climate variables
        climate_vars = np.column_stack(climate_data)
        
        # Extract other data
        local_patches = sample_date_group['local_patches'][:]
        regional_patches = sample_date_group['regional_patches'][:]
        month_encoding = sample_date_group['month_one_hot'][:]
        
        # Get input shapes
        input_shapes = {
            'climate_vars': climate_vars.shape[1:],
            'local_dem': local_patches.shape[1:],
            'regional_dem': regional_patches.shape[1:],
            'month_encoding': month_encoding.shape
        }
    
    # Initialize the model
    model = RainfallModel()
    
    # Build the model with the correct input shapes
    print(f"Building model...")
    model.build_model(input_shapes)
    
    # Load weights from the saved model
    print(f"Loading weights from {args.model_path}...")
    model.model.load_weights(args.model_path)
    
    # Process each date
    print(f"Processing data and generating predictions...")
    with h5py.File(args.data, 'r') as h5_file:
        # Initialize results dataframe
        results = []
        
        # Process each date
        for date_key in date_keys:
            date_group = h5_file[date_key]
            date_str = date_key.split('_')[-1]
            
            print(f"\nProcessing {date_str}...")
            
            # Skip if any required data is missing
            if not all(key in date_group for key in 
                       ['climate_vars', 'local_patches', 'regional_patches', 'month_one_hot', 'rainfall']):
                print(f"Skipping {date_str} due to missing data")
                continue
            
            # Get number of grid points
            n_points = len(date_group['rainfall'][:])
            
            # Extract climate variables
            climate_data = []
            for var_name in date_group['climate_vars']:
                var_data = date_group['climate_vars'][var_name][:]
                climate_data.append(var_data)
            
            # Stack climate variables
            climate_vars = np.column_stack(climate_data)
            
            # Extract other data
            local_patches = date_group['local_patches'][:]
            regional_patches = date_group['regional_patches'][:]
            month_encoding = date_group['month_one_hot'][:]
            actual_rainfall = date_group['rainfall'][:]
            
            # Repeat month encoding for each grid point
            month_encodings = np.tile(month_encoding, (n_points, 1))
            
            # Prepare input data
            X = {
                'climate_vars': climate_vars,
                'local_dem': local_patches,
                'regional_dem': regional_patches,
                'month_encoding': month_encodings
            }
            
            # Make predictions
            predictions = model.predict(X)
            
            # Reshape predictions
            predictions = predictions.flatten()
            
            # Visualize predictions
            output_path = visualize_predictions(predictions, grid_size, date_str, args.output_dir)
            
            # Calculate metrics
            mse = np.mean((actual_rainfall - predictions) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_rainfall - predictions))
            r2 = 1 - (np.sum((actual_rainfall - predictions) ** 2) / 
                      np.sum((actual_rainfall - np.mean(actual_rainfall)) ** 2))
            
            # Calculate normalized metrics
            mean_rainfall = np.mean(actual_rainfall)
            rmae = mae / mean_rainfall if mean_rainfall > 0 else float('nan')
            rrmse = rmse / mean_rainfall if mean_rainfall > 0 else float('nan')
            
            # Add to results
            results.append({
                'date': date_str,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'rrmse': rrmse,
                'rmae': rmae,
                'mean_actual': mean_rainfall,
                'mean_predicted': np.mean(predictions),
                'visualization_path': output_path
            })
            
            print(f"  RMSE: {rmse:.4f} mm")
            print(f"  MAE: {mae:.4f} mm")
            print(f"  R²: {r2:.4f}")
            print(f"  rRMSE: {rrmse:.4f}")
            print(f"  rMAE: {rmae:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_path = os.path.join(args.output_dir, 'prediction_metrics.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        # Print overall metrics
        print("\nOverall metrics:")
        print(f"  Mean RMSE: {results_df['rmse'].mean():.4f} mm")
        print(f"  Mean MAE: {results_df['mae'].mean():.4f} mm")
        print(f"  Mean R²: {results_df['r2'].mean():.4f}")
        print(f"  Mean rRMSE: {results_df['rrmse'].mean():.4f}")
        print(f"  Mean rMAE: {results_df['rmae'].mean():.4f}")
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
