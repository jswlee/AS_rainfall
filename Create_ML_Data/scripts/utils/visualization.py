"""
Visualization utilities for the American Samoa Rainfall Prediction project.

This module contains functions for visualizing various aspects of the rainfall prediction
pipeline, including DEM patches, interpolated rainfall, and other data visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_dem_patches(local_patches, regional_patches, grid_size, 
                          local_patch_km, regional_patch_km, output_dir):
    """
    Visualize DEM patches in a grid layout.
    
    Args:
        local_patches (list): List of local DEM patches
        regional_patches (list): List of regional DEM patches
        grid_size (int): Size of the grid (e.g., 5 for a 5x5 grid)
        local_patch_km (float): Size of local patches in km
        regional_patch_km (float): Size of regional patches in km
        output_dir (str): Directory to save the visualization
        
    Returns:
        str: Path to the saved visualization file
    """
    plt.figure(figsize=(14, 6))
    
    # Arrange local patches in a grid
    local_patches_array = np.array(local_patches)
    n_patches, local_patch_height, local_patch_width = local_patches_array.shape
    grid_rows, grid_cols = grid_size, grid_size
    
    # Create empty grid for local patches
    local_grid = np.zeros((grid_rows * local_patch_height, grid_cols * local_patch_width))
    
    # Place local patches in grid
    for i in range(min(n_patches, grid_rows * grid_cols)):
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate position in the grid
        row_start = row * local_patch_height
        row_end = (row + 1) * local_patch_height
        col_start = col * local_patch_width
        col_end = (col + 1) * local_patch_width
        
        # Place the patch
        local_grid[row_start:row_end, col_start:col_end] = local_patches_array[i]
    
    # Do the same for regional patches - but with their own dimensions
    regional_patches_array = np.array(regional_patches)
    _, regional_patch_height, regional_patch_width = regional_patches_array.shape
    
    # Create empty grid for regional patches with the regional patch dimensions
    regional_grid = np.zeros((grid_rows * regional_patch_height, grid_cols * regional_patch_width))
    
    # Place regional patches in grid
    for i in range(min(n_patches, grid_rows * grid_cols)):
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate position in the grid using regional patch dimensions
        row_start = row * regional_patch_height
        row_end = (row + 1) * regional_patch_height
        col_start = col * regional_patch_width
        col_end = (col + 1) * regional_patch_width
        
        # Place the patch
        regional_grid[row_start:row_end, col_start:col_end] = regional_patches_array[i]
    
    # Plot local patches grid
    plt.subplot(121)
    plt.imshow(local_grid, cmap='terrain', origin='lower')
    plt.title(f'Local DEM Patches ({grid_size}x{grid_size} grid, {local_patch_km}km each)')
    plt.colorbar()
    
    # Plot regional patches grid
    plt.subplot(122)
    plt.imshow(regional_grid, cmap='terrain', origin='lower')
    plt.title(f'Regional DEM Patches ({grid_size}x{grid_size} grid, {regional_patch_km}km each)')
    plt.colorbar()
    
    plt.tight_layout()
    
    # Create figures directory inside the output directory
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    dem_patches_path = os.path.join(figures_dir, 'dem_patches.png')
    plt.savefig(dem_patches_path)
    
    return dem_patches_path


def visualize_interpolated_rainfall(grid_points, interpolated_rainfall, sample_date, output_dir):
    """
    Visualize interpolated rainfall for a specific date using a grid-based style.
    
    Args:
        grid_points (list): List of grid point coordinates (lon, lat)
        interpolated_rainfall (dict): Dictionary mapping date strings to rainfall values
        sample_date (str): Date string for the sample to visualize
        output_dir (str): Directory to save the visualization
        
    Returns:
        str: Path to the saved visualization file
    """
    sample_rainfall = interpolated_rainfall[sample_date]
    
    # Determine grid dimensions
    lons = [p[0] for p in grid_points]
    lats = [p[1] for p in grid_points]
    unique_lons = sorted(set(lons))
    unique_lats = sorted(set(lats), reverse=True)  # Reverse to match image orientation
    
    # Create a 2D grid for rainfall data
    grid_size = int(len(grid_points) ** 0.5)  # Assuming square grid
    rainfall_grid = np.zeros((grid_size, grid_size))
    
    # Fill the grid with rainfall values
    for i, point in enumerate(grid_points):
        lon, lat = point
        x = unique_lons.index(lon)
        y = unique_lats.index(lat)
        rainfall_grid[y, x] = sample_rainfall[i]
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(rainfall_grid, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Rainfall (mm)')
    plt.title(f'Interpolated Rainfall for {sample_date}')
    
    # Add longitude and latitude labels
    plt.xticks(range(len(unique_lons)), [f'{lon:.2f}' for lon in unique_lons], rotation=45)
    plt.yticks(range(len(unique_lats)), [f'{lat:.2f}' for lat in unique_lats])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Create figures directory inside the output directory
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    rainfall_plot_path = os.path.join(figures_dir, f'interpolated_rainfall_{sample_date}.png')
    plt.savefig(rainfall_plot_path)
    
    return rainfall_plot_path
