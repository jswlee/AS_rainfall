"""
Gaussian Process Utilities for Rainfall Interpolation

This module provides functions for Gaussian Process-based interpolation
of rainfall data, which produces smoother and more realistic rainfall fields
compared to simpler interpolation methods, especially with sparse data.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_gp_model(optimize=True):
    """
    Create a Gaussian Process model for rainfall interpolation.
    
    Parameters
    ----------
    optimize : bool, optional
        Whether to optimize hyperparameters during fitting
        
    Returns
    -------
    GaussianProcessRegressor
        Configured GP model
    """
    # Matern kernel is often good for spatial data as it's less smooth than RBF
    # The length_scale parameter controls how quickly correlation decreases with distance
    # nu=1.5 provides a reasonable balance of smoothness
    kernel = 1.0 * Matern(length_scale=0.1, nu=1.5)
    
    # Add a white kernel to account for noise in observations
    kernel = kernel + WhiteKernel(noise_level=0.1)
    
    # Create the GP model
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,  # Small alpha to avoid numerical issues
        optimizer='fmin_l_bfgs_b',  # Standard optimizer
        n_restarts_optimizer=5 if optimize else 0,  # Multiple restarts to avoid local minima
        normalize_y=True,  # Normalize target values
        random_state=42  # For reproducibility
    )
    
    return gp

def gp_interpolate(station_locs, station_values, grid_points, optimize=True):
    """
    Interpolate rainfall using Gaussian Process regression.
    
    Parameters
    ----------
    station_locs : list or array
        List of (lon, lat) coordinates for stations
    station_values : list or array
        Rainfall values at each station
    grid_points : list or array
        List of (lon, lat) coordinates for grid points to interpolate to
    optimize : bool, optional
        Whether to optimize GP hyperparameters
        
    Returns
    -------
    tuple
        (mean_predictions, std_predictions) - mean and standard deviation
        of the GP predictions at each grid point
    """
    # Convert inputs to numpy arrays
    X_train = np.array(station_locs)
    y_train = np.array(station_values)
    X_test = np.array(grid_points)
    
    # Check for valid inputs
    if len(X_train) == 0 or len(y_train) == 0:
        print("No rainfall data points available for GP interpolation")
        return np.zeros(len(X_test)), np.zeros(len(X_test))
    
    # Handle NaN values
    mask = ~np.isnan(y_train)
    if not np.all(mask):
        print("WARNING: Input rainfall data contains NaN values. Removing them.")
        X_train = X_train[mask]
        y_train = y_train[mask]
    
    # If we have too few points, fall back to simpler methods
    if len(X_train) < 3:
        print(f"Only {len(X_train)} valid rainfall data points. GP requires at least 3 points.")
        if len(X_train) == 1:
            # With one station, use the same value everywhere
            return np.full(len(X_test), y_train[0]), np.zeros(len(X_test))
        elif len(X_train) == 2:
            # With two stations, use a very simple GP with fixed parameters
            kernel = 1.0 * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
            gp.fit(X_train, y_train)
            mean, std = gp.predict(X_test, return_std=True)
            return mean, std
    
    try:
        # Create and fit the GP model
        gp = create_gp_model(optimize=optimize)
        gp.fit(X_train, y_train)
        
        # Make predictions
        mean, std = gp.predict(X_test, return_std=True)
        
        # Ensure non-negative predictions
        mean = np.maximum(mean, 0)
        
        return mean, std
        
    except Exception as e:
        print(f"Error in GP interpolation: {e}")
        # Return zeros as fallback
        return np.zeros(len(X_test)), np.zeros(len(X_test))

def tune_gp_hyperparams(station_locs, station_values, cv=5):
    """
    Tune GP hyperparameters using cross-validation.
    
    Parameters
    ----------
    station_locs : list or array
        List of (lon, lat) coordinates for stations
    station_values : list or array
        Rainfall values at each station
    cv : int, optional
        Number of cross-validation folds
        
    Returns
    -------
    dict
        Best hyperparameters
    """
    # Convert inputs to numpy arrays
    X = np.array(station_locs)
    y = np.array(station_values)
    
    # Check if we have enough data for cross-validation
    if len(X) < cv:
        print(f"Not enough data points ({len(X)}) for {cv}-fold CV. Using default hyperparameters.")
        return None
    
    # Define parameter grid to search
    param_grid = {
        "kernel": [
            1.0 * Matern(length_scale=0.1, nu=0.5) + WhiteKernel(noise_level=0.1),
            1.0 * Matern(length_scale=0.1, nu=1.5) + WhiteKernel(noise_level=0.1),
            1.0 * Matern(length_scale=0.1, nu=2.5) + WhiteKernel(noise_level=0.1),
            1.0 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
        ]
    }
    
    # Create base GP model
    gp = GaussianProcessRegressor(
        alpha=1e-10,
        n_restarts_optimizer=2,
        normalize_y=True,
        random_state=42
    )
    
    # Set up grid search
    grid_search = GridSearchCV(
        gp, 
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=0,
        n_jobs=-1  # Use all available cores
    )
    
    try:
        # Perform grid search
        grid_search.fit(X, y)
        print(f"Best GP parameters: {grid_search.best_params_}")
        return grid_search.best_params_
    except Exception as e:
        print(f"Error in GP hyperparameter tuning: {e}")
        return None

def visualize_gp_interpolation(station_locs, station_values, grid_points, mean_pred, std_pred, 
                              title="Gaussian Process Interpolation", output_path=None):
    """
    Visualize GP interpolation results.
    
    Parameters
    ----------
    station_locs : list or array
        List of (lon, lat) coordinates for stations
    station_values : list or array
        Rainfall values at each station
    grid_points : list or array
        List of (lon, lat) coordinates for grid points
    mean_pred : array
        Mean predictions at grid points
    std_pred : array
        Standard deviation of predictions at grid points
    title : str, optional
        Plot title
    output_path : str, optional
        Path to save the visualization
    """
    # Convert inputs to numpy arrays
    stations = np.array(station_locs)
    values = np.array(station_values)
    grid = np.array(grid_points)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot mean predictions
    sc1 = ax1.scatter(grid[:, 0], grid[:, 1], c=mean_pred, cmap='Blues', 
                     s=50, edgecolor='none', alpha=0.8)
    ax1.scatter(stations[:, 0], stations[:, 1], c=values, cmap='Blues',
               s=100, edgecolor='black', marker='o')
    ax1.set_title(f"{title} - Mean")
    plt.colorbar(sc1, ax=ax1, label='Rainfall (inches)')
    
    # Plot uncertainty (standard deviation)
    sc2 = ax2.scatter(grid[:, 0], grid[:, 1], c=std_pred, cmap='Reds', 
                     s=50, edgecolor='none', alpha=0.8)
    ax2.scatter(stations[:, 0], stations[:, 1], c='black',
               s=100, edgecolor='black', marker='o')
    ax2.set_title(f"{title} - Uncertainty (Std Dev)")
    plt.colorbar(sc2, ax=ax2, label='Std Dev (inches)')
    
    # Set common labels and adjust layout
    fig.text(0.5, 0.04, 'Longitude', ha='center')
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical')
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
