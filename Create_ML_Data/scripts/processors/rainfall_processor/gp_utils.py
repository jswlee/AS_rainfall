"""
Spatiotemporal Gaussian Process Utilities for Rainfall Interpolation

This module provides a simplified spatiotemporal GP interpolation approach
that follows the methodology described in research papers, including
log transformation and zero clipping for improved rainfall predictions.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from datetime import datetime
import pandas as pd

def parse_date_to_numeric(date_str):
    """
    Convert date string to numeric value for temporal modeling.
    
    Parameters
    ----------
    date_str : str
        Date string in format 'YYYY-MM' or similar
        
    Returns
    -------
    float
        Numeric representation of date (years since reference)
    """
    try:
        # Handle different date formats
        if '-' in date_str:
            year, month = date_str.split('-')
        elif '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:
                month, day, year = parts
            else:
                month, year = parts
        else:
            # Assume it's already a year
            year = date_str
            month = '1'
        
        # Convert to decimal year (e.g., 1980.5 for July 1980)
        year_val = float(year)
        month_val = float(month)
        return year_val + (month_val - 1) / 12.0
    except:
        # Fallback: use hash of string for consistent numeric value
        return hash(date_str) % 10000

def spatiotemporal_gp_interpolate(all_station_data, prediction_points, prediction_times):
    """
    Spatiotemporal Gaussian Process interpolation with log transformation and zero clipping.
    
    Parameters
    ----------
    all_station_data : dict
        Dictionary where keys are date strings and values are dicts with
        'locations' (list of (lon, lat)) and 'values' (list of rainfall values)
    prediction_points : list
        List of (lon, lat) coordinates for spatial prediction
    prediction_times : list
        List of date strings for temporal prediction
        
    Returns
    -------
    dict
        Dictionary with date strings as keys and predicted rainfall arrays as values
    """
    # Collect all spatiotemporal training data
    X_train = []  # (lon, lat, time)
    y_train = []  # rainfall values
    
    for date_str, data in all_station_data.items():
        if not data['locations'] or not data['values']:
            continue
            
        time_val = parse_date_to_numeric(date_str)
        
        for (lon, lat), value in zip(data['locations'], data['values']):
            if not np.isnan(value) and value >= 0:
                X_train.append([lon, lat, time_val])
                y_train.append(value)
    
    if len(X_train) < 3:
        print(f"Insufficient data for spatiotemporal GP: {len(X_train)} points")
        # Return zeros for all predictions
        return {date: np.zeros(len(prediction_points)) for date in prediction_times}
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Apply log transformation as described in the paper: y = log(y + 1)
    y_train_log = np.log(y_train + 1)
    
    # Create spatiotemporal kernel with proper dimensionality
    # Use a single kernel that handles all 3 dimensions: lon, lat, time
    # Use more balanced length scales to avoid vertical/horizontal artifacts
    # Increase spatial length scales for smoother interpolation
    kernel = Matern(length_scale=[0.3, 0.3, 0.5], nu=2.5) + WhiteKernel(noise_level=0.05)
    
    # Create and fit GP model
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        random_state=42
    )
    
    try:
        gp.fit(X_train, y_train_log)
        print(f"Fitted spatiotemporal GP with {len(X_train)} training points")
    except Exception as e:
        print(f"Error fitting spatiotemporal GP: {e}")
        return {date: np.zeros(len(prediction_points)) for date in prediction_times}
    
    # Make predictions for each time point
    predictions = {}
    
    for date_str in prediction_times:
        time_val = parse_date_to_numeric(date_str)
        
        # Create prediction points with time dimension
        X_pred = np.array([[lon, lat, time_val] for lon, lat in prediction_points])
        
        try:
            # Get predictions in log space
            y_pred_log, y_std_log = gp.predict(X_pred, return_std=True)
            
            # Transform back from log space: exp(log(y+1)) - 1 = y
            y_pred = np.exp(y_pred_log) - 1
            
            # Apply zero clipping as described in the paper
            y_pred = np.maximum(y_pred, 0.0)
            
            predictions[date_str] = y_pred
            
        except Exception as e:
            print(f"Error predicting for {date_str}: {e}")
            predictions[date_str] = np.zeros(len(prediction_points))
    
    return predictions

def gp_interpolate(station_locs, station_values, grid_points, optimize=True):
    """
    Simple spatial-only GP interpolation for backward compatibility.
    
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
        (mean_predictions, std_predictions)
    """
    X_train = np.array(station_locs)
    y_train = np.array(station_values)
    X_test = np.array(grid_points)
    
    if len(X_train) == 0 or len(y_train) == 0:
        return np.zeros(len(X_test)), np.zeros(len(X_test))
    
    # Remove NaN values
    mask = ~np.isnan(y_train)
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    if len(X_train) < 2:
        if len(X_train) == 1:
            return np.full(len(X_test), y_train[0]), np.zeros(len(X_test))
        else:
            return np.zeros(len(X_test)), np.zeros(len(X_test))
    
    # Apply log transformation
    y_train_log = np.log(y_train + 1)
    
    # Simple spatial kernel - use similar parameters to spatiotemporal for consistency
    kernel = Matern(length_scale=[0.3, 0.3], nu=2.5) + WhiteKernel(noise_level=0.05)
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        random_state=42
    )
    
    try:
        gp.fit(X_train, y_train_log)
        y_pred_log, y_std_log = gp.predict(X_test, return_std=True)
        
        # Transform back and apply zero clipping
        y_pred = np.maximum(np.exp(y_pred_log) - 1, 0.0)
        y_std = y_std_log  # Keep std in log space for now
        
        return y_pred, y_std
        
    except Exception as e:
        print(f"Error in GP interpolation: {e}")
        return np.zeros(len(X_test)), np.zeros(len(X_test))
