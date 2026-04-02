"""
Utility Functions for ML Data Preprocessing

This module contains utility functions used across the ML data preprocessing pipeline.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt, atan2
from . import config


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth specified in decimal degrees.
    
    Args:
        lon1 (float): Longitude of point 1
        lat1 (float): Latitude of point 1
        lon2 (float): Longitude of point 2
        lat2 (float): Latitude of point 2
        
    Returns:
        float: Distance in kilometers
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    
    # Radius of earth in kilometers
    r = 6371.0
    return c * r


def find_nearest_point(target_lat, target_lon, lat_array, lon_array):
    """
    Find the index of the nearest point in a grid to a target point.
    
    Args:
        target_lat (float): Target latitude
        target_lon (float): Target longitude
        lat_array (numpy.ndarray): Array of latitudes
        lon_array (numpy.ndarray): Array of longitudes
        
    Returns:
        tuple: (lat_idx, lon_idx) indices of the nearest point
    """
    # Calculate distances to all points
    distances = np.zeros((len(lat_array), len(lon_array)))
    for i, lat in enumerate(lat_array):
        for j, lon in enumerate(lon_array):
            distances[i, j] = haversine(target_lon, target_lat, lon, lat)
    
    # Find the minimum distance
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return min_idx

def filter_outliers(array, threshold=3.0):
    """
    Filter outliers from an array using z-score method.
    
    Args:
        array (numpy.ndarray): Array to filter
        threshold (float): Z-score threshold for outlier detection
        
    Returns:
        numpy.ndarray: Array with outliers removed
    """
    z_scores = np.abs((array - np.mean(array)) / np.std(array))
    return array[z_scores < threshold]

def visualize_patches(patches, titles=None, cmap='viridis', save_path=None):
    """
    Visualize a list of patches.
    
    Args:
        patches (list): List of 2D arrays (patches)
        titles (list, optional): List of titles for each patch
        cmap (str): Colormap to use
        save_path (str, optional): Path to save the visualization
    """
    n_patches = len(patches)
    _, axes = plt.subplots(1, n_patches, figsize=(4 * n_patches, 4))
    
    if n_patches == 1:
        axes = [axes]
    
    for i, (patch, ax) in enumerate(zip(patches, axes)):
        im = ax.imshow(patch, cmap=cmap)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
 
def visualize_grid(patches, titles=None, cmap='viridis', n_cols=None, n_rows=None, suptitle=None, save_path=None):
    """
    Visualize a collection of 2D patches arranged in a grid.
    
    Args:
        patches (Union[list, dict]): List of 2D arrays or dict mapping title->2D array
        titles (list, optional): Explicit titles when patches is a list
        cmap (str): Colormap name
        n_cols (int, optional): Number of columns. If None, uses a square-ish layout
        n_rows (int, optional): Number of rows. If None, uses a square-ish layout
        suptitle (str, optional): Figure-level title
        save_path (str, optional): If provided, saves to path; otherwise displays
    """
    # Normalize input to an ordered list of (title, patch)
    if isinstance(patches, dict):
        keys = list(patches.keys())
        data = [(k, patches[k]) for k in keys]
    else:
        data = list(enumerate(patches))
    n = len(data)
    if n == 0:
        return
    
    # Determine grid size
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n)))
    if n_rows is None:
        n_rows = int(np.ceil(n / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot
    for i, ax in enumerate(axes):
        if i < n:
            title, patch = data[i]
            im = ax.imshow(patch, cmap=cmap)
            if isinstance(patches, dict):
                ax.set_title(str(title))
            elif titles and i < len(titles):
                ax.set_title(titles[i])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def discover_station_months(station_metadata):
    """
    Discover available (year, month) combos per station from rainfall CSVs.
    Only include rows where the rainfall value is present (non-empty, non-NaN).
    Returns a dict: station_name -> list[(year, month)]
    """
    station_months = {}
    for station_name in station_metadata.keys():
        csv_path = os.path.join(str(config.MONTHLY_RAINFALL_DATA_DIR), f"{station_name}_monthly.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            if 'year_month' not in df.columns:
                continue
            if 'monthly_total_precip_in' in df.columns:
                mask = (~df['monthly_total_precip_in'].isna()) & (df['monthly_total_precip_in'] != '')
                df = df[mask]
            ym = df['year_month'].astype(str).str.split('-', expand=True)
            if ym.shape[1] >= 2:
                df['year'] = ym[0].astype(int)
                df['month'] = ym[1].astype(int)
                pairs = [(int(y), int(m)) for y, m in zip(df['year'].tolist(), df['month'].tolist())]
                station_months[station_name] = sorted(list(set(pairs)))
        except Exception as e:
            print(f"Warning: Failed to parse rainfall CSV for {station_name}: {e}")
    return station_months

def discover_station_days(station_metadata, start_date=None, end_date=None):
    """
    Discover available (year, month, day) combos per station from daily rainfall CSVs.
    Only includes rows where the rainfall value is present and non-negative.
    
    Args:
        station_metadata (dict): Mapping of station names to metadata.
        start_date (str, optional): The start date for the time filter (e.g., '1980-01-01').
        end_date (str, optional): The end date for the time filter (e.g., '1984-12-31').

    Returns:
        dict: station_name -> list[(year, month, day)]
    """
    station_days = {}
    daily_rainfall_dir = config.DAILY_RAINFALL_DATA_DIR

    for station_name in station_metadata.keys():
        csv_path = daily_rainfall_dir + f"/{station_name}.csv"
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            df["datetime"] = pd.to_datetime(df["datetime"])

            # Filter the DataFrame to the specified date range if provided
            if start_date and end_date:
                mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
                df = df[mask]
                if df.empty:
                    # Print a diagnostic message if no data remains after filtering
                    print(f"  - No rainfall data for {station_name} within {start_date} to {end_date}")
                    continue

            # Filter for valid rainfall data
            if "precip_in" in df.columns:
                mask = df["precip_in"].notna() & (df["precip_in"] >= 0)
                df = df[mask]

            if not df.empty:
                pairs = [
                    (d.year, d.month, d.day)
                    for d in df["datetime"].tolist()
                ]
                station_days[station_name] = sorted(list(set(pairs)))
        except Exception as e:
            print(f"Warning: Failed to parse daily rainfall CSV for {station_name}: {e}")
    return station_days
