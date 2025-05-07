#!/usr/bin/env python3
"""
Convert H5 rainfall prediction data to CSV format.

This script extracts data from the H5 file and saves it as CSV files:
- features.csv: Contains all input features (climate variables, DEM data, month encoding)
- targets.csv: Contains the rainfall target values
- metadata.csv: Contains metadata about the features (names, shapes, etc.)
"""

import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_data_from_h5(h5_path):
    """
    Extract data from H5 file.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 file containing the processed data
        
    Returns
    -------
    tuple
        (features_df, targets_df, metadata_dict)
    """
    # Load data from H5 file
    with h5py.File(h5_path, 'r') as h5_file:
        # Get all date keys
        date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
        
        # Initialize lists to store data
        climate_vars_list = []
        local_patches_list = []
        regional_patches_list = []
        month_encodings_list = []
        rainfall_list = []
        date_indices = []
        
        # Extract data for each date
        for date_key in date_keys:
            date_group = h5_file[date_key]
            
            # Skip if any required data is missing
            if not all(key in date_group for key in 
                       ['climate_vars', 'local_patches', 'regional_patches', 'month_one_hot', 'rainfall']):
                continue
            
            # Get date index (e.g., 'date_2020_01' -> '2020_01')
            date_idx = date_key.replace('date_', '')
            
            # Extract climate variables
            climate_data = []
            climate_var_names = []
            for var_name in date_group['climate_vars']:
                var_data = date_group['climate_vars'][var_name][:]
                climate_data.append(var_data)
                climate_var_names.append(var_name)
            
            # Stack climate variables
            climate_vars = np.column_stack(climate_data)
            
            # Extract other data
            local_patches = date_group['local_patches'][:]
            regional_patches = date_group['regional_patches'][:]
            month_encoding = date_group['month_one_hot'][:]
            rainfall = date_group['rainfall'][:]
            
            # Repeat month encoding for each grid point
            month_encoding = np.tile(month_encoding, (len(rainfall), 1))
            
            # Repeat date index for each grid point
            date_indices.extend([date_idx] * len(rainfall))
            
            # Append to lists
            climate_vars_list.append(climate_vars)
            local_patches_list.append(local_patches)
            regional_patches_list.append(regional_patches)
            month_encodings_list.append(month_encoding)
            rainfall_list.append(rainfall)
    
    # Concatenate data from all dates
    climate_vars = np.vstack(climate_vars_list)
    local_patches = np.vstack(local_patches_list)
    regional_patches = np.vstack(regional_patches_list)
    month_encodings = np.vstack(month_encodings_list)
    rainfall = np.vstack(rainfall_list).reshape(-1)  # Flatten to 1D array
    
    # Create metadata dictionary
    metadata = {
        'climate_var_names': climate_var_names,
        'climate_var_count': len(climate_var_names),
        'local_patch_shape': local_patches.shape[1:],
        'regional_patch_shape': regional_patches.shape[1:],
        'month_encoding_shape': month_encodings.shape[1],
        'total_samples': len(rainfall)
    }
    
    # Prepare feature data for CSV
    # Flatten the local and regional patches
    local_patches_flat = local_patches.reshape(local_patches.shape[0], -1)
    regional_patches_flat = regional_patches.reshape(regional_patches.shape[0], -1)
    
    # Create column names
    climate_cols = [f'climate_{name}' for name in climate_var_names]
    local_dem_cols = [f'local_dem_{i}' for i in range(local_patches_flat.shape[1])]
    regional_dem_cols = [f'regional_dem_{i}' for i in range(regional_patches_flat.shape[1])]
    month_cols = [f'month_{i}' for i in range(month_encodings.shape[1])]
    
    # Create DataFrames
    features_df = pd.DataFrame(climate_vars, columns=climate_cols)
    
    # Add local DEM patches
    for i in range(local_patches_flat.shape[1]):
        features_df[local_dem_cols[i]] = local_patches_flat[:, i]
    
    # Add regional DEM patches
    for i in range(regional_patches_flat.shape[1]):
        features_df[regional_dem_cols[i]] = regional_patches_flat[:, i]
    
    # Add month encoding
    for i in range(month_encodings.shape[1]):
        features_df[month_cols[i]] = month_encodings[:, i]
    
    # Add date index
    features_df['date'] = date_indices
    
    # Create targets DataFrame
    targets_df = pd.DataFrame({'rainfall': rainfall, 'date': date_indices})
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame([{
        'climate_var_names': ','.join(climate_var_names),
        'climate_var_count': len(climate_var_names),
        'local_patch_shape': f'{local_patches.shape[1]}x{local_patches.shape[2]}',
        'regional_patch_shape': f'{regional_patches.shape[1]}x{regional_patches.shape[2]}',
        'month_encoding_shape': month_encodings.shape[1],
        'total_samples': len(rainfall),
        'creation_date': datetime.now().isoformat()
    }])
    
    return features_df, targets_df, metadata_df


def main():
    parser = argparse.ArgumentParser(description='Convert H5 rainfall prediction data to CSV format')
    parser.add_argument('--h5_path', type=str, default='output/rainfall_prediction_data.h5',
                        help='Path to H5 file with processed data')
    parser.add_argument('--output_dir', type=str, default='csv_data',
                        help='Directory to save CSV files')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract data from H5 file
    print(f"Extracting data from {args.h5_path}...")
    features_df, targets_df, metadata_df = extract_data_from_h5(args.h5_path)
    
    # Save to CSV
    features_path = os.path.join(args.output_dir, 'features.csv')
    targets_path = os.path.join(args.output_dir, 'targets.csv')
    metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    
    print(f"Saving features to {features_path}...")
    features_df.to_csv(features_path, index=False)
    
    print(f"Saving targets to {targets_path}...")
    targets_df.to_csv(targets_path, index=False)
    
    print(f"Saving metadata to {metadata_path}...")
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"\nConversion complete. CSV files saved to {args.output_dir}")
    print(f"Total samples: {len(targets_df)}")
    print(f"Features shape: {features_df.shape}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
