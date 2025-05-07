#!/usr/bin/env python3
"""
Normalize CSV data for rainfall prediction model.

This script loads the CSV data, applies StandardScaler to numerical features,
and saves the normalized data to new CSV files.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def normalize_data(features_path, targets_path, output_dir):
    """
    Normalize the data using StandardScaler.
    
    Parameters
    ----------
    features_path : str
        Path to features CSV file
    targets_path : str
        Path to targets CSV file
    output_dir : str
        Directory to save normalized data
        
    Returns
    -------
    None
    """
    print(f"Loading features from {features_path}...")
    features_df = pd.read_csv(features_path)
    
    print(f"Loading targets from {targets_path}...")
    targets_df = pd.read_csv(targets_path)
    
    # Extract date column for reference (not used as a feature)
    date_column = features_df['date']
    
    # Identify column groups
    climate_cols = [col for col in features_df.columns if col.startswith('climate_')]
    local_dem_cols = [col for col in features_df.columns if col.startswith('local_dem_')]
    regional_dem_cols = [col for col in features_df.columns if col.startswith('regional_dem_')]
    month_cols = [col for col in features_df.columns if col.startswith('month_')]
    
    # Columns to normalize (all except month encoding and date)
    cols_to_normalize = climate_cols + local_dem_cols + regional_dem_cols
    
    # Create a copy of the features DataFrame
    normalized_features = features_df.copy()
    
    # Initialize and fit the scaler
    print("Normalizing features...")
    scaler = StandardScaler()
    normalized_features[cols_to_normalize] = scaler.fit_transform(features_df[cols_to_normalize])
    
    # Save the scaler for later use
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save normalized features
    normalized_features_path = os.path.join(output_dir, 'normalized_features.csv')
    normalized_features.to_csv(normalized_features_path, index=False)
    print(f"Normalized features saved to {normalized_features_path}")
    
    # Also save a version without the date column for direct model input
    features_for_model = normalized_features.drop(columns=['date'])
    features_for_model_path = os.path.join(output_dir, 'features_for_model.csv')
    features_for_model.to_csv(features_for_model_path, index=False)
    print(f"Model-ready features saved to {features_for_model_path}")
    
    # Save targets (no normalization needed)
    targets_for_model_path = os.path.join(output_dir, 'targets_for_model.csv')
    targets_df.to_csv(targets_for_model_path, index=False)
    print(f"Targets saved to {targets_for_model_path}")
    
    # Print some statistics
    print("\nData statistics:")
    print(f"Total samples: {len(normalized_features)}")
    print(f"Number of climate variables: {len(climate_cols)}")
    print(f"Number of local DEM features: {len(local_dem_cols)}")
    print(f"Number of regional DEM features: {len(regional_dem_cols)}")
    print(f"Number of month encoding features: {len(month_cols)}")
    
    # Print sample of normalized data
    print("\nSample of normalized data:")
    print(normalized_features[cols_to_normalize[:5]].head(3))
    
    return normalized_features, targets_df

def main():
    parser = argparse.ArgumentParser(description='Normalize CSV data for rainfall prediction model')
    parser.add_argument('--features', type=str, default='csv_data/features.csv',
                        help='Path to features CSV file')
    parser.add_argument('--targets', type=str, default='csv_data/targets.csv',
                        help='Path to targets CSV file')
    parser.add_argument('--output_dir', type=str, default='csv_data',
                        help='Directory to save normalized data')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Normalize the data
    normalize_data(args.features, args.targets, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
