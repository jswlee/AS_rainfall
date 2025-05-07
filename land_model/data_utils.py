#!/usr/bin/env python3
"""
Data preprocessing and utility functions for the LAND-inspired rainfall prediction model.
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def load_and_reshape_data(features_path, targets_path, test_indices_path=None, 
                          test_size=0.1, val_size=0.1, random_state=None):
    """
    Load data from CSV files, reshape into appropriate format for the LAND model,
    and split into train, validation, and test sets.
    
    Parameters
    ----------
    features_path : str
        Path to features CSV file
    targets_path : str
        Path to targets CSV file
    test_indices_path : str, optional
        Path to save or load test indices
    test_size : float, optional
        Fraction of data to use for testing
    val_size : float, optional
        Fraction of training data to use for validation
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing all data splits and metadata
    """
    # Load features and targets
    print(f"Loading features from {features_path}...")
    features_df = pd.read_csv(features_path)
    
    print(f"Loading targets from {targets_path}...")
    targets_df = pd.read_csv(targets_path)
    
    # Extract rainfall values and scale as per LAND model (divide by 100)
    y = targets_df['rainfall'].values / 100.0
    
    # Identify column groups
    climate_cols = [col for col in features_df.columns if col.startswith('climate_')]
    local_dem_cols = [col for col in features_df.columns if col.startswith('local_dem_')]
    regional_dem_cols = [col for col in features_df.columns if col.startswith('regional_dem_')]
    month_cols = [col for col in features_df.columns if col.startswith('month_')]
    
    # Extract and reshape data
    climate_data = features_df[climate_cols].values
    local_dem_data = features_df[local_dem_cols].values / 4000.0  # Scale as per LAND model
    regional_dem_data = features_df[regional_dem_cols].values / 4000.0  # Scale as per LAND model
    month_data = features_df[month_cols].values
    
    # Reshape local DEM data to 3x3 grid (if it isn't already)
    if local_dem_data.shape[1] == 9:  # Assuming 9 values for a 3x3 grid
        local_dem_data = local_dem_data.reshape(-1, 3, 3)
    
    # Reshape regional DEM data to 5x5 grid (if it isn't already)
    # For now, we'll keep it as 3x3 to match the local DEM
    if regional_dem_data.shape[1] == 9:  # Assuming 9 values for a 3x3 grid
        regional_dem_data = regional_dem_data.reshape(-1, 3, 3)
    
    # Reshape climate data to match LAND model's reanalysis data format
    # Instead of duplicating values, we'll keep the original values and reshape to 3D
    num_climate_vars = len(climate_cols)
    # Normalize climate data to improve training
    climate_data = climate_data / np.max(np.abs(climate_data), axis=0)
    climate_data_reshaped = climate_data.reshape(-1, num_climate_vars, 1, 1)
    # Repeat the values to create a 3x3 grid (but with identical values)
    climate_data_reshaped = np.tile(climate_data_reshaped, (1, 1, 3, 3))
    
    # Check if we should load existing test indices
    if test_indices_path and os.path.exists(test_indices_path):
        print(f"Loading existing test indices from {test_indices_path}")
        with open(test_indices_path, 'rb') as f:
            test_indices = pickle.load(f)
        
        # Create mask for train+val indices
        mask = np.ones(len(y), dtype=bool)
        mask[test_indices] = False
        
        # Split data
        train_val_indices = np.where(mask)[0]
        test_indices = np.where(~mask)[0]
        
        # Further split train_val into train and validation
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=val_size/(1-test_size),
            random_state=random_state
        )
    else:
        # Split data into train+val and test
        train_val_indices, test_indices = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=random_state
        )
        
        # Further split train_val into train and validation
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=val_size/(1-test_size),
            random_state=random_state
        )
        
        # Save test indices if path is provided
        if test_indices_path:
            os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
            with open(test_indices_path, 'wb') as f:
                pickle.dump(test_indices, f)
            print(f"Test indices saved to {test_indices_path}")
    
    # Create data dictionary with all splits
    data = {
        'climate': {
            'train': climate_data_reshaped[train_indices],
            'val': climate_data_reshaped[val_indices],
            'test': climate_data_reshaped[test_indices],
            'shape': climate_data_reshaped.shape[1:]  # (channels, height, width)
        },
        'local_dem': {
            'train': local_dem_data[train_indices],
            'val': local_dem_data[val_indices],
            'test': local_dem_data[test_indices],
            'shape': local_dem_data.shape[1:]  # (height, width)
        },
        'regional_dem': {
            'train': regional_dem_data[train_indices],
            'val': regional_dem_data[val_indices],
            'test': regional_dem_data[test_indices],
            'shape': regional_dem_data.shape[1:]  # (height, width)
        },
        'month': {
            'train': month_data[train_indices],
            'val': month_data[val_indices],
            'test': month_data[test_indices],
            'shape': month_data.shape[1:]  # (num_months,)
        },
        'targets': {
            'train': y[train_indices],
            'val': y[val_indices],
            'test': y[test_indices]
        },
        'metadata': {
            'num_climate_vars': num_climate_vars,
            'num_month_encodings': month_data.shape[1],
            'local_dem_shape': local_dem_data.shape[1:],
            'regional_dem_shape': regional_dem_data.shape[1:],
            'climate_shape': climate_data_reshaped.shape[1:],
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices)
        }
    }
    
    print(f"Train set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")
    
    return data


def create_tf_dataset(data, batch_size=32, shuffle=True, buffer_size=10000):
    """
    Create TensorFlow datasets for training, validation, and testing.
    
    Parameters
    ----------
    data : dict
        Dictionary containing data splits from load_and_reshape_data
    batch_size : int, optional
        Batch size for training
    shuffle : bool, optional
        Whether to shuffle the training data
    buffer_size : int, optional
        Buffer size for shuffling
        
    Returns
    -------
    dict
        Dictionary containing TensorFlow datasets
    """
    import tensorflow as tf
    
    def create_dataset(split):
        # Create dataset with all inputs
        ds = tf.data.Dataset.from_tensor_slices((
            {
                'climate': data['climate'][split],
                'local_dem': data['local_dem'][split],
                'regional_dem': data['regional_dem'][split],
                'month': data['month'][split]
            },
            data['targets'][split]
        ))
        
        # Apply shuffling to training data
        if shuffle and split == 'train':
            ds = ds.shuffle(buffer_size)
        
        # Batch and prefetch
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    # Create datasets for each split
    datasets = {
        'train': create_dataset('train'),
        'val': create_dataset('val'),
        'test': create_dataset('test')
    }
    
    return datasets
