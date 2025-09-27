#!/usr/bin/env python3
"""
PyTorch data utilities for the assembled NPZ data.

This module provides:
- RainfallDataset: PyTorch Dataset for multi-modal rainfall prediction
- Data loading functions with train/val/test splitting
- DataLoader creation with optimized settings
- Test indices management for reproducible experiments
"""

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from pprint import pprint


# ================================================================
# PyTorch Dataset Implementation
# ================================================================

class RainfallDataset(Dataset):
    """
    PyTorch Dataset for rainfall prediction using climate, DEM, and temporal features.
    
    This dataset handles:
    - Climate/reanalysis patches (16 variables on 3x3 grid)
    - Local and regional DEM patches 
    - Month one-hot encodings
    - Rainfall targets
    """
    
    def __init__(self, climate_data: np.ndarray, local_dem_data: np.ndarray, 
                 regional_dem_data: np.ndarray, month_data: np.ndarray, 
                 targets: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            climate_data: Climate/reanalysis patches (N, 16, 3, 3)
            local_dem_data: Local DEM patches (N, H, W)
            regional_dem_data: Regional DEM patches (N, H, W)
            month_data: Month one-hot encodings (N, 12)
            targets: Rainfall targets (N,)
        """
        self.climate_data = torch.FloatTensor(data=climate_data)
        self.local_dem_data = torch.FloatTensor(data=local_dem_data)
        self.regional_dem_data = torch.FloatTensor(data=regional_dem_data)
        self.month_data = torch.FloatTensor(data=month_data)
        self.targets = torch.FloatTensor(data=targets)
        self.length = len(self.targets)
        
        # Verify all arrays have the same length
        lengths = [len(self.climate_data), len(self.local_dem_data), 
                  len(self.regional_dem_data), len(self.month_data), len(self.targets)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError(f"All arrays must have the same length, got: {lengths}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (features_dict, target) where features_dict contains:
            - 'climate': Climate patches (16, 3, 3)
            - 'local_dem': Local DEM patch (H, W)  
            - 'regional_dem': Regional DEM patch (H, W)
            - 'month': Month one-hot encoding (12,)
        """
        features = {
            'climate': self.climate_data[idx],
            'local_dem': self.local_dem_data[idx],
            'regional_dem': self.regional_dem_data[idx],
            'month': self.month_data[idx]
        }
        target = self.targets[idx]
        
        return features, target


# ================================================================
# Data Loading and Splitting Functions
# ================================================================

def load_assembled_npz_data_pytorch(npz_path: str,
                                   test_indices_path: Optional[str] = None,
                                   test_size: float = 0.1,
                                   val_size: float = 0.1,
                                   random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Load assembled NPZ data and return PyTorch-compatible data structure.
    
    Args:
        npz_path: Path to the assembled NPZ file
        test_indices_path: Path to save/load test indices for reproducibility
        test_size: Fraction of data for testing
        val_size: Fraction of remaining data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing PyTorch datasets and metadata
    """
    # ----------------------------------------------------------------
    # Load NPZ Data and Validate Structure
    # ----------------------------------------------------------------
    print(f"Loading NPZ data from {npz_path}...")
    z = np.load(str(npz_path), allow_pickle=True)
    
    # ----------------------------------------------------------------
    # Extract Climate/Reanalysis Data
    # ----------------------------------------------------------------
    if 'reanalysis_patches' not in z.files:
        raise KeyError("NPZ missing 'reanalysis_patches'")
    climate = z['reanalysis_patches'].astype(np.float32)
    
    # ----------------------------------------------------------------
    # Extract DEM Data (Local and Regional)
    # ----------------------------------------------------------------
    required_local = 'dem_local_divstd'
    required_regional = 'dem_regional_divstd'
    if required_local not in z.files or required_regional not in z.files:
        raise KeyError(
            f"NPZ missing required DEM keys '{required_local}' and/or '{required_regional}'. "
            f"Keys present: {sorted(z.files)}"
        )
    local_dem = z[required_local].astype(np.float32)
    regional_dem = z[required_regional].astype(np.float32)
    
    # ----------------------------------------------------------------
    # Extract Temporal Features (Month One-Hot)
    # ----------------------------------------------------------------
    if "month_onehot" not in z.files:
        raise KeyError("NPZ missing 'month_onehot'")
    month = z["month_onehot"].astype(np.float32)
    
    # ----------------------------------------------------------------
    # Extract Target Variable (Rainfall)
    # ----------------------------------------------------------------
    y = z["rainfall_mm_divstd"].astype(np.float32)
    
    n = int(y.shape[0])
    if not (climate.shape[0] == local_dem.shape[0] == regional_dem.shape[0] == month.shape[0] == n):
        raise ValueError(
            f"Mismatched N: climate {climate.shape[0]}, local {local_dem.shape[0]}, "
            f"regional {regional_dem.shape[0]}, month {month.shape[0]}, y {n}"
        )
    
    print(f"Loaded {n} samples with shapes:")
    print(f"  Climate: {climate.shape}")
    print(f"  Local DEM: {local_dem.shape}")
    print(f"  Regional DEM: {regional_dem.shape}")
    print(f"  Month: {month.shape}")
    print(f"  Targets: {y.shape}")
    
    # ----------------------------------------------------------------
    # Train/Val/Test Splitting with Reproducible Test Indices
    # ----------------------------------------------------------------
    indices = np.arange(n)
    
    def _generate_and_save_test_indices():
        _, ti = train_test_split(indices, test_size=test_size, random_state=random_state)
        ti = np.unique(np.asarray(ti, dtype=np.int64))
        ti.sort()
        if test_indices_path:
            os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
            with open(test_indices_path, 'wb') as f:
                pickle.dump(ti, f)
        return ti
    
    test_indices = None
    if test_indices_path and os.path.exists(test_indices_path):
        try:
            with open(test_indices_path, 'rb') as f:
                loaded = pickle.load(f)
            ti = np.asarray(loaded, dtype=np.int64).ravel()
            valid = (ti.size > 0 and np.all(ti >= 0) and np.all(ti < n))
            if not valid:
                test_indices = _generate_and_save_test_indices()
            else:
                test_indices = np.unique(ti)
                test_indices.sort()
        except Exception:
            test_indices = _generate_and_save_test_indices()
    else:
        test_indices = _generate_and_save_test_indices()
    
    # ----------------------------------------------------------------
    # Create Train/Validation Split from Remaining Data
    # ----------------------------------------------------------------
    train_val_indices = np.setdiff1d(indices, test_indices)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=random_state
    )
    
    print(f"Data splits:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    
    # ----------------------------------------------------------------
    # Extract Metadata for Denormalization
    # ----------------------------------------------------------------
    rainfall_mm_std = float(z['rainfall_mm_std']) if 'rainfall_mm_std' in z.files else 1.0
    
    # ----------------------------------------------------------------
    # Create PyTorch Datasets for Each Split
    # ----------------------------------------------------------------
    indices_dict = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }
    datasets = {
        name: RainfallDataset(
            climate[idx],
            local_dem[idx],
            regional_dem[idx],
            month[idx],
            y[idx],
        )
        for name, idx in indices_dict.items()
    }
    
    # Prepare metadata
    metadata = {
        'num_climate_vars': int(climate.shape[1]),
        'num_month_encodings': int(month.shape[1]),
        'local_dem_shape': tuple(local_dem.shape[1:]),
        'regional_dem_shape': tuple(regional_dem.shape[1:]),
        'climate_shape': tuple(climate.shape[1:]),
        'train_size': int(len(train_indices)),
        'val_size': int(len(val_indices)),
        'test_size': int(len(test_indices)),
        'rainfall_mm_std': float(rainfall_mm_std),
    }
    
    return {
        'datasets': datasets,
        'metadata': metadata,
        'indices': {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    }


# ================================================================
# DataLoader Creation Functions
# ================================================================

def create_pytorch_dataloaders(datasets: Dict[str, Dataset], 
                              batch_size: int = 32,
                              shuffle_train: bool = True,
                              num_workers: int = 0,
                              pin_memory: bool = False,
                              persistent_workers: Optional[bool] = None,
                              prefetch_factor: Optional[int] = None) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from datasets.
    
    Args:
        datasets: Dictionary with 'train', 'val', 'test' datasets
        batch_size: Batch size for all dataloaders
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}
    
    for split, dataset in datasets.items():
        shuffle = shuffle_train if split == 'train' else False
        
        # Build DataLoader kwargs safely to support different PyTorch versions
        dl_kwargs: Dict[str, Any] = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': False,
        }

        # persistent_workers is only meaningful when num_workers > 0
        # If not provided, default to (num_workers > 0) to preserve prior behavior
        if num_workers > 0:
            dl_kwargs['persistent_workers'] = (num_workers > 0) if (persistent_workers is None) else bool(persistent_workers)
            # prefetch_factor is only supported when num_workers > 0 in PyTorch
            if prefetch_factor is not None:
                dl_kwargs['prefetch_factor'] = int(prefetch_factor)
        
        dataloaders[split] = DataLoader(**dl_kwargs)
    
    return dataloaders


# ================================================================
# Testing and Validation
# ================================================================

if __name__ == "__main__":
    # Test the data loading
    npz_path = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz')
    test_indices_path = os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl')
    
    if os.path.exists(npz_path):
        data = load_assembled_npz_data_pytorch(
            npz_path=npz_path,
            test_indices_path=test_indices_path,
            random_state=42
        )
        
        # Create dataloaders
        dataloaders = create_pytorch_dataloaders(data['datasets'], batch_size=32)
        
        # Test a sample batch
        features, targets = next(iter(dataloaders['train']))
        print(f"\nSample batch shapes:")
        for key, tensor in features.items():
            print(f"  {key}: {tensor.shape}")
        print(f"  targets: {targets.shape}")
        
        pprint(f"Metadata: {data['metadata']}")

        # Show a sample from each dataset
        print("\nSample from each dataset:")
        for split, dataset in data['datasets'].items():
            print(f"\n{split} dataset:")
            pprint(dataset[0])
    else:
        print(f"NPZ file not found at {npz_path}")
