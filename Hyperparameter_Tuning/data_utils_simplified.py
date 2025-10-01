# data_utils_simplified.py

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split

# ================================================================
# Simplified and Memory-Efficient PyTorch Dataset
# ================================================================

class RainfallDataset(Dataset):
    """
    PyTorch Dataset that uses indices to select data from shared tensors.
    This avoids duplicating data in memory for train, val, and test splits.
    """
    def __init__(self, tensors: Dict[str, torch.Tensor], indices: np.ndarray):
        """
        Args:
            tensors: A dictionary of the full data tensors (e.g., 'climate', 'targets').
            indices: The array of indices that this dataset should expose.
        """
        self.tensors = tensors
        self.indices = indices
        # The feature keys are all keys in the tensor dict except for 'targets'
        self.feature_keys = [k for k in self.tensors.keys() if k != 'targets']

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Retrieves a single data point using the internal index mapping.
        """
        # Map the requested index to the actual index in the full tensors
        data_idx = self.indices[idx]
        
        features = {key: self.tensors[key][data_idx] for key in self.feature_keys}
        target = self.tensors['targets'][data_idx]
        
        return features, target

# ================================================================
# Encapsulated Data Loading and Management Class
# ================================================================

class DataManager:
    """

    Handles loading NPZ data, performing train/val/test splits,
    and creating PyTorch Datasets and DataLoaders.
    """
    def __init__(self, npz_path: str, test_size: float = 0.1, val_size: float = 0.1, 
                 random_state: int = 42, test_indices_path: str = None, **kwargs):
        """
        Initializes the DataManager by loading and splitting the data.
        """
        self.random_state = random_state
        self._load_npz(npz_path)
        self._split_data(test_size, val_size, test_indices_path)
        self._extract_metadata()
        print("DataManager initialized successfully.")

    def _load_npz(self, npz_path: str):
        """Loads all data arrays from the NPZ file into float32 tensors."""
        print(f"Loading and converting data from {npz_path}...")
        with np.load(npz_path) as z:
            # Define the mapping from expected keys to NPZ file keys
            key_map = {
                'climate': 'reanalysis_patches',
                'local_dem': 'dem_local_divstd',
                'regional_dem': 'dem_regional_divstd',
                'month': 'month_onehot',
                'targets': 'rainfall_mm_divstd'
            }
            self.tensors = {}
            for key, npz_key in key_map.items():
                if npz_key not in z.files:
                    raise KeyError(f"Required key '{npz_key}' not found in NPZ file.")
                self.tensors[key] = torch.from_numpy(z[npz_key].astype(np.float32))
            
            # Store original std for denormalization later if needed
            self.rainfall_mm_std = float(z.get('rainfall_mm_std', 1.0))
        
        print("Data loaded into tensors.")

    def _split_data(self, test_size: float, val_size: float, test_indices_path: str):
        """Manages reproducible train/val/test splits."""
        n_samples = len(self.tensors['targets'])
        all_indices = np.arange(n_samples)

        # 1. Determine test indices (load from file or create new)
        if test_indices_path and os.path.exists(test_indices_path):
            print(f"Loading test indices from {test_indices_path}")
            with open(test_indices_path, 'rb') as f:
                test_indices = pickle.load(f)
        else:
            print("Generating new test indices...")
            _, test_indices = train_test_split(all_indices, test_size=test_size, random_state=self.random_state)
            if test_indices_path:
                os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
                with open(test_indices_path, 'wb') as f:
                    pickle.dump(test_indices, f)
        
        # 2. Create train/val split from the remaining indices
        train_val_indices = np.setdiff1d(all_indices, test_indices)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=self.random_state
        )

        self.indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        print(f"Data splits created: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test.")
        
    def _extract_metadata(self):
        """Extracts metadata required by the model."""
        self.metadata = {
            'num_climate_vars': self.tensors['climate'].shape[1],
            'num_month_encodings': self.tensors['month'].shape[1],
            'local_dem_shape': tuple(self.tensors['local_dem'].shape[1:]),
            'regional_dem_shape': tuple(self.tensors['regional_dem'].shape[1:]),
            'climate_shape': tuple(self.tensors['climate'].shape[1:]),
            'rainfall_mm_std': self.rainfall_mm_std,
        }

    def get_datasets(self) -> Dict[str, RainfallDataset]:
        """Returns a dictionary of RainfallDataset objects for each split."""
        return {
            split: RainfallDataset(self.tensors, idx)
            for split, idx in self.indices.items()
        }

    def get_cv_tensors(self) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """
        Returns all tensors and indices needed for cross-validation (train + val).
        This is a helper for the hyperparameter tuning script.
        """
        cv_indices = np.concatenate([self.indices['train'], self.indices['val']])
        return self.tensors, cv_indices

# ================================================================
# DataLoader Creation
# ================================================================

def create_pytorch_dataloaders(datasets: Dict[str, Dataset], **kwargs) -> Dict[str, DataLoader]:
    """Creates PyTorch DataLoaders with optimized and safe settings."""
    dataloaders = {}
    # Sensible defaults that can be overridden by kwargs
    dl_defaults = {
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': False,
    }
    
    for split, dataset in datasets.items():
        # Update defaults with any user-provided settings
        split_kwargs = dl_defaults.copy()
        split_kwargs.update(kwargs)
        
        # Training loader should shuffle, others should not
        split_kwargs['shuffle'] = (split == 'train')
        
        # Ensure persistent_workers and prefetch_factor are only used when appropriate
        if split_kwargs['num_workers'] > 0:
            split_kwargs.setdefault('persistent_workers', True)
            split_kwargs.setdefault('prefetch_factor', 2)
        else:
            # These args are invalid if num_workers is 0
            split_kwargs.pop('persistent_workers', None)
            split_kwargs.pop('prefetch_factor', None)
            
        dataloaders[split] = DataLoader(dataset, **split_kwargs)
        
    return dataloaders