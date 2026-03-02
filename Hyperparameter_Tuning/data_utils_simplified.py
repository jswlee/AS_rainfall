# data_utils_simplified.py

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split

# ================================================================
# Simplified and Memory-Efficient PyTorch Dataset
# ================================================================

class RainfallDataset(Dataset):
    """
    PyTorch Dataset that uses indices to select data from shared tensors.
    This avoids duplicating data in memory for train, val, and test splits.
    """
    def __init__(self, tensors: Dict[str, torch.Tensor], indices: np.ndarray, target_scale: Optional[float] = None):
        """
        Args:
            tensors: A dictionary of the full data tensors (e.g., 'climate', 'targets').
            indices: The array of indices that this dataset should expose.
            target_scale: If provided, targets are divided by this value (e.g., train-only std).
        """
        self.tensors = tensors
        self.indices = indices
        self.target_scale = float(target_scale) if target_scale is not None else None
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

        if self.target_scale is not None and self.target_scale > 0.0:
            target = target / self.target_scale
        
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
                 random_state: int = 42, test_indices_path: str = None, device: torch.device = None, **kwargs):
        """
        Initializes the DataManager by loading and splitting the data.
        """
        self.random_state = random_state
        self.device = device if device is not None else torch.device('cpu') # Store the device
        self._load_npz(npz_path)
        self._split_data(test_size, val_size, test_indices_path)
        self._compute_target_scale()  # Compute train-only scaling without modifying raw targets
        self._normalize_features()  # Fit on train split, apply everywhere
        self._extract_metadata()
        print("DataManager initialized successfully.")

    def _load_npz(self, npz_path: str):
        """Loads all data arrays from the NPZ file into float32 tensors.
        
        Note: Targets are loaded as RAW rainfall (mm) and will be normalized
        in _normalize_targets() using train-only statistics to avoid data leakage.
        """
        print(f"Loading and converting data from {npz_path}...")
        with np.load(npz_path) as z:
            def _first_nonempty_key(keys: List[str]) -> Optional[str]:
                for k in keys:
                    if k in z.files and np.asarray(z[k]).size > 0:
                        return k
                return None

            # Define the mapping from expected keys to NPZ file keys
            # Prefer cyclical daily encoding if present; otherwise use month one-hot
            if 'day_cyc' in z.files:
                temporal_key = 'day_cyc'
            elif 'month_cyc' in z.files:
                temporal_key = 'month_cyc'
            else:
                temporal_key = 'month_onehot'
            
            # For targets: prefer raw rainfall_mm_raw for proper train-only normalization
            # Fall back to pre-normalized rainfall_mm_divstd for legacy NPZ files
            if 'rainfall_mm_raw' in z.files:
                targets_key = 'rainfall_mm_raw'
                self._targets_are_raw = True
                print("  Found raw rainfall data - will normalize using train-only statistics.")
            else:
                targets_key = 'rainfall_mm_divstd'
                self._targets_are_raw = False
                print("  WARNING: Using pre-normalized rainfall (potential data leakage).")
                print("           Re-run assemble_training_data.py to generate rainfall_mm_raw.")

            local_dem_key = _first_nonempty_key(['dem_local_raw', 'dem_local_divstd', 'dem_local_minmax'])
            regional_dem_key = _first_nonempty_key(['dem_regional_raw', 'dem_regional_divstd', 'dem_regional_minmax'])
            if local_dem_key is None or regional_dem_key is None:
                raise KeyError("Missing DEM arrays in NPZ. Expected one of dem_*_raw, dem_*_divstd, dem_*_minmax")
            
            key_map = {
                'climate': 'reanalysis_patches',
                'local_dem': local_dem_key,
                'regional_dem': regional_dem_key,
                'temporal': temporal_key,
                'targets': targets_key
            }
            self.tensors = {}
            for key, npz_key in key_map.items():
                if npz_key not in z.files:
                    raise KeyError(f"Required key '{npz_key}' not found in NPZ file.")
                # Create tensor and immediately move it to the target device
                self.tensors[key] = torch.from_numpy(z[npz_key].astype(np.float32)).to(self.device)

            # Optional temporal metadata for time-based splits
            # These remain as NumPy arrays on CPU as they are only used for indexing
            self.years = z['years'].astype(int) if 'years' in z.files else None
            self.months = z['months'].astype(int) if 'months' in z.files else None
            self.days = z['days'].astype(int) if 'days' in z.files else None
            
            # Store global std from NPZ (for legacy/fallback only)
            self._global_rainfall_std = float(z.get('rainfall_mm_std', 1.0))
        
        print("Data loaded into tensors.")

    def _normalize_features(self):
        train_indices = self.indices['train']

        # Climate: normalize per-variable channel using train-only stats.
        climate = self.tensors['climate']
        if climate.ndim != 4:
            raise ValueError(f"Expected climate tensor with shape (N,C,H,W); got {tuple(climate.shape)}")

        climate_train = climate[train_indices]
        c = int(climate.shape[1])
        means = torch.zeros(c, device=self.device, dtype=climate.dtype)
        stds = torch.ones(c, device=self.device, dtype=climate.dtype)

        for i in range(c):
            vals = climate_train[:, i, :, :].reshape(-1)
            mask = torch.isfinite(vals)
            if mask.any():
                m = vals[mask].mean()
                s = vals[mask].std()
            else:
                m = torch.tensor(0.0, device=self.device, dtype=climate.dtype)
                s = torch.tensor(1.0, device=self.device, dtype=climate.dtype)
            if float(s.item()) == 0.0:
                s = torch.tensor(1.0, device=self.device, dtype=climate.dtype)
            means[i] = m
            stds[i] = s

        self.climate_mean = means
        self.climate_std = stds
        self.tensors['climate'] = (climate - means[None, :, None, None]) / stds[None, :, None, None]

        # DEM: normalize each scale using train-only global stats.
        for dem_key in ('local_dem', 'regional_dem'):
            dem = self.tensors[dem_key]
            dem_train = dem[train_indices].reshape(-1)
            mask = torch.isfinite(dem_train)
            if mask.any():
                m = dem_train[mask].mean()
                s = dem_train[mask].std()
            else:
                m = torch.tensor(0.0, device=self.device, dtype=dem.dtype)
                s = torch.tensor(1.0, device=self.device, dtype=dem.dtype)
            if float(s.item()) == 0.0:
                s = torch.tensor(1.0, device=self.device, dtype=dem.dtype)

            setattr(self, f"{dem_key}_mean", m)
            setattr(self, f"{dem_key}_std", s)
            self.tensors[dem_key] = (dem - m) / s

    def _split_data(self, test_size: float, val_size: float, test_indices_path: str):
        """Manages reproducible train/val/test splits.

        If year metadata is available, perform deterministic time-based splits:
          - Train: 1980-2015
          - Val:   2016-2020
          - Test:  2021-2024

        Otherwise, fall back to the previous stratified random split.
        """
        n_samples = len(self.tensors['targets'])
        all_indices = np.arange(n_samples)

        # Time-based split using year metadata, if available
        if getattr(self, 'years', None) is not None:
            years = self.years
            if len(years) != n_samples:
                raise ValueError(f"Length of years array ({len(years)}) does not match number of samples ({n_samples}).")

            train_mask = (years >= 1980) & (years <= 2015)
            val_mask = (years >= 2016) & (years <= 2020)
            test_mask = (years >= 2021) & (years <= 2024)

            train_indices = all_indices[train_mask]
            val_indices = all_indices[val_mask]
            test_indices = all_indices[test_mask]

            if test_indices_path:
                # Persist test indices for compatibility with existing tooling
                os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
                with open(test_indices_path, 'wb') as f:
                    pickle.dump(test_indices, f)

            self.indices = {
                'train': train_indices,
                'val': val_indices,
                'test': test_indices
            }
            print(
                f"Time-based data splits created: "
                f"{len(train_indices)} train (1980-2015), "
                f"{len(val_indices)} val (2016-2020), "
                f"{len(test_indices)} test (2021-2024)."
            )
            return

        # ------------------------------------------------------------------
        # Previous stratified split logic (kept for reference, now unused)
        # ------------------------------------------------------------------
        # # Create stratification bins based on rainfall distribution
        # # Use quantile-based binning to handle skewed rainfall distribution
        # y = self.tensors['targets'].cpu().numpy()
        # n_bins = 5
        # try:
        #     # Use quantiles of non-zero rainfall for binning
        #     edges = np.quantile(y[y > 0], np.linspace(0, 1, n_bins + 1))
        #     edges = np.unique(edges)
        #     if len(edges) < 2:
        #         raise ValueError("Not enough unique quantile edges.")
        #     y_bins = np.digitize(y, edges[1:-1])
        # except Exception as e:
        #     print(f"Warning: stratification binning failed ({e}); falling back to random split.")
        #     y_bins = None

        # # 1. Determine test indices (load from file or create new)
        # if test_indices_path and os.path.exists(test_indices_path):
        #     print(f"Loading test indices from {test_indices_path}")
        #     with open(test_indices_path, 'rb') as f:
        #         test_indices = pickle.load(f)
        # else:
        #     print("Generating new stratified test indices...")
        #     if y_bins is not None:
        #         _, test_indices = train_test_split(
        #             all_indices, test_size=test_size, random_state=self.random_state,
        #             stratify=y_bins
        #         )
        #     else:
        #         _, test_indices = train_test_split(
        #             all_indices, test_size=test_size, random_state=self.random_state
        #         )
        #     if test_indices_path:
        #         os.makedirs(os.path.dirname(test_indices_path), exist_ok=True)
        #         with open(test_indices_path, 'wb') as f:
        #             pickle.dump(test_indices, f)
        # 
        # # 2. Create stratified train/val split from the remaining indices
        # train_val_indices = np.setdiff1d(all_indices, test_indices)
        # if y_bins is not None:
        #     train_val_bins = y_bins[train_val_indices]
        #     train_indices, val_indices = train_test_split(
        #         train_val_indices, test_size=val_size, random_state=self.random_state,
        #         stratify=train_val_bins
        #     )
        # else:
        #     train_indices, val_indices = train_test_split(
        #         train_val_indices, test_size=val_size, random_state=self.random_state
        #     )
        #
        # self.indices = {
        #     'train': train_indices,
        #     'val': val_indices,
        #     'test': test_indices
        # }
        # print(f"Data splits created: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test.")
        
    def _compute_target_scale(self):
        """Compute train-only target scaling to avoid data leakage.

        Targets remain raw in self.tensors['targets']. Normalization is applied
        in RainfallDataset by dividing by self.target_scale.
        """
        if not getattr(self, '_targets_are_raw', False):
            # Targets already normalized in legacy NPZ (typically rainfall_mm_divstd).
            # Do not re-normalize; keep target_scale=1 but retain global std for denorm.
            self.rainfall_mm_std = self._global_rainfall_std
            self.target_scale = 1.0
            print(f"Using pre-normalized targets with global std={self.rainfall_mm_std:.4f}")
            return

        train_indices = self.indices['train']
        train_targets = self.tensors['targets'][train_indices]

        train_std = float(train_targets.std().item())
        if train_std == 0.0:
            train_std = 1.0
            print("WARNING: Train target std is 0, using 1.0 to avoid division by zero.")

        self.rainfall_mm_std = train_std
        self.target_scale = train_std
        print(f"Computed TRAIN-ONLY target std: {train_std:.4f} mm (targets will be divided by this during training)")

    def _extract_metadata(self):
        """Extracts metadata required by the model."""
        self.metadata = {
            'num_climate_vars': self.tensors['climate'].shape[1],
            'num_temporal_encodings': self.tensors['temporal'].shape[1],
            'local_dem_shape': tuple(self.tensors['local_dem'].shape[1:]),
            'regional_dem_shape': tuple(self.tensors['regional_dem'].shape[1:]),
            'climate_shape': tuple(self.tensors['climate'].shape[1:]),
            'rainfall_mm_std': self.rainfall_mm_std,  # Now train-only std
        }

    def get_datasets(self) -> Dict[str, RainfallDataset]:
        """Returns a dictionary of RainfallDataset objects for each split."""
        return {
            split: RainfallDataset(self.tensors, idx, target_scale=getattr(self, 'target_scale', None))
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