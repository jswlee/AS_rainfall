#!/usr/bin/env python3
"""
NPZ data loader for the assembled `full_training_data.npz`.

Returns a dictionary shaped like the outputs from
`4_Train_Best_Model/scripts/data_utils.py` so downstream code can consume it
directly. Splits are created as Train/Val/Test with persisted test indices for
reproducibility across runs.
"""
import os
import numpy as np
import pickle
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split

def load_assembled_npz_data(npz_path: str,
                            test_indices_path: Optional[str] = None,
                            test_size: float = 0.1,
                            val_size: float = 0.1,
                            random_state: Optional[int] = None) -> Dict[str, Any]:
    z = np.load(str(npz_path), allow_pickle=True)

    # Reanalysis/climate patches (N, V, H, W)
    if 'reanalysis_patches' not in z.files:
        raise KeyError("NPZ missing 'reanalysis_patches'")
    climate = z['reanalysis_patches'].astype(np.float32)

    required_local = 'dem_local_divstd'
    required_regional = 'dem_regional_divstd'
    if required_local not in z.files or required_regional not in z.files:
        raise KeyError(
            "NPZ missing required DEM keys 'dem_local_divstd' and/or 'dem_regional_divstd'. "
            f"Keys present: {sorted(z.files)}"
        )
    local_dem = z[required_local].astype(np.float32)
    regional_dem = z[required_regional].astype(np.float32)
    
    # Month one-hot (N, M)
    if "month_onehot" not in z.files:
        raise KeyError("NPZ missing 'month_onehot'")
    month = z["month_onehot"].astype(np.float32)

    # Targets (rainfall)
    y = z["rainfall_mm_divstd"].astype(np.float32)

    n = int(y.shape[0])
    if not (climate.shape[0] == local_dem.shape[0] == regional_dem.shape[0] == month.shape[0] == n):
        raise ValueError(
            f"Mismatched N: climate {climate.shape[0]}, local {local_dem.shape[0]}, regional {regional_dem.shape[0]}, month {month.shape[0]}, y {n}"
        )

    # Split with persisted test indices for reproducibility
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
            valid = (
                ti.size > 0 and
                np.all(ti >= 0) and
                np.all(ti < n)
            )
            if not valid:
                # Regenerate if indices are out-of-bounds for current NPZ
                test_indices = _generate_and_save_test_indices()
            else:
                test_indices = np.unique(ti)
                test_indices.sort()
        except Exception:
            test_indices = _generate_and_save_test_indices()
    else:
        test_indices = _generate_and_save_test_indices()

    # Remaining pool forms train+val, then split by val_size
    train_val_indices = np.setdiff1d(indices, test_indices)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size, random_state=random_state)

    # Extract rainfall std (used to de-standardize labels back to raw mm)
    rainfall_mm_std = float(z['rainfall_mm_std']) if 'rainfall_mm_std' in z.files else 1.0

    data = {
        'climate': {
            'train': climate[train_indices],
            'val': climate[val_indices],
            'test': climate[test_indices],
            'shape': climate.shape[1:],
        },
        'local_dem': {
            'train': local_dem[train_indices],
            'val': local_dem[val_indices],
            'test': local_dem[test_indices],
            'shape': local_dem.shape[1:],
        },
        'regional_dem': {
            'train': regional_dem[train_indices],
            'val': regional_dem[val_indices],
            'test': regional_dem[test_indices],
            'shape': regional_dem.shape[1:],
        },
        'month': {
            'train': month[train_indices],
            'val': month[val_indices],
            'test': month[test_indices],
            'shape': month.shape[1:],
        },
        'targets': {
            'train': y[train_indices],
            'val': y[val_indices],
            'test': y[test_indices],
        },
        'metadata': {
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
    }

    return data
