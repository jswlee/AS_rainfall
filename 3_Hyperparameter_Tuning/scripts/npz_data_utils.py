#!/usr/bin/env python3
"""
NPZ data loader for assembled full_training_data.npz.
Returns a dict compatible with 4_Train_Best_Model/scripts/data_utils.py.
"""
import os
import numpy as np
import pickle
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split


def _first_key(z, keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        if k in z.files:
            return k
    return None


essential_keys = (
    "month_onehot",
)


def load_assembled_npz_data(npz_path: str,
                            test_indices_path: Optional[str] = None,
                            test_size: float = 0.1,
                            val_size: float = 0.1,
                            random_state: Optional[int] = None):
    z = np.load(str(npz_path), allow_pickle=True)

    # Reanalysis/climate patches (N, V, H, W)
    climate_key = _first_key(z, ("reanalysis_patches", "patches"))
    if climate_key is None:
        raise KeyError("NPZ missing 'reanalysis_patches' or 'patches'")
    climate = z[climate_key].astype(np.float32)

    # DEM patches (N, 3, 3) — prefer standardized variants if available
    local_key = _first_key(
        z,
        (
            "dem_local_divstd",  # preferred standardized
            "dem_local_std",
            "dem_local_minmax",
            "dem_local_patches",
            "dem_local_3x3",
            "dem_local",
        ),
    )
    regional_key = _first_key(
        z,
        (
            "dem_regional_divstd",  # preferred standardized
            "dem_regional_std",
            "dem_regional_minmax",
            "dem_regional_patches",
            "dem_regional_3x3",
            "dem_regional",
        ),
    )
    if local_key is None or regional_key is None:
        raise KeyError(f"NPZ missing DEM patches. Keys present: {sorted(z.files)}")
    local_dem = z[local_key].astype(np.float32)
    regional_dem = z[regional_key].astype(np.float32)

    # Month one-hot (N, M)
    if "month_onehot" not in z.files:
        raise KeyError("NPZ missing 'month_onehot'")
    month = z["month_onehot"].astype(np.float32)

    # Targets (rainfall)
    # Prefer standardized labels if provided
    if "rainfall_mm_divstd" in z.files:
        y = z["rainfall_mm_divstd"].astype(np.float32)
    elif "rainfall_mm" in z.files:
        # Backward-compat: original pipeline trained on rainfall scaled by 1/100
        y = z["rainfall_mm"].astype(np.float32) / 100.0
    elif "rainfall" in z.files:
        y = z["rainfall"].astype(np.float32) / 100.0
    else:
        raise KeyError("NPZ missing 'rainfall_mm_divstd', 'rainfall_mm' or 'rainfall'")

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

    train_val_indices = np.setdiff1d(indices, test_indices)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size, random_state=random_state)

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
        }
    }

    return data
