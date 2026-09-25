from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from LAND_AS import config


@dataclass
class DataBundle:
    arrays: dict
    metadata: dict
    splits: dict
    stats: dict


class RainDataset(Dataset):
    def __init__(self, bundle, indices):
        self.arrays = bundle.arrays
        self.indices = np.asarray(indices, dtype=int)
        self.target_scale = bundle.stats["target_scale"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        dem_index = self.arrays["station_dem_idx"][index]
        features = {
            "climate": self.arrays["climate"][index],
            "local_dem": self.arrays["local_dem"][dem_index],
            "regional_dem": self.arrays["regional_dem"][dem_index],
            "month": self.arrays["month"][index],
        }
        return features, self.arrays["target"][index] / self.target_scale


def _station_roles(stations):
    """Assign each station a role based on config.TEST_STATIONS.

    Test stations are held out temporally (only years after TRAIN_YEAR_END).
    All remaining stations are train stations. There is no separate 'val'
    role -- validation is done via leave-one-station-out CV over the train
    stations.
    """
    test = set(config.TEST_STATIONS)
    names = sorted(set(stations.astype(str)))
    for name in test:
        if name not in names:
            raise ValueError(f"Configured test station '{name}' not found in dataset")
    return {name: "test" if name in test else "train" for name in names}


def _split(stations, years):
    roles = _station_roles(stations)
    role = np.asarray([roles[str(station)] for station in stations])
    index = np.arange(len(years))
    splits = {
        "train": index[(role == "train") & (years <= config.TRAIN_YEAR_END)],
        "test": index[(role == "test") & (years > config.TRAIN_YEAR_END)],
    }
    return splits, roles


def _channel_stats(values, land_only=False):
    source = np.where(values > -0.5, values, np.nan) if land_only else values
    axes = (0, 2, 3)
    mean = np.nanmean(source, axis=axes).astype(np.float32)
    std = np.nanstd(source, axis=axes).astype(np.float32)
    std[std < 1e-6] = 1
    return mean, std


def load_data(path=config.DATASET_PATH):
    with np.load(path, allow_pickle=True) as source:
        raw = {key: source[key] for key in source.files}

    stations = raw["stations"].astype(str)
    years = raw["years"].astype(int)
    splits, roles = _split(stations, years)
    climate = raw["reanalysis_patches"].astype(np.float32)
    local_dem = raw["dem_local_raw"].astype(np.float32)
    regional_dem = raw["dem_regional_raw"].astype(np.float32)

    climate_mean, climate_std = _channel_stats(climate[splits["train"]])
    climate = (climate - climate_mean[None, :, None, None]) / climate_std[None, :, None, None]
    local_mean, local_std = _channel_stats(local_dem, land_only=True)
    regional_mean, regional_std = _channel_stats(regional_dem, land_only=True)
    local_dem = (local_dem - local_mean[None, :, None, None]) / local_std[None, :, None, None]
    regional_dem = (regional_dem - regional_mean[None, :, None, None]) / regional_std[None, :, None, None]
    target_scale = float(np.std(raw["rainfall_mm_raw"][splits["train"]]))

    arrays = {
        "climate": torch.from_numpy(np.nan_to_num(climate)),
        "local_dem": torch.from_numpy(np.nan_to_num(local_dem)),
        "regional_dem": torch.from_numpy(np.nan_to_num(regional_dem)),
        "month": torch.from_numpy(raw["month_onehot"].astype(np.float32)),
        "target": torch.from_numpy(raw["rainfall_mm_raw"].astype(np.float32)),
        "station_dem_idx": raw["station_dem_idx"].astype(int),
    }
    stats = {
        "climate_mean": climate_mean.tolist(), "climate_std": climate_std.tolist(),
        "local_dem_mean": local_mean.tolist(), "local_dem_std": local_std.tolist(),
        "regional_dem_mean": regional_mean.tolist(), "regional_dem_std": regional_std.tolist(),
        "target_scale": target_scale,
    }
    metadata = {
        "stations": stations, "years": years, "months": raw["months"].astype(int),
        "variables": raw.get("variables", np.asarray([])), "roles": roles,
    }
    return DataBundle(arrays, metadata, splits, stats)


def loaders(bundle, split_indices, batch_size, shuffle_train=True):
    return {
        name: DataLoader(
            RainDataset(bundle, indices), batch_size=batch_size,
            shuffle=shuffle_train and name == "train", num_workers=0,
        )
        for name, indices in split_indices.items()
    }


def cv_folds(bundle, count=None, mode="loso"):
    """Build cross-validation folds over train stations.

    Parameters
    ----------
    bundle : DataBundle
        Data bundle returned by ``load_data``.
    count : int | None
        Number of folds. For ``mode='loso'`` this is ignored (one fold per
        train station). For ``mode='kfold'`` it defaults to 3.
    mode : {'loso', 'kfold'}
        - 'loso': Leave-One-Station-Out -- each fold holds out one train station.
        - 'kfold': Spatial k-fold -- train stations are partitioned into ``count``
          disjoint groups; each fold uses one group as validation and the rest
          as training.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        (train_indices, val_indices) for each fold.
    """
    train_indices = bundle.splits["train"]
    stations = bundle.metadata["stations"]
    train_stations = np.asarray(sorted(set(stations[train_indices].astype(str))), dtype=str)

    if mode == "loso":
        folds = []
        for held_out in train_stations:
            val_mask = np.isin(stations[train_indices], held_out)
            fold_train = train_indices[~val_mask]
            fold_val = train_indices[val_mask]
            if len(fold_val) > 0:
                folds.append((fold_train, fold_val))
        return folds

    if mode == "kfold":
        count = count if count is not None else 3
        if count > len(train_stations):
            count = len(train_stations)
        # Balance groups by sample count: sort stations descending by size and
        # deal them into groups serpentine-style so validation sets are ~equal.
        counts = {
            station: int(np.sum(stations[train_indices] == station))
            for station in train_stations
        }
        ordered = sorted(train_stations, key=lambda s: (-counts[s], s))
        groups = [[] for _ in range(count)]
        for position, station in enumerate(ordered):
            cycle, offset = divmod(position, count)
            group_index = offset if cycle % 2 == 0 else count - 1 - offset
            groups[group_index].append(station)
        folds = []
        for group in groups:
            val_mask = np.isin(stations[train_indices], group)
            fold_train = train_indices[~val_mask]
            fold_val = train_indices[val_mask]
            if len(fold_val) > 0:
                folds.append((fold_train, fold_val))
        return folds

    raise ValueError(f"Unknown cv_folds mode: {mode!r}. Use 'loso' or 'kfold'.")


def model_metadata(bundle):
    return {
        "climate_shape": tuple(bundle.arrays["climate"].shape[1:]),
        "dem_channels": int(bundle.arrays["local_dem"].shape[1]),
    }
