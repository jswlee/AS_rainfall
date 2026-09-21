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


def _year_ranges(years):
    ordered = np.sort(years.astype(int))
    train_end = int(ordered[int(len(ordered) * config.TRAIN_FRACTION) - 1])
    val_end = int(ordered[int(len(ordered) * (config.TRAIN_FRACTION + config.VAL_FRACTION)) - 1])
    return (int(ordered[0]), train_end), (train_end + 1, val_end), (val_end + 1, int(ordered[-1]))


def _station_roles(stations, years, val_years, test_years):
    names = sorted(set(stations.astype(str)))
    ranges = {
        name: (int(years[stations.astype(str) == name].min()), int(years[stations.astype(str) == name].max()))
        for name in names
    }
    rng = np.random.default_rng(config.SEED)
    test_eligible = [name for name in names if ranges[name][0] <= test_years[1] and ranges[name][1] >= test_years[0]]
    test = set(rng.permutation(test_eligible)[:config.N_TEST_STATIONS])
    val_eligible = [name for name in names if name not in test and ranges[name][0] <= val_years[1] and ranges[name][1] >= val_years[0]]
    val = set(rng.permutation(val_eligible)[:config.N_VAL_STATIONS])
    return {name: "test" if name in test else "val" if name in val else "train" for name in names}


def _split(stations, years):
    train_years, val_years, test_years = _year_ranges(years)
    roles = _station_roles(stations, years, val_years, test_years)
    role = np.asarray([roles[str(station)] for station in stations])
    index = np.arange(len(years))
    between = lambda values, bounds: (values >= bounds[0]) & (values <= bounds[1])
    splits = {
        "train": index[(role == "train") & between(years, train_years)],
        "val_temporal": index[(role == "train") & between(years, val_years)],
        "val_spatial": index[(role == "val") & between(years, val_years)],
        "test_temporal": index[(role == "train") & between(years, test_years)],
        "test_spatial": index[(role == "test") & between(years, test_years)],
    }
    return splits, roles, {"train": train_years, "val": val_years, "test": test_years}


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
    splits, roles, year_ranges = _split(stations, years)
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
        "variables": raw.get("variables", np.asarray([])), "roles": roles, "year_ranges": year_ranges,
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


def cv_folds(bundle, count=3):
    if count < 1:
        raise ValueError("count must be at least 1")
    temporal_count = count // 2
    spatial_count = count - temporal_count
    temporal = np.array_split(bundle.splits["val_temporal"], temporal_count) if temporal_count else []
    spatial = np.array_split(bundle.splits["val_spatial"], spatial_count) if spatial_count else []
    return [(bundle.splits["train"], fold) for fold in temporal + spatial if len(fold)]


def model_metadata(bundle):
    return {
        "climate_shape": tuple(bundle.arrays["climate"].shape[1:]),
        "dem_channels": int(bundle.arrays["local_dem"].shape[1]),
    }
