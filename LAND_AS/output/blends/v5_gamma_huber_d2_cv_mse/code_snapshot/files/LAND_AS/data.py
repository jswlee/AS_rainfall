from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from LAND_AS import config


@dataclass
class DataBundle:
    arrays: dict
    metadata: dict
    splits: dict
    stats: dict


class RainDataset(Dataset):
    def __init__(self, bundle, indices, crop=None):
        # Crops/lag slices are identical for every sample, so apply them once
        # to the selected arrays here -- __getitem__ is then pure indexing.
        metadata = bundle.metadata
        arrays = bundle.arrays
        self.indices = np.asarray(indices, dtype=int)
        crop = crop or {}

        self.climate = arrays["climate"][self.indices]
        if "climate" in crop:
            self.climate = _crop_patch(self.climate, *crop["climate"])
        # Lag depth is channel-level: [current][lag-1]...[lag-L] blocks.
        if "climate_lag" in crop:
            block = int(metadata["climate_block"])
            self.climate = self.climate[:, : (1 + crop["climate_lag"]) * block]

        dem_index = arrays["station_dem_idx"][self.indices]
        self.local_dem = arrays["local_dem"][dem_index]
        if "local" in crop:
            self.local_dem = _crop_patch(self.local_dem, *crop["local"])
        self.regional_dem = arrays["regional_dem"][dem_index]
        if "regional" in crop:
            self.regional_dem = _crop_patch(self.regional_dem, *crop["regional"])

        # Lag vector is [rain_1..L, mask_1..L]; slicing selects fewer lags.
        self.lag = arrays["lag"][self.indices]
        if "rain_lag" in crop:
            n = crop["rain_lag"]
            lmax = int(metadata["lag_max"])
            self.lag = torch.cat([self.lag[:, :n], self.lag[:, lmax:lmax + n]], dim=1)

        self.month = arrays["month"][self.indices]
        self.target = arrays["target"][self.indices] / bundle.stats["target_scale"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        features = {
            "climate": self.climate[position],
            "local_dem": self.local_dem[position],
            "regional_dem": self.regional_dem[position],
            "month": self.month[position],
            "lag": self.lag[position],
        }
        return features, self.target[position]


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


def _lag_indices(stations, years, months, days, n_lags):
    """Row index of the same station's sample ``k`` weeks earlier.

    Weeks are ISO weeks stamped by their Monday (the ``days`` column), so the
    sample for lag ``k`` is the one whose week ordinal is ``w - k``.
    Returns an (N, n_lags) int array; -1 where the lag week has no sample
    (record gaps, record starts, or before the dataset start).
    """
    week = pd.to_datetime({"year": years, "month": months, "day": days})
    ordinal = week.values.astype("datetime64[D]").astype(np.int64) // 7
    row_of = {(str(s), int(w)): i for i, (s, w) in enumerate(zip(stations, ordinal))}
    index = np.full((len(stations), n_lags), -1, dtype=np.int64)
    for i, (s, w) in enumerate(zip(stations, ordinal)):
        for lag in range(1, n_lags + 1):
            index[i, lag - 1] = row_of.get((str(s), int(w) - lag), -1)
    return index


def _crop_patch(patch, size, stride=1):
    """Center-crop a (C, H, W) patch, sampling cells ``stride`` apart.

    Indices are clamped to the patch edge (same convention as
    Daily_Modeling.crop_dem_patch). ``stride`` on a 1 km base patch acts as
    km-per-cell, so (size, stride) controls both resolution and extent.
    """
    h, w = patch.shape[-2], patch.shape[-1]
    ch, cw = h // 2, w // 2
    half = size // 2
    rows = [min(max(ch + (i - half) * stride, 0), h - 1) for i in range(size)]
    cols = [min(max(cw + (j - half) * stride, 0), w - 1) for j in range(size)]
    return patch[..., rows, :][..., :, cols]


# Tunable (size, stride) DEM crops. Extent = (size - 1) * stride + 1 cells,
# which must fit the stored base patches (local 11x11 @1km, regional 25x25 @1km).
DEM_LOCAL_CHOICES = [(3, 1), (5, 1), (7, 1), (9, 1), (11, 1), (3, 2), (5, 2), (3, 3)]
DEM_REGIONAL_CHOICES = [(9, 1), (13, 1), (17, 1), (21, 1), (25, 1), (9, 2), (11, 2), (7, 3), (9, 3), (5, 4), (7, 4)]


def crop_from_hp(hp):
    """Resolve a per-sample crop spec from hyperparameters, or None.

    Keys: ``local_dem_cfg`` / ``regional_dem_cfg`` (indices into the CHOICES
    tables above), ``climate_patch`` (center-crop side length, stride 1),
    ``climate_lag_weeks`` / ``rain_lag_weeks`` (lag depths to expose; the
    stored arrays always carry ``config.LAG_WEEKS`` lags).
    """
    crop = {}
    if "local_dem_cfg" in hp:
        crop["local"] = DEM_LOCAL_CHOICES[hp["local_dem_cfg"]]
    if "regional_dem_cfg" in hp:
        crop["regional"] = DEM_REGIONAL_CHOICES[hp["regional_dem_cfg"]]
    if hp.get("climate_patch"):
        crop["climate"] = (hp["climate_patch"], 1)
    if "climate_lag_weeks" in hp:
        crop["climate_lag"] = int(hp["climate_lag_weeks"])
    if "rain_lag_weeks" in hp:
        crop["rain_lag"] = int(hp["rain_lag_weeks"])
    return crop or None


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
    climate_block = int(raw["reanalysis_patches"].shape[1])
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

    # Temporal context: each sample also receives the same station's
    # reanalysis patch (appended as extra channel blocks, [current][lag-1]
    # ... [lag-L]) and observed rainfall from the previous LAG_WEEKS weeks.
    # Missing lag weeks are filled with 0 -- the train mean for the
    # normalized climate channels -- and flagged in the lag feature vector
    # ([rain_lag_1..L, has_lag_1..L]) so the model can distinguish a dry
    # week from a data gap.
    lag_index = _lag_indices(stations, years, raw["months"], raw["days"], config.LAG_WEEKS)
    has_lag = lag_index >= 0
    safe_index = np.where(has_lag, lag_index, 0)
    climate_lag = climate[safe_index]
    climate_lag[~has_lag] = 0
    climate = np.concatenate(
        [climate[:, None], climate_lag], axis=1
    ).reshape(len(climate), -1, *climate.shape[2:])
    rain_lag = np.where(has_lag, raw["rainfall_mm_raw"][safe_index] / target_scale, 0)
    lag_features = np.concatenate(
        [rain_lag.astype(np.float32), has_lag.astype(np.float32)], axis=1
    )

    arrays = {
        "climate": torch.from_numpy(np.nan_to_num(climate)),
        "local_dem": torch.from_numpy(np.nan_to_num(local_dem)),
        "regional_dem": torch.from_numpy(np.nan_to_num(regional_dem)),
        "month": torch.from_numpy(raw["month_onehot"].astype(np.float32)),
        "lag": torch.from_numpy(lag_features),
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
        "climate_block": climate_block, "lag_max": int(config.LAG_WEEKS),
    }
    return DataBundle(arrays, metadata, splits, stats)


def loaders(bundle, split_indices, batch_size, shuffle_train=True, crop=None,
            balance_stations=False):
    result = {}
    stations = bundle.metadata["stations"]
    for name, indices in split_indices.items():
        dataset = RainDataset(bundle, indices, crop)
        sampler = None
        shuffle = shuffle_train and name == "train"
        if balance_stations and name == "train":
            station_names = stations[np.asarray(indices, dtype=int)].astype(str)
            _, inverse = np.unique(station_names, return_inverse=True)
            counts = np.bincount(inverse)
            weights = 1.0 / counts[inverse]
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True
            )
            shuffle = False
        result[name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=0,
        )
    return result


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


def model_metadata(bundle, climate_patch=None, climate_lag=None, rain_lag=None):
    channels, height, width = bundle.arrays["climate"].shape[1:]
    lag_dim = int(bundle.arrays["lag"].shape[1])
    if climate_lag is not None:
        channels = (1 + climate_lag) * int(bundle.metadata["climate_block"])
    if rain_lag is not None:
        lag_dim = 2 * rain_lag
    if climate_patch:
        height = width = climate_patch
    return {
        "climate_shape": (channels, height, width),
        "dem_channels": int(bundle.arrays["local_dem"].shape[1]),
        "lag_dim": lag_dim,
    }
