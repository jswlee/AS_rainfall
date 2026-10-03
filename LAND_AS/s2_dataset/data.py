"""Stage 2: runtime dataset -- splits, lag features, normalization, folds.

load_data() reads a dataset NPZ (weekly_dataset.npz by default, or
daily_dataset.npz with freq="daily"; both built by s1_prepare) and returns a
DataBundle: raw arrays plus the strict train/test split from config.py. The
other helpers derive everything downstream stages need: lagged rainfall and
climate context (clamped to the stored lag depth), DEM/climate crop
resolution from hyperparameters, fold-local normalization stats, PyTorch
DataLoaders, and the cv_folds() iterators (LOSO / kfold / temporal) consumed
by s4_train and s5_evaluate.
"""
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
        # Requests deeper than the stored depth are clamped so slicing always
        # matches model_metadata()'s reported channel count.
        lag_max = int(metadata["lag_max"])
        if "climate_lag" in crop:
            block = int(metadata["climate_block"])
            n = min(crop["climate_lag"], lag_max)
            self.climate = self.climate[:, : (1 + n) * block]

        dem_index = arrays["station_dem_idx"][self.indices]
        self.local_dem = arrays["local_dem"][dem_index]
        if "local" in crop:
            self.local_dem = _crop_patch(self.local_dem, *crop["local"])
        self.regional_dem = arrays["regional_dem"][dem_index]
        if "regional" in crop:
            self.regional_dem = _crop_patch(self.regional_dem, *crop["regional"])
        # Channel subset for DEM ablation: channels are
        # [elev, slope, sin(aspect), cos(aspect)]; dem_channels=1 keeps
        # elevation only. Applied after cropping (channel dim unaffected).
        if "dem_channels" in crop:
            n = crop["dem_channels"]
            self.local_dem = self.local_dem[:, :n]
            self.regional_dem = self.regional_dem[:, :n]

        # Lag vector is [rain_1..L, mask_1..L]; slicing selects fewer lags.
        self.lag = arrays["lag"][self.indices]
        if "rain_lag" in crop:
            n = min(crop["rain_lag"], lag_max)
            self.lag = torch.cat([self.lag[:, :n], self.lag[:, lag_max:lag_max + n]], dim=1)

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


def _lag_indices(stations, years, months, days, n_lags, step_days=7):
    """Row index of the same station's sample ``k`` periods earlier.

    Periods are ``step_days`` calendar days: weekly samples are stamped by
    their ISO-week Monday (``step_days=7``) so lag ``k`` resolves to the
    sample ``k`` weeks earlier; daily samples use ``step_days=1`` so lag
    ``k`` is the sample ``k`` days earlier.
    Returns an (N, n_lags) int array; -1 where the lag period has no sample
    (record gaps, record starts, or before the dataset start).
    """
    dates = pd.to_datetime({"year": years, "month": months, "day": days})
    ordinal = dates.values.astype("datetime64[D]").astype(np.int64) // step_days
    row_of = {(str(s), int(w)): i for i, (s, w) in enumerate(zip(stations, ordinal))}
    index = np.full((len(stations), n_lags), -1, dtype=np.int64)
    for i, (s, w) in enumerate(zip(stations, ordinal)):
        for lag in range(1, n_lags + 1):
            index[i, lag - 1] = row_of.get((str(s), int(w) - lag), -1)
    return index


def _crop_patch(patch, size, stride=1):
    """Center-crop a (C, H, W) patch, sampling cells ``stride`` apart.

    Indices are clamped to the patch edge (same convention as
    Daily_Modeling.crop_dem_patch). ``stride`` is in base-grid cells;
    crop_from_hp() converts the tunable km-per-cell choices to cell strides
    using the dataset's stored dem_cell_km.
    """
    h, w = patch.shape[-2], patch.shape[-1]
    ch, cw = h // 2, w // 2
    half = size // 2
    rows = [min(max(ch + (i - half) * stride, 0), h - 1) for i in range(size)]
    cols = [min(max(cw + (j - half) * stride, 0), w - 1) for j in range(size)]
    return patch[..., rows, :][..., :, cols]


# Tunable (size, km_per_cell) DEM crops. Extent ~= (size - 1) * km_per_cell,
# which must fit the stored base patches (~11 km local, ~25 km regional).
# crop_from_hp() converts km_per_cell to a cell stride via the dataset's
# dem_cell_km, so these keep the same physical meaning at any base resolution.
DEM_LOCAL_CHOICES = [(3, 1), (5, 1), (7, 1), (9, 1), (11, 1), (3, 2), (5, 2), (3, 3)]
DEM_REGIONAL_CHOICES = [(9, 1), (13, 1), (17, 1), (21, 1), (25, 1), (9, 2), (11, 2), (7, 3), (9, 3), (5, 4), (7, 4)]


def dem_choices(dem_cell_km=1.0):
    """(size, km_per_cell) DEM crop candidates for a dataset's base cell size.

    The base lists assume the 1 km grid. On a finer base grid the crop is
    fixed to native cell spacing -- only the window size varies -- so every
    candidate reads pixels at dem_cell_km resolution (sizes span the stored
    ~11 km / ~25 km base extents).
    """
    if dem_cell_km >= 1.0:
        return DEM_LOCAL_CHOICES, DEM_REGIONAL_CHOICES
    local = [(n, dem_cell_km) for n in (3, 5, 7, 9, 11, 15, 21, 29, 45)
             if n * dem_cell_km <= 11.5]
    regional = [(n, dem_cell_km) for n in (9, 13, 17, 21, 25, 33, 49, 65, 97)
                if n * dem_cell_km <= 25.5]
    return local, regional


def crop_from_hp(hp, dem_cell_km=None):
    """Resolve a per-sample crop spec from hyperparameters, or None.

    Keys: ``local_dem_cfg`` / ``regional_dem_cfg`` (indices into the tables
    from dem_choices()), ``climate_patch`` (center-crop side length, stride 1),
    ``climate_lag_weeks`` / ``rain_lag_weeks`` (lag depths to expose; "weeks"
    means lag periods -- days for the daily dataset. Requests deeper than the
    stored depth are clamped by RainDataset/model_metadata to lag_max).

    ``dem_cell_km`` is the base-grid cell size recorded in the dataset NPZ
    (``bundle.metadata['dem_cell_km']``; 1.0 for legacy datasets). It converts
    the km-per-cell choices above into cell strides and selects the matching
    choice table. Falls back to ``hp['dem_cell_km']`` then 1.0.
    """
    if dem_cell_km is None:
        dem_cell_km = hp.get("dem_cell_km", 1.0)
    local_choices, regional_choices = dem_choices(dem_cell_km)

    def _stride(km):
        return max(1, int(round(km / dem_cell_km)))

    crop = {}
    # Explicit (size, km) crops (stored in run hyperparameters.json) take
    # precedence over the resolution-dependent choice-table indices.
    if "local_dem_crop" in hp:
        size, km = hp["local_dem_crop"]
        crop["local"] = (size, _stride(km))
    elif "local_dem_cfg" in hp:
        size, km = local_choices[hp["local_dem_cfg"]]
        crop["local"] = (size, _stride(km))
    if "regional_dem_crop" in hp:
        size, km = hp["regional_dem_crop"]
        crop["regional"] = (size, _stride(km))
    elif "regional_dem_cfg" in hp:
        size, km = regional_choices[hp["regional_dem_cfg"]]
        crop["regional"] = (size, _stride(km))
    if hp.get("climate_patch"):
        crop["climate"] = (hp["climate_patch"], 1)
    if "climate_lag_weeks" in hp:
        crop["climate_lag"] = int(hp["climate_lag_weeks"])
    if "rain_lag_weeks" in hp:
        crop["rain_lag"] = int(hp["rain_lag_weeks"])
    if hp.get("dem_channels"):
        crop["dem_channels"] = int(hp["dem_channels"])
    return crop or None


def _channel_stats(values, land_only=False):
    source = np.where(values > -0.5, values, np.nan) if land_only else values
    axes = (0, 2, 3)
    mean = np.nanmean(source, axis=axes).astype(np.float32)
    std = np.nanstd(source, axis=axes).astype(np.float32)
    std[std < 1e-6] = 1
    return mean, std


def load_data(path=None, freq="weekly"):
    """Load a dataset NPZ into a normalized DataBundle.

    ``freq`` selects the sampling period: "weekly" reads
    ``config.DATASET_PATH`` with ``config.LAG_WEEKS`` week-long lags;
    "daily" reads ``config.DAILY_DATASET_PATH`` with ``config.LAG_DAYS``
    day-long lags. ``path`` overrides the default location either way.
    """
    if freq not in ("weekly", "daily"):
        raise ValueError(f"Unknown freq {freq!r}; use 'weekly' or 'daily'")
    if path is None:
        path = config.dataset_path_for(freq)
    step_days = 1 if freq == "daily" else 7
    n_lags = config.LAG_DAYS if freq == "daily" else config.LAG_WEEKS
    with np.load(path, allow_pickle=True) as source:
        raw = {key: source[key] for key in source.files}

    stations = raw["stations"].astype(str)
    years = raw["years"].astype(int)
    splits, roles = _split(stations, years)
    lag_index = _lag_indices(stations, years, raw["months"], raw["days"], n_lags, step_days=step_days)
    metadata = {
        "stations": stations, "years": years, "months": raw["months"].astype(int),
        "variables": raw.get("variables", np.asarray([])), "roles": roles,
        "freq": freq,
        "dataset_path": str(path),
        "dem_cell_km": float(raw["dem_cell_km"]) if "dem_cell_km" in raw else 1.0,
        "climate_block": int(raw["reanalysis_patches"].shape[1]),
        "lag_max": n_lags,
        "_lag_index": lag_index,
        "_raw": {
            "climate": raw["reanalysis_patches"].astype(np.float32),
            "local_dem": raw["dem_local_raw"].astype(np.float32),
            "regional_dem": raw["dem_regional_raw"].astype(np.float32),
            "month": raw["month_onehot"].astype(np.float32),
            "target": raw["rainfall_mm_raw"].astype(np.float32),
            "station_dem_idx": raw["station_dem_idx"].astype(int),
        },
    }
    bundle = DataBundle({}, metadata, splits, {})
    return normalized_bundle(bundle, splits["train"])


def normalized_bundle(bundle, normalization_indices=None, stats=None):
    """Rebuild normalized arrays from fold indices or explicit saved stats."""
    raw = bundle.metadata["_raw"]
    lag_index = bundle.metadata["_lag_index"]

    if stats is None:
        index = np.asarray(normalization_indices, dtype=int)
        climate_mean, climate_std = _channel_stats(raw["climate"][index])
        dem_rows = np.unique(raw["station_dem_idx"][index])
        local_mean, local_std = _channel_stats(raw["local_dem"][dem_rows], land_only=True)
        regional_mean, regional_std = _channel_stats(raw["regional_dem"][dem_rows], land_only=True)
        target_scale = float(np.std(raw["target"][index]))
    else:
        climate_mean = np.asarray(stats["climate_mean"], dtype=np.float32)
        climate_std = np.asarray(stats["climate_std"], dtype=np.float32)
        local_mean = np.asarray(stats["local_dem_mean"], dtype=np.float32)
        local_std = np.asarray(stats["local_dem_std"], dtype=np.float32)
        regional_mean = np.asarray(stats["regional_dem_mean"], dtype=np.float32)
        regional_std = np.asarray(stats["regional_dem_std"], dtype=np.float32)
        target_scale = float(stats["target_scale"])

    climate = (raw["climate"] - climate_mean[None, :, None, None]) / climate_std[None, :, None, None]
    local_dem = (raw["local_dem"] - local_mean[None, :, None, None]) / local_std[None, :, None, None]
    regional_dem = (raw["regional_dem"] - regional_mean[None, :, None, None]) / regional_std[None, :, None, None]

    # Temporal context: each sample also receives the same station's
    # reanalysis patch (appended as extra channel blocks, [current][lag-1]
    # ... [lag-L]) and observed rainfall from the previous lag_max periods
    # (weeks or days, per the bundle's freq). Missing lag periods are filled
    # with 0 -- the fold train mean for the normalized climate channels --
    # and flagged in the lag feature vector ([rain_lag_1..L, has_lag_1..L])
    # so the model can distinguish a dry period from a data gap.
    has_lag = lag_index >= 0
    safe_index = np.where(has_lag, lag_index, 0)
    climate_lag = climate[safe_index]
    climate_lag[~has_lag] = 0
    climate = np.concatenate(
        [climate[:, None], climate_lag], axis=1
    ).reshape(len(climate), -1, *climate.shape[2:])
    rain_lag = np.where(has_lag, raw["target"][safe_index] / target_scale, 0)
    lag_features = np.concatenate(
        [rain_lag.astype(np.float32), has_lag.astype(np.float32)], axis=1
    )

    arrays = {
        "climate": torch.from_numpy(np.nan_to_num(climate)),
        "local_dem": torch.from_numpy(np.nan_to_num(local_dem)),
        "regional_dem": torch.from_numpy(np.nan_to_num(regional_dem)),
        "month": torch.from_numpy(raw["month"]),
        "lag": torch.from_numpy(lag_features),
        "target": torch.from_numpy(raw["target"]),
        "station_dem_idx": raw["station_dem_idx"],
    }
    stats = {
        "climate_mean": climate_mean.tolist(), "climate_std": climate_std.tolist(),
        "local_dem_mean": local_mean.tolist(), "local_dem_std": local_std.tolist(),
        "regional_dem_mean": regional_mean.tolist(), "regional_dem_std": regional_std.tolist(),
        "target_scale": target_scale,
    }
    return DataBundle(arrays, bundle.metadata, bundle.splits, stats)


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
    mode : {'loso', 'loso_recent', 'kfold', 'temporal', 'both'}
        - 'loso': Leave-One-Station-Out -- each fold holds out one train station.
        - 'loso_recent': LOSO, but the validation index is restricted to the
          held-out station's years >= config.LOSO_RECENT_YEAR_START so early
          stopping measures unseen-station + late-era generalization, matching
          the test split. Train indices are identical to 'loso'; folds whose
          restricted val would be empty fall back to the full LOSO val.
        - 'kfold': Spatial k-fold -- train stations are partitioned into ``count``
          disjoint groups; each fold uses one group as validation and the rest
          as training.
        - 'temporal': contiguous blocks of pre-2017 training years are held out;
          all stations remain in both sides of each fold.
        - 'both': LOSO folds plus ``count`` temporal folds.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        (train_indices, val_indices) for each fold.
    """
    train_indices = bundle.splits["train"]
    stations = bundle.metadata["stations"]
    train_stations = np.asarray(sorted(set(stations[train_indices].astype(str))), dtype=str)

    if mode in ("loso", "loso_recent"):
        years = bundle.metadata["years"]
        folds = []
        for held_out in train_stations:
            val_mask = np.isin(stations[train_indices], held_out)
            fold_train = train_indices[~val_mask]
            fold_val = train_indices[val_mask]
            if mode == "loso_recent":
                recent = fold_val[years[fold_val] >= config.LOSO_RECENT_YEAR_START]
                if len(recent) > 0:
                    fold_val = recent
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

    if mode == "temporal":
        count = count if count is not None else 3
        years = bundle.metadata["years"]
        train_years = np.asarray(sorted(set(years[train_indices].astype(int))), dtype=int)
        groups = np.array_split(train_years, min(count, len(train_years)))
        folds = []
        for group in groups:
            val_mask = np.isin(years[train_indices], group)
            fold_train = train_indices[~val_mask]
            fold_val = train_indices[val_mask]
            if len(fold_train) > 0 and len(fold_val) > 0:
                folds.append((fold_train, fold_val))
        return folds

    if mode == "both":
        return (
            cv_folds(bundle, mode="loso")
            + cv_folds(bundle, count=count, mode="temporal")
        )

    raise ValueError(f"Unknown cv_folds mode: {mode!r}. Use 'loso', 'loso_recent', 'kfold', 'temporal', or 'both'.")


def model_metadata(bundle, climate_patch=None, climate_lag=None, rain_lag=None, dem_channels=None):
    channels, height, width = bundle.arrays["climate"].shape[1:]
    lag_dim = int(bundle.arrays["lag"].shape[1])
    dem_ch = int(bundle.arrays["local_dem"].shape[1])
    # Requests deeper than the stored lag depth are clamped so the reported
    # shapes always match what RainDataset's slicing actually yields.
    max_lag = int(bundle.metadata["lag_max"])
    if climate_lag is not None:
        channels = (1 + min(climate_lag, max_lag)) * int(bundle.metadata["climate_block"])
    if rain_lag is not None:
        lag_dim = 2 * min(rain_lag, max_lag)
    if dem_channels is not None:
        dem_ch = min(dem_channels, dem_ch)
    if climate_patch:
        height = width = climate_patch
    return {
        "climate_shape": (channels, height, width),
        "dem_channels": dem_ch,
        "lag_dim": lag_dim,
    }
