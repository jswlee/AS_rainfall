"""Spatio-temporal cross-validation splits for American Samoa daily rainfall.

Split modes
-----------
1. **spatiotemporal** (LAND): hold out entire stations *and* use different
   year ranges.  Produces five named splits:

   - ``train``           train stations x train years
   - ``val_spatial``     held-out val stations x val years  (spatial + temporal)
   - ``test_spatial``    held-out test stations x test years
   - ``val_temporal``    train stations x val years  (temporal only)
   - ``test_temporal``   train stations x test years

2. **station_proportional** (site-specific models): per-station chronological
   70/20/10 split.

Year boundaries can be computed from data so that ~70 % of samples fall in
train years, ~20 % in val years, and ~10 % in test years.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np

from Daily_Modeling import config


# ── helpers ────────────────────────────────────────────────────────────────

def compute_station_year_ranges(
    stations: np.ndarray,
    years: np.ndarray,
) -> Dict[str, Tuple[int, int]]:
    """Compute actual (min_year, max_year) for each station from the data arrays."""
    ranges: Dict[str, Tuple[int, int]] = {}
    for s in np.unique(stations):
        mask = stations == s
        yrs = years[mask].astype(int)
        if len(yrs) > 0:
            ranges[str(s)] = (int(yrs.min()), int(yrs.max()))
    return ranges


def compute_year_boundaries(
    years: np.ndarray,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Find chronological year cutoffs so that sample counts match target fractions.

    Returns (train_range, val_range, test_range) as ``(start_year, end_year)``
    tuples (inclusive).
    """
    yr = np.sort(years.astype(int))
    n = len(yr)
    train_end_idx = int(n * train_frac) - 1
    val_end_idx = int(n * (train_frac + val_frac)) - 1

    train_end_year = int(yr[train_end_idx])
    val_end_year = int(yr[val_end_idx])
    min_yr, max_yr = int(yr[0]), int(yr[-1])

    train_range = (min_yr, train_end_year)
    val_range = (train_end_year + 1, val_end_year)
    test_range = (val_end_year + 1, max_yr)

    # Print summary
    tr_n = int(np.sum((yr >= train_range[0]) & (yr <= train_range[1])))
    va_n = int(np.sum((yr >= val_range[0]) & (yr <= val_range[1])))
    te_n = int(np.sum((yr >= test_range[0]) & (yr <= test_range[1])))
    print(f"  Year boundaries (data-driven):")
    print(f"    train: {train_range[0]}–{train_range[1]}  ({tr_n:,d} samples, {100*tr_n/n:.1f}%)")
    print(f"    val:   {val_range[0]}–{val_range[1]}  ({va_n:,d} samples, {100*va_n/n:.1f}%)")
    print(f"    test:  {test_range[0]}–{test_range[1]}  ({te_n:,d} samples, {100*te_n/n:.1f}%)")
    return train_range, val_range, test_range


# ── station assignment ─────────────────────────────────────────────────────

def assign_station_groups(
    station_names: List[str],
    n_val: int = config.N_VAL_STATIONS,
    n_test: int = config.N_TEST_STATIONS,
    seed: int = config.RANDOM_SEED,
    station_year_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    val_years: Tuple[int, int] = (0, 0),
    test_years: Tuple[int, int] = (0, 0),
) -> Dict[str, str]:
    """Deterministically assign each station to 'train', 'val', or 'test'.

    If *station_year_ranges* is provided (``{name: (start_yr, end_yr)}``),
    only stations whose data overlaps the val / test year ranges are
    eligible for those roles.  This avoids empty splits.

    Returns ``{station_name: role}``.
    """
    if station_year_ranges is not None:
        def _overlaps(syr, target_range):
            return syr[0] <= target_range[1] and syr[1] >= target_range[0]
        eligible_test = sorted([
            s for s in station_names
            if s in station_year_ranges and _overlaps(station_year_ranges[s], test_years)
        ])
        eligible_val = sorted([
            s for s in station_names
            if s in station_year_ranges and _overlaps(station_year_ranges[s], val_years)
        ])
    else:
        eligible_test = sorted(station_names)
        eligible_val = sorted(station_names)

    rng = np.random.RandomState(seed)

    et = list(eligible_test); rng.shuffle(et)
    test_stations = set(et[:n_test])

    ev = [s for s in eligible_val if s not in test_stations]
    rng.shuffle(ev)
    val_stations = set(ev[:n_val])

    mapping: Dict[str, str] = {}
    for n in station_names:
        if n in test_stations:
            mapping[n] = "test"
        elif n in val_stations:
            mapping[n] = "val"
        else:
            mapping[n] = "train"
    return mapping


# ── spatiotemporal split (LAND) ────────────────────────────────────────────

def spatiotemporal_split(
    stations: np.ndarray,
    years: np.ndarray,
    station_groups: Dict[str, str],
    train_years: Tuple[int, int] = (0, 0),
    val_years: Tuple[int, int] = (0, 0),
    test_years: Tuple[int, int] = (0, 0),
) -> Dict[str, np.ndarray]:
    """Return index arrays for a spatio-temporal split.

    Five splits are returned:

    - **train**           train stations in train years
    - **val_spatial**     held-out *val* stations in val years
                          (tests spatial + temporal generalisation)
    - **test_spatial**    held-out *test* stations in test years
    - **val_temporal**    *train* stations in val years
                           (tests temporal-only generalisation)
    - **test_temporal**   *train* stations in test years
    """
    n = len(stations)
    idx = np.arange(n)

    roles = np.array([station_groups.get(str(s), "train") for s in stations])
    yr = years.astype(int)

    train_mask = (roles == "train") & (yr >= train_years[0]) & (yr <= train_years[1])
    val_sp_mask = (roles == "val") & (yr >= val_years[0]) & (yr <= val_years[1])
    test_sp_mask = (roles == "test") & (yr >= test_years[0]) & (yr <= test_years[1])
    val_tm_mask = (roles == "train") & (yr >= val_years[0]) & (yr <= val_years[1])
    test_tm_mask = (roles == "train") & (yr >= test_years[0]) & (yr <= test_years[1])

    splits = {
        "train": idx[train_mask],
        "val_spatial": idx[val_sp_mask],
        "test_spatial": idx[test_sp_mask],
        "val_temporal": idx[val_tm_mask],
        "test_temporal": idx[test_tm_mask],
    }
    total = sum(len(v) for v in splits.values())
    for k, v in splits.items():
        pct = 100 * len(v) / total if total else 0
        print(f"  {k:16s}: {len(v):>7,d} samples  ({pct:5.1f}%)")
    return splits


# ── temporal-only split ────────────────────────────────────────────────────

def temporal_split(
    years: np.ndarray,
    train_years: Tuple[int, int] = (0, 0),
    val_years: Tuple[int, int] = (0, 0),
    test_years: Tuple[int, int] = (0, 0),
) -> Dict[str, np.ndarray]:
    """Simple year-based split (for site-specific models)."""
    yr = years.astype(int)
    idx = np.arange(len(yr))
    splits = {
        "train": idx[(yr >= train_years[0]) & (yr <= train_years[1])],
        "val": idx[(yr >= val_years[0]) & (yr <= val_years[1])],
        "test": idx[(yr >= test_years[0]) & (yr <= test_years[1])],
    }
    for k, v in splits.items():
        print(f"  {k:12s}: {len(v):>7,d} samples")
    return splits


# ── per-station chronological split (site-specific) ───────────────────────

def station_temporal_split(
    stations: np.ndarray,
    years: np.ndarray,
    target_station: str,
    train_years: Tuple[int, int] = (0, 0),
    val_years: Tuple[int, int] = (0, 0),
    test_years: Tuple[int, int] = (0, 0),
) -> Dict[str, np.ndarray]:
    """Return indices for a single station, split only by year."""
    mask = np.array([str(s) == target_station for s in stations])
    yr = years.astype(int)
    idx = np.arange(len(stations))
    splits = {
        "train": idx[mask & (yr >= train_years[0]) & (yr <= train_years[1])],
        "val": idx[mask & (yr >= val_years[0]) & (yr <= val_years[1])],
        "test": idx[mask & (yr >= test_years[0]) & (yr <= test_years[1])],
    }
    return splits


# ── per-station chronological split (site-specific) ───────────────────────

def station_proportional_split(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    target_station: str,
    train_frac: float = config.SITE_TRAIN_FRAC,
    val_frac: float = config.SITE_VAL_FRAC,
) -> Dict[str, np.ndarray]:
    """Split a single station's data chronologically by proportion.

    Sorts the station's samples by date, then assigns the first
    *train_frac* to train, the next *val_frac* to val, and the
    remainder to test.  This guarantees every station with enough
    data gets all three splits regardless of its year range.
    """
    mask = np.array([str(s) == target_station for s in stations])
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"train": np.array([], dtype=int),
                "val": np.array([], dtype=int),
                "test": np.array([], dtype=int)}

    yr = years[idx].astype(int)
    mo = months[idx].astype(int)
    dy = days[idx].astype(int)
    date_order = np.lexsort((dy, mo, yr))
    sorted_idx = idx[date_order]

    n = len(sorted_idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": sorted_idx[:n_train],
        "val": sorted_idx[n_train:n_train + n_val],
        "test": sorted_idx[n_train + n_val:],
    }


# ── CV fold construction ───────────────────────────────────────────────────

def _shuffle_into_folds(val_indices, n_folds, train_idx, rng):
    """Shuffle *val_indices*, split into *n_folds* chunks, pair each with *train_idx*."""
    shuffled = rng.permutation(val_indices)
    chunks = np.array_split(shuffled, n_folds)
    return [(train_idx.astype(int), chunk.astype(int)) for chunk in chunks]


def make_cv_folds(splits, n_folds, cv_mode, rng_seed):
    """Build CV folds based on mode: temporal, spatial, or both.

    - temporal: folds split val_temporal only (train stations, held-out years)
    - spatial: folds split val_spatial only (held-out stations)
    - both: folds alternate between temporal and spatial validation
    """
    rng = np.random.RandomState(rng_seed)
    train_idx = splits.get("train", np.array([], dtype=int))
    val_temporal = splits.get("val_temporal", np.array([], dtype=int))
    val_spatial = splits.get("val_spatial", np.array([], dtype=int))

    if n_folds <= 1:
        if cv_mode == "spatial":
            val_idx = val_spatial if len(val_spatial) > 0 else val_temporal
        else:
            val_idx = val_temporal if len(val_temporal) > 0 else val_spatial
        return [(train_idx.astype(int), val_idx.astype(int))]

    if cv_mode == "temporal":
        if len(val_temporal) == 0:
            raise ValueError("cv_mode=temporal but val_temporal is empty")
        if len(val_temporal) < n_folds:
            print(f"Warning: val_temporal has only {len(val_temporal)} samples for {n_folds} folds; using single fold")
            return [(train_idx.astype(int), val_temporal.astype(int))]
        return _shuffle_into_folds(val_temporal, n_folds, train_idx, rng)

    elif cv_mode == "spatial":
        if len(val_spatial) == 0:
            raise ValueError("cv_mode=spatial but val_spatial is empty")
        if len(val_spatial) < n_folds:
            print(f"Warning: val_spatial has only {len(val_spatial)} samples for {n_folds} folds; using single fold")
            return [(train_idx.astype(int), val_spatial.astype(int))]
        return _shuffle_into_folds(val_spatial, n_folds, train_idx, rng)

    elif cv_mode == "both":
        if len(val_temporal) == 0 or len(val_spatial) == 0:
            raise ValueError("cv_mode=both requires both val_temporal and val_spatial to be non-empty")
        n_temp = n_folds // 2
        n_spat = n_folds - n_temp
        folds = []
        if n_temp > 0:
            if len(val_temporal) < n_temp:
                n_temp = 1
            folds.extend(_shuffle_into_folds(val_temporal, n_temp, train_idx, rng))
        if n_spat > 0:
            if len(val_spatial) < n_spat:
                n_spat = 1
            folds.extend(_shuffle_into_folds(val_spatial, n_spat, train_idx, rng))
        return folds

    else:
        raise ValueError(f"Unknown cv_mode: {cv_mode}")


# ── index sorting & expanding-window folds ─────────────────────────────────

def sorted_sample_indices(indices, years, months, days):
    """Sort sample indices chronologically by (year, month, day)."""
    return sorted(indices, key=lambda i: (int(years[i]), int(months[i]), int(days[i])))


def expanding_time_folds(indices_sorted, n_folds: int):
    """Forward-chaining (expanding-window) folds.

    For fold k:
      train = [0 : b_k]
      val   = [b_k : b_{k+1}]
    where b are evenly spaced boundaries.
    """
    if n_folds <= 1:
        return []
    n = len(indices_sorted)
    val_size = max(n // (n_folds + 1), 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_end = k * val_size
        val_end = min((k + 1) * val_size, n)
        if train_end >= n or train_end >= val_end:
            break
        folds.append((indices_sorted[:train_end], indices_sorted[train_end:val_end]))
    return folds


# ── split validation ───────────────────────────────────────────────────────

def validate_test_separation(splits, stations, years, train_yr, val_yr, test_yr):
    """Validate that test sets are temporally and spatially distinct from train/val."""
    train_idx = splits.get("train", np.array([], dtype=int))
    test_temporal = splits.get("test_temporal", np.array([], dtype=int))
    test_spatial = splits.get("test_spatial", np.array([], dtype=int))

    if len(test_temporal) > 0:
        test_temp_years = years[test_temporal]
        if np.any(test_temp_years < test_yr[0]):
            raise ValueError(f"test_temporal contains years before {test_yr[0]}")

    if len(test_spatial) > 0 and len(train_idx) > 0:
        test_spatial_stations = set(stations[test_spatial])
        train_stations = set(stations[train_idx])
        overlap = test_spatial_stations & train_stations
        if overlap:
            raise ValueError(f"test_spatial shares {len(overlap)} stations with train: {overlap}")

    print("\u2713 Test sets are temporally and spatially distinct from train/val")
