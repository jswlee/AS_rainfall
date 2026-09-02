"""
Assemble reanalysis patches, DEM patches, rainfall targets, and month
one-hot encodings into a single NPZ file for model consumption.

Loads the intermediate NPZ files produced by ``01_build_features.py``
(reanalysis_patches_daily.npz and dem_patches.npz) and aligns them with
rainfall data to produce a ready-to-model ``daily_dataset.npz``.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from Daily_Modeling import config
from Daily_Modeling.data_utils.load_raw import (
    load_station_metadata,
    load_daily_rainfall,
)


def _month_onehot(months: np.ndarray) -> np.ndarray:
    """Convert integer months (1-12) to one-hot + cyclical features (N, 14).

    Columns 0-11: one-hot (month 1=Jan in col 0).
    Column 12:    sin(2π * month / 12) — captures circular seasonality.
    Column 13:    cos(2π * month / 12) — paired with sin for unambiguous angle.

    Fix G: cyclical features allow the model to learn that month 12 and month 1
    are adjacent without relying on the one-hot to encode their proximity.
    """
    oh = np.zeros((len(months), 12), dtype=np.float32)
    oh[np.arange(len(months)), months - 1] = 1.0
    angle = 2.0 * np.pi * months.astype(np.float32) / 12.0
    sin_m = np.sin(angle).reshape(-1, 1)
    cos_m = np.cos(angle).reshape(-1, 1)
    return np.concatenate([oh, sin_m, cos_m], axis=1)


def aggregate_to_weekly(
    patches: np.ndarray,
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    station_dem_idx: np.ndarray,
    rainfall_mm: np.ndarray,
    min_days: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Collapse station-day samples into station-week samples.

    Weeks are ISO calendar weeks (Monday-Sunday) and are stamped with the
    date of their Monday.  Rainfall is summed over the week; each reanalysis
    channel is reduced to its within-week mean *and* standard deviation, so
    the channel axis doubles from C to 2C (means first, then stds).

    Weeks with fewer than *min_days* daily records are dropped.

    Returns ``(patches, stations, years, months, days, station_dem_idx,
    rainfall_mm, n_days)``.
    """
    dates = pd.to_datetime({"year": years, "month": months, "day": days})
    # Monday of each sample's ISO week
    week_start = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    week_ord = week_start.to_numpy().astype("int64")  # ns since epoch; unique per week

    # Group key = (station, week_start)
    keys = np.stack([
        np.unique(stations.astype(str), return_inverse=True)[1],
        np.unique(week_ord, return_inverse=True)[1],
    ], axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_groups = int(inv.max()) + 1

    # Contiguous ordering so np.add.reduceat can sum each group in one pass
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(n_groups))
    counts = np.bincount(inv, minlength=n_groups).astype(np.int32)

    p = patches[order].astype(np.float64)
    sums = np.add.reduceat(p, starts, axis=0)
    sq_sums = np.add.reduceat(p * p, starts, axis=0)
    n = counts.reshape(-1, *([1] * (patches.ndim - 1)))
    means = sums / n
    # Population variance; clip to guard against tiny negatives from rounding
    variances = np.clip(sq_sums / n - means * means, 0.0, None)
    agg_patches = np.concatenate(
        [means, np.sqrt(variances)], axis=1
    ).astype(np.float32)

    rain_sum = np.add.reduceat(rainfall_mm[order].astype(np.float64), starts).astype(np.float32)
    # Station / DEM / week stamp are constant within a group -> take the first row
    first = order[starts]
    g_stations = stations[first]
    g_dem_idx = station_dem_idx[first]
    g_week_start = week_start.to_numpy()[first]

    keep = counts >= min_days
    n_dropped = int((~keep).sum())
    print(f"  Weekly aggregation: {len(inv)} station-days -> {n_groups} station-weeks")
    if n_dropped:
        print(f"    Dropped {n_dropped} incomplete week(s) with < {min_days} daily records")

    ws = pd.DatetimeIndex(g_week_start[keep])
    return (
        agg_patches[keep],
        g_stations[keep],
        ws.year.to_numpy().astype(np.int32),
        ws.month.to_numpy().astype(np.int32),
        ws.day.to_numpy().astype(np.int32),
        g_dem_idx[keep],
        rain_sum[keep],
        counts[keep],
    )


def assemble(
    out_path: Optional[Path] = None,
    reanalysis_npz: Optional[Path] = None,
    dem_npz: Optional[Path] = None,
    freq: str = "daily",
    min_days_per_week: int = 7,
) -> Path:
    """Combine pre-built feature NPZs with rainfall into a single dataset.

    *freq* is ``"daily"`` (one sample per station-day) or ``"weekly"``
    (one sample per station ISO week; see :func:`aggregate_to_weekly`).

    Returns the path to the saved file.
    """
    freq = freq.lower()
    if freq not in ("daily", "weekly"):
        raise ValueError(f"Unsupported freq '{freq}' (expected 'daily' or 'weekly')")
    if out_path is None:
        # Keyed off *freq*, not config.FREQ, so `--freq weekly` always writes to
        # output/weekly/ even when AS_RAINFALL_FREQ is unset or set to daily.
        out_path = config.dataset_npz_for(freq)
    if reanalysis_npz is None:
        reanalysis_npz = config.FEATURES_DIR / "reanalysis_patches_daily_station_centered.npz"
    if dem_npz is None:
        dem_npz = config.FEATURES_DIR / "dem_patches.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load station metadata
    station_meta = load_station_metadata()

    # 2. Load pre-built reanalysis patches from step 01
    print(f"Loading reanalysis patches from {reanalysis_npz} ...")
    rz = np.load(str(reanalysis_npz), allow_pickle=True)
    re_patches = rz["patches"]          # (N, C, H, W)
    re_stations = rz["stations"]         # (N,)  object
    re_years = rz["years"]               # (N,)  int32
    re_months = rz["months"]             # (N,)  int32
    re_days = rz["days"]                 # (N,)  int32
    var_names = rz["variables"] if "variables" in rz.files else np.array([])
    print(f"  Reanalysis patches: {re_patches.shape}")

    # 3. Load pre-built DEM patches from step 01
    print(f"Loading DEM patches from {dem_npz} ...")
    dz = np.load(str(dem_npz), allow_pickle=True)
    dem_local_raw = dz["dem_local_raw"]      # (S, n_bands, H, W)
    dem_regional_raw = dz["dem_regional_raw"]  # (S, n_bands, H, W)
    dem_station_names = dz["stations"]
    # Build lookup: station_name -> index in DEM arrays
    dem_lookup = {str(s): i for i, s in enumerate(dem_station_names)}
    print(f"  DEM: {dem_local_raw.shape[0]} stations")

    # 4. Load rainfall per station into a fast lookup
    #    Build {station_name: {(y,m,d): rainfall_mm}}
    print("Loading rainfall data ...")
    rain_lookup: Dict[str, Dict[tuple, float]] = {}
    for sname in sorted(station_meta):
        df = load_daily_rainfall(sname)
        if df is None:
            continue
        d = {}
        for _, row in df.iterrows():
            d[(int(row["year"]), int(row["month"]), int(row["day"]))] = float(row["rainfall_mm"])
        rain_lookup[sname] = d
    print(f"  Rainfall loaded for {len(rain_lookup)} stations")

    # 5. Align: for each reanalysis sample, look up DEM + rainfall
    N = len(re_stations)
    station_dem_idx = np.full(N, -1, dtype=np.int32)   # index into dem_*_raw per sample
    rainfall_mm = np.full(N, np.nan, dtype=np.float32)
    keep = np.zeros(N, dtype=bool)

    for i in range(N):
        st = str(re_stations[i])
        y, m, d = int(re_years[i]), int(re_months[i]), int(re_days[i])

        # DEM index
        di = dem_lookup.get(st)
        if di is None:
            continue
        station_dem_idx[i] = di

        # Rainfall
        rl = rain_lookup.get(st)
        if rl is None:
            continue
        rain_val = rl.get((y, m, d))
        if rain_val is None:
            continue
        rainfall_mm[i] = rain_val
        keep[i] = True

    idx = np.where(keep)[0]
    print(f"  Aligned {len(idx)}/{N} samples (dropped {N - len(idx)} with missing DEM or rainfall)")

    re_patches = re_patches[idx]
    re_stations = re_stations[idx]
    re_years = re_years[idx]
    re_months = re_months[idx]
    re_days = re_days[idx]
    station_dem_idx = station_dem_idx[idx]   # (N_aligned,) indices into dem_*_raw
    rainfall_mm = rainfall_mm[idx]

    # 6. Optional weekly aggregation (station-days -> station-weeks)
    n_days_per_sample = None
    if freq == "weekly":
        (re_patches, re_stations, re_years, re_months, re_days,
         station_dem_idx, rainfall_mm, n_days_per_sample) = aggregate_to_weekly(
            re_patches, re_stations, re_years, re_months, re_days,
            station_dem_idx, rainfall_mm, min_days=min_days_per_week,
        )
        # Channel axis doubled: means then stds
        var_names = ([f"{v}_mean" for v in var_names]
                     + [f"{v}_std" for v in var_names])
        print(f"  Weekly patches: {re_patches.shape}")

    # 7. Month one-hot (from the week's Monday when freq == 'weekly')
    month_onehot = _month_onehot(re_months)

    # 8. Save
    #    DEM arrays are kept at station-level (S_stations, n_bands, H, W) to avoid
    #    duplicating identical patches across millions of samples.  station_dem_idx
    #    maps each sample row back to its station's DEM patch at runtime.
    extra = {} if n_days_per_sample is None else {"n_days": n_days_per_sample}
    np.savez_compressed(
        str(out_path),
        **extra,
        reanalysis_patches=re_patches,
        dem_local_raw=dem_local_raw,
        dem_regional_raw=dem_regional_raw,
        dem_stations=dem_station_names,
        station_dem_idx=station_dem_idx,
        month_onehot=month_onehot,
        rainfall_mm_raw=rainfall_mm,
        stations=re_stations,
        years=re_years,
        months=re_months,
        days=re_days,
        variables=np.array(var_names, dtype=object),
    )
    print(f"Saved assembled {freq} dataset -> {out_path}  ({len(re_stations)} samples, "
          f"{len(dem_station_names)} unique DEM stations)")
    return out_path
