"""
Assemble reanalysis patches, DEM patches, rainfall targets, and month
one-hot encodings into a single NPZ file for model consumption.

Loads the intermediate NPZ files produced by ``01_build_features.py``
(reanalysis_patches_daily.npz and dem_patches.npz) and aligns them with
rainfall data to produce a ready-to-model ``daily_dataset.npz``.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from Daily_Modeling import config
from Daily_Modeling.data_utils.load_raw import (
    load_station_metadata,
    load_daily_rainfall,
)


def _month_onehot(months: np.ndarray) -> np.ndarray:
    """Convert integer months (1-12) to one-hot (N, 12)."""
    oh = np.zeros((len(months), 12), dtype=np.float32)
    oh[np.arange(len(months)), months - 1] = 1.0
    return oh


def assemble(
    out_path: Optional[Path] = None,
    reanalysis_npz: Optional[Path] = None,
    dem_npz: Optional[Path] = None,
) -> Path:
    """Combine pre-built feature NPZs with rainfall into a single dataset.

    Returns the path to the saved file.
    """
    if out_path is None:
        out_path = config.ASSEMBLED_DIR / "daily_dataset_station_centered.npz"
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
    dem_local = np.zeros((N,) + dem_local_raw.shape[1:], dtype=np.float32)    # (N, n_bands, H, W)
    dem_regional = np.zeros((N,) + dem_regional_raw.shape[1:], dtype=np.float32)  # (N, n_bands, H, W)
    rainfall_mm = np.full(N, np.nan, dtype=np.float32)
    keep = np.zeros(N, dtype=bool)

    for i in range(N):
        st = str(re_stations[i])
        y, m, d = int(re_years[i]), int(re_months[i]), int(re_days[i])

        # DEM
        di = dem_lookup.get(st)
        if di is None:
            continue
        dem_local[i] = dem_local_raw[di]
        dem_regional[i] = dem_regional_raw[di]

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
    dem_local = dem_local[idx]
    dem_regional = dem_regional[idx]
    rainfall_mm = rainfall_mm[idx]

    # 6. Month one-hot
    month_onehot = _month_onehot(re_months)

    # 7. Save
    np.savez_compressed(
        str(out_path),
        reanalysis_patches=re_patches,
        dem_local_raw=dem_local,
        dem_regional_raw=dem_regional,
        month_onehot=month_onehot,
        rainfall_mm_raw=rainfall_mm,
        stations=re_stations,
        years=re_years,
        months=re_months,
        days=re_days,
        variables=np.array(var_names, dtype=object),
    )
    print(f"Saved assembled dataset -> {out_path}  ({len(idx)} samples)")
    return out_path
