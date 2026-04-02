"""
Step 1: Build intermediate reanalysis and DEM feature files.

Usage:
    python -m Daily_Modeling.scripts.01_build_features
"""

import argparse
import numpy as np

from Daily_Modeling import config
from Daily_Modeling.data_utils.load_raw import load_station_metadata, discover_station_days
from Daily_Modeling.data_utils.build_features import (
    load_reanalysis_datasets,
    build_reanalysis_patches,
    build_dem_patches,
)


def main(start_date: str = "1980-01-01", end_date: str = "2024-12-31"):
    output_dir = config.FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_station_metadata()
    print(f"Stations: {len(meta)}")

    # --- Reanalysis ---
    station_days = discover_station_days(meta, start_date=start_date, end_date=end_date)
    datasets = load_reanalysis_datasets()
    patches, stations, years, months, days, var_names = \
        build_reanalysis_patches(meta, station_days, datasets)

    re_path = output_dir / "reanalysis_patches_daily_station_centered.npz"
    np.savez_compressed(
        str(re_path),
        patches=patches, stations=stations, years=years, months=months, days=days,
        variables=np.array(var_names, dtype=object),
    )
    print(f"Saved reanalysis patches -> {re_path}  shape={patches.shape}")

    # --- DEM (max-size for multi-resolution HP tuning) ---
    dem = build_dem_patches(
        meta,
        local_cfg=config.DEM_MAX_LOCAL,
        regional_cfg=config.DEM_MAX_REGIONAL,
    )
    local_arr = np.stack([dem[s]["local"] for s in sorted(dem)], axis=0).astype(np.float32)
    regional_arr = np.stack([dem[s]["regional"] for s in sorted(dem)], axis=0).astype(np.float32)
    dem_stations = np.array(sorted(dem.keys()), dtype=object)
    dem_path = output_dir / "dem_patches.npz"
    np.savez_compressed(
        str(dem_path),
        dem_local_raw=local_arr, dem_regional_raw=regional_arr,
        stations=dem_stations,
    )
    print(f"Saved DEM patches -> {dem_path}  local={local_arr.shape}  regional={regional_arr.shape}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="1980-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()
    main(start_date=args.start_date, end_date=args.end_date)
