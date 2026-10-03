"""Stage 1 entry point: build the weekly (or daily) dataset NPZ from raw data.

Orchestrates load_raw.py -> build_features.py -> assemble_dataset.py in this
stage. Feature caches are reused across runs and frequencies; pass --rebuild
to regenerate them after raw inputs change (e.g. a station added to
station_locations.csv).

Usage: python -m LAND_AS.s1_prepare.prepare [--rebuild] [--daily]
"""
import argparse

import numpy as np

from LAND_AS.s1_prepare import config as daily_config
from LAND_AS.s1_prepare.assemble_dataset import assemble
from LAND_AS.s1_prepare.build_features import build_dem_patches, build_reanalysis_patches, load_reanalysis_datasets
from LAND_AS.s1_prepare.load_raw import discover_station_days, load_station_metadata
from LAND_AS import config


def prepare(start_date="1980-01-01", end_date="2024-12-31", rebuild=False, freq="weekly"):
    required = [
        daily_config.DEM_PATH,
        daily_config.STATION_METADATA_PATH,
        daily_config.REANALYSIS_DIR,
        daily_config.DAILY_RAINFALL_DIR,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing American Samoa inputs:\n" + "\n".join(missing))
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    reanalysis_path = config.FEATURE_DIR / "reanalysis_daily.npz"
    dem_path = config.FEATURE_DIR / "dem.npz"

    metadata = load_station_metadata()
    if rebuild or not reanalysis_path.exists():
        days = discover_station_days(metadata, start_date=start_date, end_date=end_date)
        patches, stations, years, months, day, variables = build_reanalysis_patches(
            metadata, days, load_reanalysis_datasets()
        )
        np.savez_compressed(
            reanalysis_path,
            patches=patches,
            stations=stations,
            years=years,
            months=months,
            days=day,
            variables=np.asarray(variables, dtype=object),
        )

    if rebuild or not dem_path.exists():
        patches = build_dem_patches(
            metadata,
            local_cfg=daily_config.DEM_MAX_LOCAL,
            regional_cfg=daily_config.DEM_MAX_REGIONAL,
        )
        stations = sorted(patches)
        np.savez_compressed(
            dem_path,
            dem_local_raw=np.stack([patches[s]["local"] for s in stations]).astype(np.float32),
            dem_regional_raw=np.stack([patches[s]["regional"] for s in stations]).astype(np.float32),
            stations=np.asarray(stations, dtype=object),
        )

    return assemble(
        out_path=config.DAILY_DATASET_PATH if freq == "daily" else config.DATASET_PATH,
        reanalysis_npz=reanalysis_path,
        dem_npz=dem_path,
        freq=freq,
    )


def main():
    parser = argparse.ArgumentParser(description="Build the weekly American Samoa LAND dataset.")
    parser.add_argument("--start-date", default="1980-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--daily", action="store_true",
                        help="assemble daily_dataset.npz (no weekly aggregation)")
    args = parser.parse_args()
    freq = "daily" if args.daily else "weekly"
    print(prepare(args.start_date, args.end_date, args.rebuild, freq))


if __name__ == "__main__":
    main()
