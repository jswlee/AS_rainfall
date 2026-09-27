import argparse

import numpy as np

from Daily_Modeling import config as daily_config
from Daily_Modeling.data_utils.assemble_dataset import assemble
from Daily_Modeling.data_utils.build_features import build_dem_patches, build_reanalysis_patches, load_reanalysis_datasets
from Daily_Modeling.data_utils.load_raw import discover_station_days, load_station_metadata
from LAND_AS import config


def prepare(start_date="1980-01-01", end_date="2024-12-31", rebuild=False):
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
        out_path=config.DATASET_PATH,
        reanalysis_npz=reanalysis_path,
        dem_npz=dem_path,
        freq="weekly",
    )


def main():
    parser = argparse.ArgumentParser(description="Build the weekly American Samoa LAND dataset.")
    parser.add_argument("--start-date", default="1980-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    print(prepare(args.start_date, args.end_date, args.rebuild))


if __name__ == "__main__":
    main()
