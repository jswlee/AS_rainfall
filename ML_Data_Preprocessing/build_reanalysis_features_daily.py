"""
Simplified Daily Reanalysis Feature Builder

This script extracts patches of climate data for given station locations from a 
directory of pre-concatenated NetCDF files. It then standardizes and exports 
the features to a single file.
"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def load_climate_data(data_dir: Path) -> xr.Dataset:
    """
    Loads and merges all NetCDF files in a directory into a single xarray Dataset.
    
    This version opens each file individually to be more stable on sensitive systems.
    """
    print(f"--> Loading all datasets from '{data_dir}' (stable mode)...")
    
    datasets = []
    nc_files = sorted(list(data_dir.glob("*.nc")))
    
    for i, file_path in enumerate(nc_files):
        print(f"    Loading file {i+1}/{len(nc_files)}: {file_path.name}")
        with xr.open_dataset(file_path) as ds:
            # Load the data fully into memory and append to our list
            datasets.append(ds.load())
            
    print("--> Merging datasets...")
    # Combine the list of in-memory datasets
    combined_ds = xr.merge(datasets)
    return combined_ds

def extract_patch(da: xr.DataArray, lat: float, lon: float, patch_size: int) -> np.ndarray:
    """
    Extracts a 2D spatial patch from a DataArray centered on the nearest point
    to the given lat/lon coordinates.
    """
    half_size = patch_size // 2
    
    # Find the nearest grid point using xarray's selection method
    point_data = da.sel(latitude=lat, longitude=lon, method="nearest")
    center_lat = point_data.latitude.item()
    center_lon = point_data.longitude.item()

    # Get the integer indices of the center point
    lat_idx = np.abs(da.latitude.values - center_lat).argmin()
    lon_idx = np.abs(da.longitude.values - center_lon).argmin()

    # Define the slice boundaries for the patch
    lat_start = max(0, lat_idx - half_size)
    lon_start = max(0, lon_idx - half_size)
    
    # Extract the patch using integer-based indexing for speed
    patch = da.isel(
        latitude=slice(lat_start, lat_start + patch_size), 
        longitude=slice(lon_start, lon_start + patch_size)
    ).values
    
    # Pad with zeros if the patch is smaller than desired (i.e., at an edge)
    pad_h = patch_size - patch.shape[0]
    pad_w = patch_size - patch.shape[1]
    if pad_h > 0 or pad_w > 0:
        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), 'constant', constant_values=0)
        
    return np.nan_to_num(patch)

def build_features_for_stations(
    ds: xr.Dataset, 
    stations_df: pd.DataFrame, 
    start_date: str, 
    end_date: str, 
    patch_size: int,
    max_workers: int
):
    """
    Orchestrates the parallel extraction of feature patches for all stations over a date range.
    """
    all_features = {name: {} for name in stations_df.index}
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    variables = list(ds.data_vars)
    
    print(f"--> Building features for {len(stations_df)} stations and {len(date_range)} days...")

    # Pre-select the time range from the main dataset for efficiency
    ds_subset = ds.sel(valid_time=slice(start_date, end_date))

    def _process_day(date):
        """Task for a single day: extract patches for all stations and variables."""
        daily_results = {}
        for station_name, station_info in stations_df.iterrows():
            station_lat, station_lon = station_info['LAT'], station_info['LONG']
            daily_results[station_name] = {}
            
            # Select the data for the specific day
            day_ds = ds_subset.sel(valid_time=date, method="nearest")

            for var in variables:
                da = day_ds[var]
                # If a variable has a pressure level, just use the first one
                if 'pressure_level' in da.dims:
                    da = da.isel(pressure_level=0, drop=True)
                daily_results[station_name][var] = extract_patch(da, station_lat, station_lon, patch_size)
        return date, daily_results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_day, date): date for date in date_range}
        
        for i, future in enumerate(as_completed(futures)):
            date, daily_results = future.result()
            for station_name, patches in daily_results.items():
                all_features[station_name][date.strftime('%Y-%m-%d')] = patches
            print(f"    Processed {date.strftime('%Y-%m-%d')} ({i+1}/{len(date_range)})")

    return all_features

def standardize_and_export(all_features: dict, out_path: Path, patch_size: int):
    """Computes statistics, standardizes all patches, and exports to a single .npz file."""
    print("--> Standardizing features and exporting to NPZ...")
    variables = list(next(iter(next(iter(all_features.values())).values())).keys())
    
    # 1. Compute statistics (mean and std) for each variable
    stats = {}
    for var in variables:
        # Flatten all patches for a variable into a single array to compute stats
        all_patches = np.array([
            patches[var] 
            for station_data in all_features.values() 
            for patches in station_data.values()
        ])
        stats[var] = {'mean': np.mean(all_patches), 'std': np.std(all_patches)}
        print(f"    {var}: mean={stats[var]['mean']:.2f}, std={stats[var]['std']:.2f}")

    # 2. Standardize and collect data for export
    export_patches, export_meta = [], []
    for station, daily_data in all_features.items():
        for date_str, patches in daily_data.items():
            standardized_patches = []
            for var in variables:
                mean, std = stats[var]['mean'], stats[var]['std']
                # Avoid division by zero
                standardized_patch = (patches[var] - mean) / (std if std > 0 else 1.0)
                standardized_patches.append(standardized_patch)
            
            export_patches.append(np.stack(standardized_patches))
            export_meta.append({'station': station, 'date': date_str})

    # 3. Save to a single compressed NPZ file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        patches=np.array(export_patches, dtype=np.float32),
        meta=np.array(export_meta),
        variables=np.array(variables)
    )
    print(f"--> Successfully saved standardized features to '{out_path}'")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Simplified daily reanalysis feature builder.")
    parser.add_argument("--data_dir", type=Path, default=Path("raw_data/climate_variables_daily_concatenated"),
                        help="Directory with concatenated daily NetCDF files.")
    parser.add_argument("--station_file", type=Path, default=Path("raw_data/station_locations.csv"),
                        help="Path to the station metadata CSV file.")
    parser.add_argument("--output_file", type=Path, default=Path("output/daily_features.npz"),
                        help="Path to save the final standardized .npz file.")
    parser.add_argument("--start_date", type=str, default="1980-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="1984-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--patch_size", type=int, default=3, help="Size of the spatial patch (e.g., 3 for 3x3).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    start_time = time.time()

    # 1. Load station metadata
    stations_df = pd.read_csv(args.station_file).set_index('Station')

    # 2. Load all climate data into one unified dataset
    climate_ds = load_climate_data(args.data_dir)
    print("    Combined dataset variables:", list(climate_ds.data_vars))

    # 3. Build features for all stations in parallel
    all_features = build_features_for_stations(
        climate_ds, stations_df, args.start_date, args.end_date, args.patch_size, args.workers
    )

    # 4. Standardize and export the final features
    standardize_and_export(all_features, args.output_file, args.patch_size)
    
    print(f"\nTotal script execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()