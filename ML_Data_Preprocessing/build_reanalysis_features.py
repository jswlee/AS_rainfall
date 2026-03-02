"""
Optimized version of build_reanalysis_features.py with:
1. Dynamic variable naming based on actual pressure levels
2. Parallelized feature extraction
"""

import os
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Optional

import ML_Data_Preprocessing.config as config
from ML_Data_Preprocessing.utils import find_nearest_point, discover_station_months, discover_station_days, visualize_grid
from ML_Data_Preprocessing.extract_station_metadata import get_station_metadata


def _get_time_dim(da: xr.DataArray) -> Optional[str]:
    if 'valid_time' in da.dims:
        return 'valid_time'
    if 'time' in da.dims:
        return 'time'
    return None


def _build_time_key_index(da: xr.DataArray, time_dim: str, time_interval: str) -> dict:
    times = np.asarray(da[time_dim].values)
    if time_interval == 'daily':
        keys = times.astype('datetime64[D]')
    else:
        keys = times.astype('datetime64[M]')

    key_to_index: dict = {}
    for i, k in enumerate(keys.tolist()):
        if k not in key_to_index:
            key_to_index[k] = i
    return key_to_index


def _filter_station_data_map_by_coverage(station_data_map: dict, climate_data: dict, time_interval: str) -> dict:
    required_vars = list(climate_data.keys())
    if not required_vars:
        return station_data_map

    available_days = None
    for var_name in required_vars:
        da = climate_data[var_name]
        time_dim = _get_time_dim(da)
        if time_dim is None:
            raise ValueError(f"Variable '{var_name}' has no supported time dimension.")
        times = np.asarray(da[time_dim].values)
        if time_interval == 'daily':
            keys = times.astype('datetime64[D]')
        else:
            keys = times.astype('datetime64[M]')
        var_keys = set(keys.tolist())
        available_days = var_keys if available_days is None else (available_days & var_keys)

    available_days = available_days or set()

    filtered = {}
    before = 0
    after = 0
    for station, tuples in station_data_map.items():
        before += len(tuples)
        kept = []
        for t in tuples:
            if time_interval == 'daily':
                y, m, d = t
                key = np.datetime64(f"{int(y)}-{int(m):02d}-{int(d):02d}").astype('datetime64[D]').tolist()
            else:
                y, m = t
                key = np.datetime64(f"{int(y)}-{int(m):02d}").astype('datetime64[M]').tolist()
            if key in available_days:
                kept.append(t)
        if kept:
            filtered[station] = kept
            after += len(kept)

    dropped = before - after
    label = "station-days" if time_interval == 'daily' else "station-months"
    print(f"[coverage] {label} before={before}, after={after}, dropped={dropped} (requires all variables present)")
    return filtered


class OptimizedReanalysisFeatureBuilder:
    """
    Optimized version with simplified code, dynamic variable names, and parallelization.
    """
    
    def __init__(self, time_interval='monthly', statistic='mean', n_workers=None):
        self.time_interval = time_interval
        
        if self.time_interval == 'monthly':
            self.reanalysis_dir = config.REANALYSIS_DIR_MONTHLY
        elif self.time_interval == 'daily':
            self.reanalysis_dir = config.REANALYSIS_DIR_DAILY
        else:
            raise ValueError(f"Unsupported time interval: {self.time_interval}")

        self.statistic = config.STATISTIC_MAPPING[statistic]    
        self.patch_size = config.REANALYSIS_PATCH_SIZE
        
        # Set number of workers
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        # Get the appropriate variable configs
        if self.time_interval == 'monthly':
            self.variable_configs = config.MONTHLY_REANALYSIS_VARIABLE_CONFIGS.copy()
        else:
            self.variable_configs = config.DAILY_REANALYSIS_VARIABLE_CONFIGS.copy()
        
        # Dictionary to store processed climate data
        self.climate_data = {}

        self._time_dim_by_var = {}
        self._time_index_by_var = {}
        
        # Variable mapping for file names
        self.variable_mapping = config.VARIABLE_MAPPING
    
    def get_file_path(self, variable):
        """Get the file path for a given variable."""
        var_name = self.variable_mapping[variable]
        time_str = config.TIME_INTERVAL_MAPPING[self.time_interval]
        filename = f"{var_name}.{time_str}.{self.statistic}.nc"
        return os.path.join(self.reanalysis_dir, filename)
    
    def load_dataset(self, file_path):
        """Load a NetCDF dataset and load data into memory immediately."""
        try:
            ds = xr.open_dataset(file_path)
            ds.load()  # Force all data into memory to avoid lazy-loading stalls later
            return ds
        except Exception as e:
            print(f"Error loading dataset {file_path}: {e}")
            return None
    
    def process_variable(self, var_name):
        """Process a climate variable based on its configuration."""
        if var_name not in self.variable_configs:
            print(f"Variable '{var_name}' not found in configurations")
            return False
        
        cfg = self.variable_configs[var_name]
        variable = cfg["variable"]
        
        # Check for dependencies
        if "depends_on" in cfg:
            for dep in cfg["depends_on"]:
                if dep not in self.climate_data:
                    if not self.process_variable(dep):
                        print(f"Failed to process dependency {dep} for {var_name}")
                        return False
        
        # Load dataset
        file_path = self.get_file_path(variable)
        ds = self.load_dataset(file_path)
        if ds is None:
            return False
        
        # Get the variable key in the dataset
        var_keys = list(ds.data_vars)
        var_key = None
        for key in var_keys:
            if key.lower() in self.variable_mapping[variable].lower():
                var_key = key
                break
        
        if var_key is None and len(var_keys) > 0:
            var_key = var_keys[0]
            print(f"Warning: Using first variable {var_key} for {var_name}")
        
        if var_key is None:
            print(f"No variables found in dataset for {var_name}")
            return False
        
        # Determine level dimension name  
        level_dim = 'pressure_level' if 'pressure_level' in ds.dims else 'level'
        
        # Process based on operation type
        result = None
        
        if "operation" in cfg and cfg["operation"] == "diff":
            # Handle difference between two levels
            level1, level2 = cfg["levels"]
            try:
                data1 = ds.sel({level_dim: level1})
                data2 = ds.sel({level_dim: level2})
                result = data1[var_key] - data2[var_key]
            except Exception as e:
                print(f"Error computing difference for {var_name}: {e}")
                return False
            
        elif "operation" in cfg and cfg["operation"] == "multiply":
            # Handle multiplication with another variable
            multiply_with = cfg["multiply_with"]
            if multiply_with not in self.climate_data:
                print(f"Multiplication variable {multiply_with} not found in climate_data")
                return False
            
            try:
                data = ds.sel({level_dim: cfg["level"]})
                prim = data[var_key]
                dep = self.climate_data[multiply_with]

                # Align on a common time-like dimension if present
                time_dims_prim = [d for d in prim.dims if d in ("valid_time", "time")]
                time_dims_dep = [d for d in dep.dims if d in ("valid_time", "time")]

                if time_dims_prim and time_dims_dep:
                    time_dim = time_dims_prim[0]
                    if time_dim != time_dims_dep[0]:
                        print(
                            f"Warning: time dim mismatch for multiply op: prim={time_dims_prim[0]}, dep={time_dims_dep[0]}"
                        )
                    prim_aligned, dep_aligned = xr.align(prim, dep, join='inner')
                    
                    # Multiply on raw NumPy arrays to avoid xarray overhead
                    prim_vals = prim_aligned.values
                    dep_vals = dep_aligned.values
                    result_values = prim_vals * dep_vals
                    # Wrap result back into a DataArray with aligned coordinates
                    result = xr.DataArray(
                        result_values,
                        coords=prim_aligned.coords,
                        dims=prim_aligned.dims,
                        attrs=prim_aligned.attrs
                    )
                else:
                    result_values = prim.values * dep.values
                    result = xr.DataArray(
                        result_values,
                        coords=prim.coords,
                        dims=prim.dims,
                        attrs=prim.attrs
                    )
            except Exception as e:
                print(f"Error computing product for {var_name}: {e}")
                return False
            
        else:
            # Handle simple selection
            try:
                if "level" in cfg:
                    data = ds.sel({level_dim: cfg["level"]})
                else:
                    data = ds
                
                result = data[var_key]
            except Exception as e:
                print(f"Error selecting data for {var_name}: {e}")
                return False
        
        # Store result
        self.climate_data[var_name] = result
        return True

    def build_time_index(self):
        self._time_dim_by_var = {}
        self._time_index_by_var = {}
        for var_name, da in self.climate_data.items():
            time_dim = _get_time_dim(da)
            if time_dim is None:
                continue
            self._time_dim_by_var[var_name] = time_dim
            self._time_index_by_var[var_name] = _build_time_key_index(da, time_dim, self.time_interval)
    
    def process_all_variables(self):
        """Process all climate variables defined in variable_configs."""
        success = True
        all_vars = list(self.variable_configs.keys())
        print(f"[process_all_variables] Starting processing of {len(all_vars)} variables")
        for idx, var_name in enumerate(all_vars, start=1):
            if var_name not in self.climate_data:
                print(f"[process_all_variables] ({idx}/{len(all_vars)}) Processing {var_name}...")
                if not self.process_variable(var_name):
                    print(f"Failed to process {var_name}")
                    success = False
            else:
                print(f"[process_all_variables] ({idx}/{len(all_vars)}) Skipping already processed {var_name}")
        print(f"[process_all_variables] Completed with success={success}")
        return success
    
    def extract_patch_for_variable_and_time(self, var_name, year, month, day, lat, lon):
        """
        Extract a single patch for a specific variable, time, and location.
        """
        if var_name not in self.climate_data:
            return None

        try:
            da = self.climate_data[var_name]
            time_dim = self._time_dim_by_var.get(var_name)
            time_index = self._time_index_by_var.get(var_name)

            if time_dim is None or time_index is None:
                return None

            if self.time_interval == 'daily' and day is not None:
                key = np.datetime64(
                    f"{int(year)}-{int(month):02d}-{int(day):02d}"
                ).astype('datetime64[D]').tolist()
            else:
                key = np.datetime64(
                    f"{int(year)}-{int(month):02d}"
                ).astype('datetime64[M]').tolist()

            if key not in time_index:
                return None

            da_t = da.isel({time_dim: int(time_index[key])})
            return self._extract_spatial_patch(da_t, lat, lon)
        except Exception as e:
            print(f"Error extracting {var_name} for {year}-{month:02d}-{day or 'XX'}: {e}")
            return None

    def _extract_spatial_patch(self, da, lat, lon):
        """Extract spatial patch around a location."""
        # Get coordinate arrays
        if 'latitude' in da.dims:
            lats = da.latitude.values
            lons = da.longitude.values
            lat_dim, lon_dim = 'latitude', 'longitude'
        elif 'lat' in da.dims:
            lats = da.lat.values
            lons = da.lon.values
            lat_dim, lon_dim = 'lat', 'lon'
        else:
            return None
        
        # Find nearest grid point using Haversine distance
        lat_idx, lon_idx = find_nearest_point(lat, lon, lats, lons)
        
        # Calculate patch bounds
        half_size = self.patch_size // 2
        lat_start = max(0, lat_idx - half_size)
        lat_end = min(len(lats), lat_idx + half_size + 1)
        lon_start = max(0, lon_idx - half_size)
        lon_end = min(len(lons), lon_idx + half_size + 1)
        
        # Extract patch
        patch = da.isel({lat_dim: slice(lat_start, lat_end), lon_dim: slice(lon_start, lon_end)}).values
        
        # Handle level dimension if present
        if len(patch.shape) > 2:
            # If there are multiple levels, take the first one (usually surface)
            patch = patch[0]
        
        # Ensure the patch has the correct shape
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded_patch = np.zeros((self.patch_size, self.patch_size))
            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded_patch
        
        return patch


def process_station_batch(args):
    """Process a batch of station-date combinations. This function runs in parallel processes."""
    station_batch, climate_data, time_interval, patch_size = args
    
    # Create a local feature builder for this process
    builder = OptimizedReanalysisFeatureBuilder(time_interval=time_interval)
    builder.climate_data = climate_data  # Use the pre-processed climate data
    builder.patch_size = patch_size
    builder.build_time_index()
    
    print(f"[process_station_batch] Starting batch with {len(station_batch)} stations; time_interval={time_interval}")
    batch_features = {}
    
    for s_idx, (station_name, metadata, date_tuples) in enumerate(station_batch, start=1):
        lat = metadata['latitude']
        lon = metadata['longitude']
        
        station_feats = {}
        for date_tuple in date_tuples:
            if time_interval == 'daily':
                year, month, day = date_tuple
            else:
                year, month = date_tuple
                day = None
            
            # Extract patches for all variables
            time_feats = {}
            ok = True
            for var_name in climate_data.keys():
                patch = builder.extract_patch_for_variable_and_time(
                    var_name, year, month, day, lat, lon
                )
                if patch is None:
                    ok = False
                    break
                time_feats[var_name] = patch
            if ok:
                station_feats[date_tuple] = time_feats
        
        batch_features[station_name] = station_feats
    
    print(f"[process_station_batch] Finished batch with {len(station_batch)} stations")
    return batch_features


def build_features_parallel(station_metadata, station_data_map, climate_data, 
                          time_interval, patch_size, n_workers=None):
    """
    Build features using parallel processing.
    """
    n_workers = n_workers or max(1, mp.cpu_count() - 1)
    print(f"Using {n_workers} parallel workers")
    
    # Create batches of stations
    stations_list = []
    for station_name, metadata in station_metadata.items():
        if station_name in station_data_map:
            date_tuples = station_data_map[station_name]
            stations_list.append((station_name, metadata, date_tuples))
    
    # Split into batches
    batch_size = max(1, len(stations_list) // n_workers)
    batches = [stations_list[i:i + batch_size] for i in range(0, len(stations_list), batch_size)]
    print(f"[build_features_parallel] Total stations={len(stations_list)}, batches={len(batches)}, batch_size={batch_size}")
    
    # Prepare arguments for each batch
    batch_args = [
        (batch, climate_data, time_interval, patch_size)
        for batch in batches
    ]
    
    # Process batches in parallel
    all_features = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_batch = {executor.submit(process_station_batch, args): i for i, args in enumerate(batch_args)}
        print(f"[build_features_parallel] Submitted {len(future_to_batch)} futures to executor")
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_features = future.result()
                all_features.update(batch_features)
                print(f"Completed batch {batch_idx + 1}/{len(batches)}")
            except Exception as e:
                print(f"Batch {batch_idx} failed: {e}")
    
    return all_features


def main(time_interval='monthly', start_date='1980-01-01', end_date='1988-07-31', n_workers=None):
    """
    Main function with optimizations.
    """
    print(f"[main] Starting optimized {time_interval} reanalysis feature building from {start_date} to {end_date}...")
    
    # Load station metadata
    station_metadata = get_station_metadata(config.STATION_METADATA_PATH)
    print(f"[main] Loaded station metadata for {len(station_metadata)} stations")
    
    # Create feature builder and process variables
    feature_builder = OptimizedReanalysisFeatureBuilder(time_interval=time_interval, n_workers=n_workers)
    
    print("[main] Processing climate variables...")
    success = feature_builder.process_all_variables()
    missing_vars = [v for v in feature_builder.variable_configs.keys() if v not in feature_builder.climate_data]
    if (not success) or missing_vars:
        raise RuntimeError(f"Missing processed variables: {missing_vars}")
    
    # Discover available station data
    if time_interval == 'monthly':
        print("[main] Discovering available station months...")
        station_data_map = discover_station_months(station_metadata)
    else:
        print("[main] Discovering available station days...")
        station_data_map = discover_station_days(
            station_metadata, 
            start_date=start_date, 
            end_date=end_date
        )

    station_data_map = _filter_station_data_map_by_coverage(
        station_data_map, feature_builder.climate_data, time_interval
    )
    
    total_pairs = sum(len(v) for v in station_data_map.values())
    print(f"[main] Building features for {len(station_data_map)} stations across {total_pairs} station-date pairs...")
    
    # Build features in parallel
    print("[main] Starting parallel feature building...")
    all_features = build_features_parallel(
        station_metadata, station_data_map, feature_builder.climate_data, 
        time_interval, config.REANALYSIS_PATCH_SIZE, n_workers
    )
    print("[main] Finished parallel feature building.")
    
    # Export results
    npz_dir = os.path.join(str(config.OUTPUT_DIR), "reanalysis_npz")
    os.makedirs(npz_dir, exist_ok=True)
    
    filename = f"reanalysis_features_all_raw_{time_interval}.npz"
    export_path = export_features_npz(all_features, feature_builder.variable_configs, npz_dir, filename, time_interval)
    
    if export_path:
        print(f"[main] Saved optimized features to {export_path}")
    
    print("[main] Done!")


def compute_variable_statistics(all_features):
    """Compute statistics for standardization."""
    variable_stats = {}
    
    # Collect all patches for each variable
    station_count = len(all_features)
    print(f"[compute_variable_statistics] Starting over {station_count} stations")
    for s_idx, station_feats in enumerate(all_features.values(), start=1):
        for time_feats in station_feats.values():
            for var_name, patch in time_feats.items():
                if var_name not in variable_stats:
                    variable_stats[var_name] = []
                variable_stats[var_name].append(patch.flatten())
        if s_idx % 10 == 0:
            print(f"[compute_variable_statistics] Processed {s_idx}/{station_count} stations")
    
    # Compute mean and std for each variable
    for var_name, patches_list in variable_stats.items():
        all_values = np.concatenate(patches_list)
        variable_stats[var_name] = {
            'mean': np.mean(all_values),
            'std': np.std(all_values)
        }
        print(f"{var_name} - Mean: {variable_stats[var_name]['mean']:.6f}, Std: {variable_stats[var_name]['std']:.6f}")
    
    return variable_stats


def standardize_features(all_features, variable_stats):
    """Standardize features using computed statistics."""
    standardized_features = {}
    
    total_stations = len(all_features)
    print(f"[standardize_features] Starting standardization for {total_stations} stations")
    for s_idx, (station_name, station_feats) in enumerate(all_features.items(), start=1):
        standardized_station = {}
        for d_idx, (date_tuple, time_feats) in enumerate(station_feats.items(), start=1):
            standardized_time = {}
            for var_name, patch in time_feats.items():
                if var_name in variable_stats:
                    mean = variable_stats[var_name]['mean']
                    std = variable_stats[var_name]['std']
                    try:
                        standardized_patch = (patch - mean) / std
                    except ZeroDivisionError:
                        print(f"Warning: Zero standard deviation for variable {var_name}")
                        return False
                    standardized_time[var_name] = standardized_patch
                else:
                    standardized_time[var_name] = patch
            standardized_station[date_tuple] = standardized_time
        if s_idx % 10 == 0:
            print(f"[standardize_features] Processed station {s_idx}/{total_stations} ('{station_name}')")
        standardized_features[station_name] = standardized_station
    
    return standardized_features


def export_features_npz(all_features, variable_configs, out_dir, filename, time_interval):
    """Export features to NPZ format."""
    os.makedirs(out_dir, exist_ok=True)
    
    var_order = list(variable_configs.keys())
    entries = []
    meta = []
    
    for station_name, station_feats in all_features.items():
        for date_tuple, time_features in station_feats.items():
            patches = []
            for v in var_order:
                if v not in time_features:
                    raise KeyError(f"Missing variable '{v}' for station '{station_name}' at {date_tuple}")
                patches.append(np.asarray(time_features[v]))
            
            arr = np.stack(patches, axis=0)  # (V,H,W)
            entries.append(arr[np.newaxis, ...])
            meta.append((station_name, *date_tuple))
    
    if not entries:
        print("No features to export.")
        return None
    
    big = np.concatenate(entries, axis=0).astype(np.float32)
    stations = np.array([m[0] for m in meta], dtype=object)
    years = np.array([m[1] for m in meta], dtype=np.int32)
    months = np.array([m[2] for m in meta], dtype=np.int32)
    
    save_path = os.path.join(out_dir, filename)
    data_to_save = {
        'patches': big,
        'stations': stations,
        'years': years,
        'months': months,
        'variables': np.array(var_order, dtype=object),
        'patch_size': np.array(config.REANALYSIS_PATCH_SIZE)
    }
    
    if time_interval == 'daily':
        days = np.array([m[3] for m in meta], dtype=np.int32)
        data_to_save['days'] = days
    
    np.savez_compressed(save_path, **data_to_save)
    return save_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build optimized reanalysis features.')
    parser.add_argument('time_interval', type=str, nargs='?', default='daily', 
                       choices=['monthly', 'daily'], help='Time interval to process')
    parser.add_argument('--workers', type=int, default=9, 
                       help='Number of parallel workers')
    parser.add_argument('--start_date', type=str, default='1980-01-01', 
                       help='Start date for station data discovery')
    parser.add_argument('--end_date', type=str, default='2024-12-31', 
                       help='End date for station data discovery')
    
    args = parser.parse_args()
    main(time_interval=args.time_interval, start_date=args.start_date, end_date=args.end_date, n_workers=args.workers)
