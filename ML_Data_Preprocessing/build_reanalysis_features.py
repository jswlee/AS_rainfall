"""
Reanalysis Features Builder Module

This module handles the extraction of reanalysis features at station locations,
creating patches of climate variables centered on the nearest grid point to each station.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import ML_Data_Preprocessing.config as config
from ML_Data_Preprocessing.utils import find_nearest_point, discover_station_months, visualize_grid
from ML_Data_Preprocessing.extract_station_metadata import get_station_metadata


class ReanalysisFeatureBuilder:
    """
    Class for building reanalysis feature patches at station locations.
    
    This class handles:
    1. Loading reanalysis data from NetCDF files using configuration-based mappings
    2. Processing climate variables with operations like differences and products
    3. Extracting patches centered on station locations
    4. Standardizing features for ML input
    
    Implementation aligns with the climate processor approach from Create_ML_Data.
    """
    
    def __init__(self, reanalysis_dir=None, time_interval="monthly", lon_slice=None, lat_slice=None, time_slice=None):
        """
        Initialize the ReanalysisFeatureBuilder.
        
        Parameters
        ----------
        reanalysis_dir : str, optional
            Directory containing climate data files. If None, uses the path from config based on time_interval.
        time_interval : str, optional
            Time interval for data processing ("monthly" or "daily"). Default is "monthly".
        lon_slice : slice, optional
            Longitude slice for data selection
        lat_slice : slice, optional
            Latitude slice for data selection
        time_slice : slice, optional
            Time slice for data selection
        """
        # Set reanalysis directory based on time interval if not provided
        if reanalysis_dir is not None:
            self.reanalysis_dir = reanalysis_dir
        elif time_interval == "daily":
            self.reanalysis_dir = config.REANALYSIS_DIR_DAILY
        else:
            self.reanalysis_dir = config.REANALYSIS_DIR_MONTHLY
        
        self.time_interval = time_interval
        self.lon_slice = lon_slice
        self.lat_slice = lat_slice
        self.time_slice = time_slice
        self.patch_size = config.REANALYSIS_PATCH_SIZE
        
        # Use the mappings from the config module
        self.variable_mapping = config.VARIABLE_MAPPING
        self.time_interval_mapping = config.TIME_INTERVAL_MAPPING
        self.statistic_mapping = config.STATISTIC_MAPPING
        
        # Use the specified time interval, or fall back to config default
        self.default_time_interval = time_interval if time_interval in self.time_interval_mapping else config.DEFAULT_TIME_INTERVAL
        self.default_statistic = config.DEFAULT_STATISTIC
        
        # Dictionary to store variable configurations
        self.variable_configs = config.REANALYSIS_VARIABLE_CONFIGS.copy()
        
        # Dictionary to store each variable's data
        self.climate_data = {}
        
        # Dictionary to store variable statistics for standardization
        self.variable_stats = {}
    
    def get_file_path(self, variable, time_interval=None, statistic=None):
        """
        Construct the file path for a climate variable.
        
        Parameters
        ----------
        variable : str
            Climate variable name (use keys from variable_mapping)
        time_interval : str, optional
            Time interval (use keys from time_interval_mapping). If None, uses default_time_interval.
        statistic : str, optional
            Statistic type (use keys from statistic_mapping). If None, uses default_statistic.
        
        Returns
        -------
        str
            Complete file path
        """
        if variable not in self.variable_mapping:
            raise ValueError(f"Unknown variable: {variable}")
        
        # Use defaults if not specified
        time_interval = time_interval or self.default_time_interval
        statistic = statistic or self.default_statistic
        
        if time_interval not in self.time_interval_mapping:
            raise ValueError(f"Unknown time interval: {time_interval}")
            
        if statistic not in self.statistic_mapping:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        # Construct file name
        var_name = self.variable_mapping[variable]
        
        # Add time interval and statistic
        file_name = f"{var_name}.{self.time_interval_mapping[time_interval]}.{self.statistic_mapping[statistic]}.nc"
        
        # Return full path
        return os.path.join(self.reanalysis_dir, file_name)
    
    def load_dataset(self, file_path):
        """
        Load a dataset from a NetCDF file.
        
        Parameters
        ----------
        file_path : str
            Path to the NetCDF file
        
        Returns
        -------
        xarray.Dataset
            Loaded dataset with time selection applied
        """
        try:
            if self.time_slice:
                ds = xr.open_dataset(file_path).sel(time=self.time_slice)
            else:
                ds = xr.open_dataset(file_path)
            return ds
        except Exception as e:
            print(f"Error loading dataset from {file_path}: {e}")
            return None
    
    def process_variable(self, var_name):
        """
        Process a climate variable based on its configuration.
        
        Parameters
        ----------
        var_name : str
            Name of the variable to process (key in variable_configs)
        
        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        if var_name not in self.variable_configs:
            print(f"Variable '{var_name}' not found in configurations")
            return False
        
        config = self.variable_configs[var_name]
        variable = config["variable"]
        
        # Check for dependencies
        if "depends_on" in config:
            for dep in config["depends_on"]:
                if dep not in self.climate_data:
                    if not self.process_variable(dep):
                        print(f"Failed to process dependency {dep} for {var_name}")
                        return False
        
        # Handle custom file path if specified
        if "custom_file" in config:
            file_path = os.path.join(self.reanalysis_dir, config["custom_file"])
        else:
            # Construct file path based on variable configuration
            try:
                file_path = self.get_file_path(
                    variable,
                )
            except ValueError as e:
                print(f"Error constructing file path for {var_name}: {e}")
                return False
        
        # Load dataset
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
        
        # Process based on operation type
        result = None
        
        if "operation" in config and config["operation"] == "diff":
            # Handle difference between two levels
            level1, level2 = config["levels"]
            try:
                data1 = ds.sel(level=level1)
                data2 = ds.sel(level=level2)
                if self.lon_slice and self.lat_slice:
                    data1 = data1.sel(lon=self.lon_slice, lat=self.lat_slice)
                    data2 = data2.sel(lon=self.lon_slice, lat=self.lat_slice)
                result = data1[var_key] - data2[var_key]
            except Exception as e:
                print(f"Error computing difference for {var_name}: {e}")
                return False
            
        elif "operation" in config and config["operation"] == "multiply":
            # Handle multiplication with another variable
            multiply_with = config["multiply_with"]
            if multiply_with not in self.climate_data:
                print(f"Multiplication variable {multiply_with} not found")
                return False
            
            try:
                # Select data based on level if specified
                if "level" in config:
                    data = ds.sel(level=config["level"])
                else:
                    data = ds
                
                if self.lon_slice and self.lat_slice:
                    data = data.sel(lon=self.lon_slice, lat=self.lat_slice)
                
                result = data[var_key] * self.climate_data[multiply_with]
            except Exception as e:
                print(f"Error computing product for {var_name}: {e}")
                return False
            
        else:
            # Handle simple selection
            try:
                if "level" in config:
                    data = ds.sel(level=config["level"])
                else:
                    data = ds
                
                if self.lon_slice and self.lat_slice:
                    data = data.sel(lon=self.lon_slice, lat=self.lat_slice)
                
                result = data[var_key]
            except Exception as e:
                print(f"Error selecting data for {var_name}: {e}")
                return False
        
        # Store result
        self.climate_data[var_name] = result
        return True
    
    def process_all_variables(self):
        """
        Process all climate variables defined in variable_configs.
        
        Returns
        -------
        bool
            True if all variables were processed successfully, False otherwise
        """
        success = True
        for var_name in self.variable_configs.keys():
            if not self.process_variable(var_name):
                print(f"Failed to process {var_name}")
                success = False
        return success
    
    def extract_features_at_location(self, var_name, year, month, lat, lon):
        """
        Extract a patch of reanalysis data for a specific climate variable centered around a rainfall station location.
        
        Similar to how DEM patches are extracted around station coordinates, this method creates patches of
        climate variables centered on the nearest grid point to each rainfall station. These reanalysis patches
        provide atmospheric context that complements the topographic context from DEM patches.
        
        Parameters
        ----------
        var_name : str
            Name of the climate variable (key in variable_configs)
        year : int
            Year of the data to extract
        month : int
            Month of the data to extract (1-12)
        lat : float
            Latitude of the rainfall station location
        lon : float
            Longitude of the rainfall station location
            
        Returns
        -------
        numpy.ndarray
            Patch of reanalysis data with shape (patch_size, patch_size) centered on the nearest
            grid point to the rainfall station location
        """
        if var_name not in self.climate_data:
            print(f"Variable '{var_name}' not processed.")
            return np.zeros((self.patch_size, self.patch_size))
        
        try:
            # Get the data array for the variable
            da = self.climate_data[var_name]
            
            # Check if the dataset has time dimension
            if 'time' in da.dims:
                # Convert year and month to datetime
                target_date = np.datetime64(f"{year}-{month:02d}")
                
                # Find the nearest time index
                time_idx = np.abs(da.time.values - target_date).argmin()
                
                # Select the data for the specific time
                da = da.isel(time=time_idx)
            
            # Get latitude and longitude arrays
            if 'latitude' in da.dims:
                lats = da.latitude.values
            elif 'lat' in da.dims:
                lats = da.lat.values
            else:
                print(f"Warning: No latitude dimension found for {var_name}")
                return np.zeros((self.patch_size, self.patch_size))
            
            if 'longitude' in da.dims:
                lons = da.longitude.values
            elif 'lon' in da.dims:
                lons = da.lon.values
            else:
                print(f"Warning: No longitude dimension found for {var_name}")
                return np.zeros((self.patch_size, self.patch_size))
            
            # Find the nearest grid point to the station
            lat_idx, lon_idx = find_nearest_point(lat, lon, lats, lons)
            
            # Calculate patch boundaries
            half_size = self.patch_size // 2
            lat_start = max(0, lat_idx - half_size)
            lat_end = min(len(lats), lat_idx + half_size + 1)
            lon_start = max(0, lon_idx - half_size)
            lon_end = min(len(lons), lon_idx + half_size + 1)
            
            # Extract the patch
            if 'latitude' in da.dims and 'longitude' in da.dims:
                patch = da.isel(latitude=slice(lat_start, lat_end), longitude=slice(lon_start, lon_end)).values
            elif 'lat' in da.dims and 'lon' in da.dims:
                patch = da.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)).values
            else:
                print(f"Warning: Incompatible dimensions for {var_name}")
                return np.zeros((self.patch_size, self.patch_size))
            
            # Handle level dimension if present
            if len(patch.shape) > 2:
                # If there are multiple levels, take the first one (usually surface)
                patch = patch[0]
            
            # Ensure the patch has the correct shape
            if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                # Pad the patch to the correct size
                padded_patch = np.zeros((self.patch_size, self.patch_size))
                padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded_patch
            elif patch.shape[0] > self.patch_size or patch.shape[1] > self.patch_size:
                # Crop the patch to the correct size
                patch = patch[:self.patch_size, :self.patch_size]
            
            # Clean the patch (handle NaN and infinite values)
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
            
            return patch
            
        except Exception as e:
            print(f"Error extracting patch for {var_name}: {e}")
            return np.zeros((self.patch_size, self.patch_size))
    
    def compute_variable_statistics(self, all_features):
        """
        Compute statistics (mean, std) for each variable across all stations and times.
        
        Args:
            all_features (dict): Dictionary with station features
            
        Returns:
            dict: Dictionary with variable statistics
        """
        # Collect all values for each variable
        variable_values = {var: [] for var in self.variable_configs.keys()}
        
        for station_features in all_features.values():
            for time_features in station_features.values():
                for var_name, patch in time_features.items():
                    variable_values[var_name].extend(patch.flatten())
        
        # Compute statistics
        variable_stats = {}
        for var_name, values in variable_values.items():
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            variable_stats[var_name] = {'mean': mean, 'std': std}
            print(f"{var_name} - Mean: {mean:.4f}, Std: {std:.4f}")
        
        self.variable_stats = variable_stats
        return variable_stats
    
    def standardize_features(self, all_features, variable_stats=None):
        """
        Standardize reanalysis features by subtracting the mean and dividing by the standard deviation.
        
        Args:
            all_features (dict): Dictionary with station features
            variable_stats (dict, optional): Dictionary with variable statistics.
                If None, computes statistics from the features.
                
        Returns:
            dict: Dictionary with standardized features
        """
        if variable_stats is None:
            variable_stats = self.compute_variable_statistics(all_features)
        
        standardized_features = {}
        
        for station_name, station_features in all_features.items():
            standardized_features[station_name] = {}
            
            for time_key, time_features in station_features.items():
                standardized_features[station_name][time_key] = {}
                
                for var_name, patch in time_features.items():
                    mean = variable_stats[var_name]['mean']
                    std = variable_stats[var_name]['std']
                    
                    # Avoid division by zero
                    if std == 0:
                        std = 1.0
                    
                    standardized_patch = (patch - mean) / std
                    standardized_features[station_name][time_key][var_name] = standardized_patch
        
        return standardized_features

    def visualize_features(self, features, station_name, year, month, output_dir=None):
        """
        Visualize reanalysis features for a specific station, year, and month.
        
        Args:
            features (dict): Dictionary with station features
            station_name (str): Name of the station
            year (int): Year
            month (int): Month
            output_dir (str, optional): Directory to save visualizations
        """
        if station_name not in features:
            print(f"Station '{station_name}' not found in features.")
            return
        
        if (year, month) not in features[station_name]:
            print(f"No data for {station_name} in {year}-{month:02d}.")
            return
        
        time_features = features[station_name][(year, month)]
        
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{station_name}_{year}_{month:02d}_reanalysis.png")
        
        # Delegate to shared grid visualizer
        suptitle = f"{station_name} - {year}-{month:02d} Reanalysis Features"
        visualize_grid(time_features, cmap='viridis', suptitle=suptitle, save_path=save_path)

    def visualize_features_from_npz(self, npz_path, station_name, year, month, output_dir=None):
        """
        Visualize reanalysis features for a specific station/year/month directly
        from an aggregate NPZ created by export_all_features_npz().

        The NPZ is expected to have keys: 'stations' [N], 'years' [N], 'months' [N],
        'patches' [N, V, H, W], and 'variables' [V].

        Args:
            npz_path (str | Path): Path to aggregate NPZ
            station_name (str): Station to visualize
            year (int): Year
            month (int): Month
            output_dir (str, optional): Directory to save visualization PNG
        """
        data = np.load(str(npz_path), allow_pickle=True)
        stations = data["stations"].astype(str)
        years = data["years"].astype(int)
        months = data["months"].astype(int)
        patches = data["patches"]  # (N, V, H, W)
        # variables can be object array; normalize to list[str]
        variables = [str(v) for v in data["variables"].tolist()]

        mask = (stations == station_name) & (years == year) & (months == month)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"No data for {station_name} in {year}-{month:02d} in NPZ.")
            return
        i = int(idxs[0])

        # Build variable->patch mapping for the selected sample
        time_features = {var: patches[i, vi, :, :] for vi, var in enumerate(variables)}

        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{station_name}_{year}_{month:02d}_reanalysis.png")

        suptitle = f"{station_name} - {year}-{month:02d} Reanalysis Features"
        visualize_grid(time_features, cmap='viridis', suptitle=suptitle, save_path=save_path)

    def export_station_month_npz(self, station_name, year, month, time_features, out_dir):
        """
        Export a single station-year-month's 16xHxW patches to an NPZ file with metadata.
        
        Args:
            station_name (str)
            year (int)
            month (int)
            time_features (dict): var_name -> patch (H x W)
            out_dir (str): directory to save
        """
        os.makedirs(out_dir, exist_ok=True)
        var_order = list(self.variable_configs.keys())
        patches = []
        actual_vars = []
        for v in var_order:
            if v in time_features:
                patches.append(np.asarray(time_features[v]))
                actual_vars.append(v)
        if len(patches) == 0:
            print(f"No patches to export for {station_name} {year}-{month:02d}")
            return
        arr = np.stack(patches, axis=0)  # (n_vars, H, W)
        save_path = os.path.join(out_dir, f"{station_name}_{year}_{month:02d}.npz")
        np.savez_compressed(
            save_path,
            patches=arr,
            variables=np.array(actual_vars, dtype=object),
            station=np.array(station_name),
            year=np.array(year),
            month=np.array(month),
            patch_size=np.array(self.patch_size)
        )
        # Also return save_path for reference
        return save_path

    def select_visualization_samples(self, station_months_map, stations=None, per_station=1):
        """
        Build a list of (station, year, month) tuples to visualize using the
        discovered station->[(year, month), ...] mapping.

        Args:
            station_months_map (dict): station -> list of (year, month)
            stations (list[str] | None): Optional subset/order of stations to consider
            per_station (int): How many (year, month) samples to take per station (from earliest)

        Returns:
            list[tuple]: [(station, year, month), ...]
        """
        samples = []
        if not station_months_map:
            return samples

        station_iter = stations if stations is not None else list(station_months_map.keys())
        for st in station_iter:
            if st not in station_months_map:
                continue
            pairs = sorted(station_months_map[st])
            if not pairs:
                continue
            take = pairs[:max(0, int(per_station)) or 1]
            for (y, m) in take:
                samples.append((st, y, m))
        return samples

    def build_features_for_all_stations_with_map(self, station_metadata, station_months_map):
        """
        Build reanalysis feature patches for all rainfall stations using a precomputed mapping of available (year, month) pairs.
        
        This method extracts climate variable patches centered on each rainfall station's coordinates for each
        available month with rainfall data. These reanalysis patches complement the DEM patches by providing
        atmospheric context around each station, while the DEM patches provide topographic context.
        
        The resulting features are organized in a nested dictionary structure:
        {station_name: {(year, month): {variable_name: patch_array, ...}, ...}, ...}
        
        Parameters
        ----------
        station_metadata : dict
            Dictionary mapping station names to metadata including latitude and longitude coordinates
            Format: {station_name: {'latitude': float, 'longitude': float, ...}, ...}
        station_months_map : dict
            Dictionary mapping station names to lists of (year, month) tuples with available rainfall data
            Format: {station_name: [(year, month), ...], ...}
            
        Returns
        -------
        dict
            Nested dictionary of reanalysis features for each station, year, month, and variable
        """
        # Initialize dictionary to store features for all stations
        all_features = {}
        
        # Process each rainfall station using its coordinates
        for station_name, metadata in station_metadata.items():
            # Skip stations with no available rainfall data months
            if station_name not in station_months_map:
                continue
                
            print(f"Building features for station {station_name}...")
            pairs = station_months_map[station_name]  # List of (year, month) tuples with rainfall data
            
            # Get the station's geographic coordinates
            lat = metadata['latitude']   # Latitude of the rainfall station
            lon = metadata['longitude']  # Longitude of the rainfall station
            
            # Initialize dictionary to store features for this station
            station_feats = {}
            
            # Process each year-month combination with rainfall data
            for (year, month) in pairs:
                key = (year, month)
                station_feats[key] = {}
                
                # Extract patches for each climate variable centered on the station's coordinates
                # These patches complement the DEM patches by providing atmospheric context
                for var_name in self.variable_configs.keys():
                    # Extract a patch centered on the station's coordinates
                    patch = self.extract_features_at_location(var_name, year, month, lat, lon)
                    station_feats[key][var_name] = patch
                    
            # Store all features for this station
            all_features[station_name] = station_feats
        return all_features

    def export_all_features_npz(self, all_features, out_dir, filename="reanalysis_features_all.npz"):
        """
        Export all station-year-month patches into a single compressed NPZ file.
        Stores:
          - patches: float32 array shaped (N, V, H, W)
          - stations: object array of station names (N,)
          - years: int array (N,)
          - months: int array (N,)
          - variables: object array (V,)
        """
        os.makedirs(out_dir, exist_ok=True)
        var_order = list(self.variable_configs.keys())
        entries = []
        meta = []
        for station_name, station_feats in all_features.items():
            for (year, month), time_features in station_feats.items():
                patches = []
                for v in var_order:
                    patches.append(np.asarray(time_features[v]))
                arr = np.stack(patches, axis=0)  # (V,H,W)
                entries.append(arr[np.newaxis, ...])
                meta.append((station_name, year, month))
        if len(entries) == 0:
            print("No features to export.")
            return None
        big = np.concatenate(entries, axis=0).astype(np.float32)
        stations = np.array([m[0] for m in meta], dtype=object)
        years = np.array([m[1] for m in meta], dtype=np.int32)
        months = np.array([m[2] for m in meta], dtype=np.int32)
        save_path = os.path.join(out_dir, filename)
        np.savez_compressed(
            save_path,
            patches=big,
            stations=stations,
            years=years,
            months=months,
            variables=np.array(var_order, dtype=object),
            patch_size=np.array(self.patch_size)
        )
        return save_path


def main(time_interval="monthly"):
    """
    Main function to demonstrate the usage of the ReanalysisFeatureBuilder class.
    
    Parameters
    ----------
    time_interval : str
        Time interval for processing ("monthly" or "daily")
    """
    # Load station metadata via unified helper
    station_metadata = get_station_metadata(config.STATION_METADATA_PATH)
    
    # Create a feature builder and process variables
    feature_builder = ReanalysisFeatureBuilder(time_interval=time_interval)
    print(f"Processing climate variables at {time_interval} scale...")
    print(f"Using data directory: {feature_builder.reanalysis_dir}")
    success = feature_builder.process_all_variables()
    if not success:
        print("Warning: Some variables could not be processed.")

    # Discover available months per station from rainfall CSVs
    print("Discovering available station months from rainfall CSVs...")
    station_months_map = discover_station_months(station_metadata)

    # Build features only for those station-year-month combinations
    total_pairs = sum(len(v) for v in station_months_map.values())
    print(f"Building features for {len(station_months_map)} stations across {total_pairs} station-month pairs...")
    all_features = feature_builder.build_features_for_all_stations_with_map(station_metadata, station_months_map)

    # Compute variable statistics and standardize
    print("Computing variable statistics...")
    variable_stats = feature_builder.compute_variable_statistics(all_features)
    print("Standardizing features...")
    standardized_features = feature_builder.standardize_features(all_features, variable_stats)

    # Export a single aggregate NPZ file with all entries
    npz_dir = os.path.join(str(config.OUTPUT_DIR), f"reanalysis_npz_{time_interval}")
    os.makedirs(npz_dir, exist_ok=True)
    filename = f"reanalysis_features_all_standardized_{time_interval}.npz"
    agg_path = feature_builder.export_all_features_npz(standardized_features, npz_dir, filename=filename)
    if agg_path:
        print(f"Saved aggregate NPZ to {agg_path}")

    viz_dir = os.path.join(str(config.OUTPUT_DIR), "reanalysis_viz")
    sample_tuple = None
    for st, feats in standardized_features.items():
        if len(feats) > 0:
            # pick the first (year, month)
            year, month = sorted(feats.keys())[0]
            sample_tuple = (st, year, month)
            break
    if sample_tuple:
        st, y, m = sample_tuple
        print(f"Saving sample visualization for {st} {y}-{m:02d}")
        feature_builder.visualize_features(standardized_features, st, y, m, output_dir=viz_dir)
    else:
        print("No available features found for sample visualization.")
    
    print("Done!")


if __name__ == "__main__":
    import sys
    
    # Allow command line argument to specify time interval
    time_interval = "monthly"  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ["daily", "monthly"]:
            time_interval = sys.argv[1]
        else:
            print("Usage: python build_reanalysis_features.py [daily|monthly]")
            sys.exit(1)
    
    main(time_interval)
