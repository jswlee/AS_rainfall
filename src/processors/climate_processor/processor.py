"""
Climate Data Processor Module

This module provides the main ClimateDataProcessor class that integrates
all the functionality for processing climate data.
"""

import os
import xarray as xr
import numpy as np
import re
from .config import (
    VARIABLE_MAPPING, TIME_INTERVAL_MAPPING, STATISTIC_MAPPING,
    DEFAULT_TIME_INTERVAL, DEFAULT_STATISTIC, DEFAULT_VARIABLE_CONFIGS
)
from .file_utils import get_file_path, load_dataset
from .data_processing_ops import (
    interpolate_to_target_grid, compute_difference, compute_product,
    select_data, validate_grid_consistency, save_to_netcdf, get_data_array
)

class ClimateDataProcessor:
    """
    A class to process climate data from NetCDF files.
    
    This class provides methods to load, process, and validate climate data
    from various NetCDF files, creating a unified dataset with consistent
    dimensions and coordinates.
    """
    
    def __init__(self, data_dir="data/climate_variables", 
                 lon_slice=slice(187.5, 192.5), 
                 lat_slice=slice(-12.5, -17.5),
                 time_slice=slice("1958-01-01", "2024-12-31"),
                 target_lons=np.array([187.5, 190.0, 192.5]),
                 target_lats=np.array([-12.5, -15.0, -17.5])):
        """
        Initialize the ClimateDataProcessor.
        
        Parameters
        ----------
        data_dir : str
            Directory containing climate data files
        lon_slice : slice
            Longitude slice for data selection
        lat_slice : slice
            Latitude slice for data selection
        time_slice : slice
            Time slice for data selection
        target_lons : numpy.ndarray
            Target longitude grid for interpolation
        target_lats : numpy.ndarray
            Target latitude grid for interpolation
        """
        self.data_dir = data_dir
        self.lon_slice = lon_slice
        self.lat_slice = lat_slice
        self.time_slice = time_slice
        self.target_lons = target_lons
        self.target_lats = target_lats
        
        # Use the mappings from the config module
        self.variable_mapping = VARIABLE_MAPPING
        self.time_interval_mapping = TIME_INTERVAL_MAPPING
        self.statistic_mapping = STATISTIC_MAPPING
        
        # Default time interval and statistic for all variables
        self.default_time_interval = DEFAULT_TIME_INTERVAL
        self.default_statistic = DEFAULT_STATISTIC
        
        # Dictionary to store variable configurations
        self.variable_configs = DEFAULT_VARIABLE_CONFIGS.copy()
        
        # Dictionary to store each variable's subset
        self.climate_data = {}
    
    def get_file_path(self, variable, time_interval=None, statistic=None, level=None, surface=False):
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
        level : int or None
            Pressure level (if applicable)
        surface : bool
            Whether the variable is a surface variable
        
        Returns
        -------
        str
            Complete file path
        """
        return get_file_path(
            self.data_dir, variable, 
            self.variable_mapping, self.time_interval_mapping, self.statistic_mapping,
            time_interval, statistic, 
            self.default_time_interval, self.default_statistic,
            surface
        )
    
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
        return load_dataset(file_path, self.time_slice)
    
    def interpolate_to_target_grid(self, data_array):
        """
        Interpolate a DataArray to the target grid.
        
        Parameters
        ----------
        data_array : xarray.DataArray
            DataArray to interpolate
        
        Returns
        -------
        xarray.DataArray
            Interpolated DataArray
        """
        return interpolate_to_target_grid(data_array, self.target_lats, self.target_lons)
    
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
            print(f"Unknown variable: {var_name}")
            return False
        
        config = self.variable_configs[var_name]
        
        # Check dependencies
        if "depends_on" in config:
            for dep in config["depends_on"]:
                if dep not in self.climate_data:
                    # Process dependency first
                    if not self.process_variable(dep):
                        print(f"Failed to process dependency {dep} for {var_name}")
                        return False
        
        # Handle custom file case
        if "custom_file" in config:
            file_path = os.path.join(self.data_dir, config["custom_file"])
        else:
            # Get file path from configuration, using defaults for time_interval and statistic
            file_path = self.get_file_path(
                config["variable"],
                config.get("time_interval", None),  # Will use default if None
                config.get("statistic", None),      # Will use default if None
                None,  # Level is not part of the file name
                config.get("surface", False)
            )
        
        # Load dataset
        ds = self.load_dataset(file_path)
        if ds is None:
            return False
        
        # Determine variable name in the dataset
        var_key = self.variable_mapping[config["variable"]].split('.')[0]
        if var_key not in ds:
            # Try to find the variable
            for key in ds.data_vars:
                if key.lower() == var_key.lower():
                    var_key = key
                    break
            else:
                print(f"Variable {var_key} not found in dataset")
                return False
        
        # Process based on operation type
        if "operation" in config and config["operation"] == "diff":
            # Handle difference between two levels
            level1, level2 = config["levels"]
            result = compute_difference(ds, var_key, level1, level2, self.lon_slice, self.lat_slice)
            
        elif "operation" in config and config["operation"] == "multiply":
            # Handle multiplication with another variable
            level = config.get("level", None)
            if level is not None:
                data = select_data(ds, var_key, level, self.lon_slice, self.lat_slice)
            else:
                data = select_data(ds, var_key, None, self.lon_slice, self.lat_slice)
            
            multiply_with = config["multiply_with"]
            if multiply_with not in self.climate_data:
                print(f"Multiplication variable {multiply_with} not found")
                return False
            
            result = compute_product(data, self.climate_data[multiply_with])
            
        else:
            # Handle simple selection
            if "level" in config:
                result = select_data(ds, var_key, config["level"], self.lon_slice, self.lat_slice)
            else:
                result = select_data(ds, var_key, None, self.lon_slice, self.lat_slice)
        
        # Apply interpolation if needed
        if config.get("interpolate", False):
            if isinstance(ds, xr.Dataset):
                ds_interp = self.interpolate_to_target_grid(ds)
                result = ds_interp[var_key]
            else:
                result = self.interpolate_to_target_grid(result)
        
        # Store result
        self.climate_data[var_name] = result
        return True
    
    def process_all_variables(self):
        """Process all climate variables defined in variable_configs."""
        for var_name in self.variable_configs.keys():
            self.process_variable(var_name)
    
    def process_by_description(self, description):
        """
        Process a climate variable by its description.
        
        Parameters
        ----------
        description : str
            Description of the variable to process
        
        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        # Normalize the description for comparison
        description_lower = description.lower()
        
        # Find the variable with the closest matching description
        best_match = None
        for var_name, config in self.variable_configs.items():
            if "description" in config:
                config_desc = config["description"].lower()
                if config_desc in description_lower or description_lower in config_desc:
                    best_match = var_name
                    break
        
        if best_match:
            return self.process_variable(best_match)
        else:
            print(f"No variable found with description matching '{description}'")
            return False
    
    def validate_grid_consistency(self):
        """
        Check that all variables are on the same grid.
        
        Returns
        -------
        bool
            True if all variables have consistent lat/lon grids, False otherwise
        """
        return validate_grid_consistency(self.climate_data)
    
    def save_to_netcdf(self, output_path):
        """
        Save the processed climate data to a NetCDF file.
        
        Parameters
        ----------
        output_path : str
            Path to save the NetCDF file
        """
        save_to_netcdf(self.climate_data, output_path)
    
    def get_data_array(self):
        """
        Convert the dataset into a DataArray with a 'variable' dimension.
        
        Returns
        -------
        xarray.DataArray
            DataArray with dimensions (time, lat, lon, variable)
        """
        return get_data_array(self.climate_data)
