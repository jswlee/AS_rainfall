"""
File Utilities Module

This module provides utilities for file path generation and dataset loading.
"""

import os
import xarray as xr
from .config import VARIABLE_MAPPING, TIME_INTERVAL_MAPPING, STATISTIC_MAPPING

def get_file_path(data_dir, variable, variable_mapping=VARIABLE_MAPPING, 
                 time_interval_mapping=TIME_INTERVAL_MAPPING,
                 statistic_mapping=STATISTIC_MAPPING,
                 time_interval=None, statistic=None, 
                 default_time_interval="monthly", default_statistic="mean",
                 surface=False):
    """
    Construct the file path for a climate variable.
    
    Parameters
    ----------
    data_dir : str
        Directory containing climate data files
    variable : str
        Climate variable name (use keys from variable_mapping)
    variable_mapping : dict
        Dictionary mapping variable names to file prefixes
    time_interval_mapping : dict
        Dictionary mapping time interval names to file components
    statistic_mapping : dict
        Dictionary mapping statistic names to file components
    time_interval : str, optional
        Time interval (use keys from time_interval_mapping). If None, uses default_time_interval.
    statistic : str, optional
        Statistic type (use keys from statistic_mapping). If None, uses default_statistic.
    default_time_interval : str
        Default time interval to use if time_interval is None
    default_statistic : str
        Default statistic to use if statistic is None
    surface : bool
        Whether the variable is a surface variable
    
    Returns
    -------
    str
        Complete file path
    """
    if variable not in variable_mapping:
        raise ValueError(f"Unknown variable: {variable}")
    
    # Use defaults if not specified
    time_interval = time_interval or default_time_interval
    statistic = statistic or default_statistic
    
    if time_interval not in time_interval_mapping:
        raise ValueError(f"Unknown time interval: {time_interval}")
        
    if statistic not in statistic_mapping:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Construct file name
    file_name = variable_mapping[variable]
    
    # Add surface indicator if needed
    if surface:
        file_name = f"{file_name}.sfc"
    
    # Add time interval and statistic
    file_name = f"{file_name}.{time_interval_mapping[time_interval]}.{statistic_mapping[statistic]}.nc"
    
    # Return full path
    return os.path.join(data_dir, file_name)

def load_dataset(file_path, time_slice=None):
    """
    Load a dataset from a NetCDF file.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file
    time_slice : slice, optional
        Time slice for data selection
    
    Returns
    -------
    xarray.Dataset
        Loaded dataset with time selection applied, or None if loading fails
    """
    try:
        if time_slice:
            ds = xr.open_dataset(file_path).sel(time=time_slice)
        else:
            ds = xr.open_dataset(file_path)
        return ds
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
