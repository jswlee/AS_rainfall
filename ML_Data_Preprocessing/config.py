"""
Configuration Module for ML Data Preprocessing

This module contains all configuration parameters for the ML data preprocessing pipeline.
"""

import os

# Input data paths
STATION_METADATA_PATH = "raw_data/station_locations.csv"
DEM_PATH = "raw_data/DEM/DEM_Tut1.tif"
REANALYSIS_DIR = "raw_data/climate_variables_monthly"
RAINFALL_DATA_DIR = "Process_Raw_Rainfall_Data/output/monthly_rainfall"

# Output paths
OUTPUT_DIR = "ML_Data_Preprocessing/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DEM patch configuration
DEM_PATCH_CONFIG = {
    'local': {
        'patch_size': 3,  # 3x3 grid
        'km_per_cell': 2  # 2km per cell (6km total; or 3 x 2km)
    },
    'regional': {
        'patch_size': 3,  # 3x3 grid
        'km_per_cell': 8  # 8km per cell (24km total; or 3 x 8km)
    }
}

# Reanalysis patch configuration
REANALYSIS_PATCH_SIZE = 3  # 3x3 grid centered on nearest grid point to station

# Data processing parameters
RAINFALL_OUTLIER_THRESHOLD = 3.0  # Standard deviations for outlier detection
RANDOM_SEED = 42  # For reproducibility

# Dictionary to store variable name mappings
VARIABLE_MAPPING = {
    "Air 2m": "air.2m",
    "Air": "air",
    "Geopotential Height": "hgt",
    "Omega": "omega",
    "Potential Temperature": "pottmp",
    "Precipitable Water": "pr_wtr.eatm",
    "Specific Humidity": "shum",
    "Skin Temperature": "skt",
    "Sea Level Pressure": "slp",
    "Zonal Wind": "uwnd",
    "Meridional Wind": "vwnd"
}

# Dictionary to store time interval mappings
TIME_INTERVAL_MAPPING = {
    "monthly": "mon",
    "daily": "day",
}

# Dictionary to store statistic mappings
STATISTIC_MAPPING = {
    "mean": "mean",
}

# Default time interval and statistic for all variables
DEFAULT_TIME_INTERVAL = "monthly"
DEFAULT_STATISTIC = "mean"

# Dictionary to store variable configurations
REANALYSIS_VARIABLE_CONFIGS = {
    "air_temp_diff_1000_500": {
        "description": "Air temperature difference between 1000 and 500 mb",
        "variable": "Air",
        "levels": [1000, 500],
        "operation": "diff"
    },
    "air_2m": {
        "description": "2m air temperature",
        "variable": "Air 2m",
        "interpolate": True
    },
    "hgt_1000": {
        "description": "Geopotential height at 1000 mb",
        "variable": "Geopotential Height",
        "level": 1000
    },
    "hgt_500": {
        "description": "Geopotential height at 500 mb",
        "variable": "Geopotential Height",
        "level": 500
    },
    "omega_500": {
        "description": "Omega (vertical velocity) at 500 mb",
        "variable": "Omega",
        "level": 500
    },
    "pottmp_diff_1000_500": {
        "description": "Potential temperature difference between 1000 and 500 mb",
        "variable": "Potential Temperature",
        "levels": [1000, 500],
        "operation": "diff"
    },
    "pottmp_diff_1000_850": {
        "description": "Potential temperature difference between 1000 and 850 mb",
        "variable": "Potential Temperature",
        "levels": [1000, 850],
        "operation": "diff"
    },
    "pr_wtr": {
        "description": "Precipitable water",
        "variable": "Precipitable Water",
        "custom_file": "pr_wtr.eatm.mon.mean.nc"
    },
    "shum_700": {
        "description": "Specific humidity at 700 mb",
        "variable": "Specific Humidity",
        "level": 700
    },
    "shum_925": {
        "description": "Specific humidity at 925 mb",
        "variable": "Specific Humidity",
        "level": 925
    },
    "zon_moist_700": {
        "description": "Zonal moisture transport at 700 mb",
        "depends_on": ["shum_700"],
        "variable": "Zonal Wind",
        "level": 700,
        "operation": "multiply",
        "multiply_with": "shum_700"
    },
    "zon_moist_925": {
        "description": "Zonal moisture transport at 925 mb",
        "depends_on": ["shum_925"],
        "variable": "Zonal Wind",
        "level": 925,
        "operation": "multiply",
        "multiply_with": "shum_925"
    },
    "merid_moist_700": {
        "description": "Meridional moisture transport at 700 mb",
        "depends_on": ["shum_700"],
        "variable": "Meridional Wind",
        "level": 700,
        "operation": "multiply",
        "multiply_with": "shum_700"
    },
    "merid_moist_925": {
        "description": "Meridional moisture transport at 925 mb",
        "depends_on": ["shum_925"],
        "variable": "Meridional Wind",
        "level": 925,
        "operation": "multiply",
        "multiply_with": "shum_925"
    },
    "skin_temp": {
        "description": "Skin temperature",
        "variable": "Skin Temperature",
        "interpolate": True
    },
    "slp": {
        "description": "Sea level pressure",
        "variable": "Sea Level Pressure"
    }
}

# List of reanalysis variables to use in the ML pipeline
REANALYSIS_VARIABLES = list(REANALYSIS_VARIABLE_CONFIGS.keys())

# Function to get the full configuration as a dictionary
def get_config():
    """
    Returns the full configuration as a dictionary.
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        'paths': {
            'station_metadata': STATION_METADATA_PATH,
            'dem': DEM_PATH,
            'reanalysis_dir': REANALYSIS_DIR,
            'rainfall_data_dir': RAINFALL_DATA_DIR,
            'output_dir': OUTPUT_DIR,
        },
        'dem_patch_config': DEM_PATCH_CONFIG,
        'reanalysis_variables': REANALYSIS_VARIABLES,
        'reanalysis_patch_size': REANALYSIS_PATCH_SIZE,
        'rainfall_outlier_threshold': RAINFALL_OUTLIER_THRESHOLD,
        'random_seed': RANDOM_SEED
    }
