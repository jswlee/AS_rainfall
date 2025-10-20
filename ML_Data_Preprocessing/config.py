"""
Configuration Module for ML Data Preprocessing

This module contains all configuration parameters for the ML data preprocessing pipeline.
"""

import os

# Input data paths
STATION_METADATA_PATH = "raw_data/station_locations.csv"
DEM_PATH = "raw_data/DEM/DEM_Tut1.tif"
REANALYSIS_DIR_MONTHLY = "raw_data/climate_variables_monthly_raw"
REANALYSIS_DIR_DAILY = "raw_data/climate_variables_daily_processed"
MONTHLY_RAINFALL_DATA_DIR = "Process_Raw_Rainfall_Data/output/monthly_rainfall"
DAILY_RAINFALL_DATA_DIR = "raw_data/rainfall_corrected"

# Backward compatibility - defaults to monthly
REANALYSIS_DIR = REANALYSIS_DIR_MONTHLY

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
MONTHLY_REANALYSIS_VARIABLE_CONFIGS = {
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
        "custom_file": "pr_wtr.eatm.mon.mean.nc",
        "custom_file_daily": "pr_wtr.eatm.day.mean.nc"
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

DAILY_REANALYSIS_VARIABLE_CONFIGS = {
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
    "pr_wtr": {
        "description": "Precipitable water",
        "variable": "Precipitable Water",
        "custom_file": "pr_wtr.eatm.mon.mean.nc",
        "custom_file_daily": "pr_wtr.eatm.day.mean.nc"
    },
    "shum_750": {
        "description": "Specific humidity at 750 mb",
        "variable": "Specific Humidity",
        "level": 750
    },
    "shum_925": {
        "description": "Specific humidity at 925 mb",
        "variable": "Specific Humidity",
        "level": 925
    },
    "zon_moist_750": {
        "description": "Zonal moisture transport at 750 mb",
        "depends_on": ["shum_750"],
        "variable": "Zonal Wind",
        "level": 750,
        "operation": "multiply",
        "multiply_with": "shum_750"
    },
    "zon_moist_925": {
        "description": "Zonal moisture transport at 925 mb",
        "depends_on": ["shum_925"],
        "variable": "Zonal Wind",
        "level": 925,
        "operation": "multiply",
        "multiply_with": "shum_925"
    },
    "merid_moist_750": {
        "description": "Meridional moisture transport at 750 mb",
        "depends_on": ["shum_750"],
        "variable": "Meridional Wind",
        "level": 750,
        "operation": "multiply",
        "multiply_with": "shum_750"
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

DAILY_FOLDER_TO_VARIABLE = {
    'gh_1980-1994': 'Geopotential Height',
    'mslp_1980-1994': 'Sea Level Pressure',
    't2m_1980-1994': 'Air 2m',
    'temp_1980-1994': 'Air',
    'omg_1980-1994': 'Omega',
    'pwat_1980-1994': 'Precipitable Water',
    'shum_1980-1994': 'Specific Humidity',
    'uwnd_1980-1994': 'Zonal Wind',
    'vwnd_1980-1994': 'Meridional Wind',
    'tskn_1980-1994': 'Skin Temperature',
    'ptmp_1980-1994': 'Potential Temperature',
}

# List of reanalysis variables to use in the ML pipeline
MONTHLY_REANALYSIS_VARIABLES = list(MONTHLY_REANALYSIS_VARIABLE_CONFIGS.keys())
DAILY_REANALYSIS_VARIABLES = list(DAILY_REANALYSIS_VARIABLE_CONFIGS.keys())

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
            'reanalysis_dir_monthly': REANALYSIS_DIR_MONTHLY,
            'reanalysis_dir_daily': REANALYSIS_DIR_DAILY,
            'rainfall_data_dir': RAINFALL_DATA_DIR,
            'output_dir': OUTPUT_DIR,
        },
        'dem_patch_config': DEM_PATCH_CONFIG,
        'reanalysis_variables_monthly': MONTHLY_REANALYSIS_VARIABLES,
        'reanalysis_variables_daily': DAILY_REANALYSIS_VARIABLES,
        'reanalysis_patch_size': REANALYSIS_PATCH_SIZE,
        'rainfall_outlier_threshold': RAINFALL_OUTLIER_THRESHOLD,
        'random_seed': RANDOM_SEED
    }
