"""
Configuration Module

This module contains default configurations and mappings for climate variables.
"""

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
    "6-hourly": "6hr"
}

# Dictionary to store statistic mappings
STATISTIC_MAPPING = {
    "mean": "mean",
    "anomaly": "anom",
    "climatology": "clim"
}

# Default time interval and statistic for all variables
DEFAULT_TIME_INTERVAL = "monthly"
DEFAULT_STATISTIC = "mean"

# Dictionary to store variable configurations
DEFAULT_VARIABLE_CONFIGS = {
    "air_temp_diff_1000_500": {
        "description": "Air temperature difference between 1000 and 500 hPa",
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
        "description": "Geopotential height at 1000 hPa",
        "variable": "Geopotential Height",
        "level": 1000
    },
    "hgt_500": {
        "description": "Geopotential height at 500 hPa",
        "variable": "Geopotential Height",
        "level": 500
    },
    "omega_500": {
        "description": "Omega (vertical velocity) at 500 hPa",
        "variable": "Omega",
        "level": 500
    },
    "pottmp_diff_1000_500": {
        "description": "Potential temperature difference between 1000 and 500 hPa",
        "variable": "Potential Temperature",
        "levels": [1000, 500],
        "operation": "diff"
    },
    "pottmp_diff_1000_850": {
        "description": "Potential temperature difference between 1000 and 850 hPa",
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
        "description": "Specific humidity at 700 hPa",
        "variable": "Specific Humidity",
        "level": 700
    },
    "shum_925": {
        "description": "Specific humidity at 925 hPa",
        "variable": "Specific Humidity",
        "level": 925
    },
    "zon_moist_700": {
        "description": "Zonal moisture transport at 700 hPa",
        "depends_on": ["shum_700"],
        "variable": "Zonal Wind",
        "level": 700,
        "operation": "multiply",
        "multiply_with": "shum_700"
    },
    "zon_moist_925": {
        "description": "Zonal moisture transport at 925 hPa",
        "depends_on": ["shum_925"],
        "variable": "Zonal Wind",
        "level": 925,
        "operation": "multiply",
        "multiply_with": "shum_925"
    },
    "merid_moist_700": {
        "description": "Meridional moisture transport at 700 hPa",
        "depends_on": ["shum_700"],
        "variable": "Meridional Wind",
        "level": 700,
        "operation": "multiply",
        "multiply_with": "shum_700"
    },
    "merid_moist_925": {
        "description": "Meridional moisture transport at 925 hPa",
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
