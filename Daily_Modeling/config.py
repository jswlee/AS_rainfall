"""
Configuration for Daily Rainfall Downscaling - American Samoa.

All paths are relative to the repository root (parent of Daily_Modeling/).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (two levels up from this file)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent

# ---------------------------------------------------------------------------
# Raw data paths
# ---------------------------------------------------------------------------
STATION_METADATA_PATH = REPO_ROOT / "raw_data" / "station_locations.csv"
DEM_PATH = REPO_ROOT / "raw_data" / "DEM" / "DEM_Tut1.tif"
REANALYSIS_DIR = REPO_ROOT / "raw_data" / "climate_variables_daily_1980-2024_updated"
DAILY_RAINFALL_DIR = REPO_ROOT / "raw_data" / "rainfall_corrected_NEW"

# ---------------------------------------------------------------------------
# Output paths (all under Daily_Modeling/output/)
# ---------------------------------------------------------------------------
OUTPUT_DIR = _THIS_DIR / "output"
FEATURES_DIR = OUTPUT_DIR / "features"
ASSEMBLED_DIR = OUTPUT_DIR / "assembled"
EDA_DIR = OUTPUT_DIR / "eda"
TUNING_DIR = OUTPUT_DIR / "tuning"
RESULTS_DIR = OUTPUT_DIR / "results"

for _d in (OUTPUT_DIR, FEATURES_DIR, ASSEMBLED_DIR, EDA_DIR, TUNING_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DEM patch configuration  (same as existing pipeline)
# ---------------------------------------------------------------------------
# Default DEM patch config (used when NOT tuning patch size)
DEM_PATCH_CONFIG = {
    "local": {"patch_size": 3, "km_per_cell": 2},      # 3x3 @ 2 km -> 6 km
    "regional": {"patch_size": 3, "km_per_cell": 8},    # 3x3 @ 8 km -> 24 km
}

# Multi-resolution DEM: generate once at max size, crop at runtime.
# Base patches are extracted at 1 km resolution at the largest extent needed.
DEM_MAX_LOCAL = {"patch_size": 11, "km_per_cell": 1}    # 11x11 @ 1 km -> 11 km
DEM_MAX_REGIONAL = {"patch_size": 25, "km_per_cell": 1}  # 25x25 @ 1 km -> 25 km

# Candidate combos for HP tuning: (patch_size, km_per_cell)
# Total box = patch_size * km_per_cell
DEM_LOCAL_CANDIDATES = [
    (1, 1),   #  1 km  (centre pixel only)
    (3, 0.5),  #  1.5 km
    (3, 1),   #  3 km
    (3, 2),   #  6 km
    (5, 1),   #  5 km
]
DEM_REGIONAL_CANDIDATES = [
    (3, 3),   #  9 km
    (3, 5),   # 15 km
    (3, 8),   # 24 km
    (5, 2),   # 10 km
    (5, 3),   # 15 km
    (5, 4),   # 20 km
    (5, 5),   # 25 km
]


def resolve_dem_crop(hp: dict) -> dict | None:
    """Build a dem_crop_config dict from hyperparameters.

    Accepts HPs that contain either:
      - ``local_dem_patch`` / ``local_dem_km`` (explicit), or
      - ``local_dem_cfg`` / ``regional_dem_cfg`` (index into candidate lists).

    Returns None if no DEM crop info is present.
    """
    if "local_dem_patch" in hp and "local_dem_km" in hp:
        return {
            "local_patch_size": hp["local_dem_patch"],
            "local_km": hp["local_dem_km"],
            "regional_patch_size": hp["regional_dem_patch"],
            "regional_km": hp["regional_dem_km"],
        }
    if "local_dem_cfg" in hp and "regional_dem_cfg" in hp:
        lp, lk = DEM_LOCAL_CANDIDATES[hp["local_dem_cfg"]]
        rp, rk = DEM_REGIONAL_CANDIDATES[hp["regional_dem_cfg"]]
        return {
            "local_patch_size": lp, "local_km": lk,
            "regional_patch_size": rp, "regional_km": rk,
        }
    return None

# ---------------------------------------------------------------------------
# Reanalysis patch configuration
# ---------------------------------------------------------------------------
REANALYSIS_PATCH_SIZE = 3  # 3x3 grid centred on nearest reanalysis grid-point

# ---------------------------------------------------------------------------
# Variable name -> NetCDF base-name mapping
# ---------------------------------------------------------------------------
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
    "Meridional Wind": "vwnd",
}

# ---------------------------------------------------------------------------
# Daily reanalysis variable configs (13 derived channels).
# Dropped hgt_1000 (r=1.00 with slp - perfectly redundant) and
# pottmp_diff_1000_500 (r=0.99 with air_temp_diff_1000_500 - near-perfectly redundant).
# ---------------------------------------------------------------------------
DAILY_VARIABLE_CONFIGS = {
    "air_temp_diff_1000_500": {
        "description": "Air temperature difference 1000-500 hPa",
        "variable": "Air", "levels": [1000, 500], "operation": "diff",
    },
    "air_2m": {
        "description": "Surface air temperature at 2 m",
        "variable": "Air 2m", "interpolate": True,
    },
    "hgt_500": {
        "description": "Geopotential height 500 hPa",
        "variable": "Geopotential Height", "level": 500,
    },
    "hgt_1000": {
        "description": "Geopotential height 1000 hPa",
        "variable": "Geopotential Height", "level": 1000,
    },
    "omega_500": {
        "description": "Omega (vertical velocity) 500 hPa",
        "variable": "Omega", "level": 500,
    },
    "pottmp_diff_500_1000": {
        "description": "Potential temperature difference 500-1000 hPa",
        "variable": "Potential Temperature", "levels": [500, 1000], "operation": "diff",
    },
    "pottmp_diff_850_1000": {
        "description": "Potential temperature difference 850-1000 hPa",
        "variable": "Potential Temperature", "levels": [850, 1000], "operation": "diff",
    },
    "pr_wtr": {
        "description": "Precipitable water",
        "variable": "Precipitable Water",
        "custom_file_daily": "pr_wtr.eatm.day.mean.nc",
    },
    "shum_750": {
        "description": "Specific humidity 750 hPa",
        "variable": "Specific Humidity", "level": 750,
    },
    "shum_925": {
        "description": "Specific humidity 925 hPa",
        "variable": "Specific Humidity", "level": 925,
    },
    "zon_moist_750": {
        "description": "Zonal moisture transport 750 hPa",
        "depends_on": ["shum_750"],
        "variable": "Zonal Wind", "level": 750,
        "operation": "multiply", "multiply_with": "shum_750",
    },
    "zon_moist_925": {
        "description": "Zonal moisture transport 925 hPa",
        "depends_on": ["shum_925"],
        "variable": "Zonal Wind", "level": 925,
        "operation": "multiply", "multiply_with": "shum_925",
    },
    "merid_moist_750": {
        "description": "Meridional moisture transport 750 hPa",
        "depends_on": ["shum_750"],
        "variable": "Meridional Wind", "level": 750,
        "operation": "multiply", "multiply_with": "shum_750",
    },
    "merid_moist_925": {
        "description": "Meridional moisture transport 925 hPa",
        "depends_on": ["shum_925"],
        "variable": "Meridional Wind", "level": 925,
        "operation": "multiply", "multiply_with": "shum_925",
    },
    "skin_temp": {
        "description": "Skin temperature",
        "variable": "Skin Temperature", "interpolate": True,
    },
    "slp": {
        "description": "Sea level pressure",
        "variable": "Sea Level Pressure",
    },
}

DAILY_VARIABLE_NAMES = list(DAILY_VARIABLE_CONFIGS.keys())

# ---------------------------------------------------------------------------
# Spatio-temporal split defaults
# ---------------------------------------------------------------------------
# Year ranges are computed from data by compute_year_boundaries() so that
# ~TRAIN_FRAC of samples fall in train years, ~VAL_FRAC in val years, and
# the rest in test years.

# Target fractions (used by data-driven year boundary computation)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.10

# Number of stations held out for spatial generalization.
# With ~26 stations, 5 val + 3 test leaves ~18 train.
# Stations are chosen from those with data overlapping the val/test years.
N_VAL_STATIONS = 5
N_TEST_STATIONS = 3

RANDOM_SEED = 42

# Site-specific model split fractions (chronological per station)
SITE_TRAIN_FRAC = 0.70
SITE_VAL_FRAC = 0.20
# SITE_TEST_FRAC = 1 - SITE_TRAIN_FRAC - SITE_VAL_FRAC = 0.10

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
# Maps loss_type -> required output_head for the LAND model
LOSS_TO_HEAD = {
    "mse": "softplus",
    "gamma": "gamma",
    "tweedie": "softplus",
    "bernoulli_gamma": "bernoulli_gamma",
}

LAND_DEFAULT_HP = {
    "climate_units": 130,
    "dem_units": 64,
    "dem_patch_size": 10,
    "temporal_units": 16,
    "na": 256,
    "nb": 64,
    "dropout_rate": 0.3,
    "learning_rate": 5e-5,
    "weight_decay": 3e-5,
    "batch_size": 256,
    "climate_processing": "conv2d",
    "output_head": "gamma",
    "loss_type": "gamma",
    "use_batch_norm": False,
}

MLP_DEFAULT_HP = {
    "hidden_sizes": [512, 512, 512],
    "dropout_rate": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "loss_type": "mse",  # one of: mse, log_mse, tweedie
}

DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True
DATALOADER_PERSISTENT_WORKERS = True
DATALOADER_PREFETCH_FACTOR = 4

MAX_EPOCHS = 1000
PATIENCE = 100
