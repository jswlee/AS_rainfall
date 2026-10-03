"""
Raw-data configuration for s1_prepare: input paths, rainfall QC rules, and
the DEM/reanalysis patch specifications used at feature-build time.

Copied from ``Daily_Modeling/config.py`` (trimmed to the pieces LAND_AS uses)
so that ``LAND_AS`` is self-contained. Raw data is located by searching, in
order:

1. ``<repo root>/raw_data/AS``   -- original monorepo layout
2. ``LAND_AS/raw_data/AS``       -- standalone-repo layout with data inside
   the package directory

Imported by load_raw.py, build_features.py, assemble_dataset.py, and
prepare.py in this stage. Distinct from the project-level ``LAND_AS/config.py``
(output paths + train/test split).
"""
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # LAND_AS/data_utils
LAND_AS_DIR = _THIS_DIR.parent               # LAND_AS/
REPO_ROOT = LAND_AS_DIR.parent               # repository root


def _find_as_dir() -> Path:
    for candidate in (REPO_ROOT / "raw_data" / "AS",
                      LAND_AS_DIR / "raw_data" / "AS"):
        if candidate.exists():
            return candidate
    return REPO_ROOT / "raw_data" / "AS"


# ---------------------------------------------------------------------------
# Raw data paths (American Samoa only)
# ---------------------------------------------------------------------------
_AS_DIR = _find_as_dir()

DEM_AS_PATH = _AS_DIR / "DEM" / "10m_tutuila_3band.tif"
DEM_PATH = DEM_AS_PATH

STATION_METADATA_PATH = _AS_DIR / "station_locations.csv"
REANALYSIS_DIR = _AS_DIR / "climate_variables_daily_1980-2024"
DAILY_RAINFALL_DIR = _AS_DIR / "final_rainfall_per_station"

# Default feature-cache location used by assemble() when the caller does not
# pass explicit npz paths (LAND_AS.s1_prepare.prepare always passes LAND_AS/data/features).
FEATURES_DIR = LAND_AS_DIR / "data" / "features"

# LAND_AS is weekly-only; retained for API parity with Daily_Modeling config.
FREQ = "weekly"

# ---------------------------------------------------------------------------
# Rainfall quality control (applied in load_daily_rainfall)
# ---------------------------------------------------------------------------
# Stations excluded entirely, based on the eda_scripts/rainfall_* audit and
# notebooks/00_qc_evidence.ipynb:
#   aunuu     reports in 0.1-inch increments (min nonzero = 2.54 mm), so
#             drizzle days read as zero -> 59% daily / 12.4% weekly zeros,
#             while co-located aunuu_UH (0.254 mm resolution) has none.
#   afono_UH  recorded exactly 0.000 for 30-53 consecutive days in three runs
#             (2022-08-14..10-05, 2022-10-18..11-16, 2024-07-12..08-15) while
#             the nearest gauges with data (toa_ridge_WRCC, vaipito_UH,
#             siufaga_WRCC, GML_SMO) recorded 0.7-16 mm/day -> gauge-offline
#             stored as zero; the record is inconsistent enough that the
#             whole station is dropped rather than masked.
# pioa_afono was tested and RETAINED: its exclusion worsened t36 test RMSE
# (49.32 -> 50.02); its zeros are largely real co-occurring dry weeks.
QC_EXCLUDE_STATIONS = frozenset({"afono_UH", "aunuu"})

# Per-station keep windows (inclusive, YYYY-MM-DD): rows outside are dropped.
# vaipito2000 is retained only where its record is corroborated by the rest
# of the network and does not duplicate vaipito_res:
#   start 1976-01-01 -- earliest record start among all other stations
#                       (GML_SMO). Its uncorroborated 1958-69 block (~17
#                       mm/day mean, vs 5.7 in 1970-91; no other station has
#                       pre-1970 data) is dropped.
#   end   1989-09-30 -- day before vaipito_res begins. Where the two overlap
#                       (1989-90) the records are identical (weekly r=1.000)
#                       -> same site, so the overlap belongs to vaipito_res.
QC_VALID_DATE_RANGES = {
    "vaipito2000": ("1976-01-01", "1989-09-30"),
}

# Per-station date ranges (inclusive, YYYY-MM-DD) reclassified to missing.
# Currently empty: afono_UH (previously masked here) is excluded entirely.
QC_MASK_DATE_RANGES = {}

# ---------------------------------------------------------------------------
# DEM patch configuration
# ---------------------------------------------------------------------------
# Number of DEM channels: 4 = elevation + slope + sin(aspect) + cos(aspect)
DEM_N_CHANNELS = 4
# Default DEM patch config (used when NOT tuning patch size)
DEM_PATCH_CONFIG = {
    "local": {"patch_size": 3, "km_per_cell": 2},      # 3x3 @ 2 km -> 6 km
    "regional": {"patch_size": 3, "km_per_cell": 8},   # 3x3 @ 8 km -> 24 km
}

# Multi-resolution DEM: generated once at max size, cropped at runtime.
# Base patches are extracted at 1 km resolution at the largest extent needed.
DEM_MAX_LOCAL = {"patch_size": 11, "km_per_cell": 1}    # 11x11 @ 1 km -> 11 km
DEM_MAX_REGIONAL = {"patch_size": 25, "km_per_cell": 1}  # 25x25 @ 1 km -> 25 km

# Candidate combos for HP tuning: (patch_size, km_per_cell)
DEM_LOCAL_CANDIDATES = [
    (3, 0.5), (3, 1), (3, 1.5), (3, 2), (3, 2.5), (5, 1), (5, 1.5),
]
DEM_REGIONAL_CANDIDATES = [
    (3, 3), (3, 5), (3, 8), (5, 2), (5, 2.5), (5, 4), (5, 5),
]


def resolve_dem_crop(hp: dict) -> dict | None:
    """Build a dem_crop_config dict from hyperparameters.

    Accepts HPs that contain either ``local_dem_patch`` / ``local_dem_km``
    (explicit) or ``local_dem_cfg`` / ``regional_dem_cfg`` (index into the
    candidate lists). Returns None if no DEM crop info is present.
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
# Daily reanalysis variable configs (15 derived channels used by LAND_AS).
# ---------------------------------------------------------------------------
DAILY_VARIABLE_CONFIGS = {
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
    "wind_div_925": {
        "description": "Horizontal wind divergence at 925 hPa (du/dx + dv/dy, finite differences)",
        "operation": "divergence",
        "u_variable": "Zonal Wind", "u_level": 925,
        "v_variable": "Meridional Wind", "v_level": 925,
    },
    "skin_temp": {
        "description": "Skin temperature",
        "variable": "Skin Temperature", "interpolate": True,
    },
}

DAILY_VARIABLE_NAMES = list(DAILY_VARIABLE_CONFIGS.keys())
