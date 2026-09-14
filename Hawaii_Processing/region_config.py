"""
Archived region-selection configuration removed from Daily_Modeling/config.py.

This file preserves the American Samoa (AS), Hawai'i (HI), and AGGREGATE
(multi-region) dispatch logic that was previously supported by the modelling
pipeline.  Daily_Modeling now assumes AS only; this copy is kept for reference
or for future Hawai'i-specific work under Hawaii_Processing.
"""
from pathlib import Path
import os as _os

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent

# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------
# Set to "AS" (American Samoa only - default), "HI" (Hawai'i only), or
# "AGGREGATE" (combined AS + HI; requires running aggregate/scripts/*.py).
# Can be overridden via the AS_RAINFALL_REGION environment variable.
REGION = _os.environ.get("AS_RAINFALL_REGION", "AS").upper()

# ---------------------------------------------------------------------------
# Raw data paths
# ---------------------------------------------------------------------------
_AS_DIR = REPO_ROOT / "raw_data" / "AS"
_HI_DIR = REPO_ROOT / "raw_data" / "HI"
_AGG_DIR = REPO_ROOT / "raw_data" / "aggregate"

# Region-specific DEM files (3-band: elevation, slope, aspect).
# These are referenced directly by build_dem_patches for multi-region dispatch.
DEM_AS_PATH = _AS_DIR / "DEM" / "10m_tutuila_3band.tif"
DEM_HI_PATH = _HI_DIR / "DEM" / "30m_hawaii.tif"

# Lookup used by get_dem_path_for_station() to route each station to the
# correct DEM file.  Keys match the prefix convention: CSV files starting
# with "HI_" are Hawaii stations; all others are American Samoa.
DEM_PATHS_BY_REGION = {
    "HI": DEM_HI_PATH,
    "AS": DEM_AS_PATH,
}

if REGION == "AGGREGATE":
    STATION_METADATA_PATH = _AGG_DIR / "station_locations.csv"
    REANALYSIS_DIR = _AGG_DIR / "reanalysis_data"
    DAILY_RAINFALL_DIR = _AGG_DIR / "final_rainfall_per_station"
    # For single-DEM callers, default to AS.  Multi-region code should use
    # get_dem_path_for_station() / DEM_PATHS_BY_REGION instead.
    DEM_PATH = DEM_AS_PATH
elif REGION == "HI":
    STATION_METADATA_PATH = _HI_DIR / "station_locations.csv"
    REANALYSIS_DIR = _HI_DIR / "hawaii_climate_variables_daily_1980-2024"
    DAILY_RAINFALL_DIR = _HI_DIR / "final_rainfall_per_station"
    DEM_PATH = DEM_HI_PATH
else:  # "AS" (default)
    STATION_METADATA_PATH = _AS_DIR / "station_locations.csv"
    DEM_PATH = DEM_AS_PATH
    REANALYSIS_DIR = _AS_DIR / "climate_variables_daily_1980-2024"
    DAILY_RAINFALL_DIR = _AS_DIR / "final_rainfall_per_station"


def get_dem_path_for_station(station_name: str):
    """Return the DEM Path for *station_name*.

    Station names that start with ``HI_`` belong to Hawaii; all others belong
    to American Samoa.  The returned value is a ``pathlib.Path``.
    """
    region = "HI" if str(station_name).startswith("HI_") else "AS"
    return DEM_PATHS_BY_REGION[region]
