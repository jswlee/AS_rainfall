"""
Archived multi-region DEM patch building logic removed from
Daily_Modeling/data_utils/build_features.py.

This snippet preserves the original ``build_dem_patches`` behaviour that
selected the correct DEM per station based on the station-name prefix
(HI_ -> Hawai'i, otherwise American Samoa).  Daily_Modeling now assumes a
single American Samoa DEM, so this dispatch code is archived here for
reference or future Hawai'i-specific processing under Hawaii_Processing.

Dependencies from the original module (rasterio, config, etc.) are left
implied; this file is not intended to be imported directly without adapting
the imports.
"""

from pathlib import Path
from typing import Dict, List, Tuple


def build_dem_patches_multi_region_archive(
    station_metadata: Dict[str, dict],
    dem_path: Path = None,
    local_cfg: dict = None,
    regional_cfg: dict = None,
    config=None,  # would be Daily_Modeling.config in original code
) -> Dict[str, dict]:
    """Build local + regional DEM patches for each station (multi-region).

    When *dem_path* is None the function uses ``config.get_dem_path_for_station``
    to select the correct DEM per station (e.g. AS vs HI in aggregate mode).
    Pass an explicit *dem_path* to force a single DEM for all stations.

    Args:
        station_metadata: {station_name: {latitude, longitude, ...}}.
        dem_path: Override DEM for all stations.  When None, each station is
                  routed to the DEM matching its region prefix.
        local_cfg: dict with 'patch_size' and 'km_per_cell'.
        regional_cfg: same as local_cfg, for regional patch.
        config: Config module providing DEM_PATH, DEM_PATCH_CONFIG,
                get_dem_path_for_station(), etc.

    Returns {station: {"local": ndarray, "regional": ndarray}}.
    """
    if local_cfg is None:
        local_cfg = config.DEM_PATCH_CONFIG["local"]
    if regional_cfg is None:
        regional_cfg = config.DEM_PATCH_CONFIG["regional"]

    patches: Dict[str, dict] = {}

    if dem_path is not None:
        # Single explicit DEM for all stations (AS-only or HI-only runs)
        dem_groups: Dict[Path, List[Tuple[str, dict]]] = {
            Path(dem_path): sorted(station_metadata.items())
        }
    else:
        # Group stations by their region-specific DEM path
        dem_groups: Dict[Path, List[Tuple[str, dict]]] = {}
        for name, meta in sorted(station_metadata.items()):
            p = config.get_dem_path_for_station(name)
            dem_groups.setdefault(p, []).append((name, meta))

    # In the original module the loop below used rasterio.open(grp_path)
    # and extract_dem_patch(...) from build_features.py; reproduced here in
    # outline form for reference.
    #
    # for grp_path, station_items in dem_groups.items():
    #     with rasterio.open(str(grp_path)) as src:
    #         for name, meta in station_items:
    #             local = extract_dem_patch(
    #                 src, meta["longitude"], meta["latitude"],
    #                 local_cfg["patch_size"], local_cfg["km_per_cell"],
    #             )
    #             regional = extract_dem_patch(
    #                 src, meta["longitude"], meta["latitude"],
    #                 regional_cfg["patch_size"], regional_cfg["km_per_cell"],
    #             )
    #             patches[name] = {"local": local, "regional": regional}

    return patches
