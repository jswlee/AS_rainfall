# Hawaii Processing Archive

This directory holds the Hawai'i and multi-region (`AGGREGATE`) dispatch logic that was removed from `Daily_Modeling` so that American Samoa is always assumed as the modelling region.

Files
- `region_config.py` — archived region selection, raw-data paths, DEM routing, and `get_dem_path_for_station()` previously in `Daily_Modeling/config.py`.
- `build_features_multi_region.py` — archived multi-region DEM grouping logic previously in `Daily_Modeling/data_utils/build_features.py`.

These files are kept for reference or for future dedicated Hawai'i processing. They are not imported by `Daily_Modeling`.
