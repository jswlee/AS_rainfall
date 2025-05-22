# ML Data Creation for Rainfall Prediction

## Overview
This component creates machine learning datasets by combining processed rainfall data with climate variables and digital elevation model (DEM) data. It's the second step in the rainfall prediction pipeline.

## Functionality
- Processes DEM data to create grid points and extract local/regional patches
  - Uses accurate distance calculations based on latitude/longitude coordinates
  - Creates appropriately sized patches for Tutuila's geography
- Processes climate variables from NetCDF files
- Interpolates rainfall data to grid points
- Combines all data sources into a unified dataset for machine learning

## Patch Sizes
- **Local Patches**: 3x3 grid with 2km per cell (6km total)
- **Regional Patches**: 3x3 grid with 8km per cell (24km total)

## Directory Structure
```
2_Create_ML_Data/
├── scripts/
│   ├── rainfall_prediction_pipeline.py       # Main pipeline script
│   ├── processors/                           # Data processing modules
│   │   ├── dem_processor/                    # DEM processing with accurate distance calculations
│   │   ├── climate_processor/                # Climate data processing
│   │   └── rainfall_processor/               # Rainfall data processing
│   └── utils/                                # Utility functions
│       └── data_generator.py                 # Generates ML datasets
├── output/                                   # Contains processed data
│   ├── processed_climate_data.nc             # Processed climate variables
│   ├── rainfall_prediction_data.h5           # Combined ML dataset
│   └── pipeline_output.log                   # Pipeline execution log
└── README.md                                 # This file
```

## Key Features
- **Robust Climate Data Handling**: Processes climate data from either raw NetCDF files or existing processed files
- **DEM Processing**: Creates local (12km) and regional (60km) patches around grid points
- **Rainfall Interpolation with Gaussian Processes**: Uses Gaussian Process (GP) regression to interpolate station rainfall data to grid points for each time step. If GP interpolation fails or produces all zeros, the pipeline falls back to RBF or IDW interpolation methods for robustness.
- **Data Integration**: Combines all data sources with proper alignment

## Usage
To create the ML dataset, run:
```bash
cd 2_Create_ML_Data/scripts
python rainfall_prediction_pipeline.py
```

The pipeline will:
1. Check for required input files
2. Process DEM data to create grid points and patches
3. Process climate data (from raw files or existing processed data)
4. Process and interpolate rainfall data
5. Generate the combined ML dataset

## Input Requirements
- DEM data (GeoTIFF format)
- Raw or processed climate variables (NetCDF files)
- Monthly rainfall data (from step 1)
- Station location data (CSV format)

## Output
- **rainfall_prediction_data.h5**: HDF5 file containing the combined dataset with features and labels
- **processed_climate_data.nc**: NetCDF file with processed climate variables
- Visualization files for DEM patches and interpolated rainfall

## Notes on Gaussian Process Interpolation
- **Randomness and Reproducibility**: Gaussian Process interpolation (with optimizer restarts) is sensitive to the state of the global random number generator (RNG). Any code that uses randomness (including visualization functions that sample or shuffle) before or during interpolation can affect results.
- **Visualization Bug**: Previously, calling the `visualize_gp_interpolation` function inside the interpolation routine altered the RNG state, causing non-reproducible and inconsistent rainfall interpolation results. This was fixed by commenting out the visualization call.
- **Best Practices**:
  - Set random seeds (`np.random.seed`, `random.seed`, etc.) at the start of the pipeline for reproducibility.
  - Avoid calling visualization or any code that uses randomness before critical interpolation steps, or save/restore the RNG state if needed.
  - If you need to visualize, do so after all data has been generated, or explicitly reset the seed/state before each interpolation.
- **Fallback Logic**: If GP interpolation produces all-zero rainfall (a sign of failure or insufficient data), the pipeline automatically tries RBF and then IDW interpolation to ensure valid outputs.

## Dependencies
- numpy, pandas
- xarray (for NetCDF handling)
- scipy (for interpolation)
- matplotlib (for visualization)
- h5py (for HDF5 file handling)

## Next Steps
After creating the ML dataset, proceed to step 3 (Hyperparameter Tuning) to find the optimal model configuration.
