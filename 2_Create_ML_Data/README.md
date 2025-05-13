# ML Data Creation for Rainfall Prediction

## Overview
This component creates machine learning datasets by combining processed rainfall data with climate variables and digital elevation model (DEM) data. It's the second step in the rainfall prediction pipeline.

## Functionality
- Processes DEM data to create grid points and extract local/regional patches
- Processes climate variables from NetCDF files
- Interpolates rainfall data to grid points
- Combines all data sources into a unified dataset for machine learning

## Directory Structure
```
2_Create_ML_Data/
├── scripts/
│   ├── rainfall_prediction_pipeline.py       # Main pipeline script
│   ├── processors/                           # Data processing modules
│   │   ├── dem_processor/                    # DEM processing
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
- **Rainfall Interpolation**: Interpolates station rainfall data to grid points
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

## Dependencies
- numpy, pandas
- xarray (for NetCDF handling)
- scipy (for interpolation)
- matplotlib (for visualization)
- h5py (for HDF5 file handling)

## Next Steps
After creating the ML dataset, proceed to step 3 (Hyperparameter Tuning) to find the optimal model configuration.
