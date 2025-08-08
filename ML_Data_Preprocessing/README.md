# ML Data Preprocessing Pipeline

This module implements a structured data preprocessing pipeline for rainfall downscaling using the LAND-style deep learning approach. The pipeline prepares training data by extracting station metadata, building DEM patches, preparing reanalysis features, one-hot encoding months, and assembling training data.

## Module Structure

- `__init__.py`: Package initialization file
- `config.py`: Configuration parameters for the pipeline
- `utils.py`: Utility functions used across the pipeline
- `extract_station_metadata.py`: Functions to extract and clean station metadata
- `build_dem_patches.py`: Functions to build local and regional DEM patches at station locations
- `build_reanalysis_features.py`: Functions to build reanalysis feature patches at station locations
- `onehot_month.py`: Functions to one-hot encode months
- `assemble_training_data.py`: Functions to assemble training data from all components

## Data Sources

The pipeline processes the following data sources:

1. **Station Rainfall Data**: CSV files located at `/Process_Rainfall_Data/output/monthly_rainfall/`, with filenames in the format `{station_name}monthly.csv`.

2. **Digital Elevation Model (DEM)**: GeoTIFF file located at `/raw_data/DEM/DEM_Tut1.tif`.

3. **Reanalysis Variables**: NetCDF files located at `/raw_data/climate_variables/`.

4. **Station Metadata**: CSV file located at `/raw_data/station_locations.csv`, containing station names, latitudes, and longitudes.

## Pipeline Steps

### 1. Extract Station Metadata

Loads and cleans station metadata from the station locations CSV file, creating a dictionary with station names as keys and metadata (latitude, longitude) as values.

```python
from ML_Data_Preprocessing.extract_station_metadata import load_station_metadata, clean_station_metadata, get_station_metadata_dict

# Load and clean station metadata
raw_metadata = load_station_metadata()
clean_metadata = clean_station_metadata(raw_metadata)
station_metadata = get_station_metadata_dict(clean_metadata)
```

### 2. Build DEM Patches

Extracts local (3x3 at 2km per cell) and regional (3x3 at 8km per cell) DEM patches at each station location, and standardizes them by subtracting the mean and dividing by the standard deviation.

```python
from ML_Data_Preprocessing.build_dem_patches import DEMPatchBuilder

# Initialize DEM patch builder and build patches
dem_builder = DEMPatchBuilder()
dem_patches = dem_builder.build_patches_for_stations(station_metadata)
standardized_dem_patches = dem_builder.standardize_patches(dem_patches)
```

### 3. Build Reanalysis Features

Extracts 3x3 patches of 16 reanalysis variables centered on the nearest grid point to each station location, and standardizes them by subtracting the mean and dividing by the standard deviation.

```python
from ML_Data_Preprocessing.build_reanalysis_features import ReanalysisFeatureBuilder

# Initialize reanalysis feature builder and build features
reanalysis_builder = ReanalysisFeatureBuilder()
reanalysis_features = reanalysis_builder.build_features_for_all_stations(
    station_metadata, start_year=1979, end_year=2023
)
standardized_reanalysis = reanalysis_builder.standardize_features(reanalysis_features)
```

### 4. One-Hot Encode Month

Creates a 12-element one-hot encoded array for each month, where the corresponding month index is set to 1 and all other elements are 0.

```python
from ML_Data_Preprocessing.onehot_month import onehot_encode_month

# One-hot encode a month (1-12)
month_onehot = onehot_encode_month(month)
```

### 5. Assemble Training Data

Combines rainfall labels, one-hot encoded months, DEM patches, and reanalysis features into a single dataset. Normalizes rainfall labels by subtracting the mean and dividing by the standard deviation after filtering out outliers.

```python
from ML_Data_Preprocessing.assemble_training_data import TrainingDataAssembler

# Initialize training data assembler
assembler = TrainingDataAssembler()

# Load station rainfall data
rainfall_data = assembler.load_all_station_rainfall(station_metadata)

# Normalize rainfall data
normalized_rainfall = assembler.normalize_rainfall(rainfall_data)

# Assemble training examples
training_data = assembler.assemble_training_examples(
    normalized_rainfall, standardized_dem_patches, standardized_reanalysis
)

# Save the dataset
assembler.save_dataset(training_data)
```

## Usage

To run the complete pipeline:

```python
from ML_Data_Preprocessing import assemble_training_data

# Run the main function to execute the complete pipeline
training_data = assemble_training_data.main()
```

Or to run individual components:

```python
from ML_Data_Preprocessing import extract_station_metadata
from ML_Data_Preprocessing import build_dem_patches
from ML_Data_Preprocessing import build_reanalysis_features
from ML_Data_Preprocessing import onehot_month

# Run individual components
station_metadata = extract_station_metadata.main()
dem_patches = build_dem_patches.main()
reanalysis_features = build_reanalysis_features.main()
onehot_month.main()  # Demonstration of month encoding
```

## Output

The pipeline produces the following outputs:

1. **ML Training Dataset**: A CSV file containing the assembled training examples, saved at `ML_Data_Preprocessing/output/ml_training_dataset.csv`.

2. **Dataset Metadata**: A text file containing metadata about the dataset, saved at `ML_Data_Preprocessing/output/dataset_metadata.txt`.

3. **Visualizations**: Various visualizations of the dataset, saved in the `ML_Data_Preprocessing/output/figures/` directory.

## Notes

- The pipeline directly interpolates to station locations rather than using a 5x5 grid as in the previous implementation.
- Rainfall values are stored in inches, consistent with the original data units.
- All features are standardized to have zero mean and unit variance for better model training.
- The pipeline handles missing or invalid data robustly, with appropriate warnings and fallbacks.
