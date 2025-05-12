# American Samoa Rainfall Prediction Project

## Project Overview
This project implements a complete machine learning pipeline for predicting rainfall in American Samoa using climate variables and topographic data. The pipeline combines climate data, digital elevation models (DEM), and historical rainfall measurements to train deep learning models that can predict monthly rainfall with high accuracy.

## Key Features
- **Non-Negative Rainfall Predictions**: Ensures physically valid predictions using ReLU or Softplus output activations
- **Multi-Modal Data Integration**: Combines climate variables, elevation data, and temporal features
- **Ensemble Modeling**: Improves prediction accuracy and robustness through model ensembling
- **Comprehensive Evaluation**: Provides detailed performance metrics and visualizations
- **Modular Pipeline**: Organized into discrete, reusable components

## Pipeline Components
The project is organized into five main components, each building on the previous:

1. **[Process Rainfall Data](./1_Process_Rainfall_Data/README.md)**: Processes raw rainfall measurements into monthly aggregates
2. **[Create ML Data](./2_Create_ML_Data/README.md)**: Combines rainfall data with climate variables and DEM data
3. **[Hyperparameter Tuning](./3_Hyperparameter_Tuning/README.md)**: Finds optimal model hyperparameters
4. **[Train Best Model](./4_Train_Best_Model/README.md)**: Trains a single model with the best hyperparameters
5. **[Train Ensemble](./5_Train_Ensemble/README.md)**: Creates an ensemble of models for improved predictions

## Data Sources
- **Climate Data**: NetCDF files containing atmospheric variables (temperature, pressure, humidity, etc.)
- **Digital Elevation Model (DEM)**: GeoTIFF files with elevation data for American Samoa
- **Rainfall Measurements**: Historical rainfall data from weather stations in American Samoa

## Model Architecture
The project uses a LAND-inspired deep learning architecture with:
- Multiple input branches for different data types
- Dense layers with batch normalization and dropout
- Non-negative output activation (ReLU or Softplus)
- Configurable hyperparameters for optimal performance

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Required packages: numpy, pandas, xarray, matplotlib, h5py, scikit-learn, keras-tuner

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/jswlee/AS_rainfall.git
   cd AS_rainfall
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
To run the complete pipeline:

1. Process rainfall data:
   ```bash
   cd 1_Process_Rainfall_Data/scripts
   python rainfall_daily_to_monthly.py
   ```

2. Create ML data:
   ```bash
   cd ../../2_Create_ML_Data/scripts
   python rainfall_prediction_pipeline.py
   ```

3. Perform hyperparameter tuning:
   ```bash
   cd ../../3_Hyperparameter_Tuning/scripts
   python extended_hyperparameter_tuning.py
   ```

4. Train the best model:
   ```bash
   cd ../../4_Train_Best_Model/scripts
   python train_best_model.py
   ```

5. Train the ensemble model:
   ```bash
   cd ../../5_Train_Ensemble/scripts
   python simple_ensemble.py
   ```

## Results
The ensemble model achieves state-of-the-art performance for rainfall prediction in American Samoa, with:
- Low root mean square error (RMSE)
- High coefficient of determination (R²)
- Physically valid (non-negative) rainfall predictions
- Robust performance across different spatial and temporal conditions

## Project Structure
```
AS_rainfall/
├── 1_Process_Rainfall_Data/       # Process raw rainfall data
├── 2_Create_ML_Data/              # Create ML datasets
├── 3_Hyperparameter_Tuning/       # Find optimal hyperparameters
├── 4_Train_Best_Model/            # Train single best model
├── 5_Train_Ensemble/              # Train ensemble model
├── raw_data/                      # Raw input data
│   ├── climate_variables/         # NetCDF climate data
│   ├── DEM/                       # Digital elevation models
│   └── AS_raingages/              # Rainfall station data
├── validation/                    # Validation scripts
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Validation
The `validation` directory contains scripts for validating the pipeline components, including:
- Comparing files to ensure data consistency

## License


## Acknowledgments

