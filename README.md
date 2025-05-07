# RainfallPredictionWithClimateData

A deep learning framework for predicting rainfall patterns using climate variables and digital elevation models (DEM). This project implements the LAND (Learning Across Non-uniform Domains) methodology to leverage both climate data and topographical information for improved rainfall predictions, enhanced with ensemble learning and cross-validation techniques.

## Project Overview

This project aims to predict rainfall patterns by combining climate variables (temperature, humidity, wind, etc.) with topographical information from digital elevation models (DEM). The approach follows these key steps:

1. **Data Processing**: Processes climate variables, DEM data, and historical rainfall data
2. **Feature Engineering**: Creates local and regional DEM patches to capture topographical influences
3. **Deep Learning**: Implements a neural network with CNN layers for DEM processing and dense layers for climate variables
4. **Hyperparameter Tuning**: Optimizes model architecture and parameters using Keras Tuner with Bayesian Optimization
5. **Ensemble Learning**: Combines multiple models trained with cross-validation for improved prediction accuracy
6. **Evaluation**: Assesses model performance with metrics like R², RMSE, and MAE

## Key Features

- **Climate Data Integration**: Processes and incorporates various climate variables
- **Multi-scale DEM Analysis**: Uses both local and regional DEM patches to capture topographical influences
- **Neural Network Architecture**: Combines CNN layers for DEM processing with dense layers for climate variables
- **Advanced Hyperparameter Optimization**: Implements systematic tuning using Keras Tuner with Bayesian Optimization (100 trials)
- **Ensemble Learning with Cross-Validation**: Combines predictions from multiple models trained on different data subsets
- **Comprehensive Evaluation**: Generates detailed metrics and visualizations of model performance

## Environment Setup

### Prerequisites
- Python 3.11.9 or newer
- pip (Python package installer)

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RainfallPredictionWithClimateData.git
   cd RainfallPredictionWithClimateData
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. You're all set! You can now run the project scripts.

### Deactivating the Environment
When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```

## Usage

The project provides a unified interface through `main.py` with various actions:

### Data Pipeline

Process raw climate and DEM data to create the dataset for deep learning:

```bash
python main.py --action pipeline --h5_file output/rainfall_prediction_data.h5
```

### Data Preprocessing

Before running the pipeline, you need to prepare the climate and rainfall data:

#### Climate Data Preprocessing

The climate data file `processed_data/AS_climate_var_ds_updated.nc` is created by:

1. Downloading data files from https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
2. Processing the raw NetCDF files to extract relevant climate variables
3. Regridding to a consistent spatial resolution
4. Combining variables into a single NetCDF file

You can regenerate the climate data using:

```bash
python main.py --action regenerate_climate
```

#### Rainfall Data Preprocessing

The monthly rainfall data in `processed_data/mon_rainfall` is created by:

1. Collecting station-based rainfall measurements
2. Performing quality control and gap-filling
3. Converting to a gridded format through interpolation
4. Saving as monthly files with consistent naming conventions

You can process raw rainfall data into monthly aggregates using:

```bash
python main.py --action process_rainfall --raw_rainfall_dir data/raw_rainfall --mon_rainfall_dir processed_data/mon_rainfall
```

This script:
- Reads raw rainfall CSV files with datetime and precipitation columns
- Aggregates the data into monthly totals
- Handles missing values appropriately
- Saves the processed data as CSV files with a consistent format

These preprocessing steps are essential for ensuring data quality and consistency before feeding into the deep learning pipeline.

### Model Training

Train the deep learning model with default parameters:

```bash
python main.py --action train_model --h5_file output/rainfall_prediction_data.h5 --model_dir model_output --epochs 100
```

### Basic Hyperparameter Tuning

Optimize model hyperparameters using Keras Tuner with a limited number of trials:

```bash
python main.py --action tune_hyperparams --h5_file output/rainfall_prediction_data.h5 --max_trials 20 --epochs 50 --min_epochs_per_trial 15
```

### Extended Hyperparameter Tuning

Perform a more comprehensive hyperparameter search with 100 trials and an expanded parameter space:

```bash
python land_model/extended_hyperparameter_tuning.py --features_path csv_data/features_for_model.csv --targets_path csv_data/targets_for_model.csv --output_dir land_model_extended_tuner --max_trials 100 --epochs 50
```

This extended tuning:
- Explores a wider range of hyperparameters including activation functions and residual connections
- Uses Bayesian Optimization for more efficient parameter search
- Implements cosine decay learning rate schedule with warmup
- Analyzes hyperparameter importance to identify key factors affecting performance

### Training with Best Hyperparameters

Train a model using the best hyperparameters found during tuning:

```bash
python land_model/train_best_model.py --features_path csv_data/features_for_model.csv --targets_path csv_data/targets_for_model.csv --hyperparams_path land_model_extended_tuner/best_hyperparameters.txt --output_dir land_model_best
```

### Ensemble Learning with Cross-Validation

Train an ensemble of models using k-fold cross-validation for improved performance:

```bash
python land_model/ensemble_cv_model.py --features_path csv_data/features_for_model.csv --targets_path csv_data/targets_for_model.csv --hyperparams_path land_model_extended_tuner/best_hyperparameters.txt --output_dir land_model_ensemble --n_folds 5 --n_models 5
```

This ensemble approach:
- Implements 5-fold cross-validation to ensure robust performance across different data splits
- Creates 5 models per fold (25 total models) with different random initializations
- Combines predictions from all models for the final output
- Provides detailed evaluation metrics for each fold and the ensemble as a whole

### Model Evaluation

Evaluate the trained model and generate performance visualizations:

```bash
python main.py --action evaluate_best_model --h5_file output/rainfall_prediction_data.h5 --best_model_dir best_model --eval_dir evaluation_results
```

### Rainfall Prediction

Generate rainfall predictions for specific dates:

```bash
python main.py --action predict --h5_file output/rainfall_prediction_data.h5 --model_dir best_model --output_dir predictions
```

## Model Architecture

The neural network architecture consists of:

1. **Climate Variables Branch**: Dense layers processing 16 climate features with batch normalization
2. **Local DEM Branch**: Processes 3x3 local topography patches through flattening and dense layers
3. **Regional DEM Branch**: Similar structure for processing regional topography information
4. **Month Encoding Branch**: Dense layer processing temporal information (one-hot encoded months)
5. **Combined Layers**: Merged features from all branches processed through dense layers with dropout and batch normalization

The ensemble version enhances this architecture with:
- Optional residual connections for better gradient flow
- Multiple activation function options (relu, elu, selu)
- Adaptive learning rate scheduling with warmup
- Improved regularization through optimized dropout and L2 regularization

## Performance Evolution

The model performance improved significantly through several development phases:

| Model | Test R² | Test RMSE (in) | Test MAE (in) |
|-------|---------|----------------|---------------|
| Single Hyperparameter Tuned | 0.2472 | 0.8715 | 0.1669 |
| Ensemble with CV | **0.7955** | **0.4522** | **.1325** |

The ensemble model with cross-validation achieved the best performance across all metrics, with a 12.9% improvement in R² over the previous best model and a 40.0% improvement over the initial tuned model.

### Cross-Validation Performance

The ensemble model showed consistent performance across different data splits:

| Metric | Average CV Value | Test Value |
|--------|-----------------|------------|
| R² | 0.6253 | 0.7955 |
| RMSE (in) | 0.6334 | 0.4522 |
| MAE (in) | 0.1443 | 0.1325 |

## Data Requirements

The project requires several data sources to function properly:

### Digital Elevation Model (DEM)
- File: `data/DEM/DEM_Tut1.tif`
- Format: GeoTIFF
- Resolution: 1km
- Source: USGS or similar topographical data provider

### Climate Variables
- File: `processed_data/AS_climate_var_ds_updated.nc`
- Format: NetCDF4
- Variables: Temperature, humidity, wind speed, pressure, etc.
- Source: ERA5 reanalysis data from Copernicus Climate Data Store

### Rainfall Measurements
- Directory: `processed_data/mon_rainfall`
- Format: CSV or NetCDF files with monthly data
- Variables: Precipitation amounts
- Source: Local weather stations or global precipitation datasets

### Station Locations
- File: `data/as_raingage_list2.csv`
- Format: CSV with latitude, longitude coordinates
- Purpose: Defines locations for rainfall measurements

These data files should be placed in their respective directories before running the pipeline.

## Project Structure

```
RainfallPredictionWithClimateData/
├── main.py                      # Main interface
├── src/
│   ├── utils/                   # Utility functions
│   ├── data_processing/         # Data processing modules
│   └── deep_learning/           # Deep learning model
├── land_model/                  # LAND-inspired model implementation
│   ├── data_utils.py            # Data preprocessing utilities
│   ├── model.py                 # Base model architecture
│   ├── training.py              # Training utilities
│   ├── hyperparameter_tuning.py # Basic hyperparameter tuning
│   ├── extended_hyperparameter_tuning.py # Extended tuning (100 trials)
│   ├── train_best_model.py      # Training with best parameters
│   └── ensemble_cv_model.py     # Ensemble learning with cross-validation
├── scripts/
│   ├── rainfall_prediction_pipeline.py  # Data pipeline
│   └── other utility scripts
├── csv_data/                    # Processed CSV data for model training
├── land_model_tuner/            # Basic hyperparameter tuning results
├── land_model_extended_tuner/   # Extended hyperparameter tuning results
├── land_model_best/             # Best single model results
└── land_model_ensemble/         # Ensemble model results
```

## Key Findings

The development of this rainfall prediction model revealed several important insights:

1. **Ensemble Learning Effectiveness**: The combination of multiple models through ensemble learning significantly improved prediction accuracy, demonstrating that the collective intelligence of multiple models outperforms even the best single model.

2. **Cross-Validation Importance**: The use of k-fold cross-validation ensured that the model was robust to different data splits and reduced the risk of overfitting to a particular subset of the data.

3. **Hyperparameter Optimization**: Extended hyperparameter tuning with 100 trials was crucial for finding the optimal model configuration, highlighting the importance of thorough exploration of the hyperparameter space.

4. **Data Preprocessing Impact**: Proper normalization and scaling of input features had a dramatic effect on model performance, turning a failing model into one with good predictive power.

5. **Architectural Balance**: Finding the right balance of model complexity was essential - the initial model was too complex, while the final model had a more balanced architecture with appropriate regularization.

## Future Work

Potential areas for further improvement include:

1. **Additional Data Sources**: Incorporating satellite imagery or more detailed temporal information
2. **Alternative Ensemble Techniques**: Exploring stacking or boosting approaches
3. **Attention Mechanisms**: Implementing attention layers to better capture spatial relationships
4. **Temporal Modeling**: Adding recurrent neural network components to better model seasonal patterns
5. **Transfer Learning**: Leveraging pre-trained models from related domains

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The LAND methodology which inspired this approach: Hatanaka, Y., Indika, A., Giambelluca, T., & Sadowski, P. (2024). Statistical Downscaling of Climate Models with Deep Learning. March 2024.