# Training the Best Rainfall Prediction Model

## Overview
This component trains the LAND-inspired deep learning model using the optimal hyperparameters identified during the tuning phase. It's the fourth step in the rainfall prediction pipeline.

## Functionality
- Loads the best hyperparameters from the tuning phase
- Builds and trains the deep learning model
- Implements non-negative output activation to ensure physically valid rainfall predictions
- Evaluates model performance on test data
- Saves the trained model and performance metrics

## Directory Structure
```
4_Train_Best_Model/
├── scripts/
│   ├── train_best_model.py   # Main script for model training
│   ├── data_utils.py         # Utilities for data loading and preprocessing
│   └── training.py           # Core training functionality
├── output/
│   ├── models/               # Saved model weights
│   ├── metrics/              # Performance metrics
│   └── plots/                # Visualizations of results
└── README.md                 # This file
```

## Key Features
- **Non-negative Output**: Uses ReLU or Softplus activation to ensure physically valid rainfall predictions
- **Comprehensive Evaluation**: Calculates multiple metrics (RMSE, MAE, R²)
- **Visualization**: Generates plots of actual vs. predicted rainfall
- **Reproducibility**: Sets random seeds for consistent results

## Usage
To train the best model, run:
```bash
cd 4_Train_Best_Model/scripts
python train_best_model.py --data_path ../../2_Create_ML_Data/output/rainfall_prediction_data.h5 --hyperparams_path ../../3_Hyperparameter_Tuning/output/land_model_extended_tuner/best_hyperparameters.py --output_dir ../output
```

## Model Architecture
The LAND-inspired model architecture includes:
- Input branches for climate variables, DEM patches, and temporal features
- Multiple dense layers with configurable units and activation functions
- Batch normalization for stable training
- Dropout for regularization
- Non-negative output activation (ReLU or Softplus)

## Output
- Trained model weights saved in HDF5 format
- Performance metrics (RMSE, MAE, R²) on validation and test sets
- Visualizations of actual vs. predicted rainfall
- Training history (loss and metrics over epochs)

## Dependencies
- tensorflow
- numpy, pandas
- matplotlib, seaborn (for visualization)
- scikit-learn (for metrics)

## Next Steps
After training the best model, proceed to step 5 (Train Ensemble) to create an ensemble of models for improved prediction accuracy.
