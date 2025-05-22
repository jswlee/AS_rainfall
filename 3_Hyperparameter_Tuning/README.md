# Hyperparameter Tuning for Rainfall Prediction

## Overview
This component performs hyperparameter tuning for the LAND-inspired deep learning model used for rainfall prediction. It's the third step in the rainfall prediction pipeline.

## Functionality
- Uses Keras Tuner with Bayesian Optimization to systematically search for optimal hyperparameters
- Evaluates model performance across various hyperparameter combinations using cross-validation
- Saves the best hyperparameter configuration for model training
- Includes non-negative output activation (ReLU or Softplus) to ensure physically valid rainfall predictions
- Supports resuming tuning sessions without losing previous trials

## Directory Structure
```
3_Hyperparameter_Tuning/
├── scripts/
│   └── extended_hyperparameter_tuning.py  # Main script for hyperparameter tuning
├── output/
│   └── land_model_extended_tuner/         # Contains tuning results
│       ├── current_best_hyperparameters.txt # Current best hyperparameters (updated during tuning)
│       ├── current_best_hyperparameters.py  # Python-importable version of best hyperparameters
│       └── land_model_cv_tuning/           # Detailed tuning results and trial history
└── README.md                              # This file
```

## Key Features
- **Extensive Search Space**: Tunes network architecture, learning rates, regularization, and more
- **Cross-Validation**: Uses 5-fold cross-validation for robust evaluation
- **Output Activation**: Includes ReLU or Softplus activation to ensure non-negative rainfall predictions
- **Early Stopping**: Implements early stopping to prevent overfitting
- **Resumable Tuning**: Can resume tuning from previous sessions without losing trial history

## Usage
To perform hyperparameter tuning, run:
```bash
cd 3_Hyperparameter_Tuning/scripts
python extended_hyperparameter_tuning.py --data_path ../../2_Create_ML_Data/output/rainfall_prediction_data.h5 --output_dir ../output
```

## Tunable Hyperparameters
- Network architecture (number of layers, units per layer)
- Learning rate and learning rate schedule
- Regularization (L1, L2, dropout)
- Batch normalization configuration
- Output activation function (ReLU or Softplus)
- Batch size

## Output
- Best hyperparameter configuration saved as a Python file
- Detailed tuning results including performance metrics for each trial
- Visualizations of hyperparameter importance

## Dependencies
- tensorflow
- keras-tuner
- numpy, pandas
- matplotlib (for visualization)

## Next Steps
After finding the optimal hyperparameters, proceed to step 4 (Train Best Model) to train the model with the best configuration.
