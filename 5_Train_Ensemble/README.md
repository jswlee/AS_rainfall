# Ensemble Model for Rainfall Prediction

## Overview
This component trains an ensemble of LAND-inspired deep learning models to improve prediction accuracy and robustness. It's the final step in the rainfall prediction pipeline.

## Functionality
- Creates an ensemble of models with the same architecture but different random initializations
- Implements k-fold cross-validation for robust evaluation
- Ensures non-negative rainfall predictions with appropriate output activation functions
- Generates comprehensive visualizations of model performance
- Combines predictions from multiple models for improved accuracy

## Directory Structure
```
5_Train_Ensemble/
├── scripts/
│   ├── simple_ensemble.py           # Main script for ensemble training
│   └── generate_visualizations.py   # Script for creating visualizations
├── output/
│   ├── models/                      # Saved ensemble model weights
│   ├── metrics/                     # Performance metrics
│   └── plots/                       # Visualizations of results
└── README.md                        # This file
```

## Key Features
- **K-Fold Cross-Validation**: Trains models on different data splits for robust evaluation
- **Multiple Models Per Fold**: Creates multiple models for each data fold
- **Non-Negative Output**: Uses ReLU or Softplus activation to ensure physically valid rainfall predictions
- **Comprehensive Visualization**: Generates detailed plots of model performance
- **Ensemble Averaging**: Combines predictions from multiple models to reduce variance

## Usage
To train the ensemble model, run:
```bash
cd 5_Train_Ensemble/scripts
python simple_ensemble.py --data_path ../../2_Create_ML_Data/output/rainfall_prediction_data.h5 --hyperparams_path ../../3_Hyperparameter_Tuning/output/land_model_extended_tuner/best_hyperparameters.py --output_dir ../output
```

To generate visualizations after training:
```bash
python generate_visualizations.py --ensemble_dir ../output
```

## Ensemble Configuration
- Number of folds: 5 (configurable)
- Models per fold: 5 (configurable)
- Training epochs: 100 (configurable)
- Batch size: 32 (configurable)

## Output
- Trained model weights for each fold and model
- Performance metrics for individual models and the ensemble
- Detailed visualizations including:
  - Actual vs. predicted rainfall
  - Error distributions
  - Temporal performance analysis
  - Spatial performance analysis

## Dependencies
- tensorflow
- numpy, pandas
- matplotlib, seaborn (for visualization)
- scikit-learn (for metrics and cross-validation)

## Final Results
The ensemble model provides the most robust and accurate rainfall predictions by combining the strengths of multiple models. This approach reduces the impact of individual model biases and improves generalization to unseen data.
