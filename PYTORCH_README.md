# PyTorch LAND Rainfall Prediction Pipeline

This document describes the PyTorch-based implementation of the LAND-inspired rainfall prediction model, which replaces the original TensorFlow implementation with a cleaner, more readable PyTorch codebase.

## Overview

The PyTorch pipeline provides:
- **Cleaner, more readable code** using modern PyTorch practices
- **Optuna-based hyperparameter tuning** instead of Keras Tuner
- **Improved training utilities** with better logging and visualization
- **Cross-validation ensemble training** with automatic resumption
- **Better error handling** and progress tracking
- **Consistent data handling** across all components

## Architecture

The LAND model processes four types of input features:
1. **Climate/Reanalysis patches** (16 variables on 3×3 grid)
2. **Local DEM patches** (fine-grained topographic context)
3. **Regional DEM patches** (broader topographic context)  
4. **Month one-hot encodings** (seasonal information)

## Pipeline Components

### 1. Hyperparameter Tuning (`Hyperparameter_Tuning/`)

**New PyTorch Files:**
- `data_utils.py` - PyTorch Dataset classes and data loading
- `model.py` - LAND model architecture in PyTorch
- `pytorch_training.py` - Training utilities with early stopping and scheduling
- `hp_tuning.py` - Optuna-based hyperparameter optimization
- `run_pytorch_tuning.py` - Main entry point for tuning

**Key Features:**
- Uses Optuna for more efficient hyperparameter search
- Cross-validation within each trial
- Automatic pruning of unpromising trials
- SQLite storage for resumable tuning sessions
- Parameter importance analysis

**Usage:**
```bash
cd Hyperparameter_Tuning
python run_pytorch_tuning.py
```

### 2. Best Model Training (`Train_Best_Model/`)

**New PyTorch Files:**
- `pytorch_train_best_model.py` - Train single best model
- `run_pytorch_training.py` - Main entry point

**Key Features:**
- Loads best hyperparameters from tuning
- Advanced learning rate scheduling (cosine annealing with warmup)
- Comprehensive evaluation and visualization
- Automatic model saving in PyTorch format

**Usage:**
```bash
cd Train_Best_Model
python run_pytorch_training.py
```

### 3. Ensemble Training (`Train_Ensemble/`)

**New PyTorch Files:**
- `pytorch_train_ensemble.py` - Cross-validation ensemble training
- `run_pytorch_ensemble.py` - Main entry point

**Key Features:**
- K-fold cross-validation with multiple models per fold
- Automatic progress saving and resumption
- Individual and ensemble predictions
- Comprehensive result analysis

**Usage:**
```bash
cd Train_Ensemble
python run_pytorch_ensemble.py
```

## Data Flow

1. **Data Loading**: NPZ files are loaded using `data_utils.py`
2. **Dataset Creation**: PyTorch `RainfallDataset` handles feature organization
3. **Model Creation**: `LANDModel` class implements the architecture
4. **Training**: Advanced training loop with early stopping and scheduling
5. **Evaluation**: Comprehensive metrics and visualizations

## Key Improvements Over TensorFlow Version

### Code Readability
- Clear separation of concerns
- Consistent naming conventions
- Comprehensive documentation
- Type hints throughout

### Training Efficiency
- Better memory management
- Faster data loading
- More efficient hyperparameter search
- Automatic mixed precision support (when available)

### Robustness
- Better error handling
- Progress saving and resumption
- Comprehensive logging
- Input validation

### Flexibility
- Modular design for easy extension
- Configurable hyperparameters
- Multiple output activation options
- Flexible batch sizing

## Model Architecture Details

The PyTorch LAND model (`LANDModel` class) includes:

### Input Processing
- **Climate features**: Flattened and passed through dense layer
- **DEM features**: Separate processing for local and regional patches
- **Month features**: Dense layer for temporal encoding

### Feature Integration
- All features concatenated after individual processing
- Batch normalization after each dense layer
- Configurable activation functions (ReLU, eLU, SeLU)

### Output Layer
- Single neuron for rainfall prediction
- Optional non-negative activation (ReLU, Softplus)
- Ensures physically meaningful predictions

### Regularization
- Dropout layers for generalization
- L2 regularization via weight decay
- Optional residual connections
- Batch normalization for stable training

## Hyperparameter Search Space

The Optuna tuner optimizes:

### Architecture Parameters
- `climate_units`: 64-256 (step 32)
- `local_dem_units`: 32-128 (step 16)
- `regional_dem_units`: 32-128 (step 16)
- `month_units`: 16-64 (step 8)
- `na`: 128-512 (step 64)
- `nb`: 64-256 (step 32)

### Regularization Parameters
- `dropout_rate`: 0.1-0.5
- `l2_reg`: 1e-5 to 1e-2 (log scale)

### Training Parameters
- `learning_rate`: 1e-4 to 1e-2 (log scale)
- `weight_decay`: 1e-5 to 1e-2 (log scale)
- `batch_size`: [16, 32, 64, 128]

### Model Choices
- `use_residual`: [True, False]
- `activation`: ['relu', 'elu', 'selu']
- `output_activation`: ['relu', 'softplus']

## Output Files

### Hyperparameter Tuning
- `best_hyperparameters.json` - Best parameters in JSON format
- `best_hyperparameters.py` - Best parameters in Python format
- `tuning_summary.txt` - Detailed tuning results
- `optimization_history.png` - Optimization progress plot
- `parameter_importances.png` - Parameter importance analysis

### Best Model Training
- `best_model.pth` - Trained PyTorch model
- `training_history.png` - Training curves
- `evaluation_metrics.csv` - Test set metrics
- `test_predictions_scatter.png` - Predictions vs actual plot
- `test_predictions.json` - Detailed predictions
- `training_summary.txt` - Training summary

### Ensemble Training
- `ensemble_summary.txt` - Ensemble results summary
- `test_predictions.json` - Ensemble predictions
- `ensemble_predictions_scatter.png` - Ensemble scatter plot
- `fold_X/` - Individual fold results
- `ensemble_progress.pkl` - Progress for resumption

## Requirements

### Core Dependencies
```
torch>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0
optuna>=3.0.0
```

### Optional Dependencies
```
plotly>=5.0.0  # For Optuna visualizations
kaleido>=0.2.1  # For saving Plotly plots
```

## Memory and Performance

### Memory Usage
- Efficient data loading with PyTorch DataLoader
- Gradient accumulation support for large models
- Automatic garbage collection between folds

### Performance Optimizations
- Automatic device detection (CPU/GPU)
- Pin memory for faster GPU transfer
- Optimized data preprocessing
- Efficient cross-validation implementation

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in hyperparameters
   - Use gradient accumulation
   - Enable automatic mixed precision

2. **Slow training**
   - Increase batch size if memory allows
   - Use more CPU workers for data loading
   - Enable GPU if available

3. **Poor convergence**
   - Adjust learning rate range in tuning
   - Increase model capacity
   - Check data preprocessing

### Debug Mode
Set environment variable for detailed logging:
```bash
export PYTORCH_DEBUG=1
```

## Migration from TensorFlow

The PyTorch version maintains compatibility with existing data:
- Uses same NPZ data format
- Preserves test set splits
- Maintains evaluation metrics
- Compatible hyperparameter formats

Key differences:
- Model weights are not directly transferable
- Different random number generation
- Slightly different numerical precision

## Future Enhancements

Potential improvements:
- Distributed training support
- Advanced architectures (Transformers, Graph Neural Networks)
- Automated feature engineering
- Real-time inference optimization
- Model interpretability tools

## Contributing

When extending the PyTorch pipeline:
1. Follow existing code style and documentation
2. Add comprehensive type hints
3. Include unit tests for new functionality
4. Update this README with new features
5. Maintain backward compatibility where possible
