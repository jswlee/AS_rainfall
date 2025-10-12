"""
Train Best Model Module for AS Rainfall Prediction

This package provides:
- Best model training with optimized hyperparameters
- Cross-validation and single-split training strategies
- MLflow experiment tracking and artifact logging
- Model evaluation and visualization utilities
- Integration with hyperparameter tuning results
PyTorch training utilities for the best LAND model.
- Simplified training entrypoints and helpers
- Integration with hyperparameter tuning results
"""

__all__ = ['train_best_model_pytorch']

def __getattr__(name):
    # Lazy import to avoid importing submodule during package import,
    # which triggers a RuntimeWarning when running `-m Train_Best_Model.train`.
    if name == 'train_best_model_pytorch':
        from .train import train_best_model_pytorch
        return train_best_model_pytorch
    raise AttributeError(f"module {__name__} has no attribute {name}")