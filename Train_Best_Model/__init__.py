"""
Train Best Model Module for AS Rainfall Prediction

This package provides:
- Best model training with optimized hyperparameters
- Cross-validation and single-split training strategies
- MLflow experiment tracking and artifact logging
- Model evaluation and visualization utilities
- Integration with hyperparameter tuning results
"""

from .train_land_model import train_best_model_pytorch

__all__ = [
    'train_best_model_pytorch'
]