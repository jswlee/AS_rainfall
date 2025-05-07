#!/usr/bin/env python3
"""
Train a deep learning model for rainfall prediction.

This script trains a neural network model using the processed data from the rainfall prediction pipeline.
The model takes climate variables, DEM data, and month encoding as input to predict rainfall.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the model
from src.deep_learning.model import RainfallModel

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model for rainfall prediction')
    parser.add_argument('--data', type=str, default='output/rainfall_prediction_data.h5',
                        help='Path to H5 file with processed data')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate an existing model without training')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for evaluation')
    
    args = parser.parse_args()
    
    # Set up GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Configure the model
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split,
        'early_stopping_patience': 15,
        'model_dir': args.model_dir,
        'random_seed': args.random_seed
    }
    
    # Initialize the model
    model = RainfallModel(config)
    
    # Load data
    print(f"Loading data from {args.data}...")
    X_train, y_train, X_val, y_val = model.load_data(args.data)
    print(f"Loaded {len(y_train)} training samples and {len(y_val)} validation samples")
    
    # Print data shapes
    print("\nInput shapes:")
    for key, value in X_train.items():
        print(f"  {key}: {value.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Print data statistics
    print("\nRainfall statistics:")
    print(f"  Training mean: {np.mean(y_train):.2f} mm")
    print(f"  Training min: {np.min(y_train):.2f} mm")
    print(f"  Training max: {np.max(y_train):.2f} mm")
    print(f"  Training std: {np.std(y_train):.2f} mm")
    
    if args.evaluate_only and args.model_path:
        # Load existing model
        print(f"\nLoading model from {args.model_path}...")
        model.load(args.model_path)
    else:
        # Build and train the model
        print("\nBuilding model...")
        model.build_model(model.input_shapes)
        model.model.summary()
        
        print("\nTraining model...")
        start_time = datetime.now()
        history = model.train(X_train, y_train, X_val, y_val)
        end_time = datetime.now()
        
        # Print training time
        training_time = end_time - start_time
        print(f"\nTraining completed in {training_time}")
        
        # Plot training history
        print("\nPlotting training history...")
        model.plot_history(history)
        
        # Save model
        model_path = os.path.join(args.model_dir, 'final_model.h5')
        print(f"\nSaving model to {model_path}...")
        model.save(model_path)
    
    # Evaluate model on validation set
    print("\nEvaluating model on validation set...")
    metrics = model.evaluate(X_val, y_val)
    
    # Print metrics
    print("\nValidation metrics:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f} mm")
    print(f"  MAE: {metrics['mae']:.4f} mm")
    print(f"  MAD: {metrics['mad']:.4f} mm")
    print(f"  rRMSE: {metrics['rrmse']:.4f}")
    print(f"  rMAE: {metrics['rmae']:.4f}")
    print(f"  rMAD: {metrics['rmad']:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(args.model_dir, 'validation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Make predictions on validation set
    print("\nGenerating predictions on validation set...")
    y_pred = model.predict(X_val)
    
    # Plot predictions
    print("\nPlotting predictions...")
    model.plot_predictions(y_val, y_pred, title='Predicted vs Actual Rainfall (Validation Set)')
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
