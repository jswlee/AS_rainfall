#!/usr/bin/env python3
"""
Hyperparameter tuning for the rainfall prediction model.

This script uses Keras Tuner to find the optimal hyperparameters for the rainfall prediction model.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import h5py
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from keras_tuner import Hyperband
from tensorflow.keras import layers, models, optimizers

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the model
from src.deep_learning.model import RainfallModel

# Global variable to store input shapes
input_shapes = None

# Custom Hyperband tuner that enforces minimum epochs
class MinEpochsHyperband(Hyperband):
    def __init__(self, min_epochs=10, **kwargs):
        super(MinEpochsHyperband, self).__init__(**kwargs)
        self.min_epochs = min_epochs
        
    def run_trial(self, trial, *args, **kwargs):
        # Enforce minimum epochs by modifying the epochs in kwargs
        if 'epochs' in kwargs:
            # Store original epochs for reporting
            original_epochs = kwargs['epochs']
            
            # Get the epochs from Hyperband's bracket scheduling
            epochs = trial.hyperparameters.values.get('tuner/epochs', original_epochs)
            
            # Enforce minimum epochs
            if epochs < self.min_epochs:
                print(f"Enforcing minimum {self.min_epochs} epochs instead of {epochs}")
                trial.hyperparameters.values['tuner/epochs'] = self.min_epochs
                kwargs['epochs'] = self.min_epochs
        
        # Call the parent run_trial method
        return super().run_trial(trial, *args, **kwargs)

def build_model_tuner(hp):
    """
    Build a model with tunable hyperparameters.
    
    Parameters
    ----------
    hp : HyperParameters
        Hyperparameters to tune
        
    Returns
    -------
    model : Model
        Keras model with tunable hyperparameters
    """
    # Input layers
    climate_input = layers.Input(shape=input_shapes['climate_vars'], name='climate_vars')
    local_dem_input = layers.Input(shape=input_shapes['local_dem'], name='local_dem')
    regional_dem_input = layers.Input(shape=input_shapes['regional_dem'], name='regional_dem')
    month_input = layers.Input(shape=input_shapes['month_encoding'], name='month_encoding')
    
    # Process climate variables
    climate_features = layers.Dense(
        hp.Int('climate_dense_units_1', min_value=32, max_value=128, step=32),
        activation=hp.Choice('climate_activation_1', values=['relu', 'elu', 'selu'])
    )(climate_input)
    
    climate_features = layers.Dropout(
        hp.Float('climate_dropout_1', min_value=0.0, max_value=0.5, step=0.1)
    )(climate_features)
    
    climate_features = layers.Dense(
        hp.Int('climate_dense_units_2', min_value=16, max_value=64, step=16),
        activation=hp.Choice('climate_activation_2', values=['relu', 'elu', 'selu'])
    )(climate_features)
    
    # Process local DEM with CNN
    local_dem_features = layers.Reshape(input_shapes['local_dem'] + (1,))(local_dem_input)
    
    local_dem_features = layers.Conv2D(
        hp.Int('local_dem_filters_1', min_value=8, max_value=32, step=8),
        (2, 2),
        activation=hp.Choice('local_dem_activation_1', values=['relu', 'elu', 'selu']),
        padding='same'
    )(local_dem_features)
    
    local_dem_features = layers.MaxPooling2D((2, 2), padding='same')(local_dem_features)
    
    local_dem_features = layers.Conv2D(
        hp.Int('local_dem_filters_2', min_value=4, max_value=16, step=4),
        (2, 2),
        activation=hp.Choice('local_dem_activation_2', values=['relu', 'elu', 'selu']),
        padding='same'
    )(local_dem_features)
    
    local_dem_features = layers.Flatten()(local_dem_features)
    
    local_dem_features = layers.Dense(
        hp.Int('local_dem_dense_units', min_value=16, max_value=64, step=16),
        activation=hp.Choice('local_dem_dense_activation', values=['relu', 'elu', 'selu'])
    )(local_dem_features)
    
    # Process regional DEM with CNN
    regional_dem_features = layers.Reshape(input_shapes['regional_dem'] + (1,))(regional_dem_input)
    
    regional_dem_features = layers.Conv2D(
        hp.Int('regional_dem_filters_1', min_value=8, max_value=32, step=8),
        (2, 2),
        activation=hp.Choice('regional_dem_activation_1', values=['relu', 'elu', 'selu']),
        padding='same'
    )(regional_dem_features)
    
    regional_dem_features = layers.MaxPooling2D((2, 2), padding='same')(regional_dem_features)
    
    regional_dem_features = layers.Conv2D(
        hp.Int('regional_dem_filters_2', min_value=4, max_value=16, step=4),
        (2, 2),
        activation=hp.Choice('regional_dem_activation_2', values=['relu', 'elu', 'selu']),
        padding='same'
    )(regional_dem_features)
    
    regional_dem_features = layers.Flatten()(regional_dem_features)
    
    regional_dem_features = layers.Dense(
        hp.Int('regional_dem_dense_units', min_value=16, max_value=64, step=16),
        activation=hp.Choice('regional_dem_dense_activation', values=['relu', 'elu', 'selu'])
    )(regional_dem_features)
    
    # Process month encoding
    month_features = layers.Dense(
        hp.Int('month_dense_units', min_value=4, max_value=16, step=4),
        activation=hp.Choice('month_activation', values=['relu', 'elu', 'selu'])
    )(month_input)
    
    # Combine all features
    combined_features = layers.Concatenate()([
        climate_features, 
        local_dem_features, 
        regional_dem_features, 
        month_features
    ])
    
    # Dense layers for prediction
    x = layers.Dense(
        hp.Int('combined_dense_units_1', min_value=32, max_value=128, step=32),
        activation=hp.Choice('combined_activation_1', values=['relu', 'elu', 'selu'])
    )(combined_features)
    
    x = layers.Dropout(
        hp.Float('combined_dropout_1', min_value=0.0, max_value=0.5, step=0.1)
    )(x)
    
    x = layers.Dense(
        hp.Int('combined_dense_units_2', min_value=16, max_value=64, step=16),
        activation=hp.Choice('combined_activation_2', values=['relu', 'elu', 'selu'])
    )(x)
    
    x = layers.Dropout(
        hp.Float('combined_dropout_2', min_value=0.0, max_value=0.5, step=0.1)
    )(x)
    
    # Output layer
    output = layers.Dense(1, name='rainfall')(x)
    
    # Create model
    model = models.Model(
        inputs=[climate_input, local_dem_input, regional_dem_input, month_input],
        outputs=output
    )
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def load_data(h5_path, validation_split=0.2, random_seed=42):
    """
    Load data from H5 file and prepare for training.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 file containing the processed data
    validation_split : float, optional
        Fraction of data to use for validation
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, input_shapes) - Training and validation data, and input shapes
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Load data from H5 file
    with h5py.File(h5_path, 'r') as h5_file:
        # Get all date keys
        date_keys = sorted([key for key in h5_file.keys() if key.startswith('date_')])
        
        # Initialize lists to store data
        climate_vars_list = []
        local_patches_list = []
        regional_patches_list = []
        month_encodings_list = []
        rainfall_list = []
        
        # Extract data for each date
        for date_key in date_keys:
            date_group = h5_file[date_key]
            
            # Skip if any required data is missing
            if not all(key in date_group for key in 
                       ['climate_vars', 'local_patches', 'regional_patches', 'month_one_hot', 'rainfall']):
                continue
            
            # Get number of grid points
            n_points = len(date_group['rainfall'][:])
            
            # Extract climate variables
            climate_data = []
            for var_name in date_group['climate_vars']:
                var_data = date_group['climate_vars'][var_name][:]
                climate_data.append(var_data)
            
            # Stack climate variables
            climate_vars = np.column_stack(climate_data)
            
            # Extract other data
            local_patches = date_group['local_patches'][:]
            regional_patches = date_group['regional_patches'][:]
            month_encoding = date_group['month_one_hot'][:]
            rainfall = date_group['rainfall'][:]
            
            # Repeat month encoding for each grid point
            month_encodings = np.tile(month_encoding, (n_points, 1))
            
            # Append to lists
            climate_vars_list.append(climate_vars)
            local_patches_list.append(local_patches)
            regional_patches_list.append(regional_patches)
            month_encodings_list.append(month_encodings)
            rainfall_list.append(rainfall)
        
        # Concatenate data
        climate_vars = np.vstack(climate_vars_list)
        local_patches = np.vstack(local_patches_list)
        regional_patches = np.vstack(regional_patches_list)
        month_encodings = np.vstack(month_encodings_list)
        rainfall = np.concatenate(rainfall_list)
        
        # Reshape rainfall to match model output
        rainfall = rainfall.reshape(-1, 1)
        
        # Split data into training and validation sets
        indices = np.arange(len(rainfall))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - validation_split))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        
        # Create training and validation sets
        X_train = {
            'climate_vars': climate_vars[train_idx],
            'local_dem': local_patches[train_idx],
            'regional_dem': regional_patches[train_idx],
            'month_encoding': month_encodings[train_idx]
        }
        
        y_train = rainfall[train_idx]
        
        X_val = {
            'climate_vars': climate_vars[val_idx],
            'local_dem': local_patches[val_idx],
            'regional_dem': regional_patches[val_idx],
            'month_encoding': month_encodings[val_idx]
        }
        
        y_val = rainfall[val_idx]
        
        # Get input shapes
        input_shapes = {
            'climate_vars': climate_vars.shape[1:],
            'local_dem': local_patches.shape[1:],
            'regional_dem': regional_patches.shape[1:],
            'month_encoding': month_encodings.shape[1:]
        }
        
        return X_train, y_train, X_val, y_val, input_shapes

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for rainfall prediction model')
    parser.add_argument('--data', type=str, default='output/rainfall_prediction_data.h5',
                        help='Path to H5 file with processed data')
    parser.add_argument('--output_dir', type=str, default='tuner_results',
                        help='Directory to save tuner results')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs for each trial')
    parser.add_argument('--max_trials', type=int, default=20,
                        help='Maximum number of trials')
    parser.add_argument('--min_epochs', type=int, default=10,
                        help='Minimum number of epochs to train each trial')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set up GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}...")
    X_train, y_train, X_val, y_val, input_shapes_local = load_data(
        args.data, 
        validation_split=args.validation_split,
        random_seed=args.random_seed
    )
    
    print(f"Loaded {len(y_train)} training samples and {len(y_val)} validation samples")
    
    # Make input_shapes global so the model builder can access it
    global input_shapes
    input_shapes = input_shapes_local
    
    # Set up the tuner
    print("\nInitializing hyperparameter tuner...")
    tuner = MinEpochsHyperband(
        min_epochs=args.min_epochs,
        hypermodel=build_model_tuner,
        objective='val_loss',
        max_epochs=args.max_epochs,
        factor=3,
        hyperband_iterations=2,
        directory=args.output_dir,
        project_name='rainfall_prediction',
        overwrite=True
    )
    
    # Print search space summary
    tuner.search_space_summary()
    
    # Set up early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased from 10
        restore_best_weights=True,
        min_delta=0.001  # Minimum change to count as improvement
    )
    
    # Add a minimum number of epochs callback
    class MinEpochsCallback(tf.keras.callbacks.Callback):
        def __init__(self, min_epochs=10):
            super(MinEpochsCallback, self).__init__()
            self.min_epochs = min_epochs
            
        def on_epoch_end(self, epoch, logs=None):
            # Allow early stopping only after min_epochs
            if epoch < self.min_epochs:
                self.model.stop_training = False
    
    min_epochs = MinEpochsCallback(min_epochs=args.min_epochs)
    
    # Start the search
    print("\nStarting hyperparameter search...")
    start_time = datetime.now()
    
    tuner.search(
        X_train, y_train,
        epochs=args.max_epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, min_epochs],
        verbose=1
    )
    
    end_time = datetime.now()
    search_time = end_time - start_time
    print(f"\nHyperparameter search completed in {search_time}")
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print best hyperparameters
    print("\nBest hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"  {param}: {value}")
    
    # Build the model with the best hyperparameters
    print("\nBuilding model with best hyperparameters...")
    best_model = tuner.hypermodel.build(best_hps)
    
    # Train the model with the best hyperparameters
    print("\nTraining model with best hyperparameters...")
    history = best_model.fit(
        X_train, y_train,
        epochs=args.max_epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, min_epochs],
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    loss, mae = best_model.evaluate(X_val, y_val, verbose=0)
    
    # Make predictions
    y_pred = best_model.predict(X_val)
    
    # Calculate metrics
    mse = np.mean((y_val - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))
    
    # Print metrics
    print("\nValidation metrics:")
    print(f"  Loss (MSE): {loss:.4f}")
    print(f"  MAE: {mae:.4f} mm")
    print(f"  RMSE: {rmse:.4f} mm")
    print(f"  R²: {r2:.4f}")
    
    # Save the best model
    model_path = os.path.join(args.output_dir, 'best_model.keras')
    print(f"\nSaving best model to {model_path}...")
    best_model.save(model_path)
    
    # Save hyperparameters to JSON
    hyperparams_path = os.path.join(args.output_dir, 'best_hyperparameters.json')
    with open(hyperparams_path, 'w') as f:
        f.write(best_hps.to_json())
    print(f"Hyperparameters saved to {hyperparams_path}")
    
    # Plot training history
    print("\nPlotting training history...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    history_path = os.path.join(args.output_dir, 'training_history.png')
    plt.savefig(history_path)
    print(f"Training history plot saved to {history_path}")
    
    # Plot predictions vs actual
    print("\nPlotting predictions...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot scatter
    ax.scatter(y_val, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    max_val = max(np.max(y_val), np.max(y_pred))
    min_val = min(np.min(y_val), np.min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add metrics to plot
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.90, f"RMSE = {rmse:.3f} mm", transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.85, f"MAE = {mae:.3f} mm", transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('Actual Rainfall (mm)')
    ax.set_ylabel('Predicted Rainfall (mm)')
    ax.set_title('Predicted vs Actual Rainfall (Validation Set)')
    
    # Save figure
    scatter_path = os.path.join(args.output_dir, 'prediction_scatter.png')
    plt.savefig(scatter_path)
    print(f"Prediction scatter plot saved to {scatter_path}")
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
