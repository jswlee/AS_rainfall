#!/usr/bin/env python3
"""
Train the best model using hyperparameters found during tuning.

This script loads the best hyperparameters from the tuner results,
builds and trains a model with those parameters, and evaluates its performance.
"""

import os
import sys
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import keras_tuner as kt
from keras_tuner import Hyperband

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model
from src.deep_learning.model import RainfallModel
from scripts.hyperparameter_tuning import load_data

# Create a custom hypermodel that includes input shapes
class RainfallHyperModel:
    def __init__(self, input_shapes):
        self.input_shapes = input_shapes
        
    def build(self, hp):
        """
        Build a model with the given hyperparameters.
        
        Parameters
        ----------
        hp : HyperParameters
            Hyperparameters to use
            
        Returns
        -------
        model : Model
            Keras model with the specified hyperparameters
        """
        # Climate variables input
        climate_input = layers.Input(shape=self.input_shapes['climate_vars'], name='climate_vars')
        climate_features = layers.Dense(
            hp.get('climate_dense_units_1'), 
            activation=hp.get('climate_activation_1')
        )(climate_input)
        climate_features = layers.Dropout(hp.get('climate_dropout_1'))(climate_features)
        climate_features = layers.Dense(
            hp.get('climate_dense_units_2'), 
            activation=hp.get('climate_activation_2')
        )(climate_features)
        
        # Local DEM input
        local_dem_input = layers.Input(shape=self.input_shapes['local_dem'], name='local_dem')
        # Reshape to add channel dimension if needed
        local_dem_reshaped = layers.Reshape((*self.input_shapes['local_dem'], 1))(local_dem_input)
        local_dem_features = layers.Conv2D(
            hp.get('local_dem_filters_1'), 
            kernel_size=(3, 3), 
            activation=hp.get('local_dem_activation_1'), 
            padding='same'
        )(local_dem_reshaped)
        local_dem_features = layers.MaxPooling2D(pool_size=(2, 2))(local_dem_features)
        local_dem_features = layers.Conv2D(
            hp.get('local_dem_filters_2'), 
            kernel_size=(3, 3), 
            activation=hp.get('local_dem_activation_2'), 
            padding='same'
        )(local_dem_features)
        local_dem_features = layers.Flatten()(local_dem_features)
        local_dem_features = layers.Dense(
            hp.get('local_dem_dense_units'), 
            activation=hp.get('local_dem_dense_activation')
        )(local_dem_features)
        
        # Regional DEM input
        regional_dem_input = layers.Input(shape=self.input_shapes['regional_dem'], name='regional_dem')
        # Reshape to add channel dimension if needed
        regional_dem_reshaped = layers.Reshape((*self.input_shapes['regional_dem'], 1))(regional_dem_input)
        regional_dem_features = layers.Conv2D(
            hp.get('regional_dem_filters_1'), 
            kernel_size=(3, 3), 
            activation=hp.get('regional_dem_activation_1'), 
            padding='same'
        )(regional_dem_reshaped)
        regional_dem_features = layers.MaxPooling2D(pool_size=(2, 2))(regional_dem_features)
        regional_dem_features = layers.Conv2D(
            hp.get('regional_dem_filters_2'), 
            kernel_size=(3, 3), 
            activation=hp.get('regional_dem_activation_2'), 
            padding='same'
        )(regional_dem_features)
        regional_dem_features = layers.Flatten()(regional_dem_features)
        regional_dem_features = layers.Dense(
            hp.get('regional_dem_dense_units'), 
            activation=hp.get('regional_dem_dense_activation')
        )(regional_dem_features)
        
        # Month encoding input
        month_input = layers.Input(shape=self.input_shapes['month_encoding'], name='month_encoding')
        month_features = layers.Dense(
            hp.get('month_dense_units'), 
            activation=hp.get('month_activation')
        )(month_input)
        
        # Combine all features
        combined_features = layers.concatenate([
            climate_features, 
            local_dem_features, 
            regional_dem_features, 
            month_features
        ])
        
        # Combined dense layers
        combined_features = layers.Dense(
            hp.get('combined_dense_units_1'), 
            activation=hp.get('combined_activation_1')
        )(combined_features)
        combined_features = layers.Dropout(hp.get('combined_dropout_1'))(combined_features)
        combined_features = layers.Dense(
            hp.get('combined_dense_units_2'), 
            activation=hp.get('combined_activation_2')
        )(combined_features)
        combined_features = layers.Dropout(hp.get('combined_dropout_2'))(combined_features)
        
        # Output layer
        output = layers.Dense(1, activation='linear')(combined_features)
        
        # Create model
        model = models.Model(
            inputs=[climate_input, local_dem_input, regional_dem_input, month_input],
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=hp.get('learning_rate')),
            loss='mse',
            metrics=['mae']
        )
        
        return model

def main():
    parser = argparse.ArgumentParser(description='Train best model from hyperparameter tuning')
    parser.add_argument('--data', type=str, default='output/rainfall_prediction_data.h5',
                        help='Path to H5 file with processed data')
    parser.add_argument('--tuner_dir', type=str, default='tuner_results',
                        help='Directory with tuner results')
    parser.add_argument('--output_dir', type=str, default='best_model',
                        help='Directory to save the best model and results')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs for training')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Configure GPU memory growth
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
    X_train, y_train, X_val, y_val, input_shapes = load_data(
        args.data, 
        validation_split=args.validation_split,
        random_seed=args.random_seed
    )
    
    print(f"Loaded {len(y_train)} training samples and {len(y_val)} validation samples")
    
    # Load the tuner
    print(f"\nLoading tuner from {args.tuner_dir}...")
    tuner = Hyperband(
        hypermodel=None,  # We don't need the hypermodel, just loading for the best params
        objective='val_loss',
        max_epochs=args.epochs,
        directory=args.tuner_dir,
        project_name='rainfall_prediction'
    )
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print best hyperparameters
    print("\nBest hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"  {param}: {value}")
    
    # Build the model with the best hyperparameters
    print("\nBuilding model with best hyperparameters...")
    hypermodel = RainfallHyperModel(input_shapes)
    best_model = hypermodel.build(best_hps)
    
    # Set up callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.output_dir, 'best_model.weights.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    # Train the model with the best hyperparameters
    print("\nTraining model with best hyperparameters...")
    start_time = datetime.now()
    
    history = best_model.fit(
        X_train, y_train,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time}")
    
    # Save the model architecture
    model_json = best_model.to_json()
    with open(os.path.join(args.output_dir, 'model_architecture.json'), 'w') as f:
        f.write(model_json)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    # Evaluate the model
    print("\nEvaluating model on validation data...")
    y_pred = best_model.predict(X_val)
    
    # Calculate metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"Validation R²: {r2:.4f}")
    print(f"Validation RMSE: {rmse:.4f} mm")
    print(f"Validation MAE: {mae:.4f} mm")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Validation R²: {r2:.4f}\n")
        f.write(f"Validation RMSE: {rmse:.4f} mm\n")
        f.write(f"Validation MAE: {mae:.4f} mm\n")
        f.write(f"Training time: {training_time}\n")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([0, np.max(y_val)], [0, np.max(y_val)], 'r--')
    plt.xlabel('Actual Rainfall (mm)')
    plt.ylabel('Predicted Rainfall (mm)')
    plt.title('Predicted vs Actual Rainfall')
    plt.savefig(os.path.join(args.output_dir, 'predictions_vs_actual.png'))
    
    print(f"\nResults saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
