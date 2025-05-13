#!/usr/bin/env python3
"""
Train the LAND-inspired rainfall prediction model with the best hyperparameters
found during hyperparameter tuning.
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))

# Add required directories to Python path
sys.path.append(PROJECT_ROOT)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Import local modules
from data_utils import load_and_reshape_data, create_tf_dataset
from training import train_model, evaluate_model, plot_training_history


def build_best_model(data_metadata, hyperparams=None):
    """
    Build the LAND model with the best hyperparameters.
    
    Parameters
    ----------
    data_metadata : dict
        Dictionary containing metadata about the input data
    hyperparams : dict, optional
        Dictionary containing hyperparameters loaded from a file
        
    Returns
    -------
    tf.keras.Model
        Compiled LAND model
    """
    # Default hyperparameters if none provided
    if hyperparams is None:
        hyperparams = {
            'na': 320,
            'nb': 256,
            'dropout_rate': 0.4,
            'l2_reg': 1.7352550593878845e-05,
            'learning_rate': 0.0007256000814102282,
            'weight_decay': 1.1574311893640013e-06,
            'local_dem_units': 128,
            'regional_dem_units': 96,
            'month_units': 32,
            'climate_units': 256,
            'use_residual': False,
            'activation': 'relu'
        }
    
    # Ensure all required hyperparameters are present
    default_params = {
        'na': 320,
        'nb': 256,
        'dropout_rate': 0.4,
        'l2_reg': 1.7352550593878845e-05,
        'learning_rate': 0.0007256000814102282,
        'weight_decay': 1.1574311893640013e-06,
        'local_dem_units': 128,
        'regional_dem_units': 96,
        'month_units': 32,
        'climate_units': 256,
        'use_residual': False,
        'activation': 'relu'
    }
    
    # Fill in any missing hyperparameters with defaults
    for key, value in default_params.items():
        if key not in hyperparams:
            print(f"Warning: Hyperparameter '{key}' not found in loaded parameters. Using default value: {value}")
            hyperparams[key] = value
    
    # Print the hyperparameters being used
    print("\nUsing hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Create input layers
    climate_input = layers.Input(shape=data_metadata['climate_shape'], name='climate')
    local_dem_input = layers.Input(shape=data_metadata['local_dem_shape'], name='local_dem')
    regional_dem_input = layers.Input(shape=data_metadata['regional_dem_shape'], name='regional_dem')
    month_input = layers.Input(shape=(data_metadata['num_month_encodings'],), name='month')
    
    # Process local DEM
    local_dem = layers.Flatten()(local_dem_input)
    local_dem = layers.Dense(
        hyperparams['local_dem_units'], 
        activation=hyperparams['activation'],
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(local_dem)
    local_dem = layers.BatchNormalization()(local_dem)
    
    # Process regional DEM
    regional_dem = layers.Flatten()(regional_dem_input)
    regional_dem = layers.Dense(
        hyperparams['regional_dem_units'], 
        activation=hyperparams['activation'],
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(regional_dem)
    regional_dem = layers.BatchNormalization()(regional_dem)
    
    # Process month
    month = layers.Dense(
        hyperparams['month_units'], 
        activation=hyperparams['activation'],
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(month_input)
    month = layers.BatchNormalization()(month)
    
    # Process climate/reanalysis data
    climate_flat = layers.Reshape((data_metadata['climate_shape'][0] * 
                                  data_metadata['climate_shape'][1] * 
                                  data_metadata['climate_shape'][2],))(climate_input)
    
    climate = layers.Dense(
        hyperparams['climate_units'], 
        activation=hyperparams['activation'],
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(climate_flat)
    climate = layers.BatchNormalization()(climate)
    
    # Concatenate all features
    concat = layers.Concatenate()([climate, local_dem, regional_dem, month])
    
    # Dense layers
    x = layers.Dense(
        hyperparams['na'], 
        activation=hyperparams['activation'],
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(hyperparams['dropout_rate'])(x)
    
    # Store residual connection if enabled and dimensions match
    residual = None
    if hyperparams['use_residual'] and hyperparams['na'] == hyperparams['nb']:
        residual = x
        
    x = layers.Dense(
        hyperparams['nb'], 
        activation=hyperparams['activation'],
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Add residual connection if enabled and dimensions match
    if hyperparams['use_residual'] and hyperparams['na'] == hyperparams['nb']:
        print("Using residual connection")
        x = layers.Add()([x, residual])
        
    x = layers.Dropout(hyperparams['dropout_rate'])(x)
    
    # Output layer with non-negative activation to ensure rainfall predictions are never negative
    # Default to 'relu' if output_activation not in hyperparams
    output_activation = hyperparams.get('output_activation', 'relu')
    output = layers.Dense(1, activation=output_activation, name='rainfall')(x)
    
    # Create model
    model = tf.keras.Model(
        inputs=[climate_input, local_dem_input, regional_dem_input, month_input],
        outputs=output
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Print model summary
    model.summary()
    
    return model


def main():
    """
    Main function to train the model with the best hyperparameters.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LAND model with best hyperparameters')
    parser.add_argument('--features_path', type=str,
        default=os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'output', 'csv_data', 'features.csv'),
        help='Path to features CSV file')
    parser.add_argument('--targets_path', type=str,
        default=os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'output', 'csv_data', 'targets.csv'),
        help='Path to targets CSV file')
    parser.add_argument('--test_indices_path', type=str,
        default=os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'output', 'test_indices.pkl'),
        help='Path to save or load test indices')
    parser.add_argument('--hyperparams_path', type=str,
        default=os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'output', 'land_model_extended_tuner', 'best_hyperparameters.pkl'),
        help='Path to hyperparameters file')
    parser.add_argument('--output_dir', type=str,
        default=os.path.join(PIPELINE_DIR, 'output', 'land_model_best'),
        help='Directory to save model weights and results')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=314,
                        help='Batch size for training')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and reshape data
    print("Loading and reshaping data...")
    data = load_and_reshape_data(
        features_path=args.features_path,
        targets_path=args.targets_path,
        test_indices_path=args.test_indices_path
    )
    
    # Create TensorFlow datasets
    print("\nCreating TensorFlow datasets...")
    datasets = create_tf_dataset(data, batch_size=args.batch_size)
    
    # Load hyperparameters if path is provided
    hyperparams = None
    if args.hyperparams_path:
        print(f"\nLoading hyperparameters from {args.hyperparams_path}...")
        # Check file extension to determine how to load
        if args.hyperparams_path.endswith('.pkl'):
            # Load binary pickle file
            import pickle
            try:
                with open(args.hyperparams_path, 'rb') as f:
                    hyperparams = pickle.load(f)
            except Exception as e:
                print(f"Error loading pickle file: {e}")
                # Try the text file version as fallback
                text_path = args.hyperparams_path.replace('.pkl', '.txt')
                if os.path.exists(text_path):
                    print(f"Falling back to text file: {text_path}")
                    args.hyperparams_path = text_path
                else:
                    print(f"No fallback file found. Using default hyperparameters.")
        
        # If it's a text file or fallback to text file
        if not hyperparams and args.hyperparams_path.endswith('.txt'):
            hyperparams = {}
            with open(args.hyperparams_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        try:
                            # Try to convert to float or int
                            value = float(value.strip())
                            if value.is_integer():
                                value = int(value)
                        except ValueError:
                            # Keep as string if not a number
                            value = value.strip()
                        hyperparams[key.strip()] = value
        
        print(f"Loaded hyperparameters: {hyperparams}")
    
    # Build the model with best hyperparameters
    print("\nBuilding model with best hyperparameters...")
    model = build_best_model(data['metadata'], hyperparams)
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    history = train_model(
        model=model,
        data=datasets,  # Changed from datasets to data to match function signature
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, data=datasets, output_dir=args.output_dir)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, args.output_dir)
    
    # Print final results
    print("\nFinal Results:")
    if 'r2' in metrics:
        print(f"Test R²: {metrics['r2']:.4f}")
    if 'rmse' in metrics:
        print(f"Test RMSE: {metrics['rmse']*100:.4f} in")
    print(f"Test MSE: {metrics['loss']*100:.4f}")
    print(f"Test MAE: {metrics['mae']*100:.4f} in")
    
    # Calculate and print training time
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    print(f"\nTraining Time: {minutes} minutes and {seconds} seconds")
    print(f"Model saved to: {args.output_dir}")
    
    # Save a summary of the run
    with open(os.path.join(args.output_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"===============\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"ALL RAINFALL VALUES ARE IN INCHES.\n\n")
        
        f.write(f"Data:\n")
        f.write(f"  Training samples: {len(data['targets']['train'])}\n")
        f.write(f"  Validation samples: {len(data['targets']['val'])}\n")
        f.write(f"  Test samples: {len(data['targets']['test'])}\n\n")
        
        f.write(f"Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write(f"Training:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Training time: {minutes} minutes and {seconds} seconds\n\n")
        
        f.write(f"Results (all in inches):\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
    
    print(f"\nTraining summary saved to {os.path.join(args.output_dir, 'training_summary.txt')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
