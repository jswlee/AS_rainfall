#!/usr/bin/env python3
"""
Train the LAND-inspired rainfall prediction model with the best hyperparameters
found during hyperparameter tuning.
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Add parent directory to path to import land_model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from land_model.data_utils import load_and_reshape_data, create_tf_dataset
from land_model.training import train_model, evaluate_model, plot_training_history


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
    # Best hyperparameters from tuning
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
            'climate_units': 256
        }
    
    # Create input layers
    climate_input = layers.Input(shape=data_metadata['climate_shape'], name='climate')
    local_dem_input = layers.Input(shape=data_metadata['local_dem_shape'], name='local_dem')
    regional_dem_input = layers.Input(shape=data_metadata['regional_dem_shape'], name='regional_dem')
    month_input = layers.Input(shape=(data_metadata['num_month_encodings'],), name='month')
    
    # Process local DEM
    local_dem = layers.Flatten()(local_dem_input)
    local_dem = layers.Dense(
        hyperparams['local_dem_units'], 
        activation='relu',
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(local_dem)
    local_dem = layers.BatchNormalization()(local_dem)
    
    # Process regional DEM
    regional_dem = layers.Flatten()(regional_dem_input)
    regional_dem = layers.Dense(
        hyperparams['regional_dem_units'], 
        activation='relu',
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(regional_dem)
    regional_dem = layers.BatchNormalization()(regional_dem)
    
    # Process month
    month = layers.Dense(
        hyperparams['month_units'], 
        activation='relu',
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(month_input)
    month = layers.BatchNormalization()(month)
    
    # Process climate/reanalysis data
    climate_flat = layers.Reshape((data_metadata['climate_shape'][0] * 
                                  data_metadata['climate_shape'][1] * 
                                  data_metadata['climate_shape'][2],))(climate_input)
    
    climate = layers.Dense(
        hyperparams['climate_units'], 
        activation='relu',
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(climate_flat)
    climate = layers.BatchNormalization()(climate)
    
    # Concatenate all features
    concat = layers.Concatenate()([climate, local_dem, regional_dem, month])
    
    # Dense layers
    x = layers.Dense(
        hyperparams['na'], 
        activation='relu',
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(hyperparams['dropout_rate'])(x)
    
    x = layers.Dense(
        hyperparams['nb'], 
        activation='relu',
        kernel_regularizer=regularizers.l2(hyperparams['l2_reg'])
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(hyperparams['dropout_rate'])(x)
    
    # Output layer
    output = layers.Dense(1, name='rainfall')(x)
    
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
    
    return model


def main():
    """
    Main function to train the model with the best hyperparameters.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LAND model with best hyperparameters')
    parser.add_argument('--features_path', type=str, default='csv_data/features.csv',
                        help='Path to features CSV file')
    parser.add_argument('--targets_path', type=str, default='csv_data/targets.csv',
                        help='Path to targets CSV file')
    parser.add_argument('--test_indices_path', type=str, default='land_model_output/test_indices.pkl',
                        help='Path to save or load test indices')
    parser.add_argument('--hyperparams_path', type=str, default=None,
                        help='Path to hyperparameters file')
    parser.add_argument('--output_dir', type=str, default='land_model_best',
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
    model, history, training_time = train_model(
        model=model,
        datasets=datasets,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, datasets, args.output_dir)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Validation R²: {metrics['validation']['r2']:.4f}")
    print(f"Validation RMSE: {metrics['validation']['rmse']:.4f} mm")
    print(f"Validation MAE: {metrics['validation']['mae']:.4f} mm")
    print(f"Test R²: {metrics['test']['r2']:.4f}")
    print(f"Test RMSE: {metrics['test']['rmse']:.4f} mm")
    print(f"Test MAE: {metrics['test']['mae']:.4f} mm")
    
    print(f"\nResults saved to {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
