#!/usr/bin/env python3
"""
Hyperparameter tuning for the LAND-inspired rainfall prediction model.

Uses Keras Tuner to find the optimal hyperparameters for the model.
"""

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import keras_tuner as kt

# Add parent directory to path to import land_model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from land_model.data_utils import load_and_reshape_data, create_tf_dataset


def build_tunable_model(hp, data_metadata):
    """
    Build a tunable LAND model with hyperparameters to optimize.
    
    Parameters
    ----------
    hp : keras_tuner.HyperParameters
        Hyperparameters object
    data_metadata : dict
        Dictionary containing metadata about the input data
        
    Returns
    -------
    tf.keras.Model
        Compiled LAND model
    """
    # Hyperparameters to tune
    na = hp.Int('na', min_value=64, max_value=512, step=64)
    nb = hp.Int('nb', min_value=128, max_value=1024, step=128)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    weight_decay = hp.Float('weight_decay', min_value=1e-6, max_value=1e-3, sampling='log')
    
    # Create input layers
    climate_input = layers.Input(shape=data_metadata['climate_shape'], name='climate')
    local_dem_input = layers.Input(shape=data_metadata['local_dem_shape'], name='local_dem')
    regional_dem_input = layers.Input(shape=data_metadata['regional_dem_shape'], name='regional_dem')
    month_input = layers.Input(shape=(data_metadata['num_month_encodings'],), name='month')
    
    # Process local DEM
    local_dem = layers.Flatten()(local_dem_input)
    local_dem = layers.Dense(
        hp.Int('local_dem_units', min_value=32, max_value=128, step=32), 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(local_dem)
    local_dem = layers.BatchNormalization()(local_dem)
    
    # Process regional DEM
    regional_dem = layers.Flatten()(regional_dem_input)
    regional_dem = layers.Dense(
        hp.Int('regional_dem_units', min_value=32, max_value=128, step=32), 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(regional_dem)
    regional_dem = layers.BatchNormalization()(regional_dem)
    
    # Process month
    month = layers.Dense(
        hp.Int('month_units', min_value=16, max_value=64, step=16), 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(month_input)
    month = layers.BatchNormalization()(month)
    
    # Process climate/reanalysis data
    climate_flat = layers.Reshape((data_metadata['climate_shape'][0] * 
                                  data_metadata['climate_shape'][1] * 
                                  data_metadata['climate_shape'][2],))(climate_input)
    
    climate = layers.Dense(
        hp.Int('climate_units', min_value=64, max_value=256, step=64), 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(climate_flat)
    climate = layers.BatchNormalization()(climate)
    
    # Concatenate all features
    concat = layers.Concatenate()([climate, local_dem, regional_dem, month])
    
    # Dense layers
    x = layers.Dense(
        na, 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(
        nb, 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    output = layers.Dense(1, name='rainfall')(x)
    
    # Create model
    model = tf.keras.Model(
        inputs=[climate_input, local_dem_input, regional_dem_input, month_input],
        outputs=output
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def main():
    """
    Main function to run hyperparameter tuning.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for LAND model')
    parser.add_argument('--features_path', type=str, default='csv_data/features.csv',
                        help='Path to features CSV file')
    parser.add_argument('--targets_path', type=str, default='csv_data/targets.csv',
                        help='Path to targets CSV file')
    parser.add_argument('--test_indices_path', type=str, default='land_model_output/test_indices.pkl',
                        help='Path to save or load test indices')
    parser.add_argument('--output_dir', type=str, default='land_model_tuner',
                        help='Directory to save tuner results')
    parser.add_argument('--max_trials', type=int, default=30,
                        help='Maximum number of hyperparameter tuning trials')
    parser.add_argument('--executions_per_trial', type=int, default=1,
                        help='Number of executions per trial')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for each trial')
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
    
    # Define the hypermodel
    def build_model(hp):
        return build_tunable_model(hp, data['metadata'])
    
    # Create the tuner
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=args.max_trials,
        executions_per_trial=args.executions_per_trial,
        directory=args.output_dir,
        project_name='land_model_tuning'
    )
    
    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Start tuning
    print("\nStarting hyperparameter tuning...")
    start_time = time.time()
    
    tuner.search(
        datasets['train'],
        validation_data=datasets['val'],
        epochs=args.epochs,
        callbacks=[early_stopping]
    )
    
    # Calculate tuning time
    tuning_time = time.time() - start_time
    print(f"\nTuning completed in {time.strftime('%H:%M:%S', time.gmtime(tuning_time))}")
    
    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("\nBest hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")
    
    # Build the model with the best hyperparameters
    best_model = tuner.hypermodel.build(best_hp)
    
    # Save the best hyperparameters
    with open(os.path.join(args.output_dir, 'best_hyperparameters.txt'), 'w') as f:
        for param, value in best_hp.values.items():
            f.write(f"{param}: {value}\n")
    
    print(f"\nBest hyperparameters saved to {os.path.join(args.output_dir, 'best_hyperparameters.txt')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
