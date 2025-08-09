#!/usr/bin/env python3
"""
Extended hyperparameter tuning for the LAND-inspired rainfall prediction model.

This refactored script is purpose-built to tune using the assembled NPZ at:
`ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz`.

All orchestration (data loading, CV, tuner creation, callbacks) is handled in
`tuning_core.py`. This module only defines the tunable model and a minimal
`main()` that invokes tuning with a fixed configuration.
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tuning_core import run_tuning

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))

# No external data loaders are needed here; `tuning_core` handles NPZ loading.


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
    l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log')
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    weight_decay = hp.Float('weight_decay', min_value=1e-7, max_value=1e-3, sampling='log')
    
    # Architecture-specific hyperparameters
    local_dem_units = hp.Int('local_dem_units', min_value=32, max_value=256, step=32)
    regional_dem_units = hp.Int('regional_dem_units', min_value=32, max_value=256, step=32)
    month_units = hp.Int('month_units', min_value=16, max_value=128, step=16)
    climate_units = hp.Int('climate_units', min_value=64, max_value=512, step=64)
    
    # Advanced hyperparameters
    use_residual = hp.Boolean('use_residual')
    activation = hp.Choice('activation', values=['relu', 'elu', 'selu'])
    
    # Output layer activation (to ensure non-negative predictions for rainfall)
    output_activation = hp.Choice('output_activation', values=['relu', 'softplus'])
    
    # Create input layers
    climate_input = layers.Input(shape=data_metadata['climate_shape'], name='climate')
    local_dem_input = layers.Input(shape=data_metadata['local_dem_shape'], name='local_dem')
    regional_dem_input = layers.Input(shape=data_metadata['regional_dem_shape'], name='regional_dem')
    month_input = layers.Input(shape=(data_metadata['num_month_encodings'],), name='month')
    
    # Process local DEM
    local_dem = layers.Flatten()(local_dem_input)
    local_dem = layers.Dense(
        local_dem_units, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(local_dem)
    local_dem = layers.BatchNormalization()(local_dem)
    
    # Process regional DEM
    regional_dem = layers.Flatten()(regional_dem_input)
    regional_dem = layers.Dense(
        regional_dem_units, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(regional_dem)
    regional_dem = layers.BatchNormalization()(regional_dem)
    
    # Process month
    month = layers.Dense(
        month_units, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(month_input)
    month = layers.BatchNormalization()(month)
    
    # Process climate/reanalysis data
    climate_flat = layers.Reshape((data_metadata['climate_shape'][0] * 
                                  data_metadata['climate_shape'][1] * 
                                  data_metadata['climate_shape'][2],))(climate_input)
    
    climate = layers.Dense(
        climate_units, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(climate_flat)
    climate = layers.BatchNormalization()(climate)
    
    # Concatenate all features
    concat = layers.Concatenate()([climate, local_dem, regional_dem, month])
    
    # Dense layers
    x = layers.Dense(
        na, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Optional residual connection
    if use_residual and na == nb:
        residual = x
        
    x = layers.Dense(
        nb, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Add residual connection if enabled and dimensions match
    if use_residual and na == nb:
        x = layers.Add()([x, residual])
        
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer with non-negative activation to ensure rainfall predictions are never negative
    output = layers.Dense(1, activation=output_activation, name='rainfall')(x)
    
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


def default_config():
    """Fixed configuration for tuning with the assembled NPZ file."""
    return {
        'project_root': PROJECT_ROOT,
        'npz_path': os.path.join(PROJECT_ROOT, 'ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz'),
        'test_indices_path': os.path.join(SCRIPT_DIR, '../output_test/test_indices.pkl'),
        'output_dir': os.path.join(SCRIPT_DIR, '../output_test/land_model_extended_tuner'),
        'max_trials': 100,
        'executions_per_trial': 1,
        'epochs': 50,
        'batch_size': 314,
        'test_size': 0.1,
        'val_size': 0.1,
        'n_folds': 5,
        'cv_seed': 42,
        'resume': True,
    }


def main():
    """Run tuning with fixed config and the model defined here."""
    config = default_config()
    run_tuning(config=config, build_model_fn=build_tunable_model)
    return 0


if __name__ == '__main__':
    sys.exit(main())
