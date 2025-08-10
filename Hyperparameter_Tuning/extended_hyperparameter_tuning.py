#!/usr/bin/env python3
"""
Extended hyperparameter tuning for the LAND-inspired rainfall prediction model.

This module defines only:
  * the tunable model function `build_tunable_model(hp, data_metadata)`; and
  * a minimal `main()`.

No data loaders needed here; `tuning_core` handles NPZ loading.

Data loading, cross-validation, tuner setup, and callbacks are orchestrated by
`tuning_core.py` using the NPZ dataset at
`ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz`.
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from Hyperparameter_Tuning.tuning_core import run_tuning

# Assume working directory is project root (AS_rainfall). No dynamic path building.

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
    # Make batch size tunable; used by CV training loop in tuning_core
    batch_size = hp.Int('batch_size', min_value=64, max_value=256, step=32)
    
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


def config(
    project_root: str = '.',
    npz_path: str = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz'),
    test_indices_path: str = os.path.join('Hyperparameter_Tuning', 'output_2', 'test_indices.pkl'),
    output_dir: str = os.path.join('Hyperparameter_Tuning', 'output_2'),
    max_trials: int = 150,
    executions_per_trial: int = 1,
    epochs: int = 150,
    batch_size: int = 64,
    n_folds: int = 10,
    cv_seed: int = 42,
    resume: bool = True,
):
    """Create a tuning config with overridable defaults.

    Pass only the params you want to change, e.g.:
        cfg = config(max_trials=200, epochs=75)
    """
    return {
        'project_root': project_root,
        'npz_path': npz_path,
        'test_indices_path': test_indices_path,
        'output_dir': output_dir,
        'max_trials': max_trials,
        'executions_per_trial': executions_per_trial,
        'epochs': epochs,
        'batch_size': batch_size,
        'n_folds': n_folds,
        'cv_seed': cv_seed,
        'resume': resume,
    }


def main():
    """Run tuning with parameterized config and the model defined here."""
    cfg = config()
    run_tuning(config=cfg, build_model_fn=build_tunable_model)
    return 0


if __name__ == '__main__':
    sys.exit(main())
