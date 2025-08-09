import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))
HPT_BASE = os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning')
HYPERPARAM_DIR = os.path.join(HPT_BASE, 'output')

def load_best_hyperparameters(base_output_dir: str):
    """
    Load the best hyperparameters from a single well-defined path within the
    provided output directory:
      <base_output_dir>/land_model_extended_tuner/land_model_cv_tuning/current_best_hyperparameters.py

    Parameters
    ----------
    base_output_dir : str
        Directory that contains 'land_model_extended_tuner/'. Defaults to
        '3_Hyperparameter_Tuning/output_test'.

    Returns
    -------
    dict
        Dictionary containing the best hyperparameters
    """
    path = os.path.join(
        base_output_dir,
        'land_model_extended_tuner',
        'land_model_cv_tuning',
        'current_best_hyperparameters.py',
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Hyperparameters file not found: {path}")
    print(f"Loading best hyperparameters from {path}")
    # Minimal, explicit load from a predefined path
    namespace = {}
    with open(path, 'r') as f:
        code = f.read()
    exec(code, namespace)
    if 'best_hyperparameters' not in namespace:
        raise ValueError(f"'best_hyperparameters' not defined in {path}")
    return namespace['best_hyperparameters']


def build_model(data_metadata, hp_dir: str = os.path.join(HPT_BASE, 'output_test'), hyperparams=None):
    """
    Build the LAND model with the given hyperparameters.
    
    Parameters
    ----------
    data_metadata : dict
        Dictionary containing metadata about the input data
    hp_dir : str, optional
        Directory containing the hyperparameters file. Defaults to '3_Hyperparameter_Tuning/output_test'.
    hyperparams : dict, optional
        Dictionary containing hyperparameters. If None, the best hyperparameters will be loaded.
        
    Returns
    -------
    tf.keras.Model
        Compiled LAND model
    """
    hyperparams = load_best_hyperparameters(hp_dir)
    if hyperparams is None:
        raise ValueError("Hyperparameters not found")
    
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
    
    # Print model summary (optional, can be commented out if not needed)
    model.summary()
    
    return model
