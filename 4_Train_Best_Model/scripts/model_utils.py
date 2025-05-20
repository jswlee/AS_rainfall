import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))
HYPERPARAM_DIR = os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'output')

def load_best_hyperparameters():
    """
    Load the best hyperparameters from the hyperparameter tuning output directory.
    
    Returns
    -------
    dict
        Dictionary containing the best hyperparameters
    """
    # Define possible paths for hyperparameter files
    possible_paths = [
        # First check for CV tuning results
        os.path.join(HYPERPARAM_DIR, 'land_model_cv_tuning', 'current_best_hyperparameters.py'),
        # Then check for extended tuning results
        os.path.join(HYPERPARAM_DIR, 'land_model_extended_tuner', 'current_best_hyperparameters.py'),
        # Finally check for best_hyperparameters.py
        os.path.join(HYPERPARAM_DIR, 'land_model_extended_tuner', 'best_hyperparameters.py')
    ]
    
    # Try to load hyperparameters from one of the paths
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading best hyperparameters from {path}")
            # Load hyperparameters from Python file
            try:
                # Get the directory containing the hyperparameters file
                hp_dir = os.path.dirname(path)
                # Get the filename without extension
                hp_file = os.path.splitext(os.path.basename(path))[0]
                # Add the directory to Python path
                sys.path.append(hp_dir)
                # Import the hyperparameters module
                hp_module = __import__(hp_file)
                # Get the hyperparameters
                hyperparams = hp_module.best_hyperparameters
                # Remove the directory from Python path
                sys.path.remove(hp_dir)
                return hyperparams
            except Exception as e:
                print(f"Error loading hyperparameters from {path}: {e}")
    
    # If no hyperparameter file was found or loaded successfully, use default values
    print("Warning: Could not load hyperparameters from any file. Using default values.")
    return {
        'na': 320,
        'nb': 768,
        'dropout_rate': 0.1,
        'l2_reg': 1e-06,
        'learning_rate': 0.006138519107514284,
        'weight_decay': 4.665590142644678e-06,
        'local_dem_units': 224,
        'regional_dem_units': 32,
        'month_units': 64,
        'climate_units': 256,
        'use_residual': True,
        'activation': 'selu',
        'output_activation': 'softplus'  # Ensures non-negative rainfall predictions
    }


def build_model(data_metadata, hyperparams=None):
    """
    Build the LAND model with the given hyperparameters.
    
    Parameters
    ----------
    data_metadata : dict
        Dictionary containing metadata about the input data
    hyperparams : dict, optional
        Dictionary containing hyperparameters. If None, the best hyperparameters will be loaded.
        
    Returns
    -------
    tf.keras.Model
        Compiled LAND model
    """
    # Load the best hyperparameters if none are provided
    if hyperparams is None:
        hyperparams = load_best_hyperparameters()
    else:
        # Ensure all required hyperparameters are present by merging with defaults
        default_params = load_best_hyperparameters()
        hyperparams = {**default_params, **hyperparams}
    
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
