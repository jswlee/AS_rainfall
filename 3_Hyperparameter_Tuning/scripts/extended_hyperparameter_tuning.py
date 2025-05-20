#!/usr/bin/env python3
"""
Extended hyperparameter tuning for the LAND-inspired rainfall prediction model.

Uses Keras Tuner with 100 trials to find the optimal hyperparameters.
"""

import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import keras_tuner as kt
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))

# Add required directories to Python path
sys.path.append(os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'scripts'))
# Add 4_Train_Best_Model/scripts to Python path to use the data_utils.py
sys.path.append(os.path.join(PROJECT_ROOT, '4_Train_Best_Model', 'scripts'))
from data_utils import load_and_reshape_data, create_tf_dataset


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


def main():
    """
    Main function to run extended hyperparameter tuning with cross-validation.
    Supports resuming from previous runs.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extended hyperparameter tuning for LAND model with cross-validation')
    parser.add_argument('--features_path', type=str, 
                        default=os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'output', 'csv_data', 'features.csv'),
                        help='Path to features CSV file')
    parser.add_argument('--targets_path', type=str, 
                        default=os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'output', 'csv_data', 'targets.csv'),
                        help='Path to targets CSV file')
    parser.add_argument('--test_indices_path', type=str, 
                        default=os.path.join(SCRIPT_DIR, '../output/test_indices.pkl'),
                        help='Path to save or load test indices')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(SCRIPT_DIR, '../output/land_model_extended_tuner'),
                        help='Directory to save tuner results')
    parser.add_argument('--max_trials', type=int, default=50,
                        help='Maximum number of hyperparameter tuning trials')
    parser.add_argument('--executions_per_trial', type=int, default=1,
                        help='Number of executions per trial')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for each trial')
    parser.add_argument('--batch_size', type=int, default=314,
                        help='Batch size for training')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--cv_seed', type=int, default=42,
                        help='Random seed for cross-validation splits')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous tuning session')
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
    
    # Extract the training data for cross-validation
    # We'll use the existing train/val split for our CV folds
    print(f"\nSetting up {args.n_folds}-fold cross-validation...")
    
    # Combine train and validation data for cross-validation
    cv_features = {}
    cv_targets = {}
    
    # Combine climate data
    cv_features['climate'] = np.concatenate([
        data['climate']['train'], 
        data['climate']['val']
    ])
    
    # Combine local DEM data
    cv_features['local_dem'] = np.concatenate([
        data['local_dem']['train'], 
        data['local_dem']['val']
    ])
    
    # Combine regional DEM data
    cv_features['regional_dem'] = np.concatenate([
        data['regional_dem']['train'], 
        data['regional_dem']['val']
    ])
    
    # Combine month data
    cv_features['month'] = np.concatenate([
        data['month']['train'], 
        data['month']['val']
    ])
    
    # Combine targets
    cv_targets = np.concatenate([
        data['targets']['train'], 
        data['targets']['val']
    ])
    
    # Create indices for cross-validation
    n_samples = len(cv_targets)
    indices = np.arange(n_samples)
    
    # Set up KFold cross-validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.cv_seed)
    
    # Define a cross-validation hypermodel
    class CVHyperModel(kt.HyperModel):
        def __init__(self, data_metadata, cv_features, cv_targets, kf, batch_size):
            self.data_metadata = data_metadata
            self.cv_features = cv_features
            self.cv_targets = cv_targets
            self.kf = kf
            self.batch_size = batch_size
            self.fold_indices = list(kf.split(np.arange(len(cv_targets))))
            
        def build(self, hp):
            return build_tunable_model(hp, self.data_metadata)
            
        def fit(self, hp, model, *args, **kwargs):
            # Get callbacks
            callbacks = kwargs.pop('callbacks', [])
            
            # Store validation scores across folds
            val_losses = []
            
            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(self.fold_indices):
                print(f"\nTraining on fold {fold+1}/{len(self.fold_indices)}")
                
                # Create fold-specific datasets
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    {
                        'climate': self.cv_features['climate'][train_idx],
                        'local_dem': self.cv_features['local_dem'][train_idx],
                        'regional_dem': self.cv_features['regional_dem'][train_idx],
                        'month': self.cv_features['month'][train_idx]
                    },
                    self.cv_targets[train_idx]
                )).batch(self.batch_size, drop_remainder=True)
                
                val_dataset = tf.data.Dataset.from_tensor_slices((
                    {
                        'climate': self.cv_features['climate'][val_idx],
                        'local_dem': self.cv_features['local_dem'][val_idx],
                        'regional_dem': self.cv_features['regional_dem'][val_idx],
                        'month': self.cv_features['month'][val_idx]
                    },
                    self.cv_targets[val_idx]
                )).batch(self.batch_size, drop_remainder=True)
                
                # Reset model weights for each fold by creating a new model with the same hyperparameters
                # This is more reliable than trying to reset weights in-place
                if fold > 0:  # Only rebuild after the first fold
                    # Clear the previous model from memory
                    tf.keras.backend.clear_session()
                    # Rebuild the model with the same hyperparameters
                    model = self.build(hp)
                
                # Train on this fold
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    **kwargs
                )
                
                # Get best validation loss from this fold
                best_val_loss = min(history.history['val_loss'])
                val_losses.append(best_val_loss)
                print(f"Fold {fold+1} best validation loss: {best_val_loss:.6f}")
            
            # Return the average validation loss across folds
            avg_val_loss = np.mean(val_losses)
            print(f"\nAverage validation loss across {len(self.fold_indices)} folds: {avg_val_loss:.6f}")
            
            # Return the last history object with the average validation loss
            history.history['val_loss'][-1] = avg_val_loss
            return history
    
    # Create the CV hypermodel
    cv_hypermodel = CVHyperModel(
        data_metadata=data['metadata'],
        cv_features=cv_features,
        cv_targets=cv_targets,
        kf=kf,
        batch_size=args.batch_size
    )
    
    # Create the tuner with the CV hypermodel
    tuner = kt.BayesianOptimization(
        cv_hypermodel,
        objective='val_loss',
        max_trials=args.max_trials,
        executions_per_trial=args.executions_per_trial,
        directory=args.output_dir,
        project_name='land_model_cv_tuning',
        overwrite=not args.resume  # Only overwrite if not resuming
    )
    
    # If resuming, load the existing trials
    if args.resume:
        print("\nResuming from previous tuning session...")
        # Get the number of completed trials
        completed_trials = len(tuner.oracle.trials)
        if completed_trials > 0:
            print(f"Found {completed_trials} completed trials")
            # Get the best trial so far
            try:
                best_trial = tuner.oracle.get_best_trials(1)[0]
                print(f"Best val_loss so far: {best_trial.score:.6f}")
                print("Best hyperparameters so far:")
                for param, value in best_trial.hyperparameters.values.items():
                    print(f"  {param}: {value}")
            except (IndexError, AttributeError) as e:
                print(f"Could not retrieve best trial information: {e}")
        else:
            print("No completed trials found. Starting from scratch.")
            # Set overwrite to True if no trials found
            tuner = kt.BayesianOptimization(
                cv_hypermodel,
                objective='val_loss',
                max_trials=args.max_trials,
                executions_per_trial=args.executions_per_trial,
                directory=args.output_dir,
                project_name='land_model_cv_tuning',
                overwrite=True
            )
    
    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Define learning rate scheduler
    def cosine_decay_with_warmup(epoch, total_epochs=args.epochs, warmup_epochs=5, 
                                initial_lr=0.001, min_lr=1e-6):
        """Cosine decay learning rate schedule with warmup."""
        import math
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, total_epochs=args.epochs),
        verbose=1
    )
    
    # Create a simpler approach using a custom callback that runs after each epoch
    class SaveBestHyperparametersCallback(tf.keras.callbacks.Callback):
        def __init__(self, tuner, output_dir):
            super(SaveBestHyperparametersCallback, self).__init__()
            self.tuner = tuner
            self.output_dir = output_dir
            self.best_val_loss = float('inf')
            self.trial_count = 0
            
        def on_train_begin(self, logs=None):
            # Increment trial counter when a new trial begins
            self.trial_count += 1
            print(f"\nStarting trial #{self.trial_count}")
            
        def on_epoch_end(self, epoch, logs=None):
            # Check if this is the last epoch (early stopping or max epochs)
            if epoch == self.params['epochs'] - 1 or logs.get('val_loss', 0) < self.best_val_loss:
                self.save_current_best()
                
        def save_current_best(self):
            # Get the best hyperparameters so far
            try:
                best_hp = self.tuner.get_best_hyperparameters(1)[0]
                best_trial = self.tuner.oracle.get_best_trials(1)[0]
                val_loss = best_trial.score
                
                # Only save if this is better than what we've seen before
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    
                    # Save the best hyperparameters to a text file
                    with open(os.path.join(self.output_dir, 'current_best_hyperparameters.txt'), 'w') as f:
                        f.write(f"Best hyperparameters after {self.trial_count} trials (val_loss: {val_loss:.6f}):\n\n")
                        for param, value in best_hp.values.items():
                            f.write(f"{param}: {value}\n")
                    
                    # Save as Python dictionary for easy import
                    with open(os.path.join(self.output_dir, 'current_best_hyperparameters.py'), 'w') as f:
                        f.write("# Current best hyperparameters\n\n")
                        f.write("best_hyperparameters = {\n")
                        for param, value in best_hp.values.items():
                            if isinstance(value, str):
                                f.write(f"    '{param}': '{value}',\n")
                            else:
                                f.write(f"    '{param}': {value},\n")
                        f.write("}\n")
                    
                    print(f"\n[SaveBestHyperparameters] Updated best hyperparameters (val_loss: {val_loss:.6f})")
            except Exception as e:
                print(f"Error saving best hyperparameters: {e}")
    
    # Create the callback
    save_best_hp_callback = SaveBestHyperparametersCallback(tuner, args.output_dir)
    
    # Start tuning
    print("\nStarting extended hyperparameter tuning with cross-validation...")
    if args.resume:
        remaining_trials = args.max_trials - len(tuner.oracle.trials)
        print(f"Resuming with {remaining_trials} remaining trials of {args.max_trials} total")
    else:
        print(f"Running {args.max_trials} trials with {args.n_folds}-fold cross-validation")
    print(f"Each fold will train for up to {args.epochs} epochs")
    print("Best hyperparameters will be saved after each trial in 'current_best_hyperparameters.txt'")
    start_time = time.time()
    
    # For CV, we don't pass datasets directly since the CV hypermodel handles that
    tuner.search(
        epochs=args.epochs,
        callbacks=[early_stopping, lr_scheduler, save_best_hp_callback]
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
        f.write(f"Best hyperparameters from {args.max_trials} trials:\n\n")
        for param, value in best_hp.values.items():
            f.write(f"{param}: {value}\n")
    
    # Save as Python dictionary for easy import
    with open(os.path.join(args.output_dir, 'best_hyperparameters.pkl'), 'wb') as f:
        pickle.dump(best_hp.values, f)
    
    # Generate Python code for the best hyperparameters
    with open(os.path.join(args.output_dir, 'best_hyperparameters.py'), 'w') as f:
        f.write("# Best hyperparameters from extended tuning\n\n")
        f.write("best_hyperparameters = {\n")
        for param, value in best_hp.values.items():
            if isinstance(value, str):
                f.write(f"    '{param}': '{value}',\n")
            else:
                f.write(f"    '{param}': {value},\n")
        f.write("}\n")
    
    print(f"\nBest hyperparameters saved to {os.path.join(args.output_dir, 'best_hyperparameters.txt')}")
    
    # Plot hyperparameter importance
    try:
        # Check if we have enough trials for meaningful importance calculation
        # Keras Tuner needs at least 10 trials for reliable importance calculation
        if len(tuner.oracle.trials) < 10:
            print(f"Not enough trials ({len(tuner.oracle.trials)}) for reliable hyperparameter importance calculation. Need at least 10.")
        else:
            # Get hyperparameter importance
            try:
                importances = tuner.results_summary.get_importance()
                if not importances:
                    print("No hyperparameter importance data available.")
                else:
                    plt.figure(figsize=(12, 8))
                    params = list(importances.keys())
                    values = list(importances.values())
                    plt.barh(params, values)
                    plt.xlabel('Importance')
                    plt.ylabel('Hyperparameter')
                    plt.title('Hyperparameter Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'hyperparameter_importance.png'))
                    plt.close()
                    print(f"Hyperparameter importance plot saved to {os.path.join(args.output_dir, 'hyperparameter_importance.png')}")
            except AttributeError:
                print("Error: results_summary.get_importance() method not available in this version of Keras Tuner.")
                print("Try upgrading keras-tuner to the latest version.")
    except Exception as e:
        print(f"Could not generate hyperparameter importance plot: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
