#!/usr/bin/env python3
"""
Ensemble model with cross-validation for the LAND-inspired rainfall prediction model.

This script implements:
1. K-fold cross-validation
2. Ensemble of multiple models
3. Extended hyperparameter tuning
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import pickle
from datetime import datetime
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add parent directory to path to import land_model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from land_model.data_utils import load_and_reshape_data, create_tf_dataset
from land_model.model import build_land_model
from land_model.training import cosine_decay_with_warmup


def build_ensemble_model(data_metadata, hyperparams, num_models=5):
    """
    Build an ensemble of LAND models.
    
    Parameters
    ----------
    data_metadata : dict
        Dictionary containing metadata about the input data
    hyperparams : dict
        Dictionary containing hyperparameters
    num_models : int, optional
        Number of models in the ensemble
        
    Returns
    -------
    list
        List of compiled LAND models
    """
    models = []
    
    for i in range(num_models):
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
        
        models.append(model)
    
    return models


def train_ensemble_with_cv(data, hyperparams, output_dir, n_folds=5, n_models=5, epochs=100, batch_size=314):
    """
    Train an ensemble of models using cross-validation.
    
    Parameters
    ----------
    data : dict
        Dictionary containing all data
    hyperparams : dict
        Dictionary containing hyperparameters
    output_dir : str
        Directory to save model weights and results
    n_folds : int, optional
        Number of cross-validation folds
    n_models : int, optional
        Number of models in each ensemble
    epochs : int, optional
        Number of training epochs
    batch_size : int, optional
        Batch size for training
        
    Returns
    -------
    dict
        Dictionary containing results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    X = {
        'climate': np.concatenate([data['climate']['train'], data['climate']['val']]),
        'local_dem': np.concatenate([data['local_dem']['train'], data['local_dem']['val']]),
        'regional_dem': np.concatenate([data['regional_dem']['train'], data['regional_dem']['val']]),
        'month': np.concatenate([data['month']['train'], data['month']['val']])
    }
    y = np.concatenate([data['targets']['train'], data['targets']['val']])
    
    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize results
    fold_results = []
    ensemble_models = []
    
    # Start timer
    start_time = time.time()
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(y)):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold+1}/{n_folds}")
        print(f"{'='*50}")
        
        # Create fold directory
        fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data
        X_train = {key: X[key][train_idx] for key in X}
        y_train = y[train_idx]
        X_val = {key: X[key][val_idx] for key in X}
        y_val = y[val_idx]
        
        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.batch(batch_size)
        if batch_size < len(y_train):
            train_ds = train_ds.shuffle(buffer_size=len(y_train))
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(batch_size)
        
        # Build ensemble models for this fold
        fold_models = build_ensemble_model(data['metadata'], hyperparams, num_models=n_models)
        fold_predictions = []
        
        # Train each model in the ensemble
        for model_idx, model in enumerate(fold_models):
            print(f"\nTraining Model {model_idx+1}/{n_models} for Fold {fold+1}")
            
            # Create model directory
            model_dir = os.path.join(fold_dir, f'model_{model_idx+1}')
            os.makedirs(model_dir, exist_ok=True)
            
            # Create callbacks
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: cosine_decay_with_warmup(epoch, total_epochs=epochs),
                verbose=1
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            )
            
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.weights.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            
            # Train model
            history = model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=[lr_scheduler, early_stopping, model_checkpoint],
                verbose=1
            )
            
            # Save training history
            with open(os.path.join(model_dir, 'history.pkl'), 'wb') as f:
                pickle.dump(history.history, f)
            
            # Plot training history
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.title('Training and Validation MAE')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'training_history.png'))
            plt.close()
            
            # Make predictions on validation set
            val_pred = model.predict(val_ds)
            fold_predictions.append(val_pred)
        
        # Ensemble predictions (average)
        ensemble_pred = np.mean(fold_predictions, axis=0)
        
        # Calculate metrics
        val_r2 = r2_score(y_val, ensemble_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        val_mae = mean_absolute_error(y_val, ensemble_pred)
        
        print(f"\nFold {fold+1} Ensemble Results:")
        print(f"Validation R²: {val_r2:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f} (scaled)")
        print(f"Validation MAE: {val_mae:.4f} (scaled)")
        
        # Convert back to in
        val_rmse_in = val_rmse * 100
        val_mae_in = val_mae * 100
        
        print(f"Validation RMSE: {val_rmse_in:.4f} in")
        print(f"Validation MAE: {val_mae_in:.4f} in")
        
        # Save fold results
        fold_result = {
            'fold': fold + 1,
            'r2': val_r2,
            'rmse': val_rmse_in,
            'mae': val_mae_in,
            'models': fold_models
        }
        fold_results.append(fold_result)
        
        # Save models for final ensemble
        ensemble_models.extend(fold_models)
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(y_val, ensemble_pred, alpha=0.5)
        plt.plot([0, max(y_val)], [0, max(y_val)], 'r--')
        plt.xlabel('Actual Rainfall (scaled)')
        plt.ylabel('Predicted Rainfall (scaled)')
        plt.title(f'Fold {fold+1}: Actual vs Predicted Rainfall\nR² = {val_r2:.4f}, RMSE = {val_rmse_in:.2f} in')
        plt.grid(True)
        plt.savefig(os.path.join(fold_dir, 'actual_vs_predicted.png'))
        plt.close()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {time.strftime('%H:%M:%S', time.gmtime(training_time))}")
    
    # Calculate average metrics across folds
    avg_r2 = np.mean([fold['r2'] for fold in fold_results])
    avg_rmse = np.mean([fold['rmse'] for fold in fold_results])
    avg_mae = np.mean([fold['mae'] for fold in fold_results])
    
    print("\nAverage Cross-Validation Results:")
    print(f"R²: {avg_r2:.4f}")
    print(f"RMSE: {avg_rmse:.4f} in")
    print(f"MAE: {avg_mae:.4f} in")
    
    # Evaluate on test set
    test_ds = tf.data.Dataset.from_tensor_slices(({
        'climate': data['climate']['test'],
        'local_dem': data['local_dem']['test'],
        'regional_dem': data['regional_dem']['test'],
        'month': data['month']['test']
    }, data['targets']['test']))
    test_ds = test_ds.batch(batch_size)
    
    # Make predictions with all models
    all_test_predictions = []
    for model in ensemble_models:
        test_pred = model.predict(test_ds)
        all_test_predictions.append(test_pred)
    
    # Ensemble predictions (average)
    ensemble_test_pred = np.mean(all_test_predictions, axis=0)
    
    # Calculate metrics
    test_r2 = r2_score(data['targets']['test'], ensemble_test_pred)
    test_rmse = np.sqrt(mean_squared_error(data['targets']['test'], ensemble_test_pred))
    test_mae = mean_absolute_error(data['targets']['test'], ensemble_test_pred)
    
    # Convert back to inches (the original data was scaled by 1/100 for model training)
    test_rmse_in = test_rmse * 100
    test_mae_in = test_mae * 100
    
    print("\nTest Set Results (Full Ensemble):")
    print(f"R²: {test_r2:.4f}")
    print(f"RMSE: {test_rmse_in:.4f} in")
    print(f"MAE: {test_mae_in:.4f} in")
    
    # Save test predictions and actual values for later visualization
    np.save(os.path.join(output_dir, 'test_predictions.npy'), ensemble_test_pred)
    np.save(os.path.join(output_dir, 'test_actual.npy'), data['targets']['test'])
    
    # Plot actual vs predicted for test set
    plt.figure(figsize=(10, 8))
    plt.scatter(data['targets']['test'], ensemble_test_pred, alpha=0.5)
    plt.plot([0, max(data['targets']['test'])], [0, max(data['targets']['test'])], 'r--')
    plt.xlabel('Actual Rainfall (scaled)')
    plt.ylabel('Predicted Rainfall (scaled)')
    plt.title(f'Test Set: Actual vs Predicted Rainfall\nR² = {test_r2:.4f}, RMSE = {test_rmse_in:.2f} in')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'test_actual_vs_predicted.png'))
    plt.close()
    
    # Save results
    results = {
        'fold_results': fold_results,
        'avg_cv_r2': avg_r2,
        'avg_cv_rmse': avg_rmse,
        'avg_cv_mae': avg_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,  # Store the raw RMSE (not scaled by 100)
        'test_mae': test_mae,    # Store the raw MAE (not scaled by 100)
        'test_rmse_in': test_rmse_in,  # Also store the inches version for convenience
        'test_mae_in': test_mae_in,    # Also store the inches version for convenience
        'hyperparams': hyperparams,
        'n_folds': n_folds,
        'n_models': n_models,
        'training_time': training_time,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'ensemble_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save a summary text file
    with open(os.path.join(output_dir, 'ensemble_summary.txt'), 'w') as f:
        f.write(f"Ensemble Model with {n_folds}-Fold Cross-Validation\n")
        f.write(f"Each fold contains {n_models} models\n\n")
        f.write(f"Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nCross-Validation Results:\n")
        for i, fold in enumerate(fold_results):
            f.write(f"  Fold {i+1}: R² = {fold['r2']:.4f}, RMSE = {fold['rmse']:.4f} in, MAE = {fold['mae']:.4f} in\n")
        f.write(f"\nAverage CV: R² = {avg_r2:.4f}, RMSE = {avg_rmse:.4f} in, MAE = {avg_mae:.4f} in\n")
        f.write(f"\nTest Set: R² = {test_r2:.4f}, RMSE = {test_rmse_in:.4f} in, MAE = {test_mae_in:.4f} in\n")
        f.write(f"\nTraining completed in {time.strftime('%H:%M:%S', time.gmtime(training_time))}\n")
    
    return results


def main():
    """
    Main function to run ensemble model with cross-validation.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ensemble model with cross-validation')
    parser.add_argument('--features_path', type=str, default='csv_data/features.csv',
                        help='Path to features CSV file')
    parser.add_argument('--targets_path', type=str, default='csv_data/targets.csv',
                        help='Path to targets CSV file')
    parser.add_argument('--test_indices_path', type=str, default='land_model_output/test_indices.pkl',
                        help='Path to save or load test indices')
    parser.add_argument('--hyperparams_path', type=str, default=None,
                        help='Path to hyperparameters file')
    parser.add_argument('--output_dir', type=str, default='land_model_ensemble',
                        help='Directory to save model weights and results')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--n_models', type=int, default=5,
                        help='Number of models in each ensemble')
    parser.add_argument('--epochs', type=int, default=100,
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
    
    # Default hyperparameters (will be used if no hyperparams_path is provided)
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
    
    # Load hyperparameters if path is provided
    if args.hyperparams_path:
        print(f"\nLoading hyperparameters from {args.hyperparams_path}...")
        loaded_hyperparams = {}
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
                    loaded_hyperparams[key.strip()] = value
        print(f"Loaded hyperparameters: {loaded_hyperparams}")
        
        # Update hyperparams with loaded values
        for key, value in loaded_hyperparams.items():
            if key in hyperparams or key not in ['Best hyperparameters from 100 trials']:
                hyperparams[key] = value
    
    # Train ensemble with cross-validation
    print("\nTraining ensemble model with cross-validation...")
    results = train_ensemble_with_cv(
        data=data,
        hyperparams=hyperparams,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        n_models=args.n_models,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Print final results
    print("\nFinal Results:")
    print(f"Average CV R²: {results['avg_cv_r2']:.4f}")
    print(f"Average CV RMSE: {results['avg_cv_rmse']:.4f} in")
    print(f"Average CV MAE: {results['avg_cv_mae']:.4f} in")
    print(f"Test R²: {results['test_r2']:.4f}")
    print(f"Test RMSE: {results['test_rmse_in']:.4f} in")
    print(f"Test MAE: {results['test_mae_in']:.4f} in")
    
    print(f"\nResults saved to {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
