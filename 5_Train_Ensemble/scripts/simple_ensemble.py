#!/usr/bin/env python3
"""
Simple ensemble model for rainfall prediction.

This script trains multiple models with the same architecture but different random seeds,
then combines their predictions to create an ensemble.
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))

# Add required directories to Python path
sys.path.append(os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'scripts'))
# Add 4_Train_Best_Model/scripts to Python path to use the existing data_utils.py
sys.path.append(os.path.join(PROJECT_ROOT, '4_Train_Best_Model', 'scripts'))
# Import data utilities and training functions from 4_Train_Best_Model/scripts
from data_utils import load_and_reshape_data, create_tf_dataset
from training import train_model, evaluate_model, plot_training_history

# Path to the hyperparameter tuning output directory
HYPERPARAM_DIR = os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'output')


# Import load_best_hyperparameters from model_utils
sys.path.append(os.path.join(PROJECT_ROOT, '4_Train_Best_Model', 'scripts'))
from model_utils import load_best_hyperparameters


# Import build_model from model_utils
from model_utils import build_model


def train_simple_ensemble(data, hyperparams, output_dir, n_folds=5, n_models_per_fold=5, epochs=100, batch_size=32, resume_training=True, start_fold=None, start_model=None):
    """
    Train a simple ensemble of models with different random seeds.
    
    Parameters
    ----------
    data : dict
        Dictionary containing all data
    hyperparams : dict
        Dictionary containing hyperparameters
    output_dir : str
        Directory to save model weights and results
    n_models : int, optional
        Number of models in the ensemble
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
    
    # Initialize lists to store results
    all_models = []
    all_histories = []
    all_test_predictions = []
    fold_results = []
    
    # Check for existing progress file
    progress_file = os.path.join(output_dir, 'training_progress.pkl')
    completed_models = {}
    
    if resume_training and os.path.exists(progress_file):
        try:
            print(f"\nFound existing training progress. Attempting to resume...")
            with open(progress_file, 'rb') as f:
                saved_data = pickle.load(f)
                completed_models = saved_data.get('completed_models', {})
            print(f"Successfully loaded progress. Will skip already trained models.")
        except Exception as e:
            print(f"Error loading progress file: {e}")
            print("Starting training from scratch.")
            completed_models = {}
            
    # Combine train and validation data for cross-validation
    X = {
        'climate': np.concatenate([data['climate']['train'], data['climate']['val']]),
        'local_dem': np.concatenate([data['local_dem']['train'], data['local_dem']['val']]),
        'regional_dem': np.concatenate([data['regional_dem']['train'], data['regional_dem']['val']]),
        'month': np.concatenate([data['month']['train'], data['month']['val']])
    }
    y = np.concatenate([data['targets']['train'], data['targets']['val']])
    
    # Initialize KFold
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Start timer
    start_time = time.time()
    
    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(y)):
        # Skip folds before start_fold if specified
        if start_fold is not None and fold_idx + 1 < start_fold:
            print(f"\n{'='*50}")
            print(f"Skipping Fold {fold_idx+1}/{n_folds} (before start fold)")
            print(f"{'='*50}")
            continue
        fold_key = f"fold_{fold_idx+1}"
        fold_dir = os.path.join(output_dir, fold_key)
        os.makedirs(fold_dir, exist_ok=True)
        
        # Check if this fold is already completed
        if resume_training and fold_key in completed_models and len(completed_models[fold_key]) == n_models_per_fold:
            print(f"\n{'='*50}")
            print(f"Skipping Fold {fold_idx+1}/{n_folds} (already completed)")
            print(f"{'='*50}")
            
            # Try to load fold results if available
            fold_ensemble_file = os.path.join(fold_dir, 'fold_ensemble_predictions.npy')
            if os.path.exists(fold_ensemble_file):
                try:
                    fold_ensemble_pred = np.load(fold_ensemble_file)
                    fold_r2 = r2_score(data['targets']['test'], fold_ensemble_pred)
                    fold_rmse = np.sqrt(mean_squared_error(data['targets']['test'], fold_ensemble_pred))
                    fold_mae = mean_absolute_error(data['targets']['test'], fold_ensemble_pred)
                    
                    fold_result = {
                        'models': [],  # We don't need the actual models
                        'test_predictions': [],  # We don't need individual predictions
                        'ensemble_prediction': fold_ensemble_pred,
                        'r2': fold_r2,
                        'rmse': fold_rmse,
                        'mae': fold_mae
                    }
                    fold_results.append(fold_result)
                except Exception as e:
                    print(f"Error loading fold results: {e}")
            
            continue
            
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx+1}/{n_folds}")
        print(f"{'='*50}")
        
        # Create fold-specific datasets
        fold_data = {
            'climate': {
                'train': X['climate'][train_idx],
                'val': X['climate'][val_idx],
                'test': data['climate']['test']
            },
            'local_dem': {
                'train': X['local_dem'][train_idx],
                'val': X['local_dem'][val_idx],
                'test': data['local_dem']['test']
            },
            'regional_dem': {
                'train': X['regional_dem'][train_idx],
                'val': X['regional_dem'][val_idx],
                'test': data['regional_dem']['test']
            },
            'month': {
                'train': X['month'][train_idx],
                'val': X['month'][val_idx],
                'test': data['month']['test']
            },
            'targets': {
                'train': y[train_idx],
                'val': y[val_idx],
                'test': data['targets']['test']
            },
            'metadata': data['metadata']
        }
        
        # Create TensorFlow datasets without batching here
        # We'll let the train_model function handle the batching
        fold_datasets = {
            'train': {
                'climate': fold_data['climate']['train'],
                'local_dem': fold_data['local_dem']['train'],
                'regional_dem': fold_data['regional_dem']['train'],
                'month': fold_data['month']['train']
            },
            'val': {
                'climate': fold_data['climate']['val'],
                'local_dem': fold_data['local_dem']['val'],
                'regional_dem': fold_data['regional_dem']['val'],
                'month': fold_data['month']['val']
            },
            'test': {
                'climate': fold_data['climate']['test'],
                'local_dem': fold_data['local_dem']['test'],
                'regional_dem': fold_data['regional_dem']['test'],
                'month': fold_data['month']['test']
            },
            'targets': fold_data['targets'],
            'metadata': fold_data['metadata']
        }
        
        # Initialize fold models and predictions
        fold_models = []
        fold_test_predictions = []
        
        # Define the specific random seeds to use
        random_seeds = [42, 43, 44, 45, 46]
        
        # Initialize fold data if not already in completed_models
        if fold_key not in completed_models:
            completed_models[fold_key] = []
            
        # Create fold-specific datasets
        fold_data = {
            'climate': {
                'train': X['climate'][train_idx],
                'val': X['climate'][val_idx],
                'test': data['climate']['test']
            },
            'local_dem': {
                'train': X['local_dem'][train_idx],
                'val': X['local_dem'][val_idx],
                'test': data['local_dem']['test']
            },
            'regional_dem': {
                'train': X['regional_dem'][train_idx],
                'val': X['regional_dem'][val_idx],
                'test': data['regional_dem']['test']
            },
            'month': {
                'train': X['month'][train_idx],
                'val': X['month'][val_idx],
                'test': data['month']['test']
            },
            'targets': {
                'train': y[train_idx],
                'val': y[val_idx],
                'test': data['targets']['test']
            },
            'metadata': data['metadata']
        }
        
        # Create TensorFlow datasets for this fold
        fold_datasets = create_tf_dataset(fold_data, batch_size=batch_size)
        
        for model_idx in range(n_models_per_fold):
            # Skip models before start_model if in start_fold
            if start_fold is not None and start_model is not None and fold_idx + 1 == start_fold and model_idx + 1 < start_model:
                print(f"\nSkipping model {model_idx+1}/{n_models_per_fold} for fold {fold_idx+1}/{n_folds} (before start model)")
                continue
                
            model_key = f"model_{model_idx+1}"
            model_dir = os.path.join(fold_dir, model_key)
            os.makedirs(model_dir, exist_ok=True)
            
            # Check if this model already exists by looking for evaluation file
            eval_file = os.path.join(model_dir, 'evaluation_metrics.csv')
            if os.path.exists(eval_file):
                print(f"\nSkipping model {model_idx+1}/{n_models_per_fold} for fold {fold_idx+1}/{n_folds} (already trained)")
                continue
            print(f"\nTraining model {model_idx+1}/{n_models_per_fold} for fold {fold_idx+1}/{n_folds}...")
            
            # Model directory already created above
            
            # Set random seed from our predefined list
            random_seed = random_seeds[model_idx]
            print(f"Using random seed: {random_seed}")
            print(f"\nTraining model with seed {random_seed}...")
            random.seed(random_seed)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)
            
            # Build model with best hyperparameters
            model = build_model(data['metadata'], hyperparams)
        
            # Train model
            history = train_model(
                model=model,
                data=fold_datasets,
                output_dir=model_dir,
                epochs=epochs,
                batch_size=batch_size
            )
        
            # Save model
            model_path = os.path.join(model_dir, 'model.h5')
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Plot training history
            plot_training_history(history, output_dir=model_dir)
        
            # Evaluate model and generate the same files as train_best_model.py
            print(f"\nEvaluating model {model_idx+1} of fold {fold_idx+1}...")
            metrics = evaluate_model(model, data=fold_datasets, output_dir=model_dir)
        
            # Create training summary
            summary_path = os.path.join(model_dir, 'training_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Training Summary for Model {model_idx+1} of Fold {fold_idx+1}\n")
                f.write(f"Random Seed: {random_seed}\n\n")
                f.write("Hyperparameters:\n")
                for key, value in hyperparams.items():
                    if key not in ['Best hyperparameters from 100 trials']:
                        f.write(f"  {key}: {value}\n")
                f.write(f"\nTraining Results:\n")
                f.write(f"  Final Loss: {history['loss'][-1]*100*100:.6f} in²\n")  # Convert to inches squared
                f.write(f"  Final MAE: {history['mae'][-1]*100:.6f} in\n")  # Convert to inches
                f.write(f"  Final Val Loss: {history['val_loss'][-1]*100*100:.6f} in²\n")  # Convert to inches squared
                f.write(f"  Final Val MAE: {history['val_mae'][-1]*100:.6f} in\n\n")  # Convert to inches
                f.write(f"Test Metrics:\n")
                f.write(f"  R²: {metrics['r2']:.4f}\n")
                f.write(f"  RMSE: {metrics['rmse']*100:.4f} in\n")  # Convert to inches
                f.write(f"  MAE: {metrics['mae']*100:.4f} in\n")
            print(f"Training summary saved to {summary_path}")
        
            # Make test predictions
            test_pred = model.predict(fold_datasets['test'], verbose=0)
            fold_test_predictions.append(test_pred)
            
            # Store model and history
            fold_models.append(model)
            all_models.append(model)
            all_histories.append(history)
            
            # Mark this model as completed
            if fold_key not in completed_models:
                completed_models[fold_key] = []
            completed_models[fold_key].append(model_key)
            
            # Save progress after each model is trained
            progress_data = {
                'completed_models': completed_models
            }
            with open(progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
        
        # Calculate fold ensemble predictions
        fold_ensemble_pred = np.mean(fold_test_predictions, axis=0)
        
        # Calculate fold ensemble metrics
        fold_r2 = r2_score(data['targets']['test'], fold_ensemble_pred)
        fold_rmse = np.sqrt(mean_squared_error(data['targets']['test'], fold_ensemble_pred))
        fold_mae = mean_absolute_error(data['targets']['test'], fold_ensemble_pred)
        
        # Store fold results
        fold_result = {
            'models': fold_models,
            'test_predictions': fold_test_predictions,
            'ensemble_prediction': fold_ensemble_pred,
            'r2': fold_r2,
            'rmse': fold_rmse,
            'mae': fold_mae
        }
        fold_results.append(fold_result)
        
        # Save fold ensemble results
        fold_summary_path = os.path.join(fold_dir, 'fold_summary.txt')
        with open(fold_summary_path, 'w') as f:
            f.write(f"Fold {fold_idx+1} Ensemble Summary\n")
            f.write(f"Number of Models: {n_models_per_fold}\n\n")
            f.write(f"Test Metrics:\n")
            f.write(f"  R²: {fold_r2:.4f}\n")
            f.write(f"  RMSE: {fold_rmse*100:.4f} in\n")  # Convert to inches
            f.write(f"  MAE: {fold_mae*100:.4f} in\n")  # Convert to inches
        
        # Create fold ensemble predictions plot
        plt.figure(figsize=(10, 8))
        # Convert to inches by multiplying by 100
        plt.scatter(data['targets']['test']*100, fold_ensemble_pred*100, alpha=0.5)
        plt.plot([data['targets']['test'].min()*100, data['targets']['test'].max()*100], 
                 [data['targets']['test'].min()*100, data['targets']['test'].max()*100], 'r--')
        plt.xlabel('Actual Rainfall (inches)')
        plt.ylabel('Predicted Rainfall (inches)')
        plt.title(f'Fold {fold_idx+1} Ensemble: Actual vs Predicted Rainfall')
        plt.grid(True)
        plt.savefig(os.path.join(fold_dir, 'fold_ensemble_predictions.png'), dpi=300)
        plt.close()
    
    # Calculate average CV metrics
    avg_r2 = np.mean([fold['r2'] for fold in fold_results])
    avg_rmse = np.mean([fold['rmse'] for fold in fold_results])
    avg_mae = np.mean([fold['mae'] for fold in fold_results])
    
    # Ensemble all test predictions from all folds
    all_test_predictions = [pred for fold in fold_results for pred in fold['test_predictions']]
    test_ensemble_pred = np.mean(all_test_predictions, axis=0)
    
    # Calculate test metrics
    test_r2 = r2_score(data['targets']['test'], test_ensemble_pred)
    test_mse = mean_squared_error(data['targets']['test'], test_ensemble_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(data['targets']['test'], test_ensemble_pred)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save test predictions to CSV
    test_pred_df = pd.DataFrame({
        'actual': data['targets']['test'].flatten(),
        'predicted': test_ensemble_pred.flatten()
    })
    test_pred_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    # Prepare results
    results = {
        'fold_results': fold_results,
        'models': all_models,
        'histories': all_histories,
        'avg_cv_r2': avg_r2,
        'avg_cv_rmse': avg_rmse,
        'avg_cv_mae': avg_mae,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'training_time': training_time
    }
    
    # Save results summary
    summary_path = os.path.join(output_dir, 'ensemble_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"K-Fold CV Ensemble Model with {n_folds} Folds\n")
        f.write(f"Each fold contains {n_models_per_fold} models\n")
        f.write(f"Total models: {n_folds * n_models_per_fold}\n\n")
        
        f.write("Hyperparameters:\n")
        for key, value in hyperparams.items():
            if key not in ['Best hyperparameters from 100 trials']:
                f.write(f"  {key}: {value}\n")
        
        f.write("\nCross-Validation Results:\n")
        for i, fold in enumerate(fold_results):
            f.write(f"  Fold {i+1}: R² = {fold['r2']:.4f}, RMSE = {fold['rmse']:.4f} in, MAE = {fold['mae']:.4f} in\n")
        f.write(f"\nAverage CV: R² = {avg_r2:.4f}, RMSE = {avg_rmse:.4f} in, MAE = {avg_mae:.4f} in\n")
        
        f.write(f"\nFinal Ensemble Test Results:\n")
        f.write(f"  R²: {test_r2:.4f}\n")
        f.write(f"  RMSE: {test_rmse*100:.4f} in\n")  
        f.write(f"  MAE: {test_mae*100:.4f} in\n")  
        f.write(f"\nTraining completed in {time.strftime('%H:%M:%S', time.gmtime(training_time))}\n")
    
    # Plot test predictions
    plt.figure(figsize=(10, 8))
    plt.scatter(data['targets']['test']*100, test_ensemble_pred*100, alpha=0.5)
    plt.plot([data['targets']['test'].min()*100, data['targets']['test'].max()*100], 
             [data['targets']['test'].min()*100, data['targets']['test'].max()*100], 'r--')
    plt.xlabel('Actual Rainfall (inches)')
    plt.ylabel('Predicted Rainfall (inches)')
    plt.title('Ensemble Model: Actual vs Predicted Rainfall (Test Set)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ensemble_test_predictions.png'), dpi=300)
    plt.close()
    
    # Plot individual model predictions
    plt.figure(figsize=(12, 8))
    for i, preds in enumerate(all_test_predictions):
        plt.scatter(data['targets']['test']*100, preds*100, alpha=0.3, label=f'Model {i+1}')
    plt.scatter(data['targets']['test']*100, test_ensemble_pred*100, alpha=0.8, color='red', label='Ensemble')
    plt.plot([data['targets']['test'].min()*100, data['targets']['test'].max()*100], 
             [data['targets']['test'].min()*100, data['targets']['test'].max()*100], 'k--')
    plt.xlabel('Actual Rainfall (inches)')
    plt.ylabel('Predicted Rainfall (inches)')
    plt.title('Individual Models vs Ensemble Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'individual_vs_ensemble.png'), dpi=300)
    plt.close()
    
    return results


def main():
    """
    Main function to run simple ensemble model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple ensemble model for rainfall prediction')
    parser.add_argument('--features_path', type=str, 
                        default=os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'output', 'csv_data', 'features.csv'),
                        help='Path to features CSV file')
    parser.add_argument('--targets_path', type=str, 
                        default=os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'output', 'csv_data', 'targets.csv'),
                        help='Path to targets CSV file')
    parser.add_argument('--test_indices_path', type=str, 
                        default=os.path.join(PROJECT_ROOT, '4_Train_Best_Model', 'output', 'land_model_best', 'test_indices.pkl'),
                        help='Path to save or load test indices')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(PIPELINE_DIR, 'output', 'simple_ensemble'),
                        help='Directory to save model weights and results')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--n_models_per_fold', type=int, default=5,
                        help='Number of models to train in each fold')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--start_fold', type=int, default=None,
                        help='Fold number to start training from (1-5)')
    parser.add_argument('--start_model', type=int, default=None,
                        help='Model number to start training from within the start fold (1-5)')
    parser.add_argument('--hyperparams_path', type=str, default=None,
                        help='Path to hyperparameters file (will use best hyperparameters from tuning if not specified)')
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
    
    # Load hyperparameters
    if args.hyperparams_path:
        # Load from specified path
        try:
            print(f"Loading hyperparameters from {args.hyperparams_path}")
            # Get the directory containing the hyperparameters file
            hp_dir = os.path.dirname(args.hyperparams_path)
            # Get the filename without extension
            hp_file = os.path.splitext(os.path.basename(args.hyperparams_path))[0]
            # Add the directory to Python path
            sys.path.append(hp_dir)
            # Import the hyperparameters module
            hp_module = __import__(hp_file)
            # Get the hyperparameters
            hyperparams = hp_module.best_hyperparameters
            # Remove the directory from Python path
            sys.path.remove(hp_dir)
        except Exception as e:
            print(f"Error loading hyperparameters from {args.hyperparams_path}: {e}")
            print("Using best hyperparameters from tuning instead.")
            hyperparams = load_best_hyperparameters()
    else:
        # Load best hyperparameters from tuning
        hyperparams = load_best_hyperparameters()
        
    print(f"Loaded hyperparameters: {hyperparams}")
    
    # Train simple ensemble model with cross-validation
    print("\nTraining ensemble model with cross-validation...")
    results = train_simple_ensemble(
        data=data,
        hyperparams=hyperparams,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        n_models_per_fold=args.n_models_per_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        start_fold=args.start_fold,
        start_model=args.start_model
    )
    
    # Print final results
    print("\nCross-Validation Results:")
    print(f"Average CV R²: {results['avg_cv_r2']:.4f}")
    print(f"Average CV RMSE: {results['avg_cv_rmse']*100:.4f} in")  
    print(f"Average CV MAE: {results['avg_cv_mae']*100:.4f} in")  
    
    print("\nFinal Ensemble Results:")
    print(f"Test R²: {results['test_r2']:.4f}")
    print(f"Test RMSE: {results['test_rmse']*100:.4f} in")  
    print(f"Test MAE: {results['test_mae']*100:.4f} in")  
    
    print(f"\nResults saved to {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
