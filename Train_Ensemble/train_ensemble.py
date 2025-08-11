#!/usr/bin/env python3
"""
Simple ensemble model for rainfall prediction.

This script trains multiple models with the same architecture but different random seeds,
then combines their predictions to create an ensemble.
"""

import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Assume working directory is project root (AS_rainfall). No dynamic path building.

# Import data utilities and training functions (package imports)
from Train_Best_Model.data_utils import create_tf_dataset
from Train_Best_Model.training import train_model, evaluate_model, plot_training_history
from Hyperparameter_Tuning.npz_data_utils import load_assembled_npz_data

# Model utilities
from Train_Best_Model.model_utils import load_best_hyperparameters, build_model

# Plotting and file I/O helpers
from Train_Ensemble.utils import (
    plot_ensemble_test_predictions,
    plot_individual_vs_ensemble,
    plot_fold_ensemble_predictions,
    save_progress,
    write_training_summary,
    write_fold_summary,
    write_test_predictions_csv,
    write_ensemble_summary,
)

def train_ensemble(data, hyperparams, output_dir, hp_dir, n_folds=5, n_models_per_fold=5, epochs=100, resume_training=True, start_fold=None, start_model=None):
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
    hp_dir : str
        Directory where best hyperparameters are stored (used by build_model)
    n_folds : int, optional
        Number of folds for cross-validation
    n_models_per_fold : int, optional
        Number of models per fold
    epochs : int, optional
        Number of training epochs
    resume_training : bool, optional
        Whether to resume training from existing progress
    start_fold : int, optional
        Fold to start training from
    start_model : int, optional
        Model to start training from
    Returns
    -------
    dict
        Dictionary containing results
    """
    # Resolve batch size strictly from best hyperparameters (no defaults)
    if 'batch_size' in hyperparams and hyperparams['batch_size'] is not None:
        batch_size = int(hyperparams['batch_size'])
    else:
        raise ValueError("batch_size must be present in best hyperparameters; no default is allowed.")

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
        
        # Initialize fold models and predictions
        fold_models = []
        fold_test_predictions = []
        
        # Define seeds dynamically based on requested models per fold
        random_seeds = [42 + i for i in range(n_models_per_fold)]
        
        # Initialize fold data if not already in completed_models
        if fold_key not in completed_models:
            completed_models[fold_key] = []
            
        # Create fold-specific datasets exactly once
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

        # Determine unit handling: prefer millimeters if NPZ std is present
        rainfall_std = float(fold_data.get('metadata', {}).get('rainfall_mm_std', 0.0))
        use_mm = rainfall_std and rainfall_std > 0.0
        
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
            
            # Build model; it will load hyperparameters from hp_dir
            model = build_model(data['metadata'], hp_dir)
        
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
            if use_mm:
                plot_training_history(history, output_dir=model_dir, rainfall_std=rainfall_std)
            else:
                plot_training_history(history, output_dir=model_dir)
        
            # Evaluate model and generate the same files as train_best_model.py
            print(f"\nEvaluating model {model_idx+1} of fold {fold_idx+1}...")
            metrics = evaluate_model(model, data=fold_datasets, output_dir=model_dir, rainfall_std=rainfall_std if use_mm else None)
        
            # Create training summary via helper
            summary_path = write_training_summary(
                model_dir=model_dir,
                model_idx=model_idx+1,
                fold_idx=fold_idx+1,
                random_seed=random_seed,
                hyperparams=hyperparams,
                history=history,
                metrics=metrics,
                use_mm=use_mm,
                rainfall_std=rainfall_std if use_mm else None,
            )
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
            save_progress(progress_file, completed_models)
        
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
        
        # Save fold ensemble results via helper
        write_fold_summary(
            fold_dir=fold_dir,
            fold_idx=fold_idx+1,
            n_models_per_fold=n_models_per_fold,
            fold_r2=fold_r2,
            fold_rmse=fold_rmse,
            fold_mae=fold_mae,
            use_mm=use_mm,
            rainfall_std=rainfall_std if use_mm else None,
        )
        
        # Create fold ensemble predictions plot via helper
        plot_fold_ensemble_predictions(data, fold_ensemble_pred, fold_dir, fold_idx)
    
    # Calculate average CV metrics (training units)
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
    
    # Save test predictions to CSV via helper (units depend on metadata)
    write_test_predictions_csv(output_dir, data, test_ensemble_pred)
    
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
    
    # Save results summary via helper
    write_ensemble_summary(
        output_dir=output_dir,
        n_folds=n_folds,
        n_models_per_fold=n_models_per_fold,
        hyperparams=hyperparams,
        fold_results=fold_results,
        avg_r2=avg_r2,
        avg_rmse=avg_rmse,
        avg_mae=avg_mae,
        test_r2=test_r2,
        test_rmse=test_rmse,
        test_mae=test_mae,
        data=data,
        training_time_sec=training_time,
    )
    
    # Plot results
    plot_ensemble_test_predictions(data, test_ensemble_pred, output_dir)
    plot_individual_vs_ensemble(data, all_test_predictions, test_ensemble_pred, output_dir)
    
    return results


def run_ensemble_cv(
    output_dir: str | None = None,
    n_folds: int = 15,
    n_models_per_fold: int = 5,
    epochs: int = 150,
    test_indices_path: str | None = None,
    hp_dir: str | None = None,
    npz_path: str | None = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz'),
):
    """Run K-fold CV simple ensemble on NPZ data; callable (no CLI).

    Parameters map 1:1 to the previous defaults dict for readability.
    Batch size is read from best hyperparameters.
    """
    # Resolve defaults if not provided (project-root-relative)
    if output_dir is None:
        output_dir = os.path.join('Train_Ensemble', 'output', 'simple_ensemble_cv')
    if test_indices_path is None:
        test_indices_path = os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl')
    if hp_dir is None:
        hp_dir = os.path.join('Hyperparameter_Tuning', 'output')

    os.makedirs(output_dir, exist_ok=True)

    # Simple completion check: if a final summary exists, assume training done
    summary_path = os.path.join(output_dir, 'ensemble_summary.txt')
    if os.path.exists(summary_path):
        print(f"Detected existing completed run at {output_dir}. Skipping training and metrics printing.")
        return {
            'status': 'already_completed',
            'output_dir': output_dir,
        }

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found at {npz_path}")
    data = load_assembled_npz_data(
        npz_path=npz_path,
        test_indices_path=test_indices_path,
        test_size=0.1,
        val_size=0.1,
        random_state=42,
    )

    # Load best hyperparameters for logging (source: hp_dir)
    hyperparams = load_best_hyperparameters(hp_dir)
    print(f"Loaded hyperparameters: {hyperparams}")

    # Train ensemble with CV
    print("\nTraining ensemble model with cross-validation...")
    results = train_ensemble(
        data=data,
        hyperparams=hyperparams,
        output_dir=output_dir,
        hp_dir=hp_dir,
        n_folds=n_folds,
        n_models_per_fold=n_models_per_fold,
        epochs=epochs,
    )

    # Print final results (units adapt inside file outputs; here we show mm if available)
    rs = float(data['metadata'].get('rainfall_mm_std', 0.0))
    print("\nCross-Validation Results:")
    print(f"Average CV R²: {results['avg_cv_r2']:.4f}")
    if rs > 0:
        print(f"Average CV RMSE: {(results['avg_cv_rmse']*rs):.4f} mm")
        print(f"Average CV MAE: {(results['avg_cv_mae']*rs):.4f} mm")
    else:
        print(f"Average CV RMSE: {results['avg_cv_rmse']*100:.4f} in")
        print(f"Average CV MAE: {results['avg_cv_mae']*100:.4f} in")

    print("\nFinal Ensemble Results:")
    print(f"Test R²: {results['test_r2']:.4f}")
    if rs > 0:
        print(f"Test RMSE: {(results['test_rmse']*rs):.4f} mm")
        print(f"Test MAE: {(results['test_mae']*rs):.4f} mm")
    else:
        print(f"Test RMSE: {results['test_rmse']*100:.4f} in")
        print(f"Test MAE: {results['test_mae']*100:.4f} in")

    print(f"\nResults saved to {output_dir}")
    return results


if __name__ == '__main__':
    run_ensemble_cv(
        output_dir=os.path.join('Train_Ensemble', 'output', 'simple_ensemble_cv'),
        test_indices_path=os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl'),
        hp_dir=os.path.join('Hyperparameter_Tuning', 'output'))
