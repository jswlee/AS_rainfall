#!/usr/bin/env python3
"""
PyTorch ensemble training for rainfall prediction using cross-validation.
Now supports:
- Passing through loss selection (e.g., 'weighted_mse') to optimize the intended criterion.
- Optional MLflow logging of per-epoch weighted/unweighted metrics, fold metrics, and final metrics.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import mlflow
import mlflow.pytorch
from datetime import datetime
import time
import tempfile

# Import MLflow utilities for robust experiment tracking
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Hyperparameter_Tuning'))
from mlflow_utils import create_mlflow_logger, MLFLOW_AVAILABLE

# Import PyTorch utilities
from Hyperparameter_Tuning.pytorch_data_utils import load_assembled_npz_data_pytorch, create_pytorch_dataloaders, RainfallDataset
from Hyperparameter_Tuning.pytorch_model import create_model_from_hyperparams
from Hyperparameter_Tuning.pytorch_training import train_model, evaluate_model
from Hyperparameter_Tuning.pytorch_hyperparameter_tuning import load_best_hyperparameters_pytorch


class EnsembleTrainer:
    """PyTorch ensemble trainer with cross-validation."""
    
    def __init__(self, 
                 npz_path: str,
                 hyperparams_dir: str,
                 output_dir: str,
                 test_indices_path: str = None,
                 n_folds: int = 5,
                 n_models_per_fold: int = 5,
                 random_state: int = 42,
                 # Loss selection
                 loss_name: str = 'mse',
                 loss_params: dict | None = None,
                 # MLflow logging
                 mlflow_enabled: bool = False,
                 mlflow_experiment: str | None = None,
                 mlflow_run_name: str | None = None):
        """
        Initialize ensemble trainer.
        
        Args:
            npz_path: Path to assembled NPZ data
            hyperparams_dir: Directory containing best hyperparameters
            output_dir: Output directory for ensemble results
            test_indices_path: Path to test indices
            n_folds: Number of CV folds
            n_models_per_fold: Number of models per fold
            random_state: Random seed
            loss_name: Loss function name ('mse' or 'weighted_mse')
            loss_params: Optional parameters for weighted loss
            mlflow_enabled: Enable MLflow logging
            mlflow_experiment: MLflow experiment name (optional)
            mlflow_run_name: MLflow run name (optional)
        """
        self.npz_path = npz_path
        self.hyperparams_dir = hyperparams_dir
        self.output_dir = output_dir
        self.test_indices_path = test_indices_path
        self.n_folds = n_folds
        self.n_models_per_fold = n_models_per_fold
        self.random_state = random_state
        self.loss_name = loss_name
        self.loss_params = loss_params or None
        self.mlflow_enabled = bool(mlflow_enabled) and MLFLOW_AVAILABLE
        self.mlflow_experiment = mlflow_experiment or "pytorch_ensemble_training"
        self.mlflow_run_name = mlflow_run_name
        
        # Initialize MLflow logger for robust experiment tracking
        # MLflow helps track ensemble training experiments with cross-validation metrics,
        # fold-level performance, and model artifacts for reproducibility
        self.mlflow_logger = None
        if self.mlflow_enabled:
            try:
                self.mlflow_logger = create_mlflow_logger(
                    experiment_name=self.mlflow_experiment,
                    tracking_uri="./mlruns"
                )
                print("MLflow logger initialized successfully for ensemble training")
            except Exception as e:
                print(f"Warning: MLflow logger initialization failed: {e}")
                self.mlflow_enabled = False
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        print(f"Loading data from {npz_path}...")
        self.data = load_assembled_npz_data_pytorch(
            npz_path=npz_path,
            test_indices_path=test_indices_path,
            random_state=random_state
        )
        
        # Load hyperparameters
        print(f"Loading hyperparameters from {hyperparams_dir}...")
        self.hyperparams = load_best_hyperparameters_pytorch(hyperparams_dir)
        
        # Combine train and val for CV
        train_dataset = self.data['datasets']['train']
        val_dataset = self.data['datasets']['val']
        
        self.cv_climate = torch.cat(tensors=[train_dataset.climate_data, val_dataset.climate_data], dim=0)
        self.cv_local_dem = torch.cat(tensors=[train_dataset.local_dem_data, val_dataset.local_dem_data], dim=0)
        self.cv_regional_dem = torch.cat(tensors=[train_dataset.regional_dem_data, val_dataset.regional_dem_data], dim=0)
        self.cv_month = torch.cat(tensors=[train_dataset.month_data, val_dataset.month_data], dim=0)
        self.cv_targets = torch.cat(tensors=[train_dataset.targets, val_dataset.targets], dim=0)
        
        print(f"CV data: {self.cv_targets.shape[0]} samples")
        print(f"Test data: {len(self.data['datasets']['test'])} samples")
        
        # Setup device - prefer CUDA if available, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: {self.device} (CUDA GPU)")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using device: {self.device} (Apple Silicon GPU)")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: {self.device} (CPU fallback)")
        print(f"Using device: {self.device}")
    
    def train_single_model(self, fold_idx: int, model_idx: int, 
                          train_dataset: RainfallDataset, val_dataset: RainfallDataset,
                          epochs: int = 150) -> tuple:
        """
        Train a single model.
        
        Args:
            fold_idx: Fold index
            model_idx: Model index within fold
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Maximum epochs
            
        Returns:
            Tuple of (model, history, test_predictions)
        """
        # Set random seed for reproducibility
        seed = self.random_state + fold_idx * 100 + model_idx
        torch.manual_seed(seed=seed)
        np.random.seed(seed=seed)
        
        # Create dataloaders
        batch_size = self.hyperparams.get('batch_size', 32)
        dataloaders = create_pytorch_dataloaders(
            {'train': train_dataset, 'val': val_dataset},
            batch_size=batch_size,
            num_workers=0
        )
        
        # Create model
        model = create_model_from_hyperparams(self.hyperparams, self.data['metadata'])
        
        # Train model
        history = train_model(
            model=model,
            dataloaders=dataloaders,
            epochs=epochs,
            learning_rate=self.hyperparams.get('learning_rate', 0.001),
            weight_decay=self.hyperparams.get('weight_decay', 0.001),
            patience=15,
            device=self.device,
            verbose=False,
            loss_name=self.loss_name,
            loss_params=self.loss_params
        )
        
        # Get test predictions
        test_dataloader = create_pytorch_dataloaders(
            {'test': self.data['datasets']['test']},
            batch_size=batch_size,
            num_workers=0
        )['test']
        
        model.eval()
        test_predictions = []
        with torch.no_grad():
            for features, _ in test_dataloader:
                features = {k: v.to(device=self.device) for k, v in features.items()}
                outputs = model(features)
                test_predictions.extend(outputs.cpu().numpy().flatten())
        
        return model, history, np.array(test_predictions)
    
    def train_ensemble(self, epochs: int = 150, resume: bool = True) -> dict:
        """
        Train ensemble with cross-validation.
        
        Args:
            epochs: Maximum epochs per model
            resume: Whether to resume from existing progress
            
        Returns:
            Dictionary with ensemble results
        """
        # Check for existing progress
        progress_file = os.path.join('./', 'ensemble_progress.pkl')
        completed_models = {}
        
        if resume and os.path.exists(progress_file):
            try:
                with open(progress_file, 'rb') as f:
                    completed_models = pickle.load(file=f)
                print(f"Resuming from existing progress: {len(completed_models)} folds")
            except Exception as e:
                print(f"Could not load progress: {e}")
                completed_models = {}
        
        # Setup cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        fold_results = []
        all_test_predictions = []
        
        start_time = time.time()
        
        # Initialize MLflow run for ensemble training with robust error handling
        # This tracks the entire ensemble training process including cross-validation
        # metrics, fold-level performance, and final ensemble results
        if self.mlflow_logger:
            try:
                # Start MLflow run for the entire ensemble training process
                run_name = self.mlflow_run_name or f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.mlflow_logger.start_run(run_name=run_name)
                
                # Log ensemble training configuration parameters
                # These parameters define the ensemble setup and training configuration
                ensemble_params = {
                    'n_folds': self.n_folds,
                    'n_models_per_fold': self.n_models_per_fold,
                    'max_epochs': epochs,
                    'device': str(self.device),
                    'loss_function': self.loss_name,
                    'random_state': self.random_state,
                    'resume_training': resume,
                }
                
                # Add loss-specific parameters if using weighted loss
                if self.loss_params:
                    for key, value in self.loss_params.items():
                        ensemble_params[f'loss_{key}'] = value
                
                # Log key hyperparameters for context
                # These are the model hyperparameters used across all ensemble members
                for param_name in ['learning_rate', 'weight_decay', 'batch_size', 'na', 'nb', 'dropout_rate', 'output_activation']:
                    if param_name in self.hyperparams:
                        ensemble_params[f'model_{param_name}'] = self.hyperparams[param_name]
                
                self.mlflow_logger.log_params(ensemble_params)
                
                # Log data information for reproducibility
                data_info = {
                    'train_samples': len(self.data['datasets']['train']),
                    'test_samples': len(self.data['datasets']['test']),
                    'n_features': self.data['metadata']['n_features'],
                }
                self.mlflow_logger.log_params(data_info)
                
                print("MLflow run started for ensemble training")
                
            except Exception as e:
                print(f"Warning: Failed to start MLflow run: {e}")
                self.mlflow_logger = None
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X=self.cv_targets)):
            fold_key = f"fold_{fold_idx + 1}"
            fold_dir = os.path.join(self.output_dir, fold_key)
            os.makedirs(fold_dir, exist_ok=True)
            
            # Check if fold is already completed
            if fold_key in completed_models and len(completed_models[fold_key]) == self.n_models_per_fold:
                print(f"Skipping fold {fold_idx + 1}/{self.n_folds} (already completed)")
                
                # Load existing fold results
                fold_predictions_path = os.path.join(fold_dir, 'fold_predictions.npy')
                if os.path.exists(fold_predictions_path):
                    fold_predictions = np.load(file=fold_predictions_path)
                    all_test_predictions.extend(fold_predictions)
                    
                    # Calculate fold metrics
                    test_targets = self.data['datasets']['test'].targets.numpy()
                    fold_ensemble = np.mean(a=fold_predictions, axis=0)
                    
                    fold_result = {
                        'fold_idx': fold_idx + 1,
                        'predictions': fold_predictions,
                        'ensemble_prediction': fold_ensemble,
                        'r2': r2_score(y_true=test_targets, y_pred=fold_ensemble),
                        'rmse': np.sqrt(mean_squared_error(y_true=test_targets, y_pred=fold_ensemble)),
                        'mae': mean_absolute_error(y_true=test_targets, y_pred=fold_ensemble)
                    }
                    fold_results.append(fold_result)
                
                continue
            
            print(f"\nTraining fold {fold_idx + 1}/{self.n_folds}")
            print("=" * 50)
            
            # Create fold datasets
            fold_train_dataset = RainfallDataset(
                self.cv_climate[train_idx].numpy(),
                self.cv_local_dem[train_idx].numpy(),
                self.cv_regional_dem[train_idx].numpy(),
                self.cv_month[train_idx].numpy(),
                self.cv_targets[train_idx].numpy()
            )
            
            fold_val_dataset = RainfallDataset(
                self.cv_climate[val_idx].numpy(),
                self.cv_local_dem[val_idx].numpy(),
                self.cv_regional_dem[val_idx].numpy(),
                self.cv_month[val_idx].numpy(),
                self.cv_targets[val_idx].numpy()
            )
            
            fold_predictions = []
            fold_models = []
            
            # Train models for this fold
            for model_idx in range(self.n_models_per_fold):
                print(f"  Training model {model_idx + 1}/{self.n_models_per_fold}...")
                
                model, history, test_pred = self.train_single_model(
                    fold_idx, model_idx, fold_train_dataset, fold_val_dataset, epochs
                )
                
                fold_predictions.append(test_pred)
                fold_models.append(model)
                
                # Log individual model training metrics with robust error handling
                # This tracks per-epoch training progress for each model in the ensemble
                if self.mlflow_logger and isinstance(history, dict):
                    try:
                        # Log training history for this specific model
                        model_prefix = f"fold_{fold_idx+1}_model_{model_idx+1}"
                        
                        # Log epoch-by-epoch metrics for detailed training analysis
                        for epoch, (train_loss, val_loss) in enumerate(zip(
                            history.get('loss', []), 
                            history.get('val_loss', [])
                        )):
                            epoch_metrics = {
                                f"{model_prefix}_train_loss": train_loss,
                                f"{model_prefix}_val_loss": val_loss
                            }
                            
                            # Add unweighted MSE metrics if available
                            if 'train_mse_unw' in history and epoch < len(history['train_mse_unw']):
                                epoch_metrics[f"{model_prefix}_train_mse_unw"] = history['train_mse_unw'][epoch]
                            if 'val_mse_unw' in history and epoch < len(history['val_mse_unw']):
                                epoch_metrics[f"{model_prefix}_val_mse_unw"] = history['val_mse_unw'][epoch]
                            
                            # Log metrics for this epoch
                            for metric_name, metric_value in epoch_metrics.items():
                                self.mlflow_logger.log_metric(metric_name, metric_value, step=epoch)
                        
                        # Log final model performance summary
                        if history.get('val_loss'):
                            final_metrics = {
                                f"{model_prefix}_final_val_loss": min(history['val_loss']),
                                f"{model_prefix}_epochs_trained": len(history['val_loss']),
                                f"{model_prefix}_best_epoch": history['val_loss'].index(min(history['val_loss'])) + 1
                            }
                            self.mlflow_logger.log_metrics(final_metrics)
                            
                    except Exception as e:
                        print(f"Warning: Failed to log model {model_idx+1} metrics for fold {fold_idx+1}: {e}")
                
                # Save individual model
                model_dir = os.path.join(fold_dir, f"model_{model_idx + 1}")
                os.makedirs(model_dir, exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'history': history,
                    'hyperparams': self.hyperparams
                }, os.path.join(model_dir, 'model.pth'))
            
            fold_predictions = np.array(object=fold_predictions)
            
            # Save fold predictions
            np.save(file=os.path.join(fold_dir, 'fold_predictions.npy'), arr=fold_predictions)
            
            # Calculate fold ensemble
            test_targets = self.data['datasets']['test'].targets.numpy()
            fold_ensemble = np.mean(a=fold_predictions, axis=0)
            
            fold_result = {
                'fold_idx': fold_idx + 1,
                'predictions': fold_predictions,
                'ensemble_prediction': fold_ensemble,
                'r2': r2_score(y_true=test_targets, y_pred=fold_ensemble),
                'rmse': np.sqrt(mean_squared_error(y_true=test_targets, y_pred=fold_ensemble)),
                'mae': mean_absolute_error(y_true=test_targets, y_pred=fold_ensemble)
            }
            fold_results.append(fold_result)
            all_test_predictions.extend(fold_predictions)
            
            # Log fold-level ensemble metrics with robust error handling
            # These metrics show how well the ensemble performs on each cross-validation fold
            if self.mlflow_logger:
                try:
                    fold_metrics = {
                        f"fold_{fold_idx+1}_r2": fold_result['r2'],
                        f"fold_{fold_idx+1}_rmse": fold_result['rmse'] * 100,  # Convert to inches
                        f"fold_{fold_idx+1}_mae": fold_result['mae'] * 100,    # Convert to inches
                        f"fold_{fold_idx+1}_n_models": len(fold_predictions)
                    }
                    self.mlflow_logger.log_metrics(fold_metrics)
                    
                except Exception as e:
                    print(f"Warning: Failed to log fold {fold_idx+1} metrics: {e}")
            
            # Save fold summary
            self.save_fold_summary(fold_dir, fold_result)
            
            # Update progress
            completed_models[fold_key] = [f"model_{i+1}" for i in range(self.n_models_per_fold)]
            with open(progress_file, 'wb') as f:
                pickle.dump(obj=completed_models, file=f)
            
            print(f"  Fold {fold_idx + 1} R²: {fold_result['r2']:.4f}")
        
        # Calculate final ensemble results
        all_test_predictions = np.array(object=all_test_predictions)
        final_ensemble = np.mean(a=all_test_predictions, axis=0)
        test_targets = self.data['datasets']['test'].targets.numpy()
        
        ensemble_results = {
            'fold_results': fold_results,
            'all_predictions': all_test_predictions,
            'ensemble_prediction': final_ensemble,
            'test_targets': test_targets,
            'final_r2': r2_score(y_true=test_targets, y_pred=final_ensemble),
            'final_rmse': np.sqrt(mean_squared_error(y_true=test_targets, y_pred=final_ensemble)),
            'final_mae': mean_absolute_error(y_true=test_targets, y_pred=final_ensemble),
            'avg_fold_r2': np.mean(a=[f['r2'] for f in fold_results]),
            'avg_fold_rmse': np.mean(a=[f['rmse'] for f in fold_results]),
            'avg_fold_mae': np.mean(a=[f['mae'] for f in fold_results]),
            'training_time': time.time() - start_time,
            'hyperparams': self.hyperparams
        }
        
        # Save final results
        self.save_ensemble_results(ensemble_results)
        
        # Log final ensemble results with robust error handling
        # These metrics summarize the overall ensemble performance across all folds
        if self.mlflow_logger:
            try:
                # Log final ensemble metrics (converted to inches for physical meaning)
                final_metrics = {
                    'ensemble_r2': ensemble_results['final_r2'],
                    'ensemble_rmse_inches': ensemble_results['final_rmse'] * 100,
                    'ensemble_mae_inches': ensemble_results['final_mae'] * 100,
                    'avg_fold_r2': ensemble_results['avg_fold_r2'],
                    'avg_fold_rmse_inches': ensemble_results['avg_fold_rmse'] * 100,
                    'avg_fold_mae_inches': ensemble_results['avg_fold_mae'] * 100,
                    'training_time_seconds': ensemble_results['training_time'],
                    'total_models_trained': self.n_folds * self.n_models_per_fold
                }
                self.mlflow_logger.log_metrics(final_metrics)
                
                # Log ensemble artifacts for model analysis and reproducibility
                artifact_paths = [
                    ('ensemble_summary.txt', 'Ensemble training summary with detailed metrics'),
                    ('ensemble_predictions_scatter.png', 'Scatter plot of ensemble predictions vs actual'),
                    ('training_summary.txt', 'Overall training configuration and results')
                ]
                
                for filename, description in artifact_paths:
                    file_path = os.path.join(self.output_dir, filename)
                    if os.path.exists(file_path):
                        self.mlflow_logger.log_artifact(file_path, description)
                
                # End MLflow run
                self.mlflow_logger.end_run()
                print("MLflow logging completed successfully for ensemble training")
                
            except Exception as e:
                print(f"Warning: Failed to log final ensemble results: {e}")
                # Ensure run is ended even if logging fails
                try:
                    if self.mlflow_logger:
                        self.mlflow_logger.end_run()
                except:
                    pass
        
        return ensemble_results
    
    def save_fold_summary(self, fold_dir: str, fold_result: dict):
        """Save fold summary."""
        summary_path = os.path.join(fold_dir, 'fold_summary.txt')
        rainfall_std = self.data['metadata'].get('rainfall_mm_std', None)
        
        with open(summary_path, 'w') as f:
            f.write(f"Fold {fold_result['fold_idx']} Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Models in fold: {self.n_models_per_fold}\n")
            f.write(f"R²: {fold_result['r2']:.4f}\n")
            f.write(f"RMSE: {fold_result['rmse']:.6f}\n")
            f.write(f"MAE: {fold_result['mae']:.6f}\n")
            
            if rainfall_std is not None and rainfall_std > 0:
                f.write(f"\nDenormalized metrics (mm):\n")
                f.write(f"RMSE: {fold_result['rmse'] * rainfall_std:.4f} mm\n")
                f.write(f"MAE: {fold_result['mae'] * rainfall_std:.4f} mm\n")
            else:
                f.write(f"\nMetrics in inches:\n")
                f.write(f"RMSE: {fold_result['rmse'] * 100:.4f} inches\n")
                f.write(f"MAE: {fold_result['mae'] * 100:.4f} inches\n")
    
    def save_ensemble_results(self, results: dict):
        """Save ensemble results."""
        rainfall_std = self.data['metadata'].get('rainfall_mm_std', None)
        
        # Save ensemble summary
        summary_path = os.path.join(self.output_dir, 'ensemble_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("PyTorch Ensemble Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Number of folds: {self.n_folds}\n")
            f.write(f"  Models per fold: {self.n_models_per_fold}\n")
            f.write(f"  Total models: {self.n_folds * self.n_models_per_fold}\n")
            f.write(f"  Training time: {results['training_time']:.2f} seconds\n\n")
            
            f.write("Cross-Validation Results:\n")
            f.write(f"  Average R²: {results['avg_fold_r2']:.4f}\n")
            f.write(f"  Average RMSE: {results['avg_fold_rmse']:.6f}\n")
            f.write(f"  Average MAE: {results['avg_fold_mae']:.6f}\n\n")
            
            f.write("Final Ensemble Results:\n")
            f.write(f"  R²: {results['final_r2']:.4f}\n")
            f.write(f"  RMSE: {results['final_rmse']:.6f}\n")
            f.write(f"  MAE: {results['final_mae']:.6f}\n")
            
            if rainfall_std is not None and rainfall_std > 0:
                f.write(f"\nDenormalized Final Results (mm):\n")
                f.write(f"  RMSE: {results['final_rmse'] * rainfall_std:.4f} mm\n")
                f.write(f"  MAE: {results['final_mae'] * rainfall_std:.4f} mm\n")
            else:
                f.write(f"\nFinal Results in inches:\n")
                f.write(f"  RMSE: {results['final_rmse'] * 100:.4f} inches\n")
                f.write(f"  MAE: {results['final_mae'] * 100:.4f} inches\n")
        
        # Save test predictions
        pred_data = {
            'ensemble_predictions_normalized': results['ensemble_prediction'].tolist(),
            'test_targets_normalized': results['test_targets'].tolist(),
            'individual_predictions_normalized': results['all_predictions'].tolist()
        }
        
        if rainfall_std is not None and rainfall_std > 0:
            pred_data.update({
                'ensemble_predictions_mm': (results['ensemble_prediction'] * rainfall_std).tolist(),
                'test_targets_mm': (results['test_targets'] * rainfall_std).tolist(),
                'rainfall_std': rainfall_std
            })
        else:
            pred_data.update({
                'ensemble_predictions_inches': (results['ensemble_prediction'] * 100).tolist(),
                'test_targets_inches': (results['test_targets'] * 100).tolist()
            })
        
        with open(os.path.join(self.output_dir, 'test_predictions.json'), 'w') as f:
            json.dump(obj=pred_data, fp=f, indent=2)
        
        # Create scatter plot
        self.create_ensemble_scatter_plot(results)
        
        print(f"\nEnsemble results saved to {self.output_dir}")
    
    def create_ensemble_scatter_plot(self, results: dict):
        """Create scatter plot of ensemble predictions."""
        rainfall_std = self.data['metadata'].get('rainfall_mm_std', None)
        
        plt.figure(figsize=(10, 8))
        
        y_true = results['test_targets']
        y_pred = results['ensemble_prediction']
        
        # Use appropriate units
        if rainfall_std is not None and rainfall_std > 0:
            y_true_plot = y_true * rainfall_std
            y_pred_plot = y_pred * rainfall_std
            unit_label = "mm"
        else:
            y_true_plot = y_true * 100
            y_pred_plot = y_pred * 100
            unit_label = "inches"
        
        plt.scatter(x=y_true_plot, y=y_pred_plot, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true_plot.min(), y_pred_plot.min())
        max_val = max(y_true_plot.max(), y_pred_plot.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel(f'Actual Rainfall ({unit_label})')
        plt.ylabel(f'Predicted Rainfall ({unit_label})')
        plt.title(f'Ensemble Predictions vs Actual Rainfall\n({self.n_folds} folds × {self.n_models_per_fold} models = {self.n_folds * self.n_models_per_fold} total models)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot
        plt.text(x=0.05, y=0.95, s=f'R² = {results["final_r2"]:.4f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(fname=os.path.join(self.output_dir, 'ensemble_predictions_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def train_ensemble_pytorch(
    npz_path: str = None,
    hyperparams_dir: str = None,
    output_dir: str = None,
    test_indices_path: str = None,
    n_folds: int = 5,
    n_models_per_fold: int = 5,
    epochs: int = 150,
    resume: bool = True,
    # Loss selection
    loss_name: str = 'mse',
    loss_params: dict | None = None,
    # MLflow logging
    mlflow_enabled: bool = False,
    mlflow_experiment: str | None = None,
    mlflow_run_name: str | None = None
):
    """
    Train PyTorch ensemble with cross-validation.
    
    Args:
        npz_path: Path to assembled NPZ data
        hyperparams_dir: Directory containing best hyperparameters
        output_dir: Output directory for ensemble results
        test_indices_path: Path to test indices
        n_folds: Number of CV folds
        n_models_per_fold: Number of models per fold
        epochs: Maximum epochs per model
        resume: Whether to resume from existing progress
    """
    # Set default paths
    if npz_path is None:
        npz_path = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz')
    if hyperparams_dir is None:
        hyperparams_dir = os.path.join('Hyperparameter_Tuning', 'output')
    if output_dir is None:
        output_dir = os.path.join('Train_Ensemble', 'output', 'pytorch_ensemble')
    if test_indices_path is None:
        test_indices_path = os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl')
    
    print("PyTorch Ensemble Training")
    print("=" * 50)
    
    # Create trainer
    trainer = EnsembleTrainer(
        npz_path=npz_path,
        hyperparams_dir=hyperparams_dir,
        output_dir=output_dir,
        test_indices_path=test_indices_path,
        n_folds=n_folds,
        n_models_per_fold=n_models_per_fold,
        loss_name=loss_name,
        loss_params=loss_params,
        mlflow_enabled=mlflow_enabled,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name
    )
    
    # Train ensemble
    results = trainer.train_ensemble(epochs=epochs, resume=resume)
    
    print(f"\nEnsemble training completed!")
    print(f"Final ensemble R²: {results['final_r2']:.4f}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    
    return results


if __name__ == "__main__":
    # Train ensemble
    results = train_ensemble_pytorch(
        n_folds=3,  # Smaller for testing
        n_models_per_fold=2,  # Smaller for testing
        epochs=50  # Fewer epochs for testing
    )
    
    print(f"Ensemble training completed with R² = {results['final_r2']:.4f}")
