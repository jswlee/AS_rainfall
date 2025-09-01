#!/usr/bin/env python3
"""
PyTorch hyperparameter tuning using Optuna for the LAND rainfall prediction model.
"""

import os
import json
import time
import numpy as np
import torch
import argparse
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, Optional
from optuna.visualization.matplotlib import plot_optimization_history as plot_hist
from optuna.visualization.matplotlib import plot_param_importances as plot_importances
import matplotlib.pyplot as plt
 
# Import robust MLflow utilities for experiment tracking
# These utilities provide comprehensive error handling and MLOps best practices
from Hyperparameter_Tuning.mlflow_utils import (
    create_mlflow_logger, log_hyperparameters, log_model_summary, 
    log_evaluation_results, MLFLOW_AVAILABLE
)

# Optional Optuna MLflow integration
try:
    from optuna.integration import MLflowCallback
except ImportError:
    try:
        from optuna.integration.mlflow import MLflowCallback
    except ImportError:
        MLflowCallback = None

# Direct MLflow imports for model registration
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None

from Hyperparameter_Tuning.data_utils import load_assembled_npz_data_pytorch, create_pytorch_dataloaders
from Hyperparameter_Tuning.model import create_model_from_hyperparams
from Hyperparameter_Tuning.model_training import train_model, evaluate_model


class OptunaTuner:
    """
    Optuna-based hyperparameter tuner for the LAND model.
    """
    
    def __init__(self, 
                 npz_path: str,
                 output_dir: str,
                 test_indices_path: Optional[str] = None,
                 n_folds: int = 3,
                 max_epochs: int = 100,
                 patience: int = 10,
                 random_state: int = 42,
                 loss_name: str = 'mse',
                 loss_params: Optional[Dict[str, Any]] = None,
                 enable_mlflow: bool = False,
                 mlflow_experiment: Optional[str] = None):
        """
        Initialize the tuner.
        
        Args:
            npz_path: Path to the assembled NPZ data file
            output_dir: Directory to save tuning results
            test_indices_path: Path for test indices (for reproducibility)
            n_folds: Number of CV folds
            max_epochs: Maximum epochs per trial
            patience: Early stopping patience
            random_state: Random seed
        """
        self.npz_path = npz_path
        self.output_dir = output_dir
        self.test_indices_path = test_indices_path
        self.n_folds = n_folds
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.loss_name = loss_name
        self.loss_params = loss_params or None
        # MLflow experiment tracking setup
        # Experiments group related runs together (e.g., all hyperparameter tuning runs)
        self.enable_mlflow = bool(enable_mlflow and MLFLOW_AVAILABLE)
        self.mlflow_experiment = mlflow_experiment or "AS_Rainfall_Hyperparameter_Tuning"
        
        # Initialize MLflow logger for robust experiment tracking
        self.mlflow_logger = None
        if self.enable_mlflow:
            self.mlflow_logger = create_mlflow_logger(
                experiment_name=self.mlflow_experiment,
                enabled=True
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        print(f"Loading data from {npz_path}...")
        self.data = load_assembled_npz_data_pytorch(
            npz_path=npz_path,
            test_indices_path=test_indices_path,
            random_state=random_state,
            test_size=0.1,
            val_size=0.1
        )
        
        # Combine train and val for CV
        train_dataset = self.data['datasets']['train']
        val_dataset = self.data['datasets']['val']
        self.cv_climate = torch.cat(tensors=[train_dataset.climate_data, val_dataset.climate_data])
        self.cv_local_dem = torch.cat(tensors=[train_dataset.local_dem_data, val_dataset.local_dem_data])
        self.cv_regional_dem = torch.cat(tensors=[train_dataset.regional_dem_data, val_dataset.regional_dem_data])
        self.cv_month = torch.cat(tensors=[train_dataset.month_data, val_dataset.month_data])
        self.cv_targets = torch.cat(tensors=[train_dataset.targets, val_dataset.targets])
        
        print(f"CV data shape: {self.cv_targets.shape[0]} samples")
        
        # Prepare stratification bins for regression targets (quantile-based)
        self._n_strata_bins = 5  # adjustable; 5 bins is a good default
        y = self.cv_targets.numpy().ravel()
        try:
            q = np.linspace(0.0, 1.0, self._n_strata_bins + 1)
            edges = np.quantile(y, q)
            # Ensure strictly increasing edges; fallback if duplicates due to ties
            uniq = np.unique(edges)
            if uniq.size < edges.size:
                edges = np.linspace(y.min(), y.max(), self._n_strata_bins + 1)
            # Digitize into bins 0..n_bins-1
            self._y_bins = np.digitize(y, edges[1:-1], right=True)
        except Exception as e:
            print(f"Warning: stratification binning failed ({e}); falling back to single bin.")
            self._y_bins = np.zeros_like(y, dtype=int)
        
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
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        return {
            # Architecture parameters
            'climate_units': trial.suggest_int(name='climate_units', low=64, high=512, step=32),
            'local_dem_units': trial.suggest_int(name='local_dem_units', low=16, high=256, step=16),
            'regional_dem_units': trial.suggest_int(name='regional_dem_units', low=32, high=128, step=16),
            'month_units': trial.suggest_int(name='month_units', low=16, high=64, step=8),
            'na': trial.suggest_int(name='na', low=128, high=512, step=64),
            'nb': trial.suggest_int(name='nb', low=64, high=1024, step=64),
            
            # Regularization parameters
            'dropout_rate': trial.suggest_float(name='dropout_rate', low=0.0, high=0.5, step=0.05),
            'l2_reg': trial.suggest_float(name='l2_reg', low=1e-5, high=1e-3, log=True),
            
            # Training parameters
            'learning_rate': trial.suggest_float(name='learning_rate', low=1e-3, high=1e-2, log=True),
            'weight_decay': trial.suggest_float(name='weight_decay', low=1e-6, high=1e-3, log=True),
            'batch_size': trial.suggest_categorical(name='batch_size', choices=[64, 128, 256]),
            
            # Model architecture choices
            'use_residual': trial.suggest_categorical(name='use_residual', choices=[True, False]),
            'climate_activation': trial.suggest_categorical(name='climate_activation', choices=['relu', 'none']),
            'output_activation': trial.suggest_categorical(name='output_activation', choices=['relu', 'softplus']),
            'climate_processing': trial.suggest_categorical(name='climate_processing', choices=['flatten', 'conv2d'])
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Average validation loss across CV folds
        """
        # Get hyperparameters for this trial
        hyperparams = self.suggest_hyperparameters(trial)
        print(f"\nTrial {trial.number}: {hyperparams}", flush=True)

        # ============================================================================
        # MLflow Experiment Tracking Setup
        # ============================================================================
        # MLflow helps track experiments by logging:
        # 1. Parameters (hyperparameters, configuration)
        # 2. Metrics (loss, accuracy over time)
        # 3. Artifacts (models, plots, results)
        # 4. Tags (metadata for organizing runs)
        
        _mlflow_run_started_here = False
        if self.enable_mlflow and self.mlflow_logger:
            # Check if Optuna's MLflowCallback already created a run
            if mlflow and mlflow.active_run() is None:
                # Start a dedicated run for this trial
                self.mlflow_logger.active_run = mlflow.start_run(
                    run_name=f"trial_{trial.number}"
                )
                _mlflow_run_started_here = True
            
            # Log trial configuration for reproducibility
            trial_config = {
                "trial_number": trial.number,
                "device": str(self.device),
                "n_folds": self.n_folds,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "loss_name": self.loss_name,
            }
            
            # Log hyperparameters with "hp_" prefix for clarity
            log_hyperparameters(self.mlflow_logger, hyperparams, prefix="hp")
            self.mlflow_logger.log_params(trial_config)
            self.mlflow_logger.set_tag("optuna_trial_number", str(trial.number))
            self.mlflow_logger.set_tag("experiment_type", "hyperparameter_tuning")
            
            # Store run ID for model registration later
            if mlflow and mlflow.active_run():
                trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
        
        # Setup cross-validation - use pre-computed splits for speed
        if not hasattr(self, '_cv_splits'):
            # Use StratifiedKFold on binned targets to balance target distribution per fold
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            # X is dummy; stratify on bins
            dummy_X = np.zeros_like(self._y_bins)
            self._cv_splits = list(skf.split(dummy_X, self._y_bins))
            # Diagnostics on bin counts
            counts = np.bincount(self._y_bins)
            print(f"Pre-computed {self.n_folds} stratified CV splits (bin counts: {counts.tolist()})", flush=True)
        
        fold_losses = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self._cv_splits):
            # Create fold datasets (faster with pre-converted numpy arrays)
            from Hyperparameter_Tuning.data_utils import RainfallDataset
            
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
            
            # Create optimized dataloaders for GPU utilization
            fold_dataloaders = create_pytorch_dataloaders(
                {'train': fold_train_dataset, 'val': fold_val_dataset},
                batch_size=hyperparams['batch_size'],
                num_workers=2,  # Use multiple workers for parallel data loading
                pin_memory=False
            )
            
            # Create model
            print(f"    Creating model with {hyperparams['na']} hidden units...", flush=True)
            model = create_model_from_hyperparams(hyperparams, self.data['metadata'])

            # Log model architecture summary for this fold
            # This helps understand model complexity and debug architecture issues
            if self.enable_mlflow and self.mlflow_logger and fold_idx == 0:  # Only log once per trial
                log_model_summary(
                    self.mlflow_logger, 
                    model, 
                    f"trial_{trial.number}_model_architecture.txt"
                )
            
            # Train model
            try:
                print(f"  Fold {fold_idx+1}/{self.n_folds}: Training with {len(fold_train_dataset)} samples...", flush=True)
                history = train_model(
                    model=model,
                    dataloaders=fold_dataloaders,
                    epochs=self.max_epochs,
                    learning_rate=hyperparams['learning_rate'],
                    weight_decay=hyperparams['weight_decay'],
                    patience=self.patience,
                    device=self.device,
                    verbose=True,  # Enable verbose to see epoch progress
                    loss_name=self.loss_name,
                    loss_params=self.loss_params
                )

                # Get best validation loss
                best_val_loss = min(history['val_loss'])
                fold_losses.append(best_val_loss)

                # ================================================================
                # MLflow Logging: Training Curves and Model Artifacts
                # ================================================================
                # Log detailed training metrics for this fold
                # This creates time-series plots in MLflow UI showing training progress
                if self.enable_mlflow and self.mlflow_logger:
                    # Log per-epoch training curves with fold prefix
                    fold_history = {}
                    for metric_name, values in history.items():
                        fold_history[f"fold{fold_idx+1}_{metric_name}"] = values
                    
                    # Log training curves (creates line plots in MLflow UI)
                    self.mlflow_logger.log_training_curves(fold_history, start_epoch=1)
                    
                    # Log fold-level summary metrics
                    fold_metrics = {
                        f"fold{fold_idx+1}_best_val_loss": float(best_val_loss),
                        f"fold{fold_idx+1}_final_train_loss": float(history['train_loss'][-1]),
                        f"fold{fold_idx+1}_epochs_trained": len(history['train_loss']),
                    }
                    
                    # Add unweighted MSE if available
                    if 'val_mse_unweighted' in history:
                        best_epoch_idx = int(np.argmin(history['val_loss']))
                        fold_metrics[f"fold{fold_idx+1}_val_mse_unweighted"] = float(
                            history['val_mse_unweighted'][best_epoch_idx]
                        )
                    
                    self.mlflow_logger.log_metrics(fold_metrics)

                # Report intermediate value for pruning
                trial.report(best_val_loss, fold_idx)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except Exception as e:
                print(f"Error in fold {fold_idx}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                
                # End MLflow run if active to prevent resource leaks
                if self.mlflow_logger:
                    try:
                        self.mlflow_logger.end_run()
                    except:
                        pass
                
                # Return inf to indicate failed trial
                return float('inf')
        
        # Return average validation loss across folds
        avg_loss = np.mean(fold_losses)

        # ================================================================
        # MLflow Logging: Trial Summary and Cleanup
        # ================================================================
        # Log aggregated metrics across all folds for this trial
        if self.enable_mlflow and self.mlflow_logger:
            # Trial-level summary metrics
            trial_summary = {
                "avg_val_loss": float(avg_loss),
                "std_val_loss": float(np.std(fold_losses)),
                "min_val_loss": float(np.min(fold_losses)),
                "max_val_loss": float(np.max(fold_losses)),
                "n_folds_completed": len(fold_losses),
            }
            
            self.mlflow_logger.log_metrics(trial_summary)
            
            # Store metadata for model registration
            best_fold_idx = int(np.argmin(fold_losses)) + 1
            trial.set_user_attr("best_fold", best_fold_idx)
            self.mlflow_logger.set_tag("best_fold", str(best_fold_idx))
            self.mlflow_logger.set_tag("trial_status", "completed")
        
        # Clean up MLflow run if we started it
        if _mlflow_run_started_here and mlflow:
            try:
                mlflow.end_run()
            except Exception as e:
                print(f"Warning: Error ending MLflow run: {e}")

        return avg_loss
    
    def run_tuning(self, 
                   n_trials: int = 100,
                   study_name: str = "land_model_tuning",
                   resume: bool = True) -> optuna.Study:
        """
        Run hyperparameter tuning.
        
        Args:
            n_trials: Number of trials to run
            study_name: Name of the study
            resume: Whether to resume from existing study
            
        Returns:
            Optuna study object
        """
        # Setup study storage
        storage_path = os.path.join(self.output_dir, f"{study_name}.db")
        storage = f"sqlite:///{storage_path}"
        
        # Create or load study
        if resume and os.path.exists(storage_path):
            print(f"Resuming study from {storage_path}")
            study = optuna.load_study(study_name=study_name, storage=storage)
            print(f"Found {len(study.trials)} existing trials")
        else:
            print(f"Creating new study: {study_name}")
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction='minimize',
                sampler=TPESampler(seed=self.random_state),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
        
        print(f"Starting hyperparameter tuning with {n_trials} trials...")
        start_time = time.time()
        
        # ================================================================
        # MLflow Integration with Optuna
        # ================================================================
        # Set up MLflow callbacks for automatic logging of Optuna trials
        callbacks = []
        if self.enable_mlflow and MLflowCallback is not None:
            try:
                # Optuna's MLflowCallback automatically logs trial parameters and results
                mlflow_cb = MLflowCallback(
                    tracking_uri=None,  # Use default (local ./mlruns)
                    experiment_name=self.mlflow_experiment,
                    metric_name="avg_val_loss"  # Primary metric to optimize
                )
                callbacks.append(mlflow_cb)
                print(f"✓ MLflow callback enabled for experiment: {self.mlflow_experiment}")
            except Exception as e:
                print(f"Warning: Could not enable MLflow callback: {e}")
                print("Falling back to direct MLflow logging")
        
        # Ensure experiment exists even without callback
        if self.enable_mlflow and mlflow:
            try:
                mlflow.set_experiment(self.mlflow_experiment)
                print(f"✓ MLflow experiment set: {self.mlflow_experiment}")
            except Exception as e:
                print(f"Warning: Could not set MLflow experiment: {e}")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True, callbacks=callbacks)
        
        tuning_time = time.time() - start_time
        print(f"Tuning completed in {tuning_time:.2f} seconds")
        
        # Save results
        self.save_results(study)
        
        return study

    def register_best_model(self, study: optuna.Study, model_name: str, stage: str = "Staging") -> Optional[str]:
        """
        Register the best model from the study in the MLflow Model Registry.

        This uses the run_id stored on the best trial (user_attr "mlflow_run_id") and
        the best fold index (user_attr "best_fold") to form the model URI
        runs:/<run_id>/model_fold<best_fold>.

        Args:
            study: The completed Optuna study.
            model_name: The registered model name to use in MLflow.
            stage: Target stage to transition the registered version to (e.g., "Staging", "Production").

        Returns:
            The registered model version string, or None if registration not performed.
        """
        # Return average validation loss across folds
        avg_loss = np.mean(fold_losses)

        # ================================================================
        # MLflow Logging: Trial Summary and Cleanup
        # ================================================================
        # Log aggregated metrics across all folds for this trial
        if self.enable_mlflow and self.mlflow_logger:
            trial_summary = {
                "avg_val_loss": float(avg_loss),
                "std_val_loss": float(np.std(fold_losses)),
                "min_val_loss": float(np.min(fold_losses)),
                "max_val_loss": float(np.max(fold_losses)),
                "n_folds_completed": len(fold_losses),
            }
            
            self.mlflow_logger.log_metrics(trial_summary)
            
            # Store metadata for model registration
            best_fold_idx = int(np.argmin(fold_losses)) + 1
            trial.set_user_attr("best_fold", best_fold_idx)
            self.mlflow_logger.set_tag("best_fold", str(best_fold_idx))
            self.mlflow_logger.set_tag("trial_status", "completed")
            
            # End MLflow run
            self.mlflow_logger.end_run()

        return avg_loss
    
    def register_best_model(self, study: optuna.Study, model_name: str, stage: str = "Staging") -> Optional[str]:
        """
        Register the best model from the study in the MLflow Model Registry.

        This uses the run_id stored on the best trial (user_attr "mlflow_run_id") and
        the best fold index (user_attr "best_fold") to form the model URI
        runs:/<run_id>/model_fold<best_fold>.

        Args:
            study: The completed Optuna study.
            model_name: The registered model name to use in MLflow.
            stage: Target stage to transition the registered version to (e.g., "Staging", "Production").

        Returns:
            The registered model version string, or None if registration not performed.
        """
        if not (self.enable_mlflow and MLFLOW_AVAILABLE):
            print("MLflow not available or not enabled; skipping model registration.")
            return None
            
        try:
            # Register best model in model registry (without deprecated staging)
            best_trial = study.best_trial
            run_id = best_trial.user_attrs.get("mlflow_run_id")
            best_fold = best_trial.user_attrs.get("best_fold", 1)
            
            if not run_id:
                print("No MLflow run ID found in best trial; skipping model registration.")
                return None
                
            model_uri = f"runs:/{run_id}/model_fold{best_fold}"
            
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            print(f"Registered model version {model_version.version} for {model_name} from {model_uri}")
            
            # Set model version alias instead of deprecated stage
            client = mlflow.tracking.MlflowClient()
            try:
                client.set_registered_model_alias(
                    name=model_name,
                    alias="champion",  # Use alias instead of deprecated stage
                    version=model_version.version
                )
                print(f"Set alias 'champion' for model version {model_version.version}")
            except AttributeError:
                # Fallback for older MLflow versions - skip staging
                print(f"Model registered without alias (MLflow version doesn't support aliases)")
            
        except Exception as e:
            self.mlflow_logger.warning(f"Failed to register model: {e}")
        return None
    
    def save_results(self, study: optuna.Study):
        """
        Save tuning results.
        
        Args:
            study: Completed Optuna study
        """
        # Get best trial
        best_trial = study.best_trial
        
        print(f"\nBest trial:")
        print(f"  Value (validation loss): {best_trial.value:.6f}")
        print(f"  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best hyperparameters with trial metadata
        best_hyperparams_path = os.path.join(self.output_dir, 'best_hyperparameters.json')
        best_trial_data = {
            'hyperparameters': best_trial.params,
            'trial_number': best_trial.number,
            'best_value': best_trial.value,
            'trial_datetime': best_trial.datetime_start.isoformat() if best_trial.datetime_start else None
        }
        with open(best_hyperparams_path, 'w') as f:
            json.dump(best_trial_data, f, indent=2)
        print(f"\nBest hyperparameters saved to {best_hyperparams_path}")
        print(f"Best trial number: {best_trial.number}")
        
        # Save best hyperparameters in Python format (for compatibility)
        python_hp_path = os.path.join(self.output_dir, 'best_hyperparameters.py')
        with open(python_hp_path, 'w') as f:
            f.write("# Best hyperparameters from Optuna tuning\n\n")
            f.write("best_hyperparameters = {\n")
            for key, value in best_trial.params.items():
                if isinstance(value, str):
                    f.write(f"    '{key}': '{value}',\n")
                else:
                    f.write(f"    '{key}': {value},\n")
            f.write("}\n")
        print(f"Best hyperparameters saved to {python_hp_path}")
        
        # Save study summary
        summary_path = os.path.join(self.output_dir, 'tuning_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Hyperparameter Tuning Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Study name: {study.study_name}\n")
            f.write(f"Number of trials: {len(study.trials)}\n")
            f.write(f"Best value: {best_trial.value:.6f}\n\n")
            f.write(f"Best hyperparameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nCompleted trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
            f.write(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
            f.write(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
        
        print(f"Tuning summary saved to {summary_path}")
        
        # Create optimization history plot (suppress experimental warnings)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="optuna")
                warnings.filterwarnings("ignore", message=".*experimental.*", module="optuna")
                # Use matplotlib backend to get a Matplotlib Axes
                ax = plot_hist(study)
                ax.figure.savefig(os.path.join(self.output_dir, 'optimization_history.png'), 
                                dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Optimization history plot saved to {self.output_dir}/optimization_history.png")
        except Exception as e:
            print(f"Could not create optimization history plot: {e}")
        
        # Create parameter importance plot (suppress experimental warnings)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="optuna")
                warnings.filterwarnings("ignore", message=".*experimental.*", module="optuna")
                # Use matplotlib backend to get a Matplotlib Axes
                ax = plot_importances(study)
                ax.figure.savefig(os.path.join(self.output_dir, 'parameter_importances.png'), 
                                dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Parameter importances plot saved to {self.output_dir}/parameter_importances.png")
        except Exception as e:
            print(f"Could not create parameter importances plot: {e}")


def load_best_hyperparameters_pytorch(output_dir: str) -> Dict[str, Any]:
    """
    Load the best hyperparameters from tuning output.
    
    Args:
        output_dir: Directory containing tuning results
        
    Returns:
        Dictionary of best hyperparameters
    """
    # Try JSON format first
    json_path = os.path.join(output_dir, 'best_hyperparameters.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Handle both old format (direct hyperparams) and new format (with metadata)
            if 'hyperparameters' in data:
                return data  # Return full metadata including trial_number
            else:
                return {'hyperparameters': data}  # Wrap old format for compatibility
    
    # Try Python format
    py_path = os.path.join(output_dir, 'best_hyperparameters.py')
    if os.path.exists(py_path):
        namespace = {}
        with open(py_path, 'r') as f:
            exec(f.read(), namespace)
        return namespace.get('best_hyperparameters', {})
    
    raise FileNotFoundError(f"No hyperparameters file found in {output_dir}")


def run_hyperparameter_tuning(
    npz_path: str = None,
    output_dir: str = None,
    test_indices_path: str = None,
    n_trials: int = 100,
    n_folds: int = 5,
    max_epochs: int = 150,
    patience: int = 10,
    resume: bool = True,
    loss_name: str = 'mse',
    loss_params: Optional[Dict[str, Any]] = None,
    enable_mlflow: bool = False,
    mlflow_experiment: Optional[str] = None,
    study_name: str = "land_model_tuning",
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning with default paths.
    
    Args:
        npz_path: Path to NPZ data file
        output_dir: Output directory for results
        test_indices_path: Path for test indices
        n_trials: Number of tuning trials
        n_folds: Number of CV folds
        max_epochs: Maximum epochs per trial
        patience: Early stopping patience
        resume: Whether to resume existing study
        
    Returns:
        Dictionary with best hyperparameters and study info
    """
    # Require explicit inputs (no internal defaults)
    if npz_path is None or output_dir is None or test_indices_path is None:
        raise ValueError(
            "npz_path, output_dir, and test_indices_path must be provided to run_hyperparameter_tuning()."
        )
    
    # Create tuner
    tuner = OptunaTuner(
        npz_path=npz_path,
        output_dir=output_dir,
        test_indices_path=test_indices_path,
        n_folds=n_folds,
        max_epochs=max_epochs,
        patience=patience,
        loss_name=loss_name,
        loss_params=loss_params,
        enable_mlflow=enable_mlflow,
        mlflow_experiment=mlflow_experiment
    )
    
    # Run tuning
    study = tuner.run_tuning(n_trials=n_trials, study_name=study_name, resume=resume)
    tuner.register_best_model(study=study, model_name="land_rainfall_model", stage="Staging")
    return {
        'best_hyperparameters': study.best_trial.params,
        'best_value': study.best_trial.value,
        'n_trials': len(study.trials),
        'study': study
    }

if __name__ == "__main__":
    # Command-line interface for running hyperparameter tuning
    parser = argparse.ArgumentParser(description="Run PyTorch hyperparameter tuning for LAND rainfall model")
    parser.add_argument("--npz-path", default=os.path.join("ML_Data_Preprocessing", "output", "assembled_npz", "full_training_data.npz"), help="Path to assembled NPZ data file")
    parser.add_argument("--output-dir", default=os.path.join("Hyperparameter_Tuning", "output_WeightedMSE2"), help="Directory to write tuning outputs")
    parser.add_argument("--test-indices-path", default=os.path.join("Hyperparameter_Tuning", "output_WeightedMSE2", "test_indices.pkl"), help="Path to test indices file for reproducibility")

    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--max-epochs", type=int, default=150, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--study-name", type=str, default="land_model_tuning", help="Optuna study name")

    parser.add_argument("--loss-name", type=str, default="mse", choices=["mse", "weighted_mse"], help="Loss function name")
    parser.add_argument("--loss-params", type=str, default=None, help="JSON string of loss params, e.g. '{\"w\": 0.5}'")

    parser.add_argument("--enable-mlflow", action="store_true", help="Enable MLflow experiment tracking")
    parser.add_argument("--mlflow-experiment", type=str, default=None, help="MLflow experiment name")

    # Resume flags
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="Resume existing study if found (default)")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Start a fresh study")
    parser.set_defaults(resume=True)

    args = parser.parse_args()

    # Parse loss params JSON if provided
    loss_params = None
    if args.loss_params:
        try:
            loss_params = json.loads(args.loss_params)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --loss-params JSON: {e}")

    run_hyperparameter_tuning(
        npz_path=args.npz_path,
        output_dir=args.output_dir,
        test_indices_path=args.test_indices_path,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        resume=args.resume,
        loss_name=args.loss_name,
        loss_params=loss_params,
        enable_mlflow=args.enable_mlflow,
        mlflow_experiment=args.mlflow_experiment,
        study_name=args.study_name,
    )


 
