#!/usr/bin/env python3
"""
PyTorch hyperparameter tuning using Optuna for the LAND rainfall prediction model.
"""

import os
import json
import time
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from typing import Dict, Any, Optional
from optuna.visualization.matplotlib import plot_optimization_history as plot_hist
from optuna.visualization.matplotlib import plot_param_importances as plot_importances
import matplotlib.pyplot as plt

from Hyperparameter_Tuning.pytorch_data_utils import load_assembled_npz_data_pytorch, create_pytorch_dataloaders
from Hyperparameter_Tuning.pytorch_model import create_model_from_hyperparams
from Hyperparameter_Tuning.pytorch_training import train_model, evaluate_model


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
                 random_state: int = 42):
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
            'batch_size': trial.suggest_categorical(name='batch_size', choices=[16, 32, 64, 128]),
            
            # Model architecture choices
            'use_residual': trial.suggest_categorical(name='use_residual', choices=[True, False]),
            'activation': trial.suggest_categorical(name='activation', choices=['relu', 'elu', 'selu']),
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
        
        # Setup cross-validation - use pre-computed splits for speed
        if not hasattr(self, '_cv_splits'):
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            self._cv_splits = list(kf.split(self.cv_targets))
            print(f"Pre-computed {self.n_folds} CV splits", flush=True)
        
        fold_losses = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self._cv_splits):
            # Create fold datasets (faster with pre-converted numpy arrays)
            from Hyperparameter_Tuning.pytorch_data_utils import RainfallDataset
            
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
                    verbose=True  # Enable verbose to see epoch progress
                )
                
                # Get best validation loss
                best_val_loss = min(history['val_loss'])
                fold_losses.append(best_val_loss)
                
                # Report intermediate value for pruning
                trial.report(best_val_loss, fold_idx)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
            except Exception as e:
                print(f"Error in fold {fold_idx}: {e}")
                # Return a large loss value for failed trials
                return float('inf')
        
        # Return average validation loss across folds
        avg_loss = np.mean(fold_losses)
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
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        tuning_time = time.time() - start_time
        print(f"Tuning completed in {tuning_time:.2f} seconds")
        
        # Save results
        self.save_results(study)
        
        return study
    
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
        
        # Save best hyperparameters
        best_hyperparams_path = os.path.join(self.output_dir, 'best_hyperparameters.json')
        with open(best_hyperparams_path, 'w') as f:
            json.dump(best_trial.params, f, indent=2)
        print(f"\nBest hyperparameters saved to {best_hyperparams_path}")
        
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
        
        # Plot optimization history
        self.plot_optimization_history(study)
        
        # Plot parameter importances
        self.plot_parameter_importances(study)
    
    def plot_optimization_history(self, study: optuna.Study):
        """Plot optimization history (matplotlib backend)."""
        try:
            ax = plot_hist(study)
            fig = ax.figure
            out_path = os.path.join(self.output_dir, 'optimization_history.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Optimization history plot saved to {out_path}")
        except Exception as e:
            print(f"Could not create optimization history plot: {e}")
    
    def plot_parameter_importances(self, study: optuna.Study):
        """Plot parameter importances (matplotlib backend)."""
        try:
            ax = plot_importances(study)
            fig = ax.figure
            out_path = os.path.join(self.output_dir, 'parameter_importances.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Parameter importances plot saved to {out_path}")
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
            return json.load(f)
    
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
    resume: bool = True
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
    # Set default paths
    if npz_path is None:
        npz_path = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz')
    if output_dir is None:
        output_dir = os.path.join('Hyperparameter_Tuning', 'output')
    if test_indices_path is None:
        test_indices_path = os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl')
    
    # Create tuner
    tuner = OptunaTuner(
        npz_path=npz_path,
        output_dir=output_dir,
        test_indices_path=test_indices_path,
        n_folds=n_folds,
        max_epochs=max_epochs,
        patience=patience
    )
    
    # Run tuning
    study = tuner.run_tuning(n_trials=n_trials, resume=resume)
    
    return {
        'best_hyperparameters': study.best_trial.params,
        'best_value': study.best_trial.value,
        'n_trials': len(study.trials),
        'study': study
    }


if __name__ == "__main__":
    # Test hyperparameter tuning
    print("Testing PyTorch hyperparameter tuning...")
    
    # Run a small test
    results = run_hyperparameter_tuning(
        output_dir=os.path.join('Hyperparameter_Tuning', 'output'),
        n_trials=5,  # Small number for testing
        n_folds=3,   # Fewer folds for testing
        max_epochs=10,  # Fewer epochs for testing
        patience=5
    )
    
    print(f"Best hyperparameters: {results['best_hyperparameters']}")
    print(f"Best validation loss: {results['best_value']:.6f}")
    print(f"Completed {results['n_trials']} trials")
