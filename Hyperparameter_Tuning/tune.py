# hp_tuning_final.py

# For daily data:
# python -m Hyperparameter_Tuning.hp_tuning_simplified --npz-path "ML_Data_Preprocessing\output\assembled_npz\full_training_data_daily.npz" --output-dir "output/daily_data_200trials" --n-trials 20 --time-interval "daily" --study-name "land_model_tuning_daily"

# For monthly data:
# python -m Hyperparameter_Tuning.hp_tuning_simplified --npz-path "ML_Data_Preprocessing\output\assembled_npz\full_training_data_monthly.npz" --output-dir "output/monthly_data_200trials" --n-trials 20 --time-interval "monthly" --study-name "land_model_tuning_monthly"

import os
import json
import numpy as np
import torch
import argparse
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any

from Hyperparameter_Tuning.data_utils_simplified import DataManager, create_pytorch_dataloaders, RainfallDataset
from Hyperparameter_Tuning.model import create_model_from_hyperparams
from Hyperparameter_Tuning.model_training import train_model

class OptunaTuner:
    """Optuna-based hyperparameter tuner."""
    def __init__(self, **kwargs):
        self.config = kwargs
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"--- Using device: {self.device} ---")

        # Instantiate the DataManager to handle all data logic
        data_manager = DataManager(device=self.device, **self.config)
        
        # Persist the exact test indices used by the tuner for reproducibility
        # If the caller did not provide a path, default to <output_dir>/test_indices.pkl
        import pickle
        ti_path = self.config.get('test_indices_path')
        if not ti_path:
            ti_path = os.path.join(self.config['output_dir'], 'test_indices.pkl')
        try:
            os.makedirs(os.path.dirname(ti_path), exist_ok=True)
            with open(ti_path, 'wb') as f:
                pickle.dump(data_manager.indices['test'], f)
            print(f"Saved tuner test indices to {ti_path}")
        except Exception as e:
            print(f"Warning: Could not save test indices to {ti_path}: {e}")
        
        # Get the metadata and the specific tensors needed for cross-validation
        self.metadata = data_manager.metadata
        self.cv_tensors, self.cv_indices = data_manager.get_cv_tensors()
        self.data_manager = data_manager  # Store for single-fold case
        if self.config['n_folds'] > 1:
            self._prepare_cv_splits()
        else:
            # For single fold, we'll use the train/val split from DataManager
            self._cv_splits = None

    def _prepare_cv_splits(self):
        """Pre-computes Stratified K-Fold splits based on target quantiles."""
        n_bins = 5
        y = self.cv_tensors['targets'][self.cv_indices].cpu().numpy().ravel()
        try:
            edges = np.quantile(y[y > 0], np.linspace(0, 1, n_bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2: raise ValueError("Not enough unique quantile edges.")
            y_bins = np.digitize(y, edges[1:-1])
        except Exception as e:
            # Stratified CV is required for robust evaluation
            raise ValueError(f"Stratified binning failed: {e}. Cannot proceed with non-stratified CV.") from e
            
        cv_seed = self.config.get('random_state', 42)
        skf = StratifiedKFold(n_splits=self.config['n_folds'], shuffle=True, random_state=cv_seed)
        # skf.split gives indices *relative to the input array* (self.cv_indices). 
        # We need to map them back to the original tensor indices.
        self._cv_splits = []
        for train_fold_idx, val_fold_idx in skf.split(np.zeros_like(y_bins), y_bins):
            train_original_indices = self.cv_indices[train_fold_idx]
            val_original_indices = self.cv_indices[val_fold_idx]
            self._cv_splits.append((train_original_indices, val_original_indices))
            
        print(f"Pre-computed {self.config['n_folds']} stratified CV splits.")

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Defines the hyperparameter search space for Optuna."""
        time_interval = self.config.get('time_interval', 'daily')
        if time_interval == 'daily':
            return {
                'climate_units': trial.suggest_int('climate_units', 150, 1350, step=15),
                'local_dem_units': trial.suggest_int('local_dem_units', 16, 256, step=16),
                'regional_dem_units': trial.suggest_int('regional_dem_units', 16, 256, step=16),
                'temporal_units': trial.suggest_int('temporal_units', 16, 128, step=16),

                'na': trial.suggest_int('na', 4096, 5120, step=128),
                'nb': trial.suggest_int('nb', 516, 628, step=16),

                'dropout_rate': trial.suggest_float('dropout_rate', 0.40, 0.45, step=0.05),
                'l2_reg': trial.suggest_float('l2_reg', 1e-7, 2e-7, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 8e-6, 5e-4, log=True),

                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'use_residual': trial.suggest_categorical('use_residual', [False]),
                'climate_activation': trial.suggest_categorical('climate_activation', ['relu']),
                'output_activation': trial.suggest_categorical('output_activation', ['softplus']),  # NEVER 'none' - must be non-negative!
                'climate_processing': trial.suggest_categorical('climate_processing', ['conv2d']),
                
                # Attention mechanism
                'use_spatial_attention': trial.suggest_categorical('use_spatial_attention', [True]),
                'use_multihead_attention': trial.suggest_categorical('use_multihead_attention', [False]),
                'attention_heads': trial.suggest_categorical('attention_heads', [3, 5]),
                'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.3, step=0.05),
                
                # Temporal branch depth
                'temporal_depth': trial.suggest_categorical('temporal_depth', [1, 2]),
                'temporal_dropout': trial.suggest_float('temporal_dropout', 0.05, 0.2, step=0.05)
            }
        else:
            return {
                # Climate variables
                'climate_units': trial.suggest_int(name='climate_units', low=64, high=512, step=32),
                'local_dem_units': trial.suggest_int(name='local_dem_units', low=16, high=1024, step=16),
                'regional_dem_units': trial.suggest_int(name='regional_dem_units', low=32, high=512, step=16),
                'temporal_units': trial.suggest_int(name='temporal_units', low=16, high=64, step=8),
                
                # Neural network architecture
                'na': trial.suggest_int(name='na', low=128, high=512, step=64),
                'nb': trial.suggest_int(name='nb', low=32, high=512, step=32),
                
                # Regularization parameters
                'dropout_rate': trial.suggest_float(name='dropout_rate', low=0.0, high=0.5, step=0.05),
                'l2_reg': trial.suggest_float(name='l2_reg', low=1e-5, high=1e-4, log=True),
                
                # Training parameters
                'learning_rate': trial.suggest_float(name='learning_rate', low=1e-5, high=1e-2, log=True),
                'weight_decay': trial.suggest_float(name='weight_decay', low=1e-6, high=1e-3, log=True),
                'batch_size': trial.suggest_categorical(name='batch_size', choices=[256, 512, 1024, 2048]),
                
                # Model architecture choices
                'use_residual': trial.suggest_categorical(name='use_residual', choices=[True, False]),
                'climate_activation': trial.suggest_categorical(name='climate_activation', choices=['relu', 'none']),
                'output_activation': trial.suggest_categorical(name='output_activation', choices=['relu', 'softplus']),
                'climate_processing': trial.suggest_categorical(name='climate_processing', choices=['flatten', 'conv2d']),
                
                # Attention mechanism hyperparameters
                'use_spatial_attention': trial.suggest_categorical(name='use_spatial_attention', choices=[True, False]),
                'use_multihead_attention': trial.suggest_categorical(name='use_multihead_attention', choices=[True, False]),
                'attention_heads': trial.suggest_categorical(name='attention_heads', choices=[2, 4, 8]),
                'attention_dropout': trial.suggest_float(name='attention_dropout', low=0.0, high=0.3, step=0.1),
                
                # Temporal branch depth
                'temporal_depth': trial.suggest_categorical(name='temporal_depth', choices=[1, 2, 3]),
                'temporal_dropout': trial.suggest_float(name='temporal_dropout', low=0.0, high=0.3, step=0.1)
            }

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for one Optuna trial, performing cross-validation."""
        hyperparams = self.suggest_hyperparameters(trial)
        print(f"\nTrial {trial.number}: Starting with params: {hyperparams}")

        fold_losses, fold_models = [], []
        
        # Handle single fold vs multi-fold
        if self._cv_splits is None:
            # Single fold: use train/val split from DataManager
            cv_iterations = [(0, (self.data_manager.indices['train'], self.data_manager.indices['val']))]
        else:
            # Multi-fold: use CV splits
            cv_iterations = enumerate(self._cv_splits)
        
        for fold_idx, (train_idx, val_idx) in cv_iterations:
            # Create datasets by passing the shared tensors and the specific indices for this fold
            train_ds = RainfallDataset(self.cv_tensors, train_idx)
            val_ds = RainfallDataset(self.cv_tensors, val_idx)
            
            # Dataloader creation now pass dataloader-specific params from config
            dataloader_params = {k: v for k, v in self.config.items() if k in ['num_workers', 'pin_memory']}
            # Set GPU-optimized defaults if not specified
            dataloader_params.setdefault('pin_memory', False)
            dataloader_params.setdefault('num_workers', 0)  # Windows: 2-4 workers max
            dataloaders = create_pytorch_dataloaders(
                {'train': train_ds, 'val': val_ds}, 
                batch_size=hyperparams['batch_size'], 
                **dataloader_params
            )

            model = create_model_from_hyperparams(hyperparams, self.metadata).to(self.device)

            try:
                # Scale LR with batch size using sqrt scaling (alpha=0.5)
                # Reference: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
                batch_ref = 1024
                alpha = 0.5
                base_lr = float(hyperparams['learning_rate'])
                bs = int(hyperparams['batch_size'])
                lr_scale = (bs / batch_ref) ** alpha
                scaled_lr = base_lr * lr_scale

                # Log base and scaled LR (once per fold for traceability)
                trial.set_user_attr("base_learning_rate", base_lr)
                trial.set_user_attr("scaled_learning_rate", scaled_lr)

                history = train_model(
                    model=model, dataloaders=dataloaders, device=self.device,
                    epochs=self.config['max_epochs'], patience=self.config['patience'],
                    learning_rate=scaled_lr, weight_decay=hyperparams['weight_decay'],
                    loss_name=self.config['loss_name'], loss_params=self.config['loss_params'],
                    verbose=5
                )
            except Exception as e:
                print(f"    ERROR in Fold {fold_idx+1}: {e}") 
                return float('inf')

            best_val_loss = min(history['val_loss'])
            fold_losses.append(best_val_loss)
            fold_models.append(model.cpu()) # Move to CPU to save GPU memory
            
            trial.report(best_val_loss, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg_loss = float(np.mean(fold_losses))
        return avg_loss

    def run_tuning(self):
        """Orchestrates a robust Optuna study."""
        study_name = self.config['study_name']
        # Choose a separate Optuna DB per interval. Allow explicit override via config['db_url'].
        db_url = self.config.get('db_url')
        if not db_url:
            interval = self.config.get('time_interval')
            if not interval:
                npz_path = str(self.config.get('npz_path', '')).lower()
                interval = 'daily' if 'daily' in npz_path else 'monthly'
            db_name = f"optuna_{interval}"
            # Default to PC-style connection; override via --db-url for Mac or other platforms
            db_url = f"postgresql://postgres:mysecretpassword@localhost:5432/{db_name}"
            # For Mac, use explicit psycopg2 driver and 127.0.0.1 to match Docker in setup_db.sh
            # db_url = f"postgresql+psycopg2://postgres:mysecretpassword@127.0.0.1:5432/{db_name}"
        storage = db_url
        print(f"Using Optuna storage: {storage}")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='minimize',
            sampler=TPESampler(seed=self.config.get('random_state', 42)),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=True
        )

        # Check how many trials have already been completed by other workers.
        completed_trials = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
        total_target_trials = self.config['n_trials']
        remaining_trials = total_target_trials - completed_trials

        if remaining_trials <= 0:
            print(f"Study '{study_name}' already has {len(study.trials)} trials. Target of {total_target_trials} met. This worker will exit.")
            return study

        print(f"Worker starting. Target: {total_target_trials}, Completed: {completed_trials}. This worker will run up to {remaining_trials} trials.")

        # Let Optuna manage the optimization loop. This is much more robust.
        study.optimize(self.objective, n_trials=remaining_trials, show_progress_bar=True)

        # Reload the study to ensure we have the absolute latest state before saving.
        final_study = optuna.load_study(study_name=study_name, storage=storage)

        print(f"\nTuning completed. Final study has {len(final_study.trials)} trials.")
        print(f"Best trial: {final_study.best_trial.number} with value {final_study.best_trial.value:.6f}")

        self.save_results(final_study)
        return final_study

    def save_results(self, study: optuna.Study):
        """Saves tuning results, including hyperparameters and plots."""
        results_dir = self.config['output_dir']
        best_trial = study.best_trial
        
        # Save best hyperparameters to JSON
        results = {'best_value': best_trial.value, 'best_params': best_trial.params}
        with open(os.path.join(results_dir, 'best_hyperparameters.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Best hyperparameters saved to {results_dir}/best_hyperparameters.json")

        # Save visualizations
        try:
            fig_hist = optuna.visualization.plot_optimization_history(study)
            fig_hist.write_image(os.path.join(results_dir, "optimization_history.png"))
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_image(os.path.join(results_dir, "param_importances.png"))
            print("Saved optimization history and parameter importance plots.")
        except (ImportError, ValueError) as e:
            print(f"Could not save plots. Install plotly and kaleido or disable. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyTorch hyperparameter tuning for LAND rainfall model")
    parser.add_argument("--npz-path", default="ML_Data_Preprocessing/output/assembled_npz/full_training_data_daily_3x3_2km8km_cyclical.npz", help="Path to data")
    parser.add_argument("--output-dir", default="Hyperparameter_Tuning/output/daily_3x3_2km8km_cyclical_attention_deeptemp_1980-1999_2", help="Directory for outputs")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="daily_3x3_2km8km_cyclical_attention_deeptemp_1980-1999_2")
    parser.add_argument("--loss-name", type=str, default="mse")
    parser.add_argument("--loss-params", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-indices-path", type=str, default="Hyperparameter_Tuning/output/daily_3x3_2km8km_cyclical_attention_deeptemp_1980-1999_2/test_indices.pkl", help="Path to save/load test indices (pkl). If omitted, will save to <output-dir>/test_indices.pkl")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible splits and CV")
    parser.add_argument("--db-url", type=str, default=None, help="Optuna database URL")
    parser.add_argument("--time-interval", type=str, default="daily", help="Time interval for the study")

    args = parser.parse_args()

    # Convert the argparse.Namespace to a dictionary for easy use
    config = vars(args)
    
    # Handle JSON parsing for loss_params
    if config['loss_params']:
        try:
            config['loss_params'] = json.loads(config['loss_params'])
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --loss-params JSON: {e}")
    
    # Auto-set loss_params based on loss_name to avoid manual configuration
    if config['loss_params'] is None:  # Only auto-set if not explicitly provided
        if config['loss_name'] == 'mse':
            config['loss_params'] = None
        elif config['loss_name'] == 'weighted_mse':
            # Default parameters for weighted MSE
            config['loss_params'] = {"alpha": 2.0, "power": 1.5, "percentile": 0.95}
    
    # Auto-generate output_dir and test_indices_path if not provided
    if not config.get('output_dir'):
        interval = config.get('time_interval', 'daily')
        config['output_dir'] = f"Hyperparameter_Tuning/output/{interval}_{config['study_name']}"
    
    if not config.get('test_indices_path'):
        config['test_indices_path'] = os.path.join(config['output_dir'], 'test_indices.pkl')

    # Instantiate the class with the config and run the tuning process
    tuner = OptunaTuner(**config)
    tuner.run_tuning()