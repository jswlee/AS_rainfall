import os
import json
import numpy as np
import torch
import argparse
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, Any
import shutil
import csv

from Hyperparameter_Tuning.data_utils_simplified import DataManager, create_pytorch_dataloaders, RainfallDataset
from Hyperparameter_Tuning.model import create_model_from_hyperparams
from Hyperparameter_Tuning.model_training import train_model

class OptunaTuner:
    """Optuna-based hyperparameter tuner."""
    def __init__(self, **kwargs):
        self.config = kwargs
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Setup device (prefer CUDA, then MPS, otherwise CPU)
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        device_type = 'cuda' if torch.cuda.is_available() else ('mps' if has_mps else 'cpu')
        self.device = torch.device(device_type)

        print(f"--- Using device: {self.device} ---")

        # Instantiate the DataManager to handle all data logic
        data_manager = DataManager(device=self.device, **self.config)

        # Persist the exact test indices used by the tuner for reproducibility
        self._save_test_indices(data_manager)

        # Get the metadata and the specific tensors needed for cross-validation
        self.metadata = data_manager.metadata
        self.cv_tensors, self.cv_indices = data_manager.get_cv_tensors()
        self.data_manager = data_manager  # Store for single-fold case

        # Persist the exact hyperparameter search space used for this run
        self._save_init_search_space()

        if self.config['n_folds'] > 1:
            self._prepare_cv_splits()
        else:
            # For single fold, we'll use the train/val split from DataManager
            self._cv_splits = None

    def _prepare_cv_splits(self):
        """Pre-computes temporal CV splits where validation occurs AFTER training in time.

        This avoids leakage from future conditions into the training set.
        Splits are created on grouped timestamps (year-month[-day]) so that all
        samples from the same timestamp stay together.
        """
        years = getattr(self.data_manager, 'years', None)
        months = getattr(self.data_manager, 'months', None)
        days = getattr(self.data_manager, 'days', None)
        if years is None or months is None:
            raise ValueError("Temporal CV requires 'years' and 'months' arrays in the NPZ.")

        time_interval = self.config.get('time_interval', 'daily')
        if time_interval == 'daily' and days is not None:
            group_all = (years.astype(np.int64) * 10000) + (months.astype(np.int64) * 100) + days.astype(np.int64)
        else:
            group_all = (years.astype(np.int64) * 100) + months.astype(np.int64)

        group_cv = group_all[self.cv_indices]
        unique_groups = np.unique(group_cv)
        unique_groups.sort()

        n_folds = int(self.config['n_folds'])
        if len(unique_groups) <= n_folds:
            raise ValueError(
                f"Not enough unique time groups ({len(unique_groups)}) for n_folds={n_folds}."
            )

        # Expanding-window splits across time groups.
        # Fold i uses earlier groups for train, immediately following block for val.
        self._cv_splits = []
        total = len(unique_groups)
        for i in range(n_folds):
            train_end = int(((i + 1) * total) / (n_folds + 1))
            val_end = int(((i + 2) * total) / (n_folds + 1))

            train_groups = unique_groups[:train_end]
            val_groups = unique_groups[train_end:val_end]

            if len(train_groups) == 0 or len(val_groups) == 0:
                continue

            train_mask = np.isin(group_cv, train_groups)
            val_mask = np.isin(group_cv, val_groups)

            train_original_indices = self.cv_indices[train_mask]
            val_original_indices = self.cv_indices[val_mask]
            self._cv_splits.append((train_original_indices, val_original_indices))

        if len(self._cv_splits) != n_folds:
            raise ValueError(
                f"Temporal CV produced {len(self._cv_splits)} folds, expected {n_folds}."
            )

        print(f"Pre-computed {n_folds} temporal CV splits (train before val).")

    def _save_test_indices(self, data_manager: DataManager) -> None:
        """Persist the test indices to a stable location for reproducibility."""
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

    def _save_init_search_space(self) -> None:
        """Persist the hyperparameter search space at initialization time.

        This captures the exact HP ranges/choices used for this run, tied to
        the current config and metadata.
        """
        try:
            search_space = self._get_search_space_definition()
            search_space_path = os.path.join(self.config['output_dir'], 'search_space.json')
            with open(search_space_path, 'w') as f:
                json.dump(
                    {
                        'search_space': search_space,
                        'time_interval': self.config.get('time_interval', 'daily'),
                        'loss_name': self.config.get('loss_name', 'mse'),
                        'num_climate_vars': int(self.metadata.get('num_climate_vars', 1)),
                    },
                    f,
                    indent=2,
                )
            print(f"Saved hyperparameter search space to {search_space_path}")
        except Exception as e:
            print(f"Warning: Could not save hyperparameter search space: {e}")

    def _get_search_space_definition(self) -> Dict[str, Dict[str, Any]]:
        """Returns the hyperparameter search space definition based on time interval.

        If loss_name is 'tweedie', constrain output_activation to 'softplus' to ensure
        strictly non-negative outputs compatible with Tweedie loss.
        """
        time_interval = self.config.get('time_interval', 'daily')
        loss_name = self.config.get('loss_name', 'mse')
        num_climate_vars = int(self.metadata.get('num_climate_vars', 1))

        def _ceil_to_multiple(x: int, m: int) -> int:
            return ((x + m - 1) // m) * m

        def _floor_to_multiple(x: int, m: int) -> int:
            return (x // m) * m
        
        if time_interval == 'daily':
            climate_low = _ceil_to_multiple(900, num_climate_vars)
            climate_high = _floor_to_multiple(1200, num_climate_vars)
            if climate_low > climate_high:
                raise ValueError(
                    f"Invalid climate_units range after enforcing divisibility: low={climate_low}, high={climate_high}, num_climate_vars={num_climate_vars}"
                )
            return {
                'climate_units': {'type': 'int', 'low': climate_low, 'high': climate_high, 'step': num_climate_vars},
                'local_dem_units': {'type': 'int', 'low': 16, 'high': 64, 'step': 16},
                'regional_dem_units': {'type': 'int', 'low': 8, 'high': 32, 'step': 8},
                'temporal_units': {'type': 'int', 'low': 4, 'high': 12, 'step': 4},
                'na': {'type': 'int', 'low': 256, 'high': 1024, 'step': 256},
                'nb': {'type': 'int', 'low': 64, 'high': 160, 'step': 16},
                'dropout_rate': {'type': 'float', 'low': 0.2, 'high': 0.35, 'step': 0.05},
                'learning_rate': {'type': 'float', 'low': 3e-5, 'high': 1e-4, 'log': True},
                'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True},
                'batch_size': {'type': 'categorical', 'choices': [64, 128, 256, 512,1024]},
                'use_residual': {'type': 'categorical', 'choices': [False]},
                'climate_activation': {'type': 'categorical', 'choices': ['relu']},
                'output_activation': {'type': 'categorical', 'choices': ['softplus']},
                'climate_processing': {'type': 'categorical', 'choices': ['conv2d']},
            }
        else:
            climate_low = _ceil_to_multiple(64, num_climate_vars)
            climate_high = _floor_to_multiple(512, num_climate_vars)
            return {
                'climate_units': {'type': 'int', 'low': climate_low, 'high': climate_high, 'step': num_climate_vars},
                'local_dem_units': {'type': 'int', 'low': 16, 'high': 1024, 'step': 16},
                'regional_dem_units': {'type': 'int', 'low': 32, 'high': 512, 'step': 16},
                'temporal_units': {'type': 'int', 'low': 16, 'high': 64, 'step': 8},
                'na': {'type': 'int', 'low': 128, 'high': 512, 'step': 64},
                'nb': {'type': 'int', 'low': 32, 'high': 512, 'step': 32},
                'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.05},
                'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
                'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True},
                'batch_size': {'type': 'categorical', 'choices': [256, 512, 1024, 2048]},
                'use_residual': {'type': 'categorical', 'choices': [True, False]},
                'climate_activation': {'type': 'categorical', 'choices': ['relu', 'none']},
                'output_activation': {'type': 'categorical', 'choices': ['softplus'] if loss_name == 'tweedie' else ['relu', 'softplus']},
                'climate_processing': {'type': 'categorical', 'choices': ['flatten', 'conv2d']},
            }

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Defines the hyperparameter search space for Optuna."""
        search_space = self._get_search_space_definition()
        
        hyperparams = {}
        for name, definition in search_space.items():
            if definition['type'] == 'int':
                hyperparams[name] = trial.suggest_int(name, definition['low'], definition['high'], step=definition['step'])
            elif definition['type'] == 'float':
                # Check if step is present (linear float) or use log scale
                if 'step' in definition:
                    hyperparams[name] = trial.suggest_float(name, definition['low'], definition['high'], step=definition['step'])
                else:
                    # Log-scaled float without step
                    hyperparams[name] = trial.suggest_float(name, definition['low'], definition['high'], log=definition.get('log', False))
            elif definition['type'] == 'categorical':
                hyperparams[name] = trial.suggest_categorical(name, definition['choices'])
        
        print(f"\nTrial {trial.number}: Starting with params: {hyperparams}")
        return hyperparams

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for one Optuna trial, performing cross-validation."""
        hyperparams = self.suggest_hyperparameters(trial)
        
        fold_losses, fold_models = [], []
        
        # Handle single fold vs multi-fold
        if self._cv_splits is None:
            # Single fold: use train/val split from DataManager
            cv_iterations = [(0, (self.data_manager.indices['train'], self.data_manager.indices['val']))]
        else:
            # Multi-fold: use CV splits
            cv_iterations = enumerate(self._cv_splits)
        
        for fold_idx, (train_idx, val_idx) in cv_iterations:
            # Train-only target scaling computed from the primary train split (DataManager)
            target_scale = getattr(self.data_manager, 'target_scale', None)
            # Create datasets by passing the shared tensors and the specific indices for this fold
            train_ds = RainfallDataset(self.cv_tensors, train_idx, target_scale=target_scale)
            val_ds = RainfallDataset(self.cv_tensors, val_idx, target_scale=target_scale)
            
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

    def _save_search_space(self, results_dir: str):
        """Extract and save the hyperparameter search space for reproducibility."""
        search_space = self._get_search_space_definition()
        output_path = os.path.join(results_dir, 'search_space.json')
        with open(output_path, 'w') as f:
            json.dump(search_space, f, indent=4)
        print(f"Search space saved to {output_path}")

    def _save_all_trials_csv(self, study: optuna.Study, results_dir: str):
        """Save all completed trials to CSV (without rank column)."""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            print("No completed trials to save.")
            return
        
        # Collect all unique param names
        param_names = set()
        for trial in completed_trials:
            param_names.update(trial.params.keys())
        param_names = sorted(param_names)
        
        output_path = os.path.join(results_dir, 'all_trials.csv')
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: trial_number, trial_id, objective_value, then all params
            header = ['trial_number', 'trial_id', 'objective_value'] + param_names
            writer.writerow(header)
            
            # Write each trial
            for trial in completed_trials:
                row = [
                    trial.number,
                    trial._trial_id,
                    trial.value
                ]
                # Add param values in same order as header
                for param_name in param_names:
                    row.append(trial.params.get(param_name, ''))
                writer.writerow(row)
        
        print(f"All trials saved to {output_path} ({len(completed_trials)} trials)")

    def _copy_model_architecture(self, results_dir: str):
        """Copy model.py to output directory for reproducibility."""
        source = os.path.join('Hyperparameter_Tuning', 'model.py')
        dest = os.path.join(results_dir, 'model_architecture.py')
        
        try:
            shutil.copy2(source, dest)
            print(f"Model architecture copied to {dest}")
        except Exception as e:
            print(f"Warning: Could not copy model.py: {e}")

    def save_results(self, study: optuna.Study):
        """Saves tuning results, including hyperparameters and plots."""
        results_dir = self.config['output_dir']
        best_trial = study.best_trial
        
        # Save best hyperparameters to JSON
        results = {'best_value': best_trial.value, 'best_params': best_trial.params}
        with open(os.path.join(results_dir, 'best_hyperparameters.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Best hyperparameters saved to {results_dir}/best_hyperparameters.json")

        # Save search space for reproducibility
        self._save_search_space(results_dir)
        
        # Save all trials to CSV
        self._save_all_trials_csv(study, results_dir)
        
        # Copy model architecture
        self._copy_model_architecture(results_dir)

        # Save visualizations
        try:
            fig_hist = optuna.visualization.plot_optimization_history(study)
            fig_hist.write_image(os.path.join(results_dir, "optimization_history.png"))
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_image(os.path.join(results_dir, "param_importances.png"))
            print("Saved optimization history and parameter importance plots.")
        except (ImportError, ValueError) as e:
            print(f"Could not save plots. Install plotly and kaleido or disable. Error: {e}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyTorch hyperparameter tuning for LAND rainfall model")
    parser.add_argument("--npz-path", default="ML_Data_Preprocessing/output/assembled_npz/full_training_data_daily_3x3_2km8km_one_hot.npz", help="Path to data")
    # Let output_dir be optional; if omitted, it will be auto-generated from study_name and time_interval
    parser.add_argument("--output-dir", default=None, help="Directory for outputs (if omitted, derived from study name and time interval)")
     # Let test_indices_path be optional; if omitted, it will default to <output_dir>/test_indices.pkl
    parser.add_argument("--test-indices-path", type=str, default=None, help="Path to save/load test indices (pkl). If omitted, will save to <output-dir>/test_indices.pkl")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="3x3_2km8km_one_hot_1980-2024")
    parser.add_argument("--loss-name", type=str, default="mse")
    parser.add_argument("--loss-params", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
   
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