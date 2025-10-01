# hp_tuning_final.py

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

# Import the simplified, robust MLflow logger
from Hyperparameter_Tuning.mlflow_utils_simplified import MLflowLogger, MLFLOW_AVAILABLE

# Direct MLflow imports for model registration and callbacks
try:
    from optuna.integration import MLflowCallback
except ImportError:
    MLflowCallback = None
try:
    import mlflow
except ImportError:
    mlflow = None

from Hyperparameter_Tuning.data_utils_simplified import DataManager, create_pytorch_dataloaders, RainfallDataset
from Hyperparameter_Tuning.model import create_model_from_hyperparams
from Hyperparameter_Tuning.model_training import train_model

class OptunaTuner:
    """Optuna-based hyperparameter tuner with simplified, robust MLflow integration."""
    def __init__(self, **kwargs):
        # Store all configuration parameters in a dictionary for easy access
        self.config = kwargs
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Setup device
        if torch.cuda.is_available(): self.device = torch.device('cuda')
        elif torch.backends.mps.is_available(): self.device = torch.device('mps')
        else: self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # MLflow setup
        self.enable_mlflow = bool(self.config.get('enable_mlflow') and MLFLOW_AVAILABLE)
        mlflow_experiment = self.config.get('mlflow_experiment') or "AS_Rainfall_Hyperparameter_Tuning"
        self.mlflow_logger = MLflowLogger(
            experiment_name=mlflow_experiment,
            enabled=self.enable_mlflow
        )
        
        # Instantiate the DataManager to handle all data logic
        data_manager = DataManager(**self.config)
        
        # Get the metadata and the specific tensors needed for cross-validation
        self.metadata = data_manager.metadata
        self.cv_tensors, self.cv_indices = data_manager.get_cv_tensors()

        self._prepare_cv_splits()

    def _prepare_cv_splits(self):
        """Pre-computes Stratified K-Fold splits based on target quantiles."""
        n_bins = 5
        y = self.cv_tensors['targets'][self.cv_indices].numpy().ravel()
        try:
            edges = np.quantile(y[y > 0], np.linspace(0, 1, n_bins + 1))
            edges = np.unique(edges)
            if len(edges) < 2: raise ValueError("Not enough unique quantile edges.")
            y_bins = np.digitize(y, edges[1:-1])
        except Exception as e:
            print(f"Warning: Stratified binning failed ({e}). Falling back to non-stratified CV.")
            y_bins = np.zeros_like(y, dtype=int)
            
        skf = StratifiedKFold(n_splits=self.config['n_folds'], shuffle=True, random_state=self.config.get('random_state', 42))
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
        return {
            # Climate variables
            'climate_units': trial.suggest_int(name='climate_units', low=64, high=512, step=32),
            'local_dem_units': trial.suggest_int(name='local_dem_units', low=16, high=256, step=16),
            'regional_dem_units': trial.suggest_int(name='regional_dem_units', low=32, high=128, step=16),
            'month_units': trial.suggest_int(name='month_units', low=16, high=64, step=8),
            
            # Neural network architecture
            'na': trial.suggest_int(name='na', low=128, high=512, step=64),
            'nb': trial.suggest_int(name='nb', low=64, high=1024, step=64),
            
            # Regularization parameters
            'dropout_rate': trial.suggest_float(name='dropout_rate', low=0.0, high=0.5, step=0.05),
            'l2_reg': trial.suggest_float(name='l2_reg', low=1e-5, high=1e-3, log=True),
            
            # Training parameters
            'learning_rate': trial.suggest_float(name='learning_rate', low=1e-4, high=1e-2, log=True),
            'weight_decay': trial.suggest_float(name='weight_decay', low=1e-6, high=1e-3, log=True),
            'batch_size': trial.suggest_categorical(name='batch_size', choices=[16, 32, 64, 128, 256]),
            
            # Model architecture choices
            'use_residual': trial.suggest_categorical(name='use_residual', choices=[True, False]),
            'climate_activation': trial.suggest_categorical(name='climate_activation', choices=['relu', 'none']),
            'output_activation': trial.suggest_categorical(name='output_activation', choices=['relu', 'softplus']),
            'climate_processing': trial.suggest_categorical(name='climate_processing', choices=['flatten', 'conv2d'])
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for one Optuna trial, performing cross-validation."""
        hyperparams = self.suggest_hyperparameters(trial)
        print(f"\nTrial {trial.number}: Starting with params: {hyperparams}")

        with self.mlflow_logger.start_run(run_name=f"trial_{trial.number}") as trial_logger:
            if trial_logger.enabled:
                # Log hyperparameters with a clean prefix
                params_to_log = {f"hp_{k}": v for k, v in hyperparams.items()}
                params_to_log["trial_number"] = trial.number
                trial_logger.log_params(params_to_log)
                trial_logger.set_tags({"optuna_trial_number": trial.number, "status": "running"})
                trial.set_user_attr("mlflow_run_id", trial_logger.get_run_id())

            fold_losses, fold_models = [], []
            for fold_idx, (train_idx, val_idx) in enumerate(self._cv_splits):
                print(f"  Fold {fold_idx+1}/{self.config['n_folds']}: Training...")
                
                # Create datasets by passing the shared tensors and the specific indices for this fold
                train_ds = RainfallDataset(self.cv_tensors, train_idx)
                val_ds = RainfallDataset(self.cv_tensors, val_idx)
                
                # Dataloader creation now pass dataloader-specific params from config
                dataloader_params = {k: v for k, v in self.config.items() if k in ['num_workers', 'pin_memory']}
                dataloaders = create_pytorch_dataloaders(
                    {'train': train_ds, 'val': val_ds}, 
                    batch_size=hyperparams['batch_size'], 
                    **dataloader_params
                )

                model = create_model_from_hyperparams(hyperparams, self.metadata).to(self.device)
                
                if trial_logger.enabled and fold_idx == 0:
                    trial_logger.log_model_summary(model, "model_architecture.txt")

                try:
                    history = train_model(
                        model=model, dataloaders=dataloaders, device=self.device,
                        epochs=self.config['max_epochs'], patience=self.config['patience'],
                        learning_rate=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'],
                        loss_name=self.config['loss_name'], loss_params=self.config['loss_params'],
                        verbose=5
                    )
                except Exception as e:
                    print(f"    ERROR in Fold {fold_idx+1}: {e}")
                    if trial_logger.enabled: trial_logger.set_tag("status", "failed")
                    return float('inf')

                best_val_loss = min(history['val_loss'])
                fold_losses.append(best_val_loss)
                fold_models.append(model.cpu()) # Move to CPU to save GPU memory
                
                trial.report(best_val_loss, fold_idx)
                if trial.should_prune():
                    if trial_logger.enabled: trial_logger.set_tag("status", "pruned")
                    raise optuna.TrialPruned()

            avg_loss = float(np.mean(fold_losses))
            if trial_logger.enabled:
                trial_logger.log_metrics({
                    "final_avg_val_loss": avg_loss,
                    "final_std_val_loss": float(np.std(fold_losses)),
                })
                trial_logger.set_tag("status", "completed")
                
                best_fold_idx = np.argmin(fold_losses)
                best_model = fold_models[best_fold_idx]
                trial.set_user_attr("best_fold", best_fold_idx + 1)

                # Use the new, safe logger method to log the model
                example_input = tuple(t[0].unsqueeze(0) for t in val_ds.tensors if t.shape)
                trial_logger.log_pytorch_model(best_model, name="best_model_in_trial", input_example=example_input[0])
            
            return avg_loss

    def run_tuning(self):
        """Orchestrates the Optuna hyperparameter tuning study."""
        study_name = self.config['study_name']
        storage_path = os.path.join(self.config['output_dir'], f"{study_name}.db")
        
        study = optuna.create_study(
            study_name=study_name, storage=f"sqlite:///{storage_path}", direction='minimize',
            sampler=TPESampler(seed=self.config.get('random_state', 42)),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=self.config['resume']
        )
        
        print(f"Starting/Resuming study '{study_name}' with {self.config['n_trials']} trials...")
        study.optimize(self.objective, n_trials=self.config['n_trials'], show_progress_bar=True)

        print(f"Tuning completed. Best trial: {study.best_trial.number} with value {study.best_trial.value:.6f}")
        self.save_results(study)
        self.register_best_model(study, "land_rainfall_model")
        return study

    def register_best_model(self, study: optuna.Study, model_name: str):
        """Registers the best model from the study to the MLflow Model Registry."""
        if not self.enable_mlflow:
            print("MLflow not enabled; skipping model registration.")
            return

        run_id = study.best_trial.user_attrs.get("mlflow_run_id")
        if not run_id:
            print("Warning: No MLflow run ID found in best trial; cannot register model.")
            return

        try:
            model_uri = f"runs:/{run_id}/best_model_in_trial"
            print(f"Registering model from URI: {model_uri}")
            model_version = mlflow.register_model(model_uri, model_name)
            print(f"Successfully registered model '{model_name}' version {model_version.version}.")
            
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(name=model_name, alias="champion", version=model_version.version)
            print(f"Set alias 'champion' for model version {model_version.version}.")
        except Exception as e:
            print(f"Error registering model: {e}")

    def save_results(self, study: optuna.Study):
        """Saves tuning results, including hyperparameters and plots."""
        results_dir = self.output_dir
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
    # Add all arguments as before...
    parser.add_argument("--npz-path", default="ML_Data_Preprocessing/output/assembled_npz/full_training_data_monthly.npz", help="Path to data")
    parser.add_argument("--output-dir", default="output/tuning", help="Directory for outputs")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="land_model_tuning")
    parser.add_argument("--loss-name", type=str, default="mse")
    parser.add_argument("--loss-params", type=str, default=None)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    # Convert the argparse.Namespace to a dictionary for easy use
    config = vars(args)
    
    # Handle JSON parsing for loss_params
    if config['loss_params']:
        try:
            config['loss_params'] = json.loads(config['loss_params'])
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --loss-params JSON: {e}")

    # Instantiate the class with the config and run the tuning process
    tuner = OptunaTuner(**config)
    tuner.run_tuning()