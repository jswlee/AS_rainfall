import os
import pickle
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def plot_ensemble_test_predictions(data: Dict[str, Any], test_ensemble_pred, output_dir: str) -> None:
    """Save scatter of actual vs predicted rainfall for ensemble on test set.
    Units: mm if metadata['rainfall_mm_std'] > 0 else inches.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    if 'metadata' in data and 'rainfall_mm_std' in data['metadata']:
        rs = float(data['metadata']['rainfall_mm_std'])
        plt.scatter(x=data['targets']['test'] * rs, y=test_ensemble_pred * rs, alpha=0.5)
        plt.plot(
            [data['targets']['test'].min() * rs, data['targets']['test'].max() * rs],
            [data['targets']['test'].min() * rs, data['targets']['test'].max() * rs],
            'r--'
        )
        plt.xlabel('Actual Rainfall (mm)')
        plt.ylabel('Predicted Rainfall (mm)')
        plt.title('Ensemble Model: Actual vs Predicted Rainfall (Test Set)')
    else:
        plt.scatter(x=data['targets']['test'] * 100, y=test_ensemble_pred * 100, alpha=0.5)
        plt.plot(
            [data['targets']['test'].min() * 100, data['targets']['test'].max() * 100],
            [data['targets']['test'].min() * 100, data['targets']['test'].max() * 100],
            'r--'
        )
        plt.xlabel('Actual Rainfall (inches)')
        plt.ylabel('Predicted Rainfall (inches)')
        plt.title('Ensemble Model: Actual vs Predicted Rainfall (Test Set)')
    plt.grid(True)
    plt.savefig(fname=os.path.join(output_dir, 'ensemble_test_predictions.png'), dpi=300)
    plt.close()


def plot_individual_vs_ensemble(data: Dict[str, Any], all_test_predictions: List, test_ensemble_pred, output_dir: str) -> None:
    """Save scatter comparing individual model predictions to ensemble on test set.
    Units: mm if metadata['rainfall_mm_std'] > 0 else inches.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    if 'metadata' in data and 'rainfall_mm_std' in data['metadata']:
        rs = float(data['metadata']['rainfall_mm_std'])
        for i, preds in enumerate(all_test_predictions):
            plt.scatter(x=data['targets']['test'] * rs, y=preds * rs, alpha=0.3, label=f'Model {i+1}')
        plt.scatter(x=data['targets']['test'] * rs, y=test_ensemble_pred * rs, alpha=0.8, color='red', label='Ensemble')
        plt.plot(
            [data['targets']['test'].min() * rs, data['targets']['test'].max() * rs],
            [data['targets']['test'].min() * rs, data['targets']['test'].max() * rs],
            'k--'
        )
        plt.xlabel('Actual Rainfall (mm)')
        plt.ylabel('Predicted Rainfall (mm)')
        plt.title('Individual Models vs Ensemble Predictions')
    else:
        for i, preds in enumerate(all_test_predictions):
            plt.scatter(x=data['targets']['test'] * 100, y=preds * 100, alpha=0.3, label=f'Model {i+1}')
        plt.scatter(x=data['targets']['test'] * 100, y=test_ensemble_pred * 100, alpha=0.8, color='red', label='Ensemble')
        plt.plot(
            [data['targets']['test'].min() * 100, data['targets']['test'].max() * 100],
            [data['targets']['test'].min() * 100, data['targets']['test'].max() * 100],
            'k--'
        )
        plt.xlabel('Actual Rainfall (inches)')
        plt.ylabel('Predicted Rainfall (inches)')
        plt.title('Individual Models vs Ensemble Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname=os.path.join(output_dir, 'individual_vs_ensemble.png'), dpi=300)
    plt.close()


def plot_fold_ensemble_predictions(data: Dict[str, Any], fold_ensemble_pred, fold_dir: str, fold_idx: int) -> None:
    """Save scatter of actual vs predicted for one fold's ensemble predictions.
    Units: mm if metadata['rainfall_mm_std'] > 0 else inches.
    """
    os.makedirs(fold_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    if 'metadata' in data and 'rainfall_mm_std' in data['metadata'] and float(data['metadata']['rainfall_mm_std']) > 0:
        rs = float(data['metadata']['rainfall_mm_std'])
        plt.scatter(x=data['targets']['test'] * rs, y=fold_ensemble_pred * rs, alpha=0.5)
        plt.plot(
            [data['targets']['test'].min() * rs, data['targets']['test'].max() * rs],
            [data['targets']['test'].min() * rs, data['targets']['test'].max() * rs],
            'r--'
        )
        plt.xlabel('Actual Rainfall (mm)')
        plt.ylabel('Predicted Rainfall (mm)')
    else:
        plt.scatter(x=data['targets']['test'] * 100, y=fold_ensemble_pred * 100, alpha=0.5)
        plt.plot(
            [data['targets']['test'].min() * 100, data['targets']['test'].max() * 100],
            [data['targets']['test'].min() * 100, data['targets']['test'].max() * 100],
            'r--'
        )
        plt.xlabel('Actual Rainfall (inches)')
        plt.ylabel('Predicted Rainfall (inches)')
    plt.title(f'Fold {fold_idx+1} Ensemble: Actual vs Predicted Rainfall')
    plt.grid(True)
    plt.savefig(fname=os.path.join(fold_dir, 'fold_ensemble_predictions.png'), dpi=300)
    plt.close()


# File I/O helpers

def save_progress(progress_file: str, completed_models: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'wb') as f:
        pickle.dump(obj={'completed_models': completed_models}, file=f)


def write_training_summary(model_dir: str, model_idx: int, fold_idx: int, random_seed: int,
                           hyperparams: Dict[str, Any], history: Dict[str, Any], metrics: Dict[str, Any],
                           use_mm: bool, rainfall_std: float | None) -> str:
    os.makedirs(model_dir, exist_ok=True)
    summary_path = os.path.join(model_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary for Model {model_idx} of Fold {fold_idx}\n")
        f.write(f"Random Seed: {random_seed}\n\n")
        f.write("Hyperparameters:\n")
        for key, value in hyperparams.items():
            if key not in ['Best hyperparameters from 100 trials']:
                f.write(f"  {key}: {value}\n")
        f.write(f"\nTraining Results:\n")
        if use_mm and rainfall_std and rainfall_std > 0:
            f.write(f"  Final Loss: {history['loss'][-1]*(rainfall_std**2):.6f} mm²\n")
            f.write(f"  Final MAE: {history['mae'][-1]*rainfall_std:.6f} mm\n")
            f.write(f"  Final Val Loss: {history['val_loss'][-1]*(rainfall_std**2):.6f} mm²\n")
            f.write(f"  Final Val MAE: {history['val_mae'][-1]*rainfall_std:.6f} mm\n\n")
            f.write(f"Test Metrics:\n")
            f.write(f"  R²: {metrics['r2']:.4f}\n")
            f.write(f"  RMSE: {metrics.get('rmse_mm', float('nan')):.4f} mm\n")
            f.write(f"  MAE: {metrics.get('mae_mm', float('nan')):.4f} mm\n")
        else:
            f.write(f"  Final Loss: {history['loss'][-1]*100*100:.6f} in²\n")
            f.write(f"  Final MAE: {history['mae'][-1]*100:.6f} in\n")
            f.write(f"  Final Val Loss: {history['val_loss'][-1]*100*100:.6f} in²\n")
            f.write(f"  Final Val MAE: {history['val_mae'][-1]*100:.6f} in\n\n")
            f.write(f"Test Metrics:\n")
            f.write(f"  R²: {metrics['r2']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']*100:.4f} in\n")
            f.write(f"  MAE: {metrics['mae']*100:.4f} in\n")
    return summary_path


def write_fold_summary(fold_dir: str, fold_idx: int, n_models_per_fold: int, fold_r2: float,
                       fold_rmse: float, fold_mae: float, use_mm: bool, rainfall_std: float | None) -> str:
    os.makedirs(fold_dir, exist_ok=True)
    fold_summary_path = os.path.join(fold_dir, 'fold_summary.txt')
    with open(fold_summary_path, 'w') as f:
        f.write(f"Fold {fold_idx} Ensemble Summary\n")
        f.write(f"Number of Models: {n_models_per_fold}\n\n")
        f.write(f"Test Metrics:\n")
        if use_mm and rainfall_std and rainfall_std > 0:
            f.write(f"  R²: {fold_r2:.4f}\n")
            f.write(f"  RMSE: {(fold_rmse*rainfall_std):.4f} mm\n")
            f.write(f"  MAE: {(fold_mae*rainfall_std):.4f} mm\n")
        else:
            f.write(f"  R²: {fold_r2:.4f}\n")
            f.write(f"  RMSE: {fold_rmse*100:.4f} in\n")
            f.write(f"  MAE: {fold_mae*100:.4f} in\n")
    return fold_summary_path


def write_test_predictions_csv(output_dir: str, data: Dict[str, Any], test_ensemble_pred) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if 'metadata' in data and 'rainfall_mm_std' in data['metadata']:
        rs = float(data['metadata']['rainfall_mm_std'])
        test_pred_df = pd.DataFrame({
            'actual_mm': (data['targets']['test']*rs).flatten(),
            'predicted_mm': (test_ensemble_pred*rs).flatten()
        })
    else:
        test_pred_df = pd.DataFrame({
            'actual_inches': (data['targets']['test']*100).flatten(),
            'predicted_inches': (test_ensemble_pred*100).flatten()
        })
    out_path = os.path.join(output_dir, 'test_predictions.csv')
    test_pred_df.to_csv(path_or_buf=out_path, index=False)
    return out_path


def write_ensemble_summary(output_dir: str, n_folds: int, n_models_per_fold: int, hyperparams: Dict[str, Any],
                           fold_results: List[Dict[str, Any]], avg_r2: float, avg_rmse: float, avg_mae: float,
                           test_r2: float, test_rmse: float, test_mae: float, data: Dict[str, Any], training_time_sec: float) -> str:
    os.makedirs(output_dir, exist_ok=True)
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
        if 'metadata' in data and 'rainfall_mm_std' in data['metadata']:
            rs = float(data['metadata']['rainfall_mm_std'])
            for i, fold in enumerate(fold_results):
                f.write(f"  Fold {i+1}: R² = {fold['r2']:.4f}, RMSE = {(fold['rmse']*rs):.4f} mm, MAE = {(fold['mae']*rs):.4f} mm\n")
            f.write(f"\nAverage CV: R² = {avg_r2:.4f}, RMSE = {(avg_rmse*rs):.4f} mm, MAE = {(avg_mae*rs):.4f} mm\n")
        else:
            for i, fold in enumerate(fold_results):
                f.write(f"  Fold {i+1}: R² = {fold['r2']:.4f}, RMSE = {fold['rmse']:.4f} in, MAE = {fold['mae']:.4f} in\n")
            f.write(f"\nAverage CV: R² = {avg_r2:.4f}, RMSE = {avg_rmse:.4f} in, MAE = {avg_mae:.4f} in\n")

        f.write(f"\nFinal Ensemble Test Results:\n")
        f.write(f"  R²: {test_r2:.4f}\n")
        if 'metadata' in data and 'rainfall_mm_std' in data['metadata']:
            rs = float(data['metadata']['rainfall_mm_std'])
            f.write(f"  RMSE: {(test_rmse*rs):.4f} mm\n")
            f.write(f"  MAE: {(test_mae*rs):.4f} mm\n")
        else:
            f.write(f"  RMSE: {test_rmse*100:.4f} in\n")
            f.write(f"  MAE: {test_mae*100:.4f} in\n")

        import time as _time
        f.write(f"\nTraining completed in {_time.strftime('%H:%M:%S', _time.gmtime(training_time_sec))}\n")

    return summary_path
