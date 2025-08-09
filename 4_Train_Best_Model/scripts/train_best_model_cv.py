#!/usr/bin/env python3
"""
K-fold CV training using best hyperparameters on the assembled NPZ dataset.
Holds out the persisted test set; splits train+val into K folds.
Saves per-fold metrics (in millimeters) and an overall summary.
"""
import os
import sys
import time
import numpy as np
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))

# Python path
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'scripts'))

# Imports
from sklearn.model_selection import KFold
import tensorflow as tf

from npz_data_utils import load_assembled_npz_data
from model_utils import build_model, load_best_hyperparameters
from training import train_model, evaluate_model, plot_training_history

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def create_tf_datasets_from_arrays(split_dict, batch_size=128, drop_remainder=False):
    import tensorflow as tf
    def ds_for(split):
        return (
            tf.data.Dataset.from_tensor_slices((
                {
                    'climate': split_dict['climate'][split],
                    'local_dem': split_dict['local_dem'][split],
                    'regional_dem': split_dict['regional_dem'][split],
                    'month': split_dict['month'][split],
                },
                split_dict['targets'][split]
            ))
            .shuffle(10000) if split == 'train' else tf.data.Dataset.from_tensor_slices((
                {
                    'climate': split_dict['climate'][split],
                    'local_dem': split_dict['local_dem'][split],
                    'regional_dem': split_dict['regional_dem'][split],
                    'month': split_dict['month'][split],
                },
                split_dict['targets'][split]
            ))
        )
    def batched(ds, split_name):
        # Ensure evaluation sets are not empty even if smaller than batch_size
        dr = drop_remainder if split_name == 'train' else False
        return ds.batch(batch_size, drop_remainder=dr).prefetch(tf.data.AUTOTUNE)
    return {
        'train': batched(ds_for('train'), 'train'),
        'val': batched(ds_for('val'), 'val'),
        'test': batched(ds_for('test'), 'test'),
    }


def run_cv_training(config: dict | None = None):
    """Run K-fold CV training using best hyperparameters; no CLI required.

    Parameters
    - config: optional dict overriding defaults. Keys:
      output_dir, n_folds, epochs, batch_size, test_indices_path.
    """
    defaults = {
        'output_dir': os.path.join(PIPELINE_DIR, 'output', 'land_model_best_cv'),
        'n_folds': 10,
        'epochs': 150,
        'batch_size': 64,
        'test_indices_path': os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'output_test', 'test_indices.pkl'),
        'hp_dir': os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'output_test'),
    }
    cfg = {**defaults, **(config or {})}

    os.makedirs(cfg['output_dir'], exist_ok=True)

    # Load NPZ
    npz_path = os.path.join(PROJECT_ROOT, 'ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found at {npz_path}")
    data = load_assembled_npz_data(
        npz_path=npz_path,
        test_indices_path=cfg['test_indices_path'],
        test_size=0.1,
        val_size=0.1,
        random_state=SEED,
    )

    # Build CV pool from original train+val
    cv_features = {
        'climate': np.concatenate([data['climate']['train'], data['climate']['val']]),
        'local_dem': np.concatenate([data['local_dem']['train'], data['local_dem']['val']]),
        'regional_dem': np.concatenate([data['regional_dem']['train'], data['regional_dem']['val']]),
        'month': np.concatenate([data['month']['train'], data['month']['val']]),
    }
    cv_targets = np.concatenate([data['targets']['train'], data['targets']['val']])
    N_cv = cv_targets.shape[0]

    # Static test set (held-out)
    heldout_test = {
        'climate': data['climate']['test'],
        'local_dem': data['local_dem']['test'],
        'regional_dem': data['regional_dem']['test'],
        'month': data['month']['test'],
        'targets': data['targets']['test'],
    }

    # Hyperparameters are loaded inside build_model() from hp_dir; no explicit load here.

    # CV
    kf = KFold(n_splits=cfg['n_folds'], shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(N_cv)), start=1):
        print(f"\n===== Fold {fold}/{cfg['n_folds']} =====")
        fold_dir = os.path.join(cfg['output_dir'], f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Build split dict arrays
        split = {
            'climate': {
                'train': cv_features['climate'][train_idx],
                'val': cv_features['climate'][val_idx],
                'test': heldout_test['climate'],
            },
            'local_dem': {
                'train': cv_features['local_dem'][train_idx],
                'val': cv_features['local_dem'][val_idx],
                'test': heldout_test['local_dem'],
            },
            'regional_dem': {
                'train': cv_features['regional_dem'][train_idx],
                'val': cv_features['regional_dem'][val_idx],
                'test': heldout_test['regional_dem'],
            },
            'month': {
                'train': cv_features['month'][train_idx],
                'val': cv_features['month'][val_idx],
                'test': heldout_test['month'],
            },
            'targets': {
                'train': cv_targets[train_idx],
                'val': cv_targets[val_idx],
                'test': heldout_test['targets'],
            },
            'metadata': data['metadata'],
        }

        # Datasets
        datasets = create_tf_datasets_from_arrays(split, batch_size=cfg['batch_size'], drop_remainder=True)

        # Model (explicit hp_dir)
        model = build_model(data['metadata'], cfg['hp_dir'])

        # Train
        start_time = time.time()
        history = train_model(
            model=model,
            data=datasets,
            output_dir=fold_dir,
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
        )
        train_time = time.time() - start_time
        print(f"Fold {fold} training time: {train_time:.2f}s")

        # Evaluate on held-out test set; convert metrics to millimeters using stored std
        rainfall_std = float(data['metadata'].get('rainfall_mm_std', 1.0))
        metrics = evaluate_model(model, data=datasets, output_dir=fold_dir, rainfall_std=rainfall_std)
        fold_metrics.append(metrics)

        # Plot history
        plot_training_history(history, fold_dir, rainfall_std=rainfall_std)

        # Save brief fold summary
        with open(os.path.join(fold_dir, 'fold_summary.txt'), 'w') as f:
            f.write(f"Fold {fold} Summary\n")
            f.write("===================\n\n")
            f.write(f"RMSE (mm): {metrics.get('rmse_mm', float('nan')):.6f}\n")
            f.write(f"MAE  (mm): {metrics.get('mae_mm', float('nan')):.6f}\n")
            f.write(f"MSE (mm²): {metrics.get('mse_mm2', float('nan')):.6f}\n")
            f.write(f"R²:        {metrics.get('r2', float('nan')):.6f}\n")
            f.write(f"Training time (s): {train_time:.2f}\n")

    # Aggregate summary
    import pandas as pd
    df = pd.DataFrame([
        {
            'rmse_mm': m.get('rmse_mm', np.nan),
            'mae_mm': m.get('mae_mm', np.nan),
            'mse_mm2': m.get('mse_mm2', np.nan),
            'r2': m.get('r2', np.nan),
        } for m in fold_metrics
    ])
    summary = df.mean().to_dict()
    summary_path = os.path.join(cfg['output_dir'], 'cv_summary.csv')
    df.to_csv(os.path.join(cfg['output_dir'], 'cv_fold_metrics.csv'), index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"\nCV summary saved to {summary_path}")
    print("Averages (mm):", summary)

    # Write human-readable summary
    with open(os.path.join(cfg['output_dir'], 'cv_summary.txt'), 'w') as f:
        f.write("CV Training Summary\n")
        f.write("====================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folds: {cfg['n_folds']}\n")
        f.write(f"Epochs: {cfg['epochs']}\n")
        f.write(f"Batch size: {cfg['batch_size']}\n\n")
        f.write("Averaged metrics (mm):\n")
        f.write(f"  RMSE: {summary.get('rmse_mm', float('nan')):.6f} mm\n")
        f.write(f"  MAE:  {summary.get('mae_mm', float('nan')):.6f} mm\n")
        f.write(f"  MSE:  {summary.get('mse_mm2', float('nan')):.6f} mm²\n")
        f.write(f"  R²:   {summary.get('r2', float('nan')):.6f}\n")
    return summary

if __name__ == '__main__':
    # Run with defaults when executed directly
    run_cv_training()
