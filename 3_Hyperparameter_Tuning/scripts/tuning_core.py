#!/usr/bin/env python3
"""
Core tuning utilities for extended hyperparameter tuning with cross-validation.
This module encapsulates data loading, CV setup, tuner configuration,
callbacks, and orchestration. It is intended to be imported by
extended_hyperparameter_tuning.py, keeping its main() minimal.
"""
import os
import time
import pickle
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Local import for NPZ loader (same directory)
from npz_data_utils import load_assembled_npz_data


class CVHyperModel(kt.HyperModel):
    def __init__(self, build_model_fn, data_metadata, cv_features, cv_targets, kf, batch_size):
        self.build_model_fn = build_model_fn
        self.data_metadata = data_metadata
        self.cv_features = cv_features
        self.cv_targets = cv_targets
        self.kf = kf
        self.batch_size = batch_size
        self.fold_indices = list(kf.split(np.arange(len(cv_targets))))

    def build(self, hp):
        return self.build_model_fn(hp, self.data_metadata)

    def fit(self, hp, model, *args, **kwargs):
        callbacks = kwargs.pop('callbacks', [])
        val_losses = []
        for fold, (train_idx, val_idx) in enumerate(self.fold_indices):
            print(f"\nTraining on fold {fold+1}/{len(self.fold_indices)}")
            train_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'climate': self.cv_features['climate'][train_idx],
                    'local_dem': self.cv_features['local_dem'][train_idx],
                    'regional_dem': self.cv_features['regional_dem'][train_idx],
                    'month': self.cv_features['month'][train_idx]
                },
                self.cv_targets[train_idx]
            )).batch(self.batch_size, drop_remainder=True)

            val_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'climate': self.cv_features['climate'][val_idx],
                    'local_dem': self.cv_features['local_dem'][val_idx],
                    'regional_dem': self.cv_features['regional_dem'][val_idx],
                    'month': self.cv_features['month'][val_idx]
                },
                self.cv_targets[val_idx]
            )).batch(self.batch_size, drop_remainder=True)

            if fold > 0:
                tf.keras.backend.clear_session()
                model = self.build(hp)

            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                callbacks=callbacks,
                **kwargs
            )
            best_val_loss = min(history.history['val_loss'])
            val_losses.append(best_val_loss)
            print(f"Fold {fold+1} best validation loss: {best_val_loss:.6f}")

        avg_val_loss = np.mean(val_losses)
        print(f"\nAverage validation loss across {len(self.fold_indices)} folds: {avg_val_loss:.6f}")
        # Report average through last history element
        history.history['val_loss'][-1] = avg_val_loss
        return history


class SaveBestHyperparametersCallback(tf.keras.callbacks.Callback):
    def __init__(self, tuner, output_dir):
        super().__init__()
        self.tuner = tuner
        self.output_dir = output_dir
        self.best_val_loss = float('inf')
        self.trial_count = 0

    def on_train_begin(self, logs=None):
        self.trial_count += 1
        print(f"\nStarting trial #{self.trial_count}")

    def on_epoch_end(self, epoch, logs=None):
        curr_val_loss = logs.get('val_loss', float('inf'))
        if epoch == self.params.get('epochs', 0) - 1 or curr_val_loss < self.best_val_loss:
            self.save_current_best()

    def save_current_best(self):
        try:
            if not self.tuner.oracle.trials:
                return
            best_hp = self.tuner.get_best_hyperparameters(1)[0]
            best_trials = self.tuner.oracle.get_best_trials(1)
            if not best_trials:
                return
            best_trial = best_trials[0]
            val_loss = best_trial.score
            if val_loss is None:
                return
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                with open(os.path.join(self.output_dir, 'current_best_hyperparameters.txt'), 'w') as f:
                    f.write(f"Best hyperparameters after {self.trial_count} trials (val_loss: {val_loss:.6f}):\n\n")
                    for param, value in best_hp.values.items():
                        f.write(f"{param}: {value}\n")
                with open(os.path.join(self.output_dir, 'current_best_hyperparameters.py'), 'w') as f:
                    f.write("# Current best hyperparameters\n\n")
                    f.write("best_hyperparameters = {\n")
                    for param, value in best_hp.values.items():
                        if isinstance(value, str):
                            f.write(f"    '{param}': '{value}',\n")
                        else:
                            f.write(f"    '{param}': {value},\n")
                    f.write("}\n")
                print(f"\n[SaveBestHyperparameters] Updated best hyperparameters (val_loss: {val_loss:.6f})")
        except Exception as e:
            print(f"Error saving best hyperparameters: {e}")


def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, initial_lr=0.001, min_lr=1e-6):
    import math
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))


def run_tuning(config, build_model_fn):
    """
    Orchestrate loading data from assembled NPZ, setting up CV tuner, and running search.

    config keys expected:
      - project_root, output_dir, test_indices_path
      - npz_path
      - max_trials, executions_per_trial, epochs, batch_size, n_folds, cv_seed, resume
    """
    os.makedirs(config['output_dir'], exist_ok=True)

    print(f"Loading assembled NPZ from {config['npz_path']}...")
    data = load_assembled_npz_data(
        npz_path=config['npz_path'],
        test_indices_path=config['test_indices_path'],
        test_size=config.get('test_size', 0.1),
        val_size=config.get('val_size', 0.1),
        random_state=config['cv_seed'],
    )

    # Build CV features/targets pool (train+val)
    cv_features = {
        'climate': np.concatenate([data['climate']['train'], data['climate']['val']]),
        'local_dem': np.concatenate([data['local_dem']['train'], data['local_dem']['val']]),
        'regional_dem': np.concatenate([data['regional_dem']['train'], data['regional_dem']['val']]),
        'month': np.concatenate([data['month']['train'], data['month']['val']]),
    }
    cv_targets = np.concatenate([data['targets']['train'], data['targets']['val']])

    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['cv_seed'])

    cv_hypermodel = CVHyperModel(
        build_model_fn=build_model_fn,
        data_metadata=data['metadata'],
        cv_features=cv_features,
        cv_targets=cv_targets,
        kf=kf,
        batch_size=config['batch_size'],
    )

    tuner_dir = os.path.join(config['output_dir'], 'land_model_cv_tuning')
    existing_trials = os.path.exists(tuner_dir)

    tuner = kt.BayesianOptimization(
        cv_hypermodel,
        objective='val_loss',
        max_trials=config['max_trials'],
        executions_per_trial=config['executions_per_trial'],
        directory=config['output_dir'],
        project_name='land_model_cv_tuning',
        overwrite=False,
    )

    if config.get('resume', False) or existing_trials:
        print("\nResuming from previous tuning session...")
        completed_trials = len(tuner.oracle.trials)
        if completed_trials > 0:
            print(f"Found {completed_trials} completed trials")
            try:
                best_trials = tuner.oracle.get_best_trials(1)
                if best_trials:
                    best_trial = best_trials[0]
                    if best_trial.score is not None:
                        print(f"Best val_loss so far: {best_trial.score:.6f}")
                        print("Best hyperparameters so far:")
                        for param, value in best_trial.hyperparameters.values.items():
                            print(f"  {param}: {value}")
                    else:
                        print("Best trial found but score is None. Will continue tuning.")
                else:
                    print("No best trials found yet. Will continue tuning.")
            except Exception as e:
                print(f"Could not retrieve best trial information: {e}")
                print("Will continue tuning with existing trials.")
        else:
            print("No completed trials found. Starting from scratch.")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, total_epochs=config['epochs']), verbose=1
    )
    save_best_hp_callback = SaveBestHyperparametersCallback(tuner, config['output_dir'])

    print("\nStarting extended hyperparameter tuning with cross-validation...")
    if config.get('resume', False):
        remaining_trials = config['max_trials'] - len(tuner.oracle.trials)
        print(f"Resuming with {remaining_trials} remaining trials of {config['max_trials']} total")
    else:
        print(f"Running {config['max_trials']} trials with {config['n_folds']}-fold cross-validation")
    print(f"Each fold will train for up to {config['epochs']} epochs")
    print("Best hyperparameters will be saved after each trial in 'current_best_hyperparameters.txt'")
    start_time = time.time()

    tuner.search(
        epochs=config['epochs'], callbacks=[early_stopping, lr_scheduler, save_best_hp_callback]
    )

    tuning_time = time.time() - start_time
    print(f"\nTuning completed in {time.strftime('%H:%M:%S', time.gmtime(tuning_time))}")

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("\nBest hyperparameters:")
    for param, value in best_hp.values.items():
        print(f"{param}: {value}")

    with open(os.path.join(config['output_dir'], 'best_hyperparameters.txt'), 'w') as f:
        f.write(f"Best hyperparameters from {config['max_trials']} trials:\n\n")
        for param, value in best_hp.values.items():
            f.write(f"{param}: {value}\n")
    with open(os.path.join(config['output_dir'], 'best_hyperparameters.pkl'), 'wb') as f:
        pickle.dump(best_hp.values, f)
    with open(os.path.join(config['output_dir'], 'best_hyperparameters.py'), 'w') as f:
        f.write("# Best hyperparameters from extended tuning\n\n")
        f.write("best_hyperparameters = {\n")
        for param, value in best_hp.values.items():
            if isinstance(value, str):
                f.write(f"    '{param}': '{value}',\n")
            else:
                f.write(f"    '{param}': {value},\n")
        f.write("}\n")

    # Optional: importance plot if available and enough trials
    try:
        if len(tuner.oracle.trials) >= 10:
            try:
                importances = tuner.results_summary.get_importance()
                if importances:
                    plt.figure(figsize=(12, 8))
                    params = list(importances.keys())
                    values = list(importances.values())
                    plt.barh(params, values)
                    plt.xlabel('Importance')
                    plt.ylabel('Hyperparameter')
                    plt.title('Hyperparameter Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(config['output_dir'], 'hyperparameter_importance.png'))
                    plt.close()
                    print(f"Hyperparameter importance plot saved to {os.path.join(config['output_dir'], 'hyperparameter_importance.png')}")
                else:
                    print("No hyperparameter importance data available.")
            except AttributeError:
                print("results_summary.get_importance() not available in this Keras Tuner version.")
    except Exception as e:
        print(f"Could not generate hyperparameter importance plot: {str(e)}")

    return best_hp
