#!/usr/bin/env python3
"""
Core tuning utilities for extended hyperparameter tuning with cross-validation.
This module encapsulates data loading, CV setup, tuner configuration,
callbacks, and orchestration. It is intended to be imported by
extended_hyperparameter_tuning.py, keeping its main() minimal.
"""
import os
import time
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Local import for NPZ loader (package import, no scripts/)
from Hyperparameter_Tuning.npz_data_utils import load_assembled_npz_data

class CVHyperModel(kt.HyperModel):
    """A HyperModel wrapper that performs K-fold CV inside a single trial.

    build_model_fn(hp, data_metadata) -> tf.keras.Model
    """
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
            # Use hp-defined batch_size if provided; fall back to config's batch_size
            try:
                hp_batch = hp.get('batch_size')
            except Exception:
                hp_batch = None
            base_batch = hp_batch if hp_batch is not None else self.batch_size
            # Ensure effective batch size is within valid range for this fold
            eff_batch = int(min(max(1, base_batch), len(train_idx)))
            # Build datasets with drop_remainder=False to avoid zero-step epochs
            train_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'climate': self.cv_features['climate'][train_idx],
                    'local_dem': self.cv_features['local_dem'][train_idx],
                    'regional_dem': self.cv_features['regional_dem'][train_idx],
                    'month': self.cv_features['month'][train_idx]
                },
                self.cv_targets[train_idx]
            )).shuffle(2048).batch(eff_batch, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

            val_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'climate': self.cv_features['climate'][val_idx],
                    'local_dem': self.cv_features['local_dem'][val_idx],
                    'regional_dem': self.cv_features['regional_dem'][val_idx],
                    'month': self.cv_features['month'][val_idx]
                },
                self.cv_targets[val_idx]
            )).batch(eff_batch, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

            # Log steps/epoch for visibility
            steps_per_epoch = int(np.ceil(len(train_idx) / eff_batch))
            val_steps = int(np.ceil(len(val_idx) / eff_batch))
            src = 'hp' if hp_batch is not None else 'config'
            print(f"Fold {fold+1}: base_batch={base_batch} ({src}), eff_batch={eff_batch}, steps/epoch={steps_per_epoch}, val_steps={val_steps}")

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
        # Path under the tuner project directory where hyperparameters are saved
        self.hp_save_dir = os.path.join(self.output_dir, 'land_model_cv_tuning')
        self.hp_file_path = os.path.join(self.hp_save_dir, 'current_best_hyperparameters.py')

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
                os.makedirs(self.hp_save_dir, exist_ok=True)
                with open(self.hp_file_path, 'w') as f:
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


def plot_hyperparameter_importance(tuner, output_dir: str):
    """Generate and save a hyperparameter importance plot.

    Returns the output image path if successful, else None.
    Works without keras_tuner.visualization by using a proxy computation.
    """
    try:
        n_trials = len(tuner.oracle.trials)
        if n_trials < 2:
            print(f"Skipping hyperparameter importance plot (need >=2 trials, have {n_trials}).")
            return None
        try:
            # Preferred API (if available)
            from keras_tuner import visualization as kt_vis
            fig = kt_vis.plot_hyperparameter_importance(tuner)
            out_path = os.path.join(output_dir, 'hyperparameter_importance.png')
            fig.savefig(out_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
            print(f"Hyperparameter importance plot saved to {out_path}")
            return out_path
        except Exception as e_vis:
            # Fallback: derive simple importances from completed trials
            print(f"keras_tuner.visualization not available; computing simple importances: {e_vis}")
            try:
                # Collect trial data
                trials = [t for t in tuner.oracle.trials.values() if t.score is not None]
                if len(trials) < 2:
                    print(f"Not enough completed trials for importance plot (have {len(trials)}).")
                    return None
                # Performance: higher is better -> use negative val_loss
                perf = np.array([-t.score for t in trials], dtype=float)
                # Gather hyperparameter values
                hp_names = sorted(list(trials[0].hyperparameters.values.keys()))
                importances = {}
                for hp in hp_names:
                    vals = [trial.hyperparameters.values.get(hp, None) for trial in trials]
                    # Skip if all values are None or constant
                    if all(v is None for v in vals):
                        continue
                    # Determine numeric vs categorical
                    numeric = all(isinstance(v, (int, float, np.floating, np.integer)) for v in vals if v is not None)
                    if numeric:
                        x = np.array([float(v) for v in vals], dtype=float)
                        if np.all(x == x[0]):
                            continue
                        # Pearson correlation magnitude as proxy importance
                        corr = np.corrcoef(x, perf)[0, 1]
                        if np.isnan(corr):
                            continue
                        importances[hp] = abs(corr)
                    else:
                        # Categorical: compute std of group means of performance
                        groups = {}
                        for v, p in zip(vals, perf):
                            groups.setdefault(str(v), []).append(p)
                        if len(groups) <= 1:
                            continue
                        means = np.array([np.mean(g) for g in groups.values()], dtype=float)
                        imp = float(np.std(means))
                        importances[hp] = imp

                if not importances:
                    print("Could not compute importances (no varying hyperparameters).")
                    return None
                # Normalize to sum to 1 for readability
                total = sum(importances.values())
                if total > 0:
                    for k in list(importances.keys()):
                        importances[k] /= total
                # Plot
                params = list(importances.keys())
                values = [importances[k] for k in params]
                plt.figure(figsize=(10, max(4, 0.4 * len(params))))
                y_pos = np.arange(len(params))
                plt.barh(y_pos, values)
                plt.yticks(y_pos, params)
                plt.xlabel('Relative importance')
                plt.title('Hyperparameter Importance')
                plt.tight_layout()
                out_path = os.path.join(output_dir, 'hyperparameter_importance.png')
                plt.savefig(out_path, dpi=200)
                plt.close()
                print(f"Hyperparameter importance plot saved to {out_path}")
                return out_path
            except Exception as e_fb:
                print(f"Fallback importance computation failed: {e_fb}")
                return None
    except Exception as e:
        print(f"Could not generate hyperparameter importance plot: {str(e)}")
        return None

def cosine_decay_with_warmup(epoch: int, total_epochs: int, warmup_epochs: int = 5, initial_lr: float = 0.001, min_lr: float = 1e-6) -> float:
    import math
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))

def run_tuning(config: dict, build_model_fn, patience: int = 10):
    """Run cross-validated hyperparameter tuning.

    Expects config keys:
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

    # Sanity assertions on loaded data
    meta = data.get('metadata', {})
    assert tuple(meta.get('climate_shape', ())) == (16, 3, 3), f"Unexpected climate_shape: {meta.get('climate_shape')}"
    n_total = sum(data['targets'][split].shape[0] for split in ('train','val','test'))
    assert n_total == 2032, f"Unexpected total N={n_total}, expected 2032"

    # Build CV features/targets pool (train+val)
    cv_features = {
        'climate': np.concatenate([data['climate']['train'], data['climate']['val']]),
        'local_dem': np.concatenate([data['local_dem']['train'], data['local_dem']['val']]),
        'regional_dem': np.concatenate([data['regional_dem']['train'], data['regional_dem']['val']]),
        'month': np.concatenate([data['month']['train'], data['month']['val']]),
    }
    cv_targets = np.concatenate([data['targets']['train'], data['targets']['val']])

    # Debug logging of shapes
    print("CV pool sizes:")
    print(f"  N_cv = {cv_targets.shape[0]}")
    for k, v in cv_features.items():
        print(f"  {k}: {v.shape}")
    print(f"  targets: {cv_targets.shape}")

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
        monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1
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
    print("Best hyperparameters will be saved after each trial to 'land_model_cv_tuning/current_best_hyperparameters.py'")
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

    # Write the single .py file under the tuner project directory
    tuner_dir = os.path.join(config['output_dir'], 'land_model_cv_tuning')
    os.makedirs(tuner_dir, exist_ok=True)
    hp_file_path = os.path.join(tuner_dir, 'current_best_hyperparameters.py')
    with open(hp_file_path, 'w') as f:
        f.write("# Current best hyperparameters\n\n")
        f.write("best_hyperparameters = {\n")
        for param, value in best_hp.values.items():
            if isinstance(value, str):
                f.write(f"    '{param}': '{value}',\n")
            else:
                f.write(f"    '{param}': {value},\n")
        f.write("}\n")
    print(f"Hyperparameters saved to: {hp_file_path}")

    # Optional: importance plot
    plot_hyperparameter_importance(tuner, config['output_dir'])

    return best_hp
