#!/usr/bin/env python3
"""
Training utilities for the LAND-inspired rainfall prediction model.

Implements cosine annealing with warm restarts and other training functionality.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
from datetime import datetime
import os
import pickle
import math

def cosine_decay_with_warmup(epoch, total_epochs=150, warmup_epochs=5, 
                            initial_lr=0.00117, min_lr=1e-6):
    """
    Cosine decay learning rate schedule with warmup.
    
    Parameters
    ----------
    epoch : int
        Current epoch
    total_epochs : int
        Total number of epochs
    warmup_epochs : int
        Number of warmup epochs
    initial_lr : float
        Initial learning rate
    min_lr : float
        Minimum learning rate
        
    Returns
    -------
    float
        Learning rate for the current epoch
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_model(model, datasets, output_dir, epochs=150, patience=25):
    """
    Train the LAND model.
    
    Parameters
    ----------
    model : tf.keras.Model
        Compiled model
    datasets : dict
        Dictionary containing TensorFlow datasets for training, validation, and testing
    output_dir : str
        Directory to save model weights and results
    epochs : int, optional
        Number of training epochs
    patience : int, optional
        Patience for early stopping
        
    Returns
    -------
    tuple
        (model, history, training_time)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create callbacks
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, total_epochs=epochs),
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, 'best_model.weights.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    # Start timer
    start_time = time.time()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        datasets['train'],
        epochs=epochs,
        validation_data=datasets['val'],
        callbacks=[lr_scheduler, early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTraining completed in {time.strftime('%H:%M:%S', time.gmtime(training_time))}")
    
    # Plot learning rate schedule
    plot_lr_schedule(epochs, output_dir)
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    return model, history, training_time


def plot_lr_schedule(total_epochs, output_dir, warmup_epochs=5, 
                    initial_lr=0.00117, min_lr=1e-6):
    """
    Plot the learning rate schedule.
    
    Parameters
    ----------
    total_epochs : int
        Total number of epochs
    output_dir : str
        Directory to save the plot
    warmup_epochs : int, optional
        Number of warmup epochs
    initial_lr : float, optional
        Initial learning rate
    min_lr : float, optional
        Minimum learning rate
        
    Returns
    -------
    None
    """
    # Generate learning rates for all epochs
    lr_schedule = [
        cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs, initial_lr, min_lr)
        for epoch in range(total_epochs)
    ]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(total_epochs), lr_schedule)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Cosine Decay with Warmup')
    ax.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'lr_schedule.png'))
    plt.close()


def evaluate_model(model, datasets, output_dir):
    """
    Evaluate the trained model.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    datasets : dict
        Dictionary containing TensorFlow datasets for training, validation, and testing
    output_dir : str
        Directory to save evaluation results
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate on validation set
    print("\nEvaluating model on validation data...")
    val_loss, val_mae = model.evaluate(datasets['val'], verbose=1)
    
    # Get validation predictions
    val_predictions = model.predict(datasets['val'], verbose=1)
    
    # Get validation targets
    val_targets = np.concatenate([y for _, y in datasets['val']], axis=0)
    
    # Calculate metrics
    val_r2 = r2_score(val_targets, val_predictions)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
    val_mae = mean_absolute_error(val_targets, val_predictions)
    
    print(f"Validation R²: {val_r2:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f} (scaled)")
    print(f"Validation MAE: {val_mae:.4f} (scaled)")
    
    # Evaluate on test set
    print("\nEvaluating model on test data...")
    test_loss, test_mae = model.evaluate(datasets['test'], verbose=1)
    
    # Get test predictions
    test_predictions = model.predict(datasets['test'], verbose=1)
    
    # Get test targets
    test_targets = np.concatenate([y for _, y in datasets['test']], axis=0)
    
    # Calculate metrics
    test_r2 = r2_score(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    test_mae = mean_absolute_error(test_targets, test_predictions)
    
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} (scaled)")
    print(f"Test MAE: {test_mae:.4f} (scaled)")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(
        val_targets, val_predictions.flatten(),
        'Validation Set: Predicted vs Actual Rainfall',
        os.path.join(output_dir, 'validation_results.png'),
        scaled=True
    )
    plot_results(
        test_targets, test_predictions.flatten(),
        'Test Set: Predicted vs Actual Rainfall',
        os.path.join(output_dir, 'test_results.png'),
        scaled=True
    )
    
    # Save results
    results = {
        'validation': {
            'r2': val_r2,
            'rmse': val_rmse * 100,  # Convert back to mm
            'mae': val_mae * 100,  # Convert back to mm
            'loss': val_loss
        },
        'test': {
            'r2': test_r2,
            'rmse': test_rmse * 100,  # Convert back to mm
            'mae': test_mae * 100,  # Convert back to mm
            'loss': test_loss
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return results


def plot_training_history(history, output_dir):
    """
    Plot training history.
    
    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history
    output_dir : str
        Directory to save plots
        
    Returns
    -------
    None
    """
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()
    
    # Plot MAE
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mae_history.png'))
    plt.close()


def plot_results(y_true, y_pred, title, output_path, scaled=False):
    """
    Plot predicted vs actual values.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    title : str
        Plot title
    output_path : str
        Path to save the plot
    scaled : bool, optional
        Whether the values are scaled (divided by 100)
        
    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 8))
    
    # Plot scatter
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot diagonal line (perfect predictions)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    if scaled:
        plt.xlabel('Actual Rainfall (scaled)')
        plt.ylabel('Predicted Rainfall (scaled)')
        factor = 100  # Convert back to mm for metrics display
    else:
        plt.xlabel('Actual Rainfall (mm)')
        plt.ylabel('Predicted Rainfall (mm)')
        factor = 1
    
    plt.title(title)
    
    # Add metrics to plot
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * factor
    mae = mean_absolute_error(y_true, y_pred) * factor
    
    plt.text(
        0.05, 0.95,
        f'R² = {r2:.4f}\nRMSE = {rmse:.4f} mm\nMAE = {mae:.4f} mm',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
