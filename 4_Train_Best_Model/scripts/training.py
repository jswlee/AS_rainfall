#!/usr/bin/env python3
"""
Training utilities for the LAND-inspired rainfall prediction model.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import math


def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, initial_lr=0.001, min_lr=0.0001):
    """
    Cosine decay learning rate schedule with warmup.
    
    Parameters
    ----------
    epoch : int
        Current epoch
    total_epochs : int
        Total number of epochs
    warmup_epochs : int, optional
        Number of warmup epochs
    initial_lr : float, optional
        Initial learning rate
    min_lr : float, optional
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
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (initial_lr - min_lr) * cosine_decay


def train_model(model, data, epochs=100, batch_size=32, output_dir=None, 
                initial_lr=0.001, min_lr=0.0001, warmup_epochs=5, patience=30):
    """
    Train the model with the given data.
    
    Parameters
    ----------
    model : tf.keras.Model
        Model to train
    data : dict
        Dictionary containing TensorFlow datasets
    epochs : int, optional
        Number of epochs to train for
    batch_size : int, optional
        Batch size for training
    output_dir : str, optional
        Directory to save model weights and training history
    initial_lr : float, optional
        Initial learning rate
    min_lr : float, optional
        Minimum learning rate
    warmup_epochs : int, optional
        Number of warmup epochs
    patience : int, optional
        Patience for early stopping
        
    Returns
    -------
    dict
        Dictionary containing training history
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        weights_path = os.path.join(output_dir, 'best_weights.weights.h5')  # Changed extension to .weights.h5
    else:
        weights_path = 'best_weights.weights.h5'  # Changed extension to .weights.h5
    
    # Define callbacks
    lr_scheduler = LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(
            epoch, epochs, warmup_epochs, initial_lr, min_lr
        )
    )
    
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print(f"\nTraining model for {epochs} epochs with batch size {batch_size}")
    print(f"Learning rate: {initial_lr} to {min_lr} with {warmup_epochs} warmup epochs")
    print(f"Early stopping patience: {patience}")
    
    # Handle different data formats
    if isinstance(data['train'], tf.data.Dataset):
        # If it's already a TensorFlow dataset
        train_ds = data['train']
        val_ds = data['val']
    elif isinstance(data['train'], dict) and 'climate' in data['train']:
        # If it's raw data from simple_ensemble.py
        # Create TensorFlow datasets from the raw data
        train_ds = tf.data.Dataset.from_tensor_slices((
            {
                'climate': data['train']['climate'],
                'local_dem': data['train']['local_dem'],
                'regional_dem': data['train']['regional_dem'],
                'month': data['train']['month']
            },
            data['targets']['train']
        )).batch(batch_size, drop_remainder=True)
        
        val_ds = tf.data.Dataset.from_tensor_slices((
            {
                'climate': data['val']['climate'],
                'local_dem': data['val']['local_dem'],
                'regional_dem': data['val']['regional_dem'],
                'month': data['val']['month']
            },
            data['targets']['val']
        )).batch(batch_size, drop_remainder=True)
    else:
        # If it's some other format, just use as is
        train_ds = data['train']
        val_ds = data['val']
        
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[lr_scheduler, checkpoint, early_stopping],
        verbose=1
    )
    
    # Save training history
    if output_dir:
        history_path = os.path.join(output_dir, 'training_history.npy')
        np.save(history_path, history.history)
    
    return history.history


def evaluate_model(model, data, output_dir=None):
    """
    Evaluate the model on the test set with detailed metrics.
    
    Parameters
    ----------
    model : tf.keras.Model
        Model to evaluate
    data : dict
        Dictionary containing TensorFlow datasets
    output_dir : str, optional
        Directory to save evaluation results
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    print("\nEvaluating model on test set...")
    
    # Handle different data formats
    if isinstance(data, tf.data.Dataset) or (isinstance(data, dict) and isinstance(data.get('test'), tf.data.Dataset)):
        # If it's a TensorFlow dataset
        test_ds = data['test'] if isinstance(data, dict) else data
        
        # Get base metrics from model.evaluate
        base_metrics = model.evaluate(test_ds, verbose=1, return_dict=True)
        
        # Get predictions on test set
        y_pred = model.predict(test_ds, verbose=0)
        
        # Extract actual values from test dataset
        y_true = np.concatenate([y for _, y in test_ds], axis=0)
    elif isinstance(data, dict) and isinstance(data.get('test'), dict) and 'climate' in data.get('test', {}):
        # If it's raw data from simple_ensemble.py
        # Create a TensorFlow dataset from the raw data
        test_ds = tf.data.Dataset.from_tensor_slices((
            {
                'climate': data['test']['climate'],
                'local_dem': data['test']['local_dem'],
                'regional_dem': data['test']['regional_dem'],
                'month': data['test']['month']
            },
            data['targets']['test']
        )).batch(32, drop_remainder=False)
        
        # Get base metrics from model.evaluate
        base_metrics = model.evaluate(test_ds, verbose=1, return_dict=True)
        
        # Get predictions on test set
        y_pred = model.predict(test_ds, verbose=0)
        
        # Use the raw targets
        y_true = data['targets']['test']
    else:
        # Assume it's some other format we can't handle
        raise ValueError("Unsupported data format for evaluation")
    
    # Calculate additional metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Combine metrics
    metrics = {
        **base_metrics,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    # Print detailed metrics
    print("\nDetailed Test Metrics:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Save evaluation metrics
    if output_dir:
        # Save as numpy array
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.npy')
        np.save(metrics_path, metrics)
        
        # Also save as CSV for easier reading
        metrics_df = pd.DataFrame([metrics])
        csv_path = os.path.join(output_dir, 'evaluation_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
        
        # Save predictions vs actual values
        results_df = pd.DataFrame({
            'actual_inches': y_true.flatten(),
            'predicted_inches': y_pred.flatten()
        })
        results_path = os.path.join(output_dir, 'test_predictions.csv')
        # Write header with units
        with open(results_path, 'w') as f:
            f.write('# Units: inches\n')
            results_df.to_csv(f, index=False)
        print(f"Test predictions saved to {results_path}")
        
        # Scatter plot: Actual vs Predicted
        plt.figure(figsize=(6,6))
        plt.scatter(y_true*100, y_pred*100, alpha=0.5, label='Predictions')
        lims = [min(y_true.min()*100, y_pred.min()*100), max(y_true.max()*100, y_pred.max()*100)]
        plt.plot(lims, lims, 'r--', label='1:1 Line')
        plt.xlabel('Actual Rainfall (inches)')
        plt.ylabel('Predicted Rainfall (inches)')
        plt.title('Actual vs Predicted Rainfall')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'actual_vs_predicted.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Actual vs Predicted plot saved to {plot_path}")
    
    return metrics


def plot_training_history(history, output_dir=None):
    """
    Plot training history.
    
    Parameters
    ----------
    history : dict
        Dictionary containing training history
    output_dir : str, optional
        Directory to save plots
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    
    # plt.show()  # Disabled to prevent popup window
