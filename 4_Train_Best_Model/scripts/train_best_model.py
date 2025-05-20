#!/usr/bin/env python3
"""
Train the LAND-inspired rainfall prediction model with the best hyperparameters
found during hyperparameter tuning.
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))

# Add required directories to Python path
sys.path.append(PROJECT_ROOT)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Import local modules
from data_utils import load_and_reshape_data, create_tf_dataset
from training import train_model, evaluate_model, plot_training_history
from model_utils import build_model, load_best_hyperparameters


# Use the build_model function from model_utils.py
build_best_model = build_model


def main():
    """
    Main function to train the model with the best hyperparameters.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LAND model with best hyperparameters')
    parser.add_argument('--features_path', type=str,
        default=os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'output', 'csv_data', 'features.csv'),
        help='Path to features CSV file')
    parser.add_argument('--targets_path', type=str,
        default=os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'output', 'csv_data', 'targets.csv'),
        help='Path to targets CSV file')
    parser.add_argument('--test_indices_path', type=str,
        default=os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'output', 'test_indices.pkl'),
        help='Path to save or load test indices')
    parser.add_argument('--hyperparams_path', type=str,
        default=os.path.join(PROJECT_ROOT, '3_Hyperparameter_Tuning', 'output', 'land_model_extended_tuner', 'current_best_hyperparameters.py'),
        help='Path to hyperparameters file')
    parser.add_argument('--output_dir', type=str,
        default=os.path.join(PIPELINE_DIR, 'output', 'land_model_best'),
        help='Directory to save model weights and results')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=314,
                        help='Batch size for training')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and reshape data
    print("Loading and reshaping data...")
    data = load_and_reshape_data(
        features_path=args.features_path,
        targets_path=args.targets_path,
        test_indices_path=args.test_indices_path
    )
    
    # Create TensorFlow datasets
    print("\nCreating TensorFlow datasets...")
    datasets = create_tf_dataset(data, batch_size=args.batch_size)
    
    # Load hyperparameters if path is provided
    hyperparams = None
    if args.hyperparams_path:
        print(f"\nLoading hyperparameters from {args.hyperparams_path}...")
        # Check file extension to determine how to load
        if args.hyperparams_path.endswith('.py'):
            # Load Python module file (current_best_hyperparameters.py)
            try:
                # Get the directory containing the hyperparameters file
                hyperparams_dir = os.path.dirname(args.hyperparams_path)
                # Add this directory to Python path temporarily
                sys.path.insert(0, hyperparams_dir)
                # Get the module name without extension
                module_name = os.path.basename(args.hyperparams_path).replace('.py', '')
                # Import the module dynamically
                hyperparams_module = __import__(module_name)
                # Get the hyperparameters dictionary
                hyperparams = hyperparams_module.best_hyperparameters
                # Remove the directory from path
                sys.path.pop(0)
            except Exception as e:
                print(f"Error loading Python hyperparameters file: {e}")
                print(f"Falling back to default hyperparameters.")
        elif args.hyperparams_path.endswith('.pkl'):
            # Load binary pickle file
            import pickle
            try:
                with open(args.hyperparams_path, 'rb') as f:
                    hyperparams = pickle.load(f)
            except Exception as e:
                print(f"Error loading pickle file: {e}")
                # Try the text file version as fallback
                text_path = args.hyperparams_path.replace('.pkl', '.txt')
                if os.path.exists(text_path):
                    print(f"Falling back to text file: {text_path}")
                    args.hyperparams_path = text_path
                else:
                    print(f"No fallback file found. Using default hyperparameters.")
        
        # If it's a text file or fallback to text file
        if not hyperparams and args.hyperparams_path.endswith('.txt'):
            hyperparams = {}
            with open(args.hyperparams_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        try:
                            # Try to convert to float or int
                            value = float(value.strip())
                            if value.is_integer():
                                value = int(value)
                        except ValueError:
                            # Keep as string if not a number
                            value = value.strip()
                        hyperparams[key.strip()] = value
        
        if hyperparams:
            print("Loaded hyperparameters:")
            for key, value in hyperparams.items():
                print(f"  {key}: {value}")
        else:
            print("No hyperparameters loaded, using defaults.")
    
    # Build the model with best hyperparameters
    print("\nBuilding model with best hyperparameters...")
    model = build_best_model(data['metadata'], hyperparams)
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    history = train_model(
        model=model,
        data=datasets,  # Changed from datasets to data to match function signature
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, data=datasets, output_dir=args.output_dir)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, args.output_dir)
    
    # Print final results
    print("\nFinal Results:")
    if 'r2' in metrics:
        print(f"Test R²: {metrics['r2']:.4f}")
    if 'rmse' in metrics:
        print(f"Test RMSE: {metrics['rmse']*100:.4f} in")
    print(f"Test MSE: {metrics['loss']*100:.4f}")
    print(f"Test MAE: {metrics['mae']*100:.4f} in")
    
    # Calculate and print training time
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    print(f"\nTraining Time: {minutes} minutes and {seconds} seconds")
    print(f"Model saved to: {args.output_dir}")
    
    # Save a summary of the run
    with open(os.path.join(args.output_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"===============\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"ALL RAINFALL VALUES ARE IN INCHES.\n\n")
        
        f.write(f"Data:\n")
        f.write(f"  Training samples: {len(data['targets']['train'])}\n")
        f.write(f"  Validation samples: {len(data['targets']['val'])}\n")
        f.write(f"  Test samples: {len(data['targets']['test'])}\n\n")
        
        f.write(f"Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write(f"Training:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Training time: {minutes} minutes and {seconds} seconds\n\n")
        
        f.write(f"Results (all in inches):\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
    
    print(f"\nTraining summary saved to {os.path.join(args.output_dir, 'training_summary.txt')}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
