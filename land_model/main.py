#!/usr/bin/env python3
"""
Main script for training and evaluating the LAND-inspired rainfall prediction model.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from land_model.data_utils import load_and_reshape_data, create_tf_dataset
from land_model.model import build_land_model
from land_model.training import train_model, evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Train LAND-inspired rainfall prediction model')
    parser.add_argument('--features', type=str, default='csv_data/features.csv',
                        help='Path to features CSV file')
    parser.add_argument('--targets', type=str, default='csv_data/targets.csv',
                        help='Path to targets CSV file')
    parser.add_argument('--output_dir', type=str, default='land_model_output',
                        help='Directory to save models and results')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=314,
                        help='Batch size (from LAND paper: 314)')
    parser.add_argument('--patience', type=int, default=25,
                        help='Patience for early stopping')
    parser.add_argument('--na', type=int, default=512,
                        help='Number of neurons in first hidden layer')
    parser.add_argument('--nb', type=int, default=1024,
                        help='Number of neurons in second hidden layer')
    parser.add_argument('--dropout_rate', type=float, default=0.45,
                        help='Dropout rate')
    parser.add_argument('--l2_reg', type=float, default=0.00645,
                        help='L2 regularization strength')
    parser.add_argument('--force_new_split', action='store_true',
                        help='Force new train/val/test split')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set test indices path
    test_indices_path = os.path.join(args.output_dir, 'test_indices.pkl')
    if args.force_new_split and os.path.exists(test_indices_path):
        os.remove(test_indices_path)
    
    # Load and reshape data
    print("Loading and reshaping data...")
    data = load_and_reshape_data(
        args.features, args.targets, 
        test_indices_path=test_indices_path,
        random_state=42
    )
    
    # Create TensorFlow datasets
    print("\nCreating TensorFlow datasets...")
    datasets = create_tf_dataset(data, batch_size=args.batch_size)
    
    # Build model
    print("\nBuilding model...")
    model = build_land_model(
        data['metadata'],
        na=args.na,
        nb=args.nb,
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    model, history, training_time = train_model(
        model,
        datasets,
        args.output_dir,
        epochs=args.epochs,
        patience=args.patience
    )
    
    # Evaluate model
    results = evaluate_model(model, datasets, args.output_dir)
    
    # Print final results
    print("\nFinal Results:")
    print(f"Validation R²: {results['validation']['r2']:.4f}")
    print(f"Validation RMSE: {results['validation']['rmse']:.4f} mm")
    print(f"Validation MAE: {results['validation']['mae']:.4f} mm")
    print(f"Test R²: {results['test']['r2']:.4f}")
    print(f"Test RMSE: {results['test']['rmse']:.4f} mm")
    print(f"Test MAE: {results['test']['mae']:.4f} mm")
    
    print(f"\nResults saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
