#!/usr/bin/env python3
"""
Main entry point for the Rainfall Prediction Pipeline.
This script provides a command-line interface to run different components of the pipeline.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Rainfall Prediction Pipeline')
    parser.add_argument('--action', type=str, required=True,
                        choices=['pipeline', 'visualize', 'regenerate_climate', 'train_model', 'predict', 'tune_hyperparams', 'train_best_model', 'evaluate_best_model', 'process_rainfall'],
                        help='Action to perform')
    parser.add_argument('--date', type=int, default=0,
                        help='Date index for visualization (0-based)')
    parser.add_argument('--grid_point', type=int, default=12,
                        help='Grid point index for visualization (0-based)')
    parser.add_argument('--h5_file', type=str, default='output/rainfall_prediction_data.h5',
                        help='Path to H5 file for visualization or model training')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save model and results')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for prediction')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs for training')
    parser.add_argument('--max_trials', type=int, default=20,
                        help='Maximum number of trials for hyperparameter tuning')
    parser.add_argument('--min_epochs_per_trial', type=int, default=10,
                        help='Minimum number of epochs to train each trial during hyperparameter tuning')
    parser.add_argument('--tuner_dir', type=str, default='tuner_results',
                        help='Directory to save hyperparameter tuning results')
    parser.add_argument('--best_model_dir', type=str, default='best_model',
                        help='Directory to save the best model and results')
    parser.add_argument('--eval_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--raw_rainfall_dir', type=str, default='data/raw_rainfall',
                        help='Directory containing raw rainfall CSV files')
    parser.add_argument('--mon_rainfall_dir', type=str, default='processed_data/mon_rainfall',
                        help='Directory to save monthly aggregated rainfall files')
    
    args = parser.parse_args()
    
    if args.action == 'pipeline':
        print("Running the complete rainfall prediction pipeline...")
        os.system(f"python3 scripts/rainfall_prediction_pipeline.py")
    
    elif args.action == 'visualize':
        print(f"Visualizing data for date index {args.date}, grid point {args.grid_point}...")
        os.system(f"python3 visualization/explore_h5_data.py --action single_point --date {args.date} --grid_point {args.grid_point} --file {args.h5_file}")
    
    elif args.action == 'regenerate_climate':
        print("Regenerating climate data...")
        os.system(f"python3 scripts/regenerate_climate_data.py")
    
    elif args.action == 'train_model':
        print("Training deep learning model for rainfall prediction...")
        cmd = f"python3 scripts/train_model.py --data {args.h5_file} --model_dir {args.model_dir} --batch_size {args.batch_size} --epochs {args.epochs}"
        if args.model_path:
            cmd += f" --evaluate_only --model_path {args.model_path}"
        os.system(cmd)
    
    elif args.action == 'predict':
        if not args.model_path:
            print("Error: --model_path is required for prediction")
            return 1
        
        print(f"Generating rainfall predictions using model {args.model_path}...")
        cmd = f"python3 scripts/predict_rainfall.py --data {args.h5_file} --model_path {args.model_path} --output_dir {args.output_dir}"
        if args.date is not None:
            cmd += f" --date_indices {args.date}"
        os.system(cmd)
    
    elif args.action == 'tune_hyperparams':
        print("Running hyperparameter tuning for rainfall prediction model...")
        cmd = f"python3 scripts/hyperparameter_tuning.py --data {args.h5_file} --output_dir {args.tuner_dir} --max_epochs {args.epochs} --max_trials {args.max_trials}"
        cmd += f" --min_epochs {args.min_epochs_per_trial}"
        os.system(cmd)
    
    elif args.action == 'train_best_model':
        print("Training the best model from hyperparameter tuning results...")
        cmd = f"python3 scripts/train_best_model.py --data {args.h5_file} --tuner_dir {args.tuner_dir} --output_dir {args.best_model_dir} --epochs {args.epochs}"
        os.system(cmd)
    
    elif args.action == 'evaluate_best_model':
        print("Evaluating the best model and generating predictions...")
        cmd = f"python3 scripts/evaluate_best_model.py --data {args.h5_file} --model_dir {args.best_model_dir} --output_dir {args.eval_dir}"
        os.system(cmd)
    
    elif args.action == 'process_rainfall':
        print("Processing raw rainfall data into monthly aggregates...")
        cmd = f"python3 scripts/process_rainfall_data.py --input_dir {args.raw_rainfall_dir} --output_dir {args.mon_rainfall_dir}"
        os.system(cmd)
    
    else:
        print(f"Unknown action: {args.action}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
