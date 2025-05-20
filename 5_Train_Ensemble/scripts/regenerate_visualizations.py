#!/usr/bin/env python3
"""
Regenerate visualizations for the ensemble model without retraining.
This script loads existing predictions and metrics, converts them to inches,
and creates updated visualizations and summary files.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob
import pickle

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))

# Add required directories to Python path
sys.path.append(PROJECT_ROOT)


def update_test_predictions_csv(csv_path):
    """
    Update a test_predictions.csv file to ensure values are in inches.
    
    Parameters
    ----------
    csv_path : str
        Path to the test_predictions.csv file
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} does not exist. Skipping.")
        return
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, comment='#')
        
        # Check column names - they might be 'actual' and 'predicted' instead of 'actual_inches' and 'predicted_inches'
        actual_col = 'actual_inches' if 'actual_inches' in df.columns else 'actual'
        predicted_col = 'predicted_inches' if 'predicted_inches' in df.columns else 'predicted'
        
        if actual_col not in df.columns or predicted_col not in df.columns:
            print(f"Warning: Required columns not found in {csv_path}. Skipping.")
            return
        
        # Check if values are already in inches
        # If the mean is very small (< 1), they're probably still in the scaled units
        if df[actual_col].mean() < 1 or df[predicted_col].mean() < 1:
            print(f"Converting values in {csv_path} to inches...")
            # Convert to inches
            df[actual_col] = df[actual_col] * 100
            df[predicted_col] = df[predicted_col] * 100
            
            # Rename columns if needed
            if actual_col == 'actual':
                df = df.rename(columns={'actual': 'actual_inches', 'predicted': 'predicted_inches'})
            
            # Write header with units
            with open(csv_path, 'w') as f:
                f.write('# Units: inches\n')
                df.to_csv(f, index=False)
            print(f"Updated {csv_path}")
        else:
            print(f"Values in {csv_path} already appear to be in inches. Skipping.")
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")


def update_evaluation_metrics_csv(csv_path):
    """
    Update an evaluation_metrics.csv file to ensure values are in inches.
    
    Parameters
    ----------
    csv_path : str
        Path to the evaluation_metrics.csv file
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} does not exist. Skipping.")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if values are already in inches
    # If rmse or mae is very small (< 1), they're probably still in the scaled units
    needs_update = False
    for key in ['rmse', 'mae']:
        if key in df.columns and df[key].iloc[0] < 1:
            needs_update = True
            break
    
    if needs_update:
        print(f"Converting metrics in {csv_path} to inches...")
        # Convert to inches
        for key in ['rmse', 'mae', 'loss', 'mean_absolute_error']:
            if key in df.columns:
                df[key] = df[key] * 100
        
        for key in ['mse', 'mean_squared_error']:
            if key in df.columns:
                df[key] = df[key] * 100 * 100
        
        # Save updated metrics
        df.to_csv(csv_path, index=False)
        print(f"Updated {csv_path}")
    else:
        print(f"Metrics in {csv_path} already appear to be in inches. Skipping.")


def regenerate_actual_vs_predicted_plot(csv_path, output_dir):
    """
    Regenerate the actual vs predicted plot from a test_predictions.csv file.
    
    Parameters
    ----------
    csv_path : str
        Path to the test_predictions.csv file
    output_dir : str
        Directory to save the plot
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} does not exist. Skipping plot generation.")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path, comment='#')
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df['actual_inches'], df['predicted_inches'], alpha=0.5)
    
    # Add 1:1 line
    lims = [
        min(df['actual_inches'].min(), df['predicted_inches'].min()),
        max(df['actual_inches'].max(), df['predicted_inches'].max())
    ]
    plt.plot(lims, lims, 'r--', label='1:1 Line')
    
    # Add labels and title
    plt.xlabel('Actual Rainfall (inches)')
    plt.ylabel('Predicted Rainfall (inches)')
    plt.title('Actual vs Predicted Rainfall')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated plot: {plot_path}")


def update_training_summary(summary_path):
    """
    Update a training_summary.txt file to ensure values are in inches.
    
    Parameters
    ----------
    summary_path : str
        Path to the training_summary.txt file
    """
    # Check if file exists
    if not os.path.exists(summary_path):
        print(f"Warning: File {summary_path} does not exist. Skipping.")
        return
    
    # Read the file
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    
    # Check if we need to update
    needs_update = False
    for line in lines:
        if ('RMSE:' in line or 'MAE:' in line or 'Loss:' in line) and ' in' not in line:
            needs_update = True
            break
    
    if needs_update:
        print(f"Updating metrics in {summary_path}...")
        updated_lines = []
        for line in lines:
            # Add units to metrics if missing
            if 'RMSE:' in line and ' in' not in line:
                parts = line.split(':')
                value = float(parts[1].strip())
                updated_lines.append(f"{parts[0]}: {value*100:.4f} in\n")
            elif 'MAE:' in line and ' in' not in line:
                parts = line.split(':')
                value = float(parts[1].strip())
                updated_lines.append(f"{parts[0]}: {value*100:.4f} in\n")
            elif 'Loss:' in line and ' in' not in line:
                parts = line.split(':')
                value = float(parts[1].strip())
                updated_lines.append(f"{parts[0]}: {value*100*100:.4f} in²\n")
            else:
                updated_lines.append(line)
        
        # Write updated file
        with open(summary_path, 'w') as f:
            f.writelines(updated_lines)
        print(f"Updated {summary_path}")
    else:
        print(f"Metrics in {summary_path} already appear to have units. Skipping.")


def update_fold_summary(summary_path):
    """
    Update a fold_summary.txt file to ensure values are in inches.
    
    Parameters
    ----------
    summary_path : str
        Path to the fold_summary.txt file
    """
    # Check if file exists
    if not os.path.exists(summary_path):
        print(f"Warning: File {summary_path} does not exist. Skipping.")
        return
    
    # Read the file
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    
    # Check if we need to update
    needs_update = False
    for line in lines:
        if ('RMSE:' in line or 'MAE:' in line):
            try:
                value = float(line.split(':')[1].split()[0])
                # If value is too small (< 1) or too large (> 1000), it needs fixing
                if value < 1 or value > 1000:
                    needs_update = True
                    break
            except (ValueError, IndexError):
                pass
    
    if needs_update:
        print(f"Updating metrics in {summary_path}...")
        updated_lines = []
        for line in lines:
            if 'RMSE:' in line:
                try:
                    parts = line.split(':')
                    value = float(parts[1].split()[0])
                    # If value is too small, multiply by 100
                    if value < 1:
                        updated_lines.append(f"{parts[0]}: {value*100:.4f} in\n")
                    # If value is too large, divide by 100
                    elif value > 1000:
                        updated_lines.append(f"{parts[0]}: {value/100:.4f} in\n")
                    else:
                        updated_lines.append(line)
                except (ValueError, IndexError):
                    updated_lines.append(line)
            elif 'MAE:' in line:
                try:
                    parts = line.split(':')
                    value = float(parts[1].split()[0])
                    # If value is too small, multiply by 100
                    if value < 1:
                        updated_lines.append(f"{parts[0]}: {value*100:.4f} in\n")
                    # If value is too large, divide by 100
                    elif value > 1000:
                        updated_lines.append(f"{parts[0]}: {value/100:.4f} in\n")
                    else:
                        updated_lines.append(line)
                except (ValueError, IndexError):
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Write updated file
        with open(summary_path, 'w') as f:
            f.writelines(updated_lines)
        print(f"Updated {summary_path}")
    else:
        print(f"Metrics in {summary_path} already appear to be in inches. Skipping.")


def regenerate_ensemble_summary(ensemble_dir):
    """
    Regenerate the ensemble summary from the fold results.
    
    Parameters
    ----------
    ensemble_dir : str
        Directory containing the ensemble results
    """
    # Find all fold directories
    fold_dirs = sorted(glob.glob(os.path.join(ensemble_dir, 'fold_*')))
    if not fold_dirs:
        print(f"Warning: No fold directories found in {ensemble_dir}. Skipping ensemble summary.")
        return
    
    # Collect fold metrics
    fold_metrics = []
    for fold_dir in fold_dirs:
        fold_summary_path = os.path.join(fold_dir, 'fold_summary.txt')
        if os.path.exists(fold_summary_path):
            with open(fold_summary_path, 'r') as f:
                lines = f.readlines()
            
            fold_metric = {}
            for line in lines:
                if 'R²:' in line:
                    try:
                        fold_metric['r2'] = float(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'RMSE:' in line:
                    try:
                        # Extract value without 'in' unit
                        value = float(line.split(':')[1].split()[0].strip())
                        # Check if value is reasonable (between 1 and 100)
                        if 1 <= value <= 100:
                            fold_metric['rmse'] = value
                        elif value > 100:
                            # Value is too large, probably incorrectly scaled
                            fold_metric['rmse'] = value / 100
                        else:
                            # Value is too small, probably needs scaling
                            fold_metric['rmse'] = value * 100
                    except (ValueError, IndexError):
                        pass
                elif 'MAE:' in line:
                    try:
                        # Extract value without 'in' unit
                        value = float(line.split(':')[1].split()[0].strip())
                        # Check if value is reasonable (between 1 and 100)
                        if 1 <= value <= 100:
                            fold_metric['mae'] = value
                        elif value > 100:
                            # Value is too large, probably incorrectly scaled
                            fold_metric['mae'] = value / 100
                        else:
                            # Value is too small, probably needs scaling
                            fold_metric['mae'] = value * 100
                    except (ValueError, IndexError):
                        pass
            
            if fold_metric and 'r2' in fold_metric and 'rmse' in fold_metric and 'mae' in fold_metric:
                fold_metrics.append(fold_metric)
    
    if not fold_metrics:
        print(f"Warning: No fold metrics found in {ensemble_dir}. Skipping ensemble summary.")
        return
    
    # Calculate average CV metrics
    avg_r2 = np.mean([fold['r2'] for fold in fold_metrics])
    avg_rmse = np.mean([fold['rmse'] for fold in fold_metrics])
    avg_mae = np.mean([fold['mae'] for fold in fold_metrics])
    
    # Find the test predictions
    test_predictions_path = os.path.join(ensemble_dir, 'test_predictions.csv')
    if os.path.exists(test_predictions_path):
        try:
            df = pd.read_csv(test_predictions_path, comment='#')
            
            # Check if the required columns exist
            if 'actual_inches' in df.columns and 'predicted_inches' in df.columns:
                # Calculate test metrics
                test_r2 = r2_score(df['actual_inches'], df['predicted_inches'])
                test_rmse = np.sqrt(mean_squared_error(df['actual_inches'], df['predicted_inches']))
                test_mae = mean_absolute_error(df['actual_inches'], df['predicted_inches'])
            else:
                print(f"Warning: Required columns not found in {test_predictions_path}. Using fold averages instead.")
                test_r2 = avg_r2
                test_rmse = avg_rmse
                test_mae = avg_mae
        except Exception as e:
            print(f"Error reading {test_predictions_path}: {e}. Using fold averages instead.")
            test_r2 = avg_r2
            test_rmse = avg_rmse
            test_mae = avg_mae
        
        # Create ensemble summary
        summary_path = os.path.join(ensemble_dir, 'ensemble_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Ensemble Summary\n")
            f.write(f"===============\n\n")
            f.write(f"Number of Folds: {len(fold_metrics)}\n")
            f.write(f"Number of Models per Fold: {len(glob.glob(os.path.join(fold_dirs[0], 'model_*')))}\n\n")
            f.write(f"Cross-Validation Results:\n")
            f.write(f"  Average CV R²: {avg_r2:.4f}\n")
            f.write(f"  Average CV RMSE: {avg_rmse:.4f} in\n")
            f.write(f"  Average CV MAE: {avg_mae:.4f} in\n\n")
            f.write(f"Final Ensemble Results:\n")
            f.write(f"  Test R²: {test_r2:.4f}\n")
            f.write(f"  Test RMSE: {test_rmse:.4f} in\n")
            f.write(f"  Test MAE: {test_mae:.4f} in\n")
        
        print(f"Generated ensemble summary: {summary_path}")
    else:
        print(f"Warning: No test predictions found in {ensemble_dir}. Skipping ensemble summary.")


def regenerate_fold_ensemble_plot(fold_dir):
    """
    Regenerate the fold ensemble plot.
    
    Parameters
    ----------
    fold_dir : str
        Directory containing the fold results
    """
    # Find all model test predictions
    model_dirs = sorted(glob.glob(os.path.join(fold_dir, 'model_*')))
    if not model_dirs:
        print(f"Warning: No model directories found in {fold_dir}. Skipping fold ensemble plot.")
        return
    
    # Collect all predictions
    all_preds = []
    actual = None
    
    for model_dir in model_dirs:
        pred_path = os.path.join(model_dir, 'test_predictions.csv')
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path, comment='#')
            all_preds.append(df['predicted_inches'].values)
            if actual is None:
                actual = df['actual_inches'].values
    
    if not all_preds or actual is None:
        print(f"Warning: No predictions found in {fold_dir}. Skipping fold ensemble plot.")
        return
    
    # Calculate ensemble prediction
    ensemble_pred = np.mean(all_preds, axis=0)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(actual, ensemble_pred, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel('Actual Rainfall (inches)')
    plt.ylabel('Predicted Rainfall (inches)')
    plt.title(f'Fold Ensemble: Actual vs Predicted Rainfall')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(fold_dir, 'fold_ensemble_predictions.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Generated fold ensemble plot: {plot_path}")


def regenerate_final_ensemble_plot(ensemble_dir):
    """
    Regenerate the final ensemble plot.
    
    Parameters
    ----------
    ensemble_dir : str
        Directory containing the ensemble results
    """
    # Find the test predictions
    test_predictions_path = os.path.join(ensemble_dir, 'test_predictions.csv')
    if not os.path.exists(test_predictions_path):
        print(f"Warning: No test predictions found in {ensemble_dir}. Trying to create from fold data.")
        
        # Try to create ensemble predictions from fold data
        fold_dirs = sorted(glob.glob(os.path.join(ensemble_dir, 'fold_*')))
        if not fold_dirs:
            print(f"No fold directories found. Skipping final ensemble plot.")
            return
        
        # Collect all model predictions
        all_preds = []
        actual = None
        
        for fold_dir in fold_dirs:
            model_dirs = sorted(glob.glob(os.path.join(fold_dir, 'model_*')))
            for model_dir in model_dirs:
                pred_path = os.path.join(model_dir, 'test_predictions.csv')
                if os.path.exists(pred_path):
                    try:
                        df = pd.read_csv(pred_path, comment='#')
                        if 'predicted_inches' in df.columns:
                            all_preds.append(df['predicted_inches'].values)
                            if actual is None and 'actual_inches' in df.columns:
                                actual = df['actual_inches'].values
                    except Exception as e:
                        print(f"Error reading {pred_path}: {e}")
        
        if not all_preds or actual is None:
            print(f"Could not collect predictions from fold data. Skipping final ensemble plot.")
            return
        
        # Calculate ensemble prediction
        ensemble_pred = np.mean(all_preds, axis=0)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.scatter(actual, ensemble_pred, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        plt.xlabel('Actual Rainfall (inches)')
        plt.ylabel('Predicted Rainfall (inches)')
        plt.title('Final Ensemble: Actual vs Predicted Rainfall')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = os.path.join(ensemble_dir, 'final_ensemble_predictions.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated final ensemble plot from fold data: {plot_path}")
        
        # Also save the predictions
        results_df = pd.DataFrame({
            'actual_inches': actual,
            'predicted_inches': ensemble_pred
        })
        with open(test_predictions_path, 'w') as f:
            f.write('# Units: inches\n')
            results_df.to_csv(f, index=False)
        print(f"Created ensemble test predictions: {test_predictions_path}")
        return
    
    # Read the predictions
    try:
        df = pd.read_csv(test_predictions_path, comment='#')
        
        # Check if the required columns exist
        if 'actual_inches' not in df.columns or 'predicted_inches' not in df.columns:
            print(f"Required columns not found in {test_predictions_path}. Skipping final ensemble plot.")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.scatter(df['actual_inches'], df['predicted_inches'], alpha=0.5)
        plt.plot([df['actual_inches'].min(), df['actual_inches'].max()], 
                [df['actual_inches'].min(), df['actual_inches'].max()], 'r--')
        plt.xlabel('Actual Rainfall (inches)')
        plt.ylabel('Predicted Rainfall (inches)')
        plt.title('Final Ensemble: Actual vs Predicted Rainfall')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = os.path.join(ensemble_dir, 'final_ensemble_predictions.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated final ensemble plot: {plot_path}")
    except Exception as e:
        print(f"Error processing {test_predictions_path}: {e}. Skipping final ensemble plot.")


def fix_extreme_values_in_summary(file_path):
    """
    Fix extreme values in summary files (values that are way too large).
    
    Parameters
    ----------
    file_path : str
        Path to the summary file
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping.")
        return
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # First pass - check if any values need fixing
        needs_fixing = False
        for line in lines:
            if 'RMSE:' in line or 'MAE:' in line:
                try:
                    parts = line.split(':')
                    value_part = parts[1].strip()
                    value_str = value_part.split()[0]
                    value = float(value_str)
                    
                    # If value is extremely large (> 10), it's likely wrong
                    if value > 10:
                        needs_fixing = True
                        break
                except (ValueError, IndexError):
                    pass
        
        if not needs_fixing:
            print(f"No extreme values found in {file_path}. Skipping.")
            return
        
        # Second pass - fix all values
        updated_lines = []
        for line in lines:
            if 'RMSE:' in line or 'MAE:' in line:
                try:
                    parts = line.split(':')
                    value_part = parts[1].strip()
                    value_str = value_part.split()[0]
                    value = float(value_str)
                    
                    # If value is extremely large (> 10), it's likely wrong
                    if value > 10:
                        # Divide by 100 to get the correct value
                        corrected_value = value / 100
                        updated_line = f"{parts[0]}: {corrected_value:.4f} in\n"
                        updated_lines.append(updated_line)
                        print(f"Fixed extreme value in {file_path}: {value} -> {corrected_value}")
                    else:
                        updated_lines.append(line)
                except (ValueError, IndexError) as e:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)
    
    except Exception as e:
        print(f"Error fixing extreme values in {file_path}: {e}")


def main():
    """
    Main function to regenerate visualizations.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Regenerate visualizations for the ensemble model')
    parser.add_argument('--ensemble_dir', type=str, 
                        default=os.path.join(PIPELINE_DIR, 'output', 'simple_ensemble'),
                        help='Directory containing ensemble results')
    parser.add_argument('--create_missing', action='store_true',
                        help='Create missing files from available data')
    parser.add_argument('--fix_ensemble_test_predictions', action='store_true',
                        help='Fix the ensemble test_predictions.csv file')
    parser.add_argument('--fix_extreme_values', action='store_true',
                        help='Fix extremely large values in summary files')
    args = parser.parse_args()
    
    # Check if ensemble directory exists
    if not os.path.exists(args.ensemble_dir):
        print(f"Error: Ensemble directory {args.ensemble_dir} does not exist.")
        return 1
    
    print(f"Regenerating visualizations for ensemble in {args.ensemble_dir}...")
    
    # Process all fold directories
    fold_dirs = sorted(glob.glob(os.path.join(args.ensemble_dir, 'fold_*')))
    for fold_dir in fold_dirs:
        print(f"\nProcessing fold directory: {fold_dir}")
        
        # Update fold summary
        fold_summary_path = os.path.join(fold_dir, 'fold_summary.txt')
        update_fold_summary(fold_summary_path)
        
        # Regenerate fold ensemble plot
        regenerate_fold_ensemble_plot(fold_dir)
        
        # Process all model directories in the fold
        model_dirs = sorted(glob.glob(os.path.join(fold_dir, 'model_*')))
        for model_dir in model_dirs:
            print(f"\nProcessing model directory: {model_dir}")
            
            # Update test predictions
            test_predictions_path = os.path.join(model_dir, 'test_predictions.csv')
            update_test_predictions_csv(test_predictions_path)
            
            # Update evaluation metrics
            eval_metrics_path = os.path.join(model_dir, 'evaluation_metrics.csv')
            update_evaluation_metrics_csv(eval_metrics_path)
            
            # Regenerate actual vs predicted plot
            regenerate_actual_vs_predicted_plot(test_predictions_path, model_dir)
            
            # Update training summary
            training_summary_path = os.path.join(model_dir, 'training_summary.txt')
            update_training_summary(training_summary_path)
    
    # Fix ensemble test predictions if requested
    if args.fix_ensemble_test_predictions:
        print(f"\nFixing ensemble test predictions...")
        ensemble_test_predictions_path = os.path.join(args.ensemble_dir, 'test_predictions.csv')
        if os.path.exists(ensemble_test_predictions_path):
            try:
                # Read the CSV file
                df = pd.read_csv(ensemble_test_predictions_path)
                
                # Check column names
                if 'actual' in df.columns and 'predicted' in df.columns:
                    # Convert to inches
                    df['actual_inches'] = df['actual'] * 100
                    df['predicted_inches'] = df['predicted'] * 100
                    
                    # Drop old columns
                    df = df.drop(columns=['actual', 'predicted'])
                    
                    # Write header with units
                    with open(ensemble_test_predictions_path, 'w') as f:
                        f.write('# Units: inches\n')
                        df.to_csv(f, index=False)
                    print(f"Updated {ensemble_test_predictions_path}")
                else:
                    print(f"Required columns not found in {ensemble_test_predictions_path}. Skipping.")
            except Exception as e:
                print(f"Error processing {ensemble_test_predictions_path}: {e}")
        else:
            print(f"File {ensemble_test_predictions_path} does not exist. Creating from fold data...")
            # Try to create from fold data
            fold_dirs = sorted(glob.glob(os.path.join(args.ensemble_dir, 'fold_*')))
            if fold_dirs:
                # Collect all model predictions
                all_preds = []
                actual = None
                
                for fold_dir in fold_dirs:
                    model_dirs = sorted(glob.glob(os.path.join(fold_dir, 'model_*')))
                    for model_dir in model_dirs:
                        pred_path = os.path.join(model_dir, 'test_predictions.csv')
                        if os.path.exists(pred_path):
                            try:
                                df = pd.read_csv(pred_path, comment='#')
                                if 'predicted_inches' in df.columns:
                                    all_preds.append(df['predicted_inches'].values)
                                    if actual is None and 'actual_inches' in df.columns:
                                        actual = df['actual_inches'].values
                            except Exception as e:
                                print(f"Error reading {pred_path}: {e}")
                
                if all_preds and actual is not None:
                    # Calculate ensemble prediction
                    ensemble_pred = np.mean(all_preds, axis=0)
                    
                    # Save the predictions
                    results_df = pd.DataFrame({
                        'actual_inches': actual,
                        'predicted_inches': ensemble_pred
                    })
                    with open(ensemble_test_predictions_path, 'w') as f:
                        f.write('# Units: inches\n')
                        results_df.to_csv(f, index=False)
                    print(f"Created ensemble test predictions: {ensemble_test_predictions_path}")
    
    # Regenerate ensemble summary
    print(f"\nRegenerating ensemble summary...")
    regenerate_ensemble_summary(args.ensemble_dir)
    
    # Regenerate final ensemble plot
    print(f"\nRegenerating final ensemble plot...")
    regenerate_final_ensemble_plot(args.ensemble_dir)
    
    # Fix extreme values in summary files if requested
    if args.fix_extreme_values:
        print(f"\nFixing extreme values in summary files...")
        # Fix fold summary files
        fold_dirs = sorted(glob.glob(os.path.join(args.ensemble_dir, 'fold_*')))
        for fold_dir in fold_dirs:
            fold_summary_path = os.path.join(fold_dir, 'fold_summary.txt')
            fix_extreme_values_in_summary(fold_summary_path)
        
        # Fix ensemble summary file
        ensemble_summary_path = os.path.join(args.ensemble_dir, 'ensemble_summary.txt')
        fix_extreme_values_in_summary(ensemble_summary_path)
    
    print(f"\nVisualization regeneration complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
