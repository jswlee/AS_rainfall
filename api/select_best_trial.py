#!/usr/bin/env python3
"""
Utility to programmatically select the best trial from saved run directories.
"""
import os
import json
import glob
from pathlib import Path
from typing import Dict, Optional, List


def get_best_trial_from_runs(metric_name: str = "val_loss", minimize: bool = True) -> Optional[Dict]:
    """
    Find the best trial from run directories based on a metric.
    
    Args:
        metric_name: Metric to optimize (e.g., 'val_loss', 'val_r2')
        minimize: Whether to minimize the metric (True for loss, False for R²)
        
    Returns:
        Dictionary with best trial info or None if not found
    """
    # Hardcoded path to runs directory
    mlruns_dir = "/Users/jlee/Desktop/github/AS_rainfall/mlruns"
    
    if not os.path.exists(mlruns_dir):
        print(f"Runs directory not found at {mlruns_dir}")
        return None
        
    print(f"Looking for best trial in {mlruns_dir}...")
    
    # Find all experiment directories
    experiment_dirs = []
    for item in os.listdir(mlruns_dir):
        if item.isdigit() and os.path.isdir(os.path.join(mlruns_dir, item)):
            experiment_dirs.append(os.path.join(mlruns_dir, item))
    
    if not experiment_dirs:
        print("No experiment directories found")
        return None
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Track best run
    best_run = None
    best_metric_value = float('inf') if minimize else float('-inf')
    best_run_dir = None
    
    # Search through all experiment directories
    for exp_dir in experiment_dirs:
        exp_id = os.path.basename(exp_dir)
        print(f"Checking experiment {exp_id}...")
        
        # Find all run directories in this experiment
        run_dirs = []
        for item in os.listdir(exp_dir):
            run_path = os.path.join(exp_dir, item)
            if os.path.isdir(run_path) and not item.startswith('.'):
                run_dirs.append(run_path)
        
        print(f"  Found {len(run_dirs)} runs")
        
        # Check each run for the target metric
        for run_dir in run_dirs:
            run_id = os.path.basename(run_dir)
            metrics_dir = os.path.join(run_dir, 'metrics')
            
            if not os.path.exists(metrics_dir):
                continue
                
            metric_file = os.path.join(metrics_dir, metric_name)
            if not os.path.exists(metric_file):
                continue
                
            # Read the metric value (last line has the final value)
            try:
                with open(metric_file, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    # Format is: timestamp value step
                    last_line = lines[-1].strip().split()
                    if len(last_line) < 2:
                        continue
                    metric_value = float(last_line[1])
                    
                    # Check if this is the best run so far
                    is_better = (metric_value < best_metric_value) if minimize else (metric_value > best_metric_value)
                    if is_better:
                        print(f"  Found better run: {run_id} with {metric_name}={metric_value}")
                        best_metric_value = metric_value
                        best_run_dir = run_dir
                        best_run = {
                            'run_uuid': run_id,
                            'experiment_id': exp_id,
                            'best_metric': metric_name,
                            'best_metric_value': metric_value
                        }
            except Exception as e:
                print(f"  Error reading metric file {metric_file}: {e}")
                continue
    
    if not best_run:
        print(f"No runs found with metric '{metric_name}'")
        return None
        
    print(f"\nBest run found: {best_run['run_uuid']}")
    print(f"  {metric_name}: {best_run['best_metric_value']}")
    
    # Read parameters for the best run
    params_dir = os.path.join(best_run_dir, 'params')
    params = {}
    if os.path.exists(params_dir):
        for param_file in os.listdir(params_dir):
            param_path = os.path.join(params_dir, param_file)
            if os.path.isfile(param_path):
                try:
                    with open(param_path, 'r') as f:
                        params[param_file] = f.read().strip()
                except Exception:
                    pass
    
    # Read all metrics for the best run
    metrics_dir = os.path.join(best_run_dir, 'metrics')
    metrics = {}
    if os.path.exists(metrics_dir):
        for metric_file in os.listdir(metrics_dir):
            metric_path = os.path.join(metrics_dir, metric_file)
            if os.path.isfile(metric_path):
                try:
                    with open(metric_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip().split()
                            if len(last_line) >= 2:
                                metrics[metric_file] = float(last_line[1])
                except Exception:
                    pass
    
    # Add parameters and metrics to the result
    best_run['parameters'] = params
    best_run['metrics'] = metrics
    
    print(f"  Parameters: {len(params)} params")
    
    return best_run


def find_trial_artifacts(trial_info: Dict, 
                        base_paths: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    Find artifact files for a given trial.
    
    Args:
        trial_info: Trial information from get_best_trial_from_mlflow
        base_paths: Dict with keys like 'hyperparams_base', 'model_base'
        
    Returns:
        Dict with artifact paths or None if not found
    """
    artifacts = {
        'hyperparams_path': None,
        'model_path': None,
        'preprocessing_path': None
    }
    
    run_uuid = trial_info['run_uuid']
    experiment_id = trial_info['experiment_id']
    
    # Construct the run artifacts path
    mlruns_dir = "/Users/jlee/Desktop/github/AS_rainfall/mlruns"
    run_dir = os.path.join(mlruns_dir, experiment_id, run_uuid)
    artifacts_dir = os.path.join(run_dir, 'artifacts')
    
    print(f"Looking for artifacts in {artifacts_dir}")
    
    # Check if artifacts directory exists
    if not os.path.exists(artifacts_dir):
        print(f"No artifacts directory found at {artifacts_dir}")
        
        # Try to find artifacts in base_paths instead
        if 'hyperparams_base' in base_paths:
            hyperparams_pattern = os.path.join(base_paths['hyperparams_base'], f"*{run_uuid}*.json")
            hyperparams_files = glob.glob(hyperparams_pattern)
            if hyperparams_files:
                artifacts['hyperparams_path'] = hyperparams_files[0]
                print(f"Found hyperparams at {artifacts['hyperparams_path']}")
        
        if 'model_base' in base_paths:
            model_pattern = os.path.join(base_paths['model_base'], f"*{run_uuid}*.pth")
            model_files = glob.glob(model_pattern)
            if model_files:
                artifacts['model_path'] = model_files[0]
                print(f"Found model at {artifacts['model_path']}")
        
        if 'preprocessing_base' in base_paths:
            preprocessing_pattern = os.path.join(base_paths['preprocessing_base'], "preprocessing.json")
            preprocessing_files = glob.glob(preprocessing_pattern)
            if preprocessing_files:
                artifacts['preprocessing_path'] = preprocessing_files[0]
                print(f"Found preprocessing at {artifacts['preprocessing_path']}")
    else:
        # Look for artifacts in the MLflow artifacts directory
        for root, dirs, files in os.walk(artifacts_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.json') and 'hyperparams' in file.lower():
                    artifacts['hyperparams_path'] = file_path
                    print(f"Found hyperparams at {file_path}")
                
                elif file.endswith('.pth') or file.endswith('.pt'):
                    artifacts['model_path'] = file_path
                    print(f"Found model at {file_path}")
                
                elif file.endswith('.json') and 'preprocessing' in file.lower():
                    artifacts['preprocessing_path'] = file_path
                    print(f"Found preprocessing at {file_path}")
    
    # If we still don't have all artifacts, look in the default locations
    if not all(artifacts.values()):
        print("Some artifacts not found, checking default locations...")
        
        # Check for hyperparams in standard locations
        if not artifacts['hyperparams_path'] and 'hyperparams_base' in base_paths:
            default_hyperparams = os.path.join(base_paths['hyperparams_base'], "best_hyperparameters.json")
            if os.path.exists(default_hyperparams):
                artifacts['hyperparams_path'] = default_hyperparams
                print(f"Using default hyperparams at {default_hyperparams}")
        
        # Check for model in standard locations
        if not artifacts['model_path'] and 'model_base' in base_paths:
            default_model = os.path.join(base_paths['model_base'], "best_model.pth")
            if os.path.exists(default_model):
                artifacts['model_path'] = default_model
                print(f"Using default model at {default_model}")
        
        # Check for preprocessing in standard locations
        if not artifacts['preprocessing_path'] and 'preprocessing_base' in base_paths:
            default_preprocessing = os.path.join(base_paths['preprocessing_base'], "preprocessing.json")
            if os.path.exists(default_preprocessing):
                artifacts['preprocessing_path'] = default_preprocessing
                print(f"Using default preprocessing at {default_preprocessing}")
    
    return artifacts


if __name__ == "__main__":
    # Example usage
    print("Finding best trial from saved runs...")
    best_trial = get_best_trial_from_runs(metric_name="val_loss", minimize=True)
    
    if best_trial:
        print("\nBest trial information:")
        print(f"Run UUID: {best_trial['run_uuid']}")
        print(f"Experiment ID: {best_trial['experiment_id']}")
        print(f"Best metric: {best_trial['best_metric']} = {best_trial['best_metric_value']}")
        
        # Find artifacts for this trial
        base_paths = {
            'hyperparams_base': '/Users/jlee/Desktop/github/AS_rainfall/api/artifacts',
            'model_base': '/Users/jlee/Desktop/github/AS_rainfall/api/artifacts',
            'preprocessing_base': '/Users/jlee/Desktop/github/AS_rainfall/api/artifacts'
        }
        
        artifacts = find_trial_artifacts(best_trial, base_paths)
        print("\nArtifact paths:")
        for key, path in artifacts.items():
            print(f"{key}: {path if path else 'Not found'}")
    else:
        print("No best trial found.")

