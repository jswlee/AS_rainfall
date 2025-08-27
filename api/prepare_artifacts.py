#!/usr/bin/env python3
"""
Prepare artifacts for the rainfall prediction API.
This script automatically selects the best trial from MLflow runs and copies
the necessary model files and preprocessing stats to the api/artifacts directory.
"""
import os
import shutil
import json
from pathlib import Path
from select_best_trial import get_best_trial_from_mlflow, find_trial_artifacts

def prepare_artifacts():
    """Copy training artifacts to api/artifacts directory."""
    
    # Create artifacts directory
    artifacts_dir = Path(__file__).parent / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    
    print("🔍 Finding best trial from MLflow runs...")
    
    # Get best trial from MLflow
    best_trial = get_best_trial_from_mlflow(
        metric_name="val_loss",
        minimize=True
    )
    
    if not best_trial:
        print("❌ Could not find best trial from MLflow. Falling back to manual paths...")
        # Fallback to manual paths
        hyperparams_src = '../Hyperparameter_Tuning/output_WeightedMSE_4/best_hyperparameters.json'
        model_src = '../Train_Best_Model/output_WeightedMSE_4/pytorch_best_model/best_model.pth'
        preprocessing_src = 'preprocessing.json'
    else:
        print(f"✅ Best trial found: {best_trial['run_uuid']}")
        print(f"   Best val_loss: {best_trial['best_metric_value']:.6f}")
        
        # Find artifact paths for the best trial
        base_paths = {
            'hyperparams_base': '../Hyperparameter_Tuning',
            'model_base': '../Train_Best_Model', 
            'preprocessing_path': 'preprocessing.json'
        }
        
        artifacts = find_trial_artifacts(best_trial, base_paths)
        hyperparams_src = artifacts['hyperparams_path']
        model_src = artifacts['model_path']
        preprocessing_src = artifacts['preprocessing_path']
        
        print(f"📁 Artifact sources:")
        print(f"   Hyperparams: {hyperparams_src}")
        print(f"   Model: {model_src}")
        print(f"   Preprocessing: {preprocessing_src}")
    
    # Destination paths
    hyperparams_dst = os.path.join(artifacts_dir, 'hyperparams.json')
    model_dst = os.path.join(artifacts_dir, 'best_model.pth')
    preprocessing_dst = os.path.join(artifacts_dir, 'preprocessing.json')
    
    # Copy files
    print("Preparing model artifacts...")
    
    if os.path.exists(hyperparams_src):
        shutil.copy2(hyperparams_src, hyperparams_dst)
        print(f"✓ Copied hyperparameters: {hyperparams_src} -> {hyperparams_dst}")
    else:
        print(f"✗ Hyperparameters not found: {hyperparams_src}")
        return False
    
    if os.path.exists(model_src):
        shutil.copy2(model_src, model_dst)
        print(f"✓ Copied model weights: {model_src} -> {model_dst}")
    else:
        print(f"✗ Model weights not found: {model_src}")
        return False
    
    if os.path.exists(preprocessing_src):
        shutil.copy2(preprocessing_src, preprocessing_dst)
        print(f"✓ Copied preprocessing stats: {preprocessing_src} -> {preprocessing_dst}")
    else:
        print(f"✗ Preprocessing stats not found: {preprocessing_src}")
        return False
    
    # Verify hyperparameters format
    with open(hyperparams_dst, 'r') as f:
        hp_data = json.load(f)
        if 'hyperparameters' in hp_data:
            print(f"✓ Hyperparameters format: new (with trial metadata)")
            print(f"  Trial number: {hp_data.get('trial_number', 'unknown')}")
        else:
            print(f"✓ Hyperparameters format: legacy (direct hyperparams)")
    
    print(f"\nAll artifacts prepared in {artifacts_dir}/")
    print("Ready for containerization!")
    return True

if __name__ == "__main__":
    prepare_artifacts()
