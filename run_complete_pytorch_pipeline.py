#!/usr/bin/env python3
"""
Complete PyTorch pipeline runner - demonstrates the full workflow.
This script shows how to run the entire PyTorch-based rainfall prediction pipeline.
"""

import os
import sys
import time
import argparse

from Hyperparameter_Tuning.pytorch_hyperparameter_tuning import (
    run_hyperparameter_tuning as run_hp_tuning,
)


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")
    
    # Check if NPZ data exists
    npz_path = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz')
    if not os.path.exists(npz_path):
        print(f"❌ NPZ data file not found at {npz_path}")
        print("Please run the data preprocessing pipeline first:")
        print("  1. cd Process_Rainfall_Data && python process_wide_format_rainfall.py")
        print("  2. cd ML_Data_Preprocessing && python build_dem_patches.py")
        print("  3. cd ML_Data_Preprocessing && python build_reanalysis_features.py")
        print("  4. cd ML_Data_Preprocessing && python assemble_training_data.py")
        return False
    
    # Check PyTorch files exist
    required_files = [
        'Hyperparameter_Tuning/pytorch_data_utils.py',
        'Hyperparameter_Tuning/pytorch_model.py',
        'Hyperparameter_Tuning/pytorch_training.py',
        'Hyperparameter_Tuning/pytorch_hyperparameter_tuning.py',
        'Train_Best_Model/pytorch_train_best_model.py',
        'Train_Ensemble/pytorch_train_ensemble.py'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing PyTorch files: {missing}")
        return False
    
    print("✅ All prerequisites met")
    return True


def run_hyperparameter_tuning(n_trials=100, quick_test=False):
    """Run hyperparameter tuning."""
    print("\n" + "="*60)
    print("STEP 1: HYPERPARAMETER TUNING")
    print("="*60)
    
    sys.path.append('Hyperparameter_Tuning')

    
    # Adjust parameters for quick testing
    if quick_test:
        config = {
            'n_trials': 5,
            'n_folds': 3,
            'max_epochs': 150,
            'patience': 10
        }
        print("Running in QUICK TEST mode (reduced trials/epochs)")
    else:
        config = {
            'n_trials': n_trials,
            'n_folds': 3,
            'max_epochs': 150,
            'patience': 30
        }
    
    print(f"Starting hyperparameter tuning with {config['n_trials']} trials and {config['n_folds']} folds...")
    start_time = time.time()
    
    try:
        results = run_hp_tuning(**config)
        tuning_time = time.time() - start_time
        
        print(f"✅ Hyperparameter tuning completed in {tuning_time:.1f} seconds")
        print(f"Best validation loss: {results['best_value']:.6f}")
        print(f"Best hyperparameters saved to: Hyperparameter_Tuning/output/")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Hyperparameter tuning failed: {e}")
        return False, None


def run_best_model_training(quick_test=False):
    """Train the best model."""
    print("\n" + "="*60)
    print("STEP 2: BEST MODEL TRAINING")
    print("="*60)
    
    from Train_Best_Model.pytorch_train_best_model import train_best_model_pytorch
    
    epochs = 20 if quick_test else 150
    print(f"Training best model for up to {epochs} epochs...")
    
    start_time = time.time()
    
    try:
        results = train_best_model_pytorch(epochs=epochs)
        training_time = time.time() - start_time
        
        print(f"✅ Best model training completed in {training_time:.1f} seconds")
        print(f"Test R²: {results['test_metrics']['r2']:.4f}")
        print(f"Results saved to: Train_Best_Model/output/pytorch_best_model/")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Best model training failed: {e}")
        return False, None


def run_ensemble_training(quick_test=False):
    """Train the ensemble."""
    print("\n" + "="*60)
    print("STEP 3: ENSEMBLE TRAINING")
    print("="*60)
    
    from Train_Ensemble.pytorch_train_ensemble import train_ensemble_pytorch
    
    if quick_test:
        config = {
            'n_folds': 3,
            'n_models_per_fold': 2,
            'epochs': 20
        }
        print("Running in QUICK TEST mode (fewer folds/models/epochs)")
    else:
        config = {
            'n_folds': 10,
            'n_models_per_fold': 10,
            'epochs': 150
        }
    
    total_models = config['n_folds'] * config['n_models_per_fold']
    print(f"Training ensemble with {total_models} models ({config['n_folds']} folds × {config['n_models_per_fold']} models/fold)...")
    
    start_time = time.time()
    
    try:
        results = train_ensemble_pytorch(**config)
        ensemble_time = time.time() - start_time
        
        print(f"✅ Ensemble training completed in {ensemble_time:.1f} seconds")
        print(f"Final ensemble R²: {results['final_r2']:.4f}")
        print(f"Average CV R²: {results['avg_fold_r2']:.4f}")
        print(f"Results saved to: Train_Ensemble/output/pytorch_ensemble/")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Ensemble training failed: {e}")
        return False, None


def print_final_summary(tuning_results, best_model_results, ensemble_results):
    """Print final summary of all results."""
    print("\n" + "="*60)
    print("PYTORCH PIPELINE COMPLETION SUMMARY")
    print("="*60)
    
    print("\n📊 RESULTS SUMMARY:")
    
    if tuning_results:
        print(f"  Hyperparameter Tuning:")
        print(f"    • Best validation loss: {tuning_results['best_value']:.6f}")
        print(f"    • Trials completed: {tuning_results['n_trials']}")
    
    if best_model_results:
        print(f"  Best Model:")
        print(f"    • Test R²: {best_model_results['test_metrics']['r2']:.4f}")
        print(f"    • Test RMSE: {best_model_results['test_metrics']['rmse']:.6f}")
        print(f"    • Training time: {best_model_results['training_time']:.1f}s")
    
    if ensemble_results:
        print(f"  Ensemble Model:")
        print(f"    • Final R²: {ensemble_results['final_r2']:.4f}")
        print(f"    • Average CV R²: {ensemble_results['avg_fold_r2']:.4f}")
        print(f"    • Training time: {ensemble_results['training_time']:.1f}s")
        
        # Calculate improvement
        if best_model_results:
            improvement = ensemble_results['final_r2'] - best_model_results['test_metrics']['r2']
            print(f"    • Improvement over single model: {improvement:+.4f}")
    
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"  • Hyperparameters: Hyperparameter_Tuning/output/")
    print(f"  • Best model: Train_Best_Model/output/pytorch_best_model/")
    print(f"  • Ensemble: Train_Ensemble/output/pytorch_ensemble/")
    
    print(f"\n📖 DOCUMENTATION:")
    print(f"  • PyTorch pipeline guide: PYTORCH_README.md")
    print(f"  • Original TensorFlow code: preserved in original files")
    
    print(f"\n🎉 PyTorch pipeline completed successfully!")


def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description='Run complete PyTorch rainfall prediction pipeline')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run in quick test mode (fewer trials/epochs)')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip hyperparameter tuning (use existing hyperparameters)')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble training')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of hyperparameter tuning trials')
    
    args = parser.parse_args()
    
    print("PyTorch Rainfall Prediction Pipeline")
    print("=" * 60)
    print(f"Mode: {'QUICK TEST' if args.quick_test else 'FULL PIPELINE'}")
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    overall_start = time.time()
    tuning_results = None
    best_model_results = None
    ensemble_results = None
    
    # Step 1: Hyperparameter tuning
    if not args.skip_tuning:
        success, tuning_results = run_hyperparameter_tuning(
            n_trials=args.n_trials, 
            quick_test=args.quick_test
        )
        if not success:
            print("❌ Pipeline failed at hyperparameter tuning step")
            return 1
    else:
        print("\n⏭️  Skipping hyperparameter tuning (using existing hyperparameters)")
    
    # Step 2: Best model training
    success, best_model_results = run_best_model_training(quick_test=args.quick_test)
    if not success:
        print("❌ Pipeline failed at best model training step")
        return 1
    
    # Step 3: Ensemble training
    if not args.skip_ensemble:
        success, ensemble_results = run_ensemble_training(quick_test=args.quick_test)
        if not success:
            print("❌ Pipeline failed at ensemble training step")
            return 1
    else:
        print("\n⏭️  Skipping ensemble training")
    
    # Final summary
    total_time = time.time() - overall_start
    print(f"\n⏱️  Total pipeline time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print_final_summary(tuning_results, best_model_results, ensemble_results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
