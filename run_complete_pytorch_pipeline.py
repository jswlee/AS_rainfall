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


def run_hyperparameter_tuning(n_trials=100, quick_test=False, *, loss_name: str = 'mse', loss_params: dict | None = None, enable_mlflow: bool = False, mlflow_experiment: str = 'AS_Rainfall'):
    """Run hyperparameter tuning."""
    print("\n" + "="*60)
    print("STEP 1: HYPERPARAMETER TUNING")
    print("="*60)

    # Adjust parameters for quick testing
    if quick_test:
        config = {
            'n_trials': 5,
            'n_folds': 3,
            'max_epochs': 150,
            'patience': 30,
            'loss_name': loss_name,
            'loss_params': loss_params
        }
        print("Running in QUICK TEST mode (reduced trials/epochs)")
    else:
        config = {
            'n_trials': n_trials,
            'n_folds': 3,
            'max_epochs': 150,
            'patience': 30,
            'loss_name': loss_name,
            'loss_params': loss_params
        }
    
    print(f"Starting hyperparameter tuning with {config['n_trials']} trials and {config['n_folds']} folds...")
    start_time = time.time()
    
    try:
        results = run_hp_tuning(**config)
        tuning_time = time.time() - start_time
        
        print(f"Hyperparameter tuning completed in {tuning_time:.1f} seconds")
        print(f"Best validation loss: {results['best_value']:.6f}")
        print(f"Best hyperparameters saved to: Hyperparameter_Tuning/output/")
        
        return True, results
        
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        return False, None


def run_best_model_training(
    quick_test=False,
    *,
    loss_name: str = 'mse',
    loss_params: dict | None = None,
    hp_dir: str = 'Hyperparameter_Tuning/output',
    enable_mlflow: bool = False,
    mlflow_experiment: str = 'AS_Rainfall',
    mlflow_run_name: str | None = None,
):
    """Train the best model."""
    print("\n" + "="*60)
    print("STEP 2: BEST MODEL TRAINING")
    print("="*60)
    
    from Train_Best_Model.pytorch_train_best_model import train_best_model_pytorch
    
    epochs = 300 if quick_test else 300
    print(f"Training best model for up to {epochs} epochs...")
    
    start_time = time.time()
    
    try:
        results = train_best_model_pytorch(
            hyperparams_dir=hp_dir,
            epochs=epochs,
            loss_name=loss_name,
            loss_params=loss_params,
            enable_mlflow=enable_mlflow,
            mlflow_experiment=mlflow_experiment,
            mlflow_run_name=mlflow_run_name,
        )
        training_time = time.time() - start_time
        
        print(f"Best model training completed in {training_time:.1f} seconds")
        print(f"Test R²: {results['test_metrics']['r2']:.4f}")
        print(f"Results saved to: Train_Best_Model/output/pytorch_best_model/")
        
        return True, results
        
    except Exception as e:
        print(f"Best model training failed: {e}")
        return False, None


def run_ensemble_training(
    quick_test=False,
    *,
    hp_dir: str = 'Hyperparameter_Tuning/output',
    loss_name: str = 'mse',
    loss_params: dict | None = None,
    enable_mlflow: bool = False,
    mlflow_experiment: str = 'AS_Rainfall',
    mlflow_run_name: str | None = None,
):
    """Train the ensemble."""
    print("\n" + "="*60)
    print("STEP 3: ENSEMBLE TRAINING")
    print("="*60)
    
    from Train_Ensemble.pytorch_train_ensemble import train_ensemble_pytorch
    
    if quick_test:
        config = {
            'hyperparams_dir': hp_dir,
            'n_folds': 3,
            'n_models_per_fold': 2,
            'epochs': 150,
            'loss_name': loss_name,
            'loss_params': loss_params,
            'mlflow_enabled': enable_mlflow,
            'mlflow_experiment': mlflow_experiment,
            'mlflow_run_name': mlflow_run_name,
        }
        print("Running in QUICK TEST mode (fewer folds/models/epochs)")
    else:
        config = {
            'hyperparams_dir': hp_dir,
            'n_folds': 10,
            'n_models_per_fold': 10,
            'epochs': 300,
            'loss_name': loss_name,
            'loss_params': loss_params,
            'mlflow_enabled': enable_mlflow,
            'mlflow_experiment': mlflow_experiment,
            'mlflow_run_name': mlflow_run_name,
        }
    
    total_models = config['n_folds'] * config['n_models_per_fold']
    print(f"Training ensemble with {total_models} models ({config['n_folds']} folds × {config['n_models_per_fold']} models/fold)...")
    
    start_time = time.time()
    
    try:
        results = train_ensemble_pytorch(**config)
        ensemble_time = time.time() - start_time
        
        print(f"Ensemble training completed in {ensemble_time:.1f} seconds")
        print(f"Final ensemble R²: {results['final_r2']:.4f}")
        print(f"Average CV R²: {results['avg_fold_r2']:.4f}")
        print(f"Results saved to: Train_Ensemble/output/pytorch_ensemble/")
        
        return True, results
        
    except Exception as e:
        print(f"Ensemble training failed: {e}")
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
    parser = argparse.ArgumentParser(description='Run PyTorch rainfall prediction pipeline (tuning, best model, ensemble)')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run in quick test mode (fewer trials/epochs)')
    # Execution mode flags
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--only-tuning', action='store_true', help='Run only hyperparameter tuning')
    mode_group.add_argument('--only-best', action='store_true', help='Run only best model training')
    mode_group.add_argument('--only-ensemble', action='store_true', help='Run only ensemble training')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of hyperparameter tuning trials')
    # MLflow tracking flags (optional)
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow experiment tracking')
    parser.add_argument('--mlflow-experiment', type=str, default='AS_Rainfall', help='MLflow experiment name')
    parser.add_argument('--mlflow-run-name', type=str, default=None, help='MLflow run name (optional)')
    # Loss configuration
    parser.add_argument('--loss-name', type=str, default='mse', choices=['mse', 'weighted_mse'],
                       help='Loss function to use')
    parser.add_argument('--loss-alpha', type=float, default=2.0,
                       help='WeightedMSE: alpha (strength of upweighting)')
    parser.add_argument('--loss-power', type=float, default=1.0,
                       help='WeightedMSE: power for exceedance growth')
    parser.add_argument('--loss-percentile', type=float, default=0.8,
                       help='WeightedMSE: percentile threshold (0-1)')
    # Set hyperparameter directory
    parser.add_argument('--hp-dir', type=str, default='Hyperparameter_Tuning/output',
                       help='Directory for hyperparameter tuning results')
    
    args = parser.parse_args()
    
    print("PyTorch Rainfall Prediction Pipeline")
    print("=" * 60)
    print(f"Mode: {'QUICK TEST' if args.quick_test else 'FULL'}")
    
    overall_start = time.time()
    tuning_results = None
    best_model_results = None
    ensemble_results = None
    
    # Build loss params
    loss_params = None
    if args.loss_name == 'weighted_mse':
        loss_params = {
            'alpha': args.loss_alpha,
            'power': args.loss_power,
            'percentile': args.loss_percentile,
        }

    # Determine execution flow based on flags
    if args.only_tuning:
        # Run tuning only
        success, tuning_results = run_hyperparameter_tuning(
            n_trials=args.n_trials,
            quick_test=args.quick_test,
            loss_name=args.loss_name,
            loss_params=loss_params,
            enable_mlflow=args.mlflow,
            mlflow_experiment=args.mlflow_experiment,
        )
        return 0 if success else 1

    if args.only_best:
        # Run best model only
        success, best_model_results = run_best_model_training(
            quick_test=args.quick_test,
            loss_name=args.loss_name,
            loss_params=loss_params,
            hp_dir=args.hp_dir,
            enable_mlflow=args.mlflow,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name=args.mlflow_run_name,
        )
        return 0 if success else 1

    if args.only_ensemble:
        # Run ensemble only (expects hyperparams already available at hp_dir)
        success, ensemble_results = run_ensemble_training(
            quick_test=args.quick_test,
            hp_dir=args.hp_dir,
            loss_name=args.loss_name,
            loss_params=loss_params,
            enable_mlflow=args.mlflow,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name=args.mlflow_run_name,
        )
        return 0 if success else 1

    # Default: run full pipeline (tuning -> best -> ensemble)
    success, tuning_results = run_hyperparameter_tuning(
        n_trials=args.n_trials,
        quick_test=args.quick_test,
        loss_name=args.loss_name,
        loss_params=loss_params,
        enable_mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
    )
    if not success:
        print("Pipeline failed at hyperparameter tuning step")
        return 1

    success, best_model_results = run_best_model_training(
        quick_test=args.quick_test,
        loss_name=args.loss_name,
        loss_params=loss_params,
        hp_dir=args.hp_dir,
        enable_mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )
    if not success:
        print("Pipeline failed at best model training step")
        return 1

    success, ensemble_results = run_ensemble_training(
        quick_test=args.quick_test,
        hp_dir=args.hp_dir,
        loss_name=args.loss_name,
        loss_params=loss_params,
        enable_mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )
    if not success:
        print("Pipeline failed at ensemble training step")
        return 1
    
    # Final summary
    total_time = time.time() - overall_start
    print(f"\n⏱️  Total pipeline time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print_final_summary(tuning_results, best_model_results, ensemble_results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
