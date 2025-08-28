#!/usr/bin/env python3
"""
Main entry point for PyTorch ensemble training.
"""

import os
import sys
from pytorch_train_ensemble import train_ensemble_pytorch


def main():
    """Run PyTorch ensemble training."""
    print("Starting PyTorch Ensemble Training")
    print("=" * 50)
    
    # Default configuration
    config = {
        'npz_path': os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz'),
        'hyperparams_dir': os.path.join('Hyperparameter_Tuning', 'output'),
        'output_dir': os.path.join('Train_Ensemble', 'output', 'pytorch_ensemble'),
        'test_indices_path': os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl'),
        'n_folds': 5,
        'n_models_per_fold': 5,
        'epochs': 150,
        'resume': True
    }
    
    # Check if required files exist
    if not os.path.exists(config['npz_path']):
        print(f"ERROR: NPZ file not found at {config['npz_path']}")
        print("Please run the data preprocessing pipeline first.")
        return 1
    
    try:
        # Run ensemble training
        results = train_ensemble_pytorch(**config)
        
        print(f"\nEnsemble training completed successfully!")
        print(f"Final ensemble R²: {results['final_r2']:.4f}")
        print(f"Average CV R²: {results['avg_fold_r2']:.4f}")
        print(f"Total models trained: {config['n_folds'] * config['n_models_per_fold']}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Results saved to: {config['output_dir']}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Ensemble training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
