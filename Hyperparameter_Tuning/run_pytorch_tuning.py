#!/usr/bin/env python3
"""
Main entry point for PyTorch hyperparameter tuning.
"""

import os
import sys
from Hyperparameter_Tuning.pytorch_hyperparameter_tuning import run_hyperparameter_tuning


def main():
    """Run PyTorch hyperparameter tuning."""
    print("Starting PyTorch Hyperparameter Tuning")
    print("=" * 50)
    
    # Default configuration
    config = {
        'npz_path': os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz'),
        'output_dir': os.path.join('Hyperparameter_Tuning', 'output'),
        'test_indices_path': os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl'),
        'n_trials': 100,
        'n_folds': 5,
        'max_epochs': 100,
        'patience': 10,
        'resume': True
    }
    
    # Check if NPZ file exists
    if not os.path.exists(config['npz_path']):
        print(f"ERROR: NPZ file not found at {config['npz_path']}")
        print("Please run the data preprocessing pipeline first:")
        print("  1. Process rainfall data")
        print("  2. Build DEM patches")
        print("  3. Build reanalysis features")
        print("  4. Assemble training data")
        return 1
    
    try:
        # Run hyperparameter tuning
        results = run_hyperparameter_tuning(**config)
        
        print(f"\nHyperparameter tuning completed successfully!")
        print(f"Best validation loss: {results['best_value']:.6f}")
        print(f"Completed {results['n_trials']} trials")
        print(f"Results saved to: {config['output_dir']}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Hyperparameter tuning failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
