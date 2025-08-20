#!/usr/bin/env python3
"""
Main entry point for PyTorch best model training.
"""

import os
import sys
from pytorch_train_best_model import train_best_model_pytorch


def main():
    """Run PyTorch best model training."""
    print("Starting PyTorch Best Model Training")
    print("=" * 50)
    
    # Default configuration
    config = {
        'npz_path': os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz'),
        'hyperparams_dir': os.path.join('Hyperparameter_Tuning', 'output'),
        'output_dir': os.path.join('Train_Best_Model', 'output', 'pytorch_best_model'),
        'test_indices_path': os.path.join('Hyperparameter_Tuning', 'output', 'test_indices.pkl'),
        'epochs': 150,
        'save_model': True
    }
    
    # Check if required files exist
    if not os.path.exists(config['npz_path']):
        print(f"ERROR: NPZ file not found at {config['npz_path']}")
        print("Please run the data preprocessing pipeline first.")
        return 1
    
    try:
        # Run training
        results = train_best_model_pytorch(**config)
        
        print(f"\nBest model training completed successfully!")
        print(f"Test R²: {results['test_metrics']['r2']:.4f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Results saved to: {config['output_dir']}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
