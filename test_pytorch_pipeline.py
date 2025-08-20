#!/usr/bin/env python3
"""
End-to-end test script for the PyTorch rainfall prediction pipeline.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path

def test_data_loading():
    """Test PyTorch data loading utilities."""
    print("Testing PyTorch data loading...")
    
    # Add hyperparameter tuning to path
    sys.path.append('Hyperparameter_Tuning')
    
    try:
        from pytorch_data_utils import load_assembled_npz_data_pytorch, create_pytorch_dataloaders
        
        # Check if NPZ file exists
        npz_path = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz')
        if not os.path.exists(npz_path):
            print(f"  ❌ NPZ file not found at {npz_path}")
            print("  Please run data preprocessing first")
            return False
        
        # Test data loading
        data = load_assembled_npz_data_pytorch(
            npz_path=npz_path,
            test_indices_path=None,  # Don't save test indices for testing
            random_state=42
        )
        
        # Verify data structure
        assert 'datasets' in data
        assert 'metadata' in data
        assert all(split in data['datasets'] for split in ['train', 'val', 'test'])
        
        # Test dataloader creation
        dataloaders = create_pytorch_dataloaders(data['datasets'], batch_size=16)
        
        # Test a sample batch
        for features, targets in dataloaders['train']:
            assert isinstance(features, dict)
            assert all(key in features for key in ['climate', 'local_dem', 'regional_dem', 'month'])
            assert isinstance(targets, torch.Tensor)
            break
        
        print("  ✅ Data loading works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False


def test_model_creation():
    """Test PyTorch model creation."""
    print("Testing PyTorch model creation...")
    
    try:
        from pytorch_model import LANDModel, create_model_from_hyperparams
        
        # Test basic model creation
        model = LANDModel()
        
        # Test forward pass with dummy data
        batch_size = 4
        dummy_features = {
            'climate': torch.randn(batch_size, 16, 3, 3),
            'local_dem': torch.randn(batch_size, 3, 3),
            'regional_dem': torch.randn(batch_size, 3, 3),
            'month': torch.randn(batch_size, 12)
        }
        
        with torch.no_grad():
            output = model(dummy_features)
            assert output.shape == (batch_size, 1)
        
        # Test model creation from hyperparameters
        hyperparams = {
            'climate_units': 128,
            'local_dem_units': 64,
            'regional_dem_units': 64,
            'month_units': 32,
            'na': 256,
            'nb': 128,
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'relu'
        }
        
        metadata = {
            'climate_shape': (16, 3, 3),
            'local_dem_shape': (3, 3),
            'regional_dem_shape': (3, 3),
            'num_month_encodings': 12
        }
        
        model2 = create_model_from_hyperparams(hyperparams, metadata)
        
        with torch.no_grad():
            output2 = model2(dummy_features)
            assert output2.shape == (batch_size, 1)
        
        print("  ✅ Model creation works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False


def test_training_utilities():
    """Test PyTorch training utilities."""
    print("Testing PyTorch training utilities...")
    
    try:
        from pytorch_training import train_model, evaluate_model, EarlyStopping, CosineAnnealingWarmup
        from pytorch_model import LANDModel
        from pytorch_data_utils import RainfallDataset
        from torch.utils.data import DataLoader
        
        # Create dummy datasets
        n_samples = 100
        dummy_climate = np.random.randn(n_samples, 16, 3, 3).astype(np.float32)
        dummy_local_dem = np.random.randn(n_samples, 3, 3).astype(np.float32)
        dummy_regional_dem = np.random.randn(n_samples, 3, 3).astype(np.float32)
        dummy_month = np.random.randn(n_samples, 12).astype(np.float32)
        dummy_targets = np.random.randn(n_samples).astype(np.float32)
        
        train_dataset = RainfallDataset(
            dummy_climate[:80], dummy_local_dem[:80], dummy_regional_dem[:80],
            dummy_month[:80], dummy_targets[:80]
        )
        val_dataset = RainfallDataset(
            dummy_climate[80:], dummy_local_dem[80:], dummy_regional_dem[80:],
            dummy_month[80:], dummy_targets[80:]
        )
        
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=16, shuffle=False)
        }
        
        # Test training for a few epochs
        model = LANDModel()
        history = train_model(
            model=model,
            dataloaders=dataloaders,
            epochs=3,
            learning_rate=0.001,
            verbose=False
        )
        
        # Verify history structure
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) <= 3
        
        # Test evaluation
        metrics = evaluate_model(model, dataloaders['val'])
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        
        print("  ✅ Training utilities work correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Training utilities failed: {e}")
        return False


def test_hyperparameter_tuning():
    """Test Optuna hyperparameter tuning (minimal test)."""
    print("Testing hyperparameter tuning setup...")
    
    try:
        from pytorch_hyperparameter_tuning import OptunaTuner, load_best_hyperparameters_pytorch
        import optuna
        
        # Test hyperparameter loading (should fail gracefully if no file exists)
        try:
            hyperparams = load_best_hyperparameters_pytorch('nonexistent_dir')
            print("  ⚠️  Unexpected: loaded hyperparameters from nonexistent directory")
        except FileNotFoundError:
            pass  # Expected behavior
        
        # Test Optuna trial parameter suggestion
        study = optuna.create_study(direction='minimize')
        trial = study.ask()
        
        # Create a minimal tuner instance (without actually running tuning)
        npz_path = os.path.join('ML_Data_Preprocessing', 'output', 'assembled_npz', 'full_training_data.npz')
        if os.path.exists(npz_path):
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    tuner = OptunaTuner(
                        npz_path=npz_path,
                        output_dir=temp_dir,
                        n_folds=2,  # Minimal for testing
                        max_epochs=2,
                        patience=1
                    )
                    
                    # Test hyperparameter suggestion
                    hyperparams = tuner.suggest_hyperparameters(trial)
                    assert isinstance(hyperparams, dict)
                    assert 'climate_units' in hyperparams
                    
                    print("  ✅ Hyperparameter tuning setup works correctly")
                    return True
                except Exception as e:
                    print(f"  ⚠️  Hyperparameter tuning setup failed: {e}")
                    return False
        else:
            print("  ⚠️  NPZ file not found, skipping hyperparameter tuning test")
            return True
            
    except Exception as e:
        print(f"  ❌ Hyperparameter tuning test failed: {e}")
        return False


def test_file_structure():
    """Test that all required PyTorch files exist."""
    print("Testing PyTorch file structure...")
    
    required_files = [
        'Hyperparameter_Tuning/pytorch_data_utils.py',
        'Hyperparameter_Tuning/pytorch_model.py',
        'Hyperparameter_Tuning/pytorch_training.py',
        'Hyperparameter_Tuning/pytorch_hyperparameter_tuning.py',
        'Hyperparameter_Tuning/run_pytorch_tuning.py',
        'Train_Best_Model/pytorch_train_best_model.py',
        'Train_Best_Model/run_pytorch_training.py',
        'Train_Ensemble/pytorch_train_ensemble.py',
        'Train_Ensemble/run_pytorch_ensemble.py',
        'PYTORCH_README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print("  ✅ All required PyTorch files exist")
        return True


def test_imports():
    """Test that all PyTorch modules can be imported."""
    print("Testing PyTorch module imports...")
    
    try:
        # Test core PyTorch
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        # Test Optuna
        import optuna
        print(f"  Optuna version: {optuna.__version__}")
        
        # Test scikit-learn
        import sklearn
        print(f"  Scikit-learn version: {sklearn.__version__}")
        
        # Test our modules
        sys.path.append('Hyperparameter_Tuning')
        from pytorch_data_utils import RainfallDataset
        from pytorch_model import LANDModel
        from pytorch_training import train_model
        from pytorch_hyperparameter_tuning import OptunaTuner
        
        print("  ✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("PyTorch Pipeline End-to-End Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Training Utilities", test_training_utilities),
        ("Data Loading", test_data_loading),
        ("Hyperparameter Tuning", test_hyperparameter_tuning),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("Test Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! PyTorch pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run hyperparameter tuning: cd Hyperparameter_Tuning && python run_pytorch_tuning.py")
        print("2. Train best model: cd Train_Best_Model && python run_pytorch_training.py")
        print("3. Train ensemble: cd Train_Ensemble && python run_pytorch_ensemble.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before using the pipeline.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
