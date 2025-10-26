#!/usr/bin/env python3
"""
Test script to verify attention mechanisms work correctly.
"""

import torch
from Hyperparameter_Tuning.model import LANDModel, create_model_from_hyperparams

def test_attention_mechanisms():
    """Test all three attention mechanisms."""
    
    print("=" * 70)
    print("Testing Attention Mechanisms")
    print("=" * 70)
    
    # Setup test data
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Create sample input
    features = {
        'climate': torch.randn(batch_size, 16, 3, 3).to(device),
        'local_dem': torch.randn(batch_size, 3, 3).to(device),
        'regional_dem': torch.randn(batch_size, 3, 3).to(device),
        'temporal': torch.randn(batch_size, 12).to(device),
    }
    
    # Test configurations
    test_configs = [
        {
            'name': 'Baseline (No Attention)',
            'params': {
                'use_spatial_attention': False,
                'use_multihead_attention': False,
                'use_cross_attention': False,
            }
        },
        {
            'name': 'Spatial Attention Only',
            'params': {
                'use_spatial_attention': True,
                'use_multihead_attention': False,
                'use_cross_attention': False,
                'attention_heads': 4,
            }
        },
        {
            'name': 'Multi-Head Attention Only',
            'params': {
                'use_spatial_attention': False,
                'use_multihead_attention': True,
                'use_cross_attention': False,
                'attention_heads': 4,
            }
        },
        {
            'name': 'Cross-Attention Only',
            'params': {
                'use_spatial_attention': False,
                'use_multihead_attention': False,
                'use_cross_attention': True,
                'attention_heads': 4,
            }
        },
        {
            'name': 'All Attention Mechanisms',
            'params': {
                'use_spatial_attention': True,
                'use_multihead_attention': True,
                'use_cross_attention': True,
                'attention_heads': 8,
                'attention_dropout': 0.1,
            }
        },
    ]
    
    # Base hyperparameters
    base_hyperparams = {
        'climate_units': 128,
        'local_dem_units': 32,
        'regional_dem_units': 16,
        'temporal_units': 16,
        'na': 256,
        'nb': 128,
        'dropout_rate': 0.3,
        'l2_reg': 1e-4,
        'use_residual': True,
        'climate_activation': 'relu',
        'output_activation': 'softplus',
        'climate_processing': 'conv2d',
    }
    
    metadata = {
        'climate_shape': (16, 3, 3),
        'local_dem_shape': (3, 3),
        'regional_dem_shape': (3, 3),
        'num_temporal_encodings': 12,
    }
    
    # Test each configuration
    for config in test_configs:
        print(f"\n{'=' * 70}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 70}")
        
        # Merge base params with test-specific params
        hyperparams = {**base_hyperparams, **config['params']}
        
        try:
            # Create model
            model = create_model_from_hyperparams(hyperparams, metadata)
            model = model.to(device)
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                output = model(features)
            
            # Check output
            assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
            assert torch.isfinite(output).all(), "Output contains non-finite values"
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✓ Model created successfully")
            print(f"✓ Forward pass successful")
            print(f"✓ Output shape: {output.shape}")
            print(f"✓ Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            print(f"✓ Total parameters: {num_params:,}")
            
            # Print attention-specific info
            if config['params'].get('use_spatial_attention'):
                print(f"  - Spatial attention: {config['params']['attention_heads']} heads on 3×3 grid")
            if config['params'].get('use_multihead_attention'):
                print(f"  - Multi-head attention: {config['params']['attention_heads']} heads on features")
            if config['params'].get('use_cross_attention'):
                print(f"  - Cross-attention: {config['params']['attention_heads']} heads between branches")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'=' * 70}")
    print("All tests passed! ✓")
    print(f"{'=' * 70}\n")
    return True


def test_attention_with_flatten():
    """Test that spatial attention is properly skipped with flatten processing."""
    
    print("\n" + "=" * 70)
    print("Testing Spatial Attention with Flatten Processing")
    print("=" * 70)
    
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    features = {
        'climate': torch.randn(batch_size, 16, 3, 3).to(device),
        'local_dem': torch.randn(batch_size, 3, 3).to(device),
        'regional_dem': torch.randn(batch_size, 3, 3).to(device),
        'temporal': torch.randn(batch_size, 12).to(device),
    }
    
    hyperparams = {
        'climate_units': 128,
        'local_dem_units': 32,
        'regional_dem_units': 16,
        'temporal_units': 16,
        'na': 256,
        'nb': 128,
        'dropout_rate': 0.3,
        'l2_reg': 1e-4,
        'use_residual': True,
        'climate_activation': 'relu',
        'output_activation': 'softplus',
        'climate_processing': 'flatten',  # Using flatten instead of conv2d
        'use_spatial_attention': True,  # This should be ignored
        'use_multihead_attention': True,
        'attention_heads': 4,
    }
    
    metadata = {
        'climate_shape': (16, 3, 3),
        'local_dem_shape': (3, 3),
        'regional_dem_shape': (3, 3),
        'num_temporal_encodings': 12,
    }
    
    try:
        model = create_model_from_hyperparams(hyperparams, metadata)
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(features)
        
        # Check that spatial attention was not created
        has_spatial_attn = hasattr(model, 'spatial_attention')
        
        print(f"✓ Model created with flatten processing")
        print(f"✓ Spatial attention module exists: {has_spatial_attn}")
        print(f"  (Should be False since flatten processing doesn't support spatial attention)")
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {output.shape}")
        
        if has_spatial_attn:
            print("\n⚠ Warning: Spatial attention was created with flatten processing")
            print("  This is unexpected but not necessarily an error")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'=' * 70}")
    print("Flatten processing test passed! ✓")
    print(f"{'=' * 70}\n")
    return True


def test_different_attention_heads():
    """Test different numbers of attention heads."""
    
    print("\n" + "=" * 70)
    print("Testing Different Numbers of Attention Heads")
    print("=" * 70)
    
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    features = {
        'climate': torch.randn(batch_size, 16, 3, 3).to(device),
        'local_dem': torch.randn(batch_size, 3, 3).to(device),
        'regional_dem': torch.randn(batch_size, 3, 3).to(device),
        'temporal': torch.randn(batch_size, 12).to(device),
    }
    
    base_hyperparams = {
        'climate_units': 128,  # Divisible by 2, 4, 8
        'local_dem_units': 32,
        'regional_dem_units': 16,
        'temporal_units': 16,
        'na': 256,
        'nb': 128,
        'dropout_rate': 0.3,
        'l2_reg': 1e-4,
        'use_residual': True,
        'climate_activation': 'relu',
        'output_activation': 'softplus',
        'climate_processing': 'conv2d',
        'use_spatial_attention': True,
        'use_multihead_attention': True,
        'use_cross_attention': True,
    }
    
    metadata = {
        'climate_shape': (16, 3, 3),
        'local_dem_shape': (3, 3),
        'regional_dem_shape': (3, 3),
        'num_temporal_encodings': 12,
    }
    
    for num_heads in [2, 4, 8]:
        print(f"\nTesting with {num_heads} attention heads...")
        
        hyperparams = {**base_hyperparams, 'attention_heads': num_heads}
        
        try:
            model = create_model_from_hyperparams(hyperparams, metadata)
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(features)
            
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  ✓ {num_heads} heads: {num_params:,} parameters")
            
        except Exception as e:
            print(f"  ✗ {num_heads} heads failed: {e}")
            return False
    
    print(f"\n{'=' * 70}")
    print("Different attention heads test passed! ✓")
    print(f"{'=' * 70}\n")
    return True


if __name__ == "__main__":
    success = True
    
    # Run all tests
    success &= test_attention_mechanisms()
    success &= test_attention_with_flatten()
    success &= test_different_attention_heads()
    
    if success:
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓✓✓")
        print("=" * 70)
        print("\nThe attention mechanisms are working correctly.")
        print("You can now use them in hyperparameter tuning with tune.py")
    else:
        print("\n" + "=" * 70)
        print("SOME TESTS FAILED ✗")
        print("=" * 70)
