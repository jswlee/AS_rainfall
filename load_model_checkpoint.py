"""
Utility functions for loading model checkpoints.

This module provides functions to load saved model checkpoints that include
both the model weights and the hyperparameters/metadata needed to recreate
the model architecture.
"""

import torch
import json
import os
from Hyperparameter_Tuning.model import create_model_from_hyperparams


def load_model_checkpoint(checkpoint_path: str, device: str = 'cpu', model_py_path: str = None):
    """
    Load a complete model checkpoint including architecture and weights.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on ('cpu', 'cuda', etc.)
        model_py_path: Optional path to model_architecture.py if the current model.py
                      has changed. If provided, will use this version instead.
    
    Returns:
        dict with keys:
            - 'model': The loaded model (ready for inference)
            - 'hyperparameters': The hyperparameters used to create the model
            - 'metadata': The metadata (data shapes, etc.)
            - 'best_epoch': The epoch where best validation loss was achieved
            - 'best_val_loss': The best validation loss
            - 'loss_name': The loss function used during training
            - 'loss_params': Parameters for the loss function (if any)
    
    Example:
        >>> checkpoint = load_model_checkpoint('best_model.pth', device='cuda')
        >>> model = checkpoint['model']
        >>> model.eval()
        >>> predictions = model(features)
        
        >>> # If model.py has changed, use the saved version:
        >>> checkpoint = load_model_checkpoint(
        ...     'best_model.pth',
        ...     device='cuda',
        ...     model_py_path='output_dir/model_architecture.py'
        ... )
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # If a specific model architecture version is provided, use it
    if model_py_path is not None:
        if not os.path.exists(model_py_path):
            raise FileNotFoundError(f"Model architecture file not found: {model_py_path}")
        
        print(f"⚠️  Using archived model architecture from {model_py_path}")
        print("   (Current model.py may have changed)")
        
        # Import the archived model module
        import importlib.util
        spec = importlib.util.spec_from_file_location("archived_model", model_py_path)
        archived_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(archived_model)
        create_model_fn = archived_model.create_model_from_hyperparams
    else:
        # Use current model.py
        from Hyperparameter_Tuning.model import create_model_from_hyperparams
        create_model_fn = create_model_from_hyperparams
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a new-style checkpoint (with hyperparameters) or old-style (just weights)
    if isinstance(checkpoint, dict) and 'hyperparameters' in checkpoint:
        # New-style checkpoint with full information
        print("✅ Found complete checkpoint with hyperparameters and metadata")
        
        hyperparams = checkpoint['hyperparameters']
        metadata = checkpoint['metadata']
        
        # Recreate model architecture
        model = create_model_fn(hyperparams, metadata)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"   Model loaded successfully")
        print(f"   Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
        print(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
        
        return {
            'model': model,
            'hyperparameters': hyperparams,
            'metadata': metadata,
            'best_fold': checkpoint.get('best_fold'),
            'best_epoch': checkpoint.get('best_epoch'),
            'best_val_loss': checkpoint.get('best_val_loss'),
            'val_loss': checkpoint.get('val_loss'),
            'loss_name': checkpoint.get('loss_name'),
            'loss_params': checkpoint.get('loss_params'),
        }
    else:
        # Old-style checkpoint (just state_dict)
        print("⚠️  Found old-style checkpoint (weights only)")
        print("   You need to provide hyperparameters separately to load this model")
        print("   Use load_model_from_hyperparams() instead")
        raise ValueError(
            "This checkpoint only contains model weights. "
            "Use load_model_from_hyperparams() and provide the hyperparameters file."
        )


def load_model_from_hyperparams(
    checkpoint_path: str,
    hyperparams_path: str,
    metadata: dict,
    device: str = 'cpu'
):
    """
    Load a model from separate checkpoint and hyperparameters files.
    
    Use this for old-style checkpoints that only contain weights.
    
    Args:
        checkpoint_path: Path to the .pth file (weights only)
        hyperparams_path: Path to best_hyperparameters.json
        metadata: Dictionary with data shapes (climate_shape, etc.)
        device: Device to load the model on
    
    Returns:
        dict with 'model' and 'hyperparameters' keys
    
    Example:
        >>> metadata = {
        ...     'climate_shape': (16, 3, 3),
        ...     'local_dem_shape': (3, 3),
        ...     'regional_dem_shape': (3, 3),
        ...     'num_temporal_encodings': 12
        ... }
        >>> result = load_model_from_hyperparams(
        ...     'best_model.pth',
        ...     'best_hyperparameters.json',
        ...     metadata,
        ...     device='cuda'
        ... )
        >>> model = result['model']
    """
    print(f"Loading hyperparameters from {hyperparams_path}...")
    
    # Load hyperparameters
    with open(hyperparams_path, 'r') as f:
        data = json.load(f)
    
    # Handle both old and new JSON formats
    if 'best_params' in data:
        hyperparams = data['best_params']
    else:
        hyperparams = data
    
    print(f"Loading model weights from {checkpoint_path}...")
    
    # Create model architecture
    model = create_model_from_hyperparams(hyperparams, metadata)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both old-style (state_dict) and new-style (dict with 'model_state_dict')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    print("✅ Model loaded successfully")
    
    return {
        'model': model,
        'hyperparameters': hyperparams,
        'metadata': metadata
    }


def get_model_info(checkpoint_path: str):
    """
    Get information about a saved checkpoint without loading the full model.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
    
    Returns:
        dict with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'hyperparameters' in checkpoint:
        # New-style checkpoint
        info = {
            'type': 'complete_checkpoint',
            'has_hyperparameters': True,
            'has_metadata': 'metadata' in checkpoint,
            'best_fold': checkpoint.get('best_fold'),
            'best_epoch': checkpoint.get('best_epoch'),
            'best_val_loss': checkpoint.get('best_val_loss'),
            'loss_name': checkpoint.get('loss_name'),
            'hyperparameters': checkpoint.get('hyperparameters'),
        }
    else:
        # Old-style checkpoint
        info = {
            'type': 'weights_only',
            'has_hyperparameters': False,
            'has_metadata': False,
        }
    
    return info


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_model_checkpoint.py <checkpoint_path>")
        print("\nThis will display information about the checkpoint.")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    info = get_model_info(checkpoint_path)
    
    print(f"Checkpoint type: {info['type']}")
    print(f"Has hyperparameters: {info['has_hyperparameters']}")
    print(f"Has metadata: {info['has_metadata']}")
    
    if info['has_hyperparameters']:
        print(f"\nBest fold: {info.get('best_fold')}")
        print(f"Best epoch: {info.get('best_epoch')}")
        print(f"Best val loss: {info.get('best_val_loss'):.6f}")
        print(f"Loss function: {info.get('loss_name')}")
        
        print("\nHyperparameters:")
        for key, value in info['hyperparameters'].items():
            print(f"  {key}: {value}")
    else:
        print("\n⚠️  This is an old-style checkpoint (weights only)")
        print("You need to load hyperparameters separately")
