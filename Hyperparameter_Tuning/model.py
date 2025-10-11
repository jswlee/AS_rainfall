#!/usr/bin/env python3
"""
PyTorch implementation of the LAND-inspired rainfall prediction model.

This module provides:
- LANDModel: Multi-branch neural network for rainfall prediction
- Support for different climate processing methods (flatten/conv2d)
- Configurable activation functions and output constraints
- Model factory function for hyperparameter-driven creation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ================================================================
# LAND Model Architecture
# ================================================================

class LANDModel(nn.Module):
    """
    PyTorch implementation of the LAND-inspired model for rainfall prediction.
    
    The model processes:
    - Climate/reanalysis patches (16 variables on 3x3 grid)
    - Local DEM patches (topographic context around station)
    - Regional DEM patches (broader topographic context)
    - Temporal one-hot encodings (seasonal information)
    """
    
    def __init__(self, 
                 climate_units: int,
                 local_dem_units: int,
                 regional_dem_units: int,
                 temporal_units: int,
                 na: int,
                 nb: int,
                 dropout_rate: float,
                 l2_reg: float,
                 use_residual: bool,
                 climate_activation: str,
                 output_activation: Optional[str],
                 climate_processing: str,
                 climate_shape: tuple = (16, 3, 3),
                 local_dem_shape: tuple = (3, 3),
                 regional_dem_shape: tuple = (3, 3),
                 num_temporal_encodings: int = 12):
        """
        Initialize the LAND model.
        
        Args:
            
            climate_units: Hidden units for climate processing
            local_dem_units: Hidden units for local DEM processing
            regional_dem_units: Hidden units for regional DEM processing
            temporal_units: Hidden units for temporal processing
            na: Hidden units for first dense layer
            nb: Hidden units for second dense layer
            dropout_rate: Dropout rate
            l2_reg: L2 regularization strength (applied via weight decay in optimizer)
            use_residual: Whether to use residual connection between na and nb layers
            climate_activation: Activation for climate/reanalysis branch only ('relu' or 'none')
            output_activation: Output activation ('relu', 'softplus', None)
            climate_processing: Method to process climate data ('flatten' or 'conv2d')
            climate_shape: Shape of climate/reanalysis patches (channels, height, width)
            local_dem_shape: Shape of local DEM patches (height, width)
            regional_dem_shape: Shape of regional DEM patches (height, width)
            num_temporal_encodings: Number of temporal one-hot encodings (2 for day_cyc, 12 for month_onehot)
        """
        super(LANDModel, self).__init__()
        
        self.climate_shape = climate_shape
        self.local_dem_shape = local_dem_shape
        self.regional_dem_shape = regional_dem_shape
        self.num_temporal_encodings = num_temporal_encodings
        self.use_residual = use_residual and (na == nb)  # Only use residual if dimensions match
        self.output_activation = output_activation
        self.climate_processing = climate_processing
        
        # Climate branch activation can be independently set to 'relu' or 'none'
        self.climate_activation_fn = self._get_optional_activation(climate_activation)
        
        # ----------------------------------------------------------------
        # Climate/Reanalysis Branch Architecture
        # ----------------------------------------------------------------
        # Support both flatten and conv2d processing options
        if climate_processing == 'conv2d':
            # Depthwise-style Conv2D: one group per input channel (variable)
            in_ch = climate_shape[0]
            groups = in_ch
            # Ensure out_channels is divisible by groups as required by PyTorch
            out_ch = climate_units
            if out_ch % groups != 0:
                out_ch = ((out_ch + groups - 1) // groups) * groups
            self.climate_conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(3, 3),
                stride=1,
                padding=0,   # No padding since input is exactly 3x3
                dilation=1,
                groups=groups,  # Channel-wise processing
                bias=True,
                padding_mode='zeros'
            )
            self.climate_bn = nn.BatchNorm1d(num_features=out_ch)
            self._climate_units_out = out_ch
        else:  # 'flatten' option
            # Simple flatten and dense layer processing
            climate_input_size = climate_shape[0] * climate_shape[1] * climate_shape[2]
            self.climate_fc = nn.Linear(in_features=climate_input_size, out_features=climate_units)
            self.climate_bn = nn.BatchNorm1d(num_features=climate_units)
            self._climate_units_out = climate_units
        # Second dense stage for climate branch (applied after BN+activation of the first stage)
        self.climate_fc2 = nn.Linear(in_features=self._climate_units_out, out_features=self._climate_units_out)
        self.climate_bn2 = nn.BatchNorm1d(num_features=self._climate_units_out)
        
        # ----------------------------------------------------------------
        # Local DEM Branch Architecture
        # ----------------------------------------------------------------
        local_dem_input_size = local_dem_shape[0] * local_dem_shape[1]
        self.local_dem_fc = nn.Linear(in_features=local_dem_input_size, out_features=local_dem_units)
        self.local_dem_bn = nn.BatchNorm1d(num_features=local_dem_units)
        # Second dense stage for local DEM branch
        self.local_dem_fc2 = nn.Linear(in_features=local_dem_units, out_features=local_dem_units)
        self.local_dem_bn2 = nn.BatchNorm1d(num_features=local_dem_units)
        
        # ----------------------------------------------------------------
        # Regional DEM Branch Architecture
        # ----------------------------------------------------------------
        regional_dem_input_size = regional_dem_shape[0] * regional_dem_shape[1]
        self.regional_dem_fc = nn.Linear(in_features=regional_dem_input_size, out_features=regional_dem_units)
        self.regional_dem_bn = nn.BatchNorm1d(num_features=regional_dem_units)
        # Second dense stage for regional DEM branch
        self.regional_dem_fc2 = nn.Linear(in_features=regional_dem_units, out_features=regional_dem_units)
        self.regional_dem_bn2 = nn.BatchNorm1d(num_features=regional_dem_units)
        
        # ----------------------------------------------------------------
        # Month/Temporal Branch Architecture
        # ----------------------------------------------------------------
        self.month_fc = nn.Linear(in_features=num_temporal_encodings, out_features=temporal_units)
        self.month_bn = nn.BatchNorm1d(num_features=temporal_units)
        
        # ----------------------------------------------------------------
        # Combined Feature Processing (Dense Head)
        # ----------------------------------------------------------------
        combined_size = self._climate_units_out + local_dem_units + regional_dem_units + temporal_units
        self.fc1 = nn.Linear(in_features=combined_size, out_features=na)
        self.bn1 = nn.BatchNorm1d(num_features=na)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(in_features=na, out_features=nb)
        self.bn2 = nn.BatchNorm1d(num_features=nb)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Output layer
        self.output = nn.Linear(in_features=nb, out_features=1)
        
        # ----------------------------------------------------------------
        # Weight Initialization
        # ----------------------------------------------------------------
        self._initialize_weights()
    
    def _get_optional_activation(self, activation: str):
        """Get activation function where 'none' maps to identity."""
        if activation == 'none':
            return lambda x: x
        elif activation == 'relu':
            return F.relu
        else:
            raise ValueError(f"Unknown climate_activation: {activation}. Use 'relu' or 'none'.")
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(tensor=module.weight)
                if module.bias is not None:
                    nn.init.zeros_(tensor=module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(tensor=module.weight)
                nn.init.zeros_(tensor=module.bias)
    
    # ================================================================
    # Forward Pass Implementation
    # ================================================================
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            features: Dictionary containing:
                - 'climate': Climate patches (batch_size, 16, 3, 3)
                - 'local_dem': Local DEM patches (batch_size, H, W)
                - 'regional_dem': Regional DEM patches (batch_size, H, W)
                - 'month': Month one-hot encodings (batch_size, 12)
        
        Returns:
            Rainfall predictions (batch_size, 1)
        """
        # ----------------------------------------------------------------
        # Climate/Reanalysis Branch Processing
        # ----------------------------------------------------------------
        climate = features['climate']  # Shape: (batch_size, 16, 3, 3)
        
        if self.climate_processing == 'conv2d':
            # Conv2D processing
            climate_out = self.climate_conv(climate)  # Conv2D output: (batch_size, climate_units, 1, 1)
            climate_out = climate_out.view(climate_out.size(0), -1)  # Flatten to (batch_size, climate_units)
        else:  # 'flatten' option
            # Simple flatten and dense layer
            climate_flat = climate.view(climate.size(0), -1)  # Flatten to (batch_size, 16*3*3)
            climate_out = self.climate_fc(climate_flat)
            
        # Common processing for both methods
        climate_out = self.climate_bn(climate_out)
        climate_out = self.climate_activation_fn(climate_out)
        # Second dense stage for climate branch
        climate_out = self.climate_fc2(climate_out)
        climate_out = self.climate_bn2(climate_out)
        climate_out = self.climate_activation_fn(climate_out)
        
        # ----------------------------------------------------------------
        # Local DEM Branch Processing
        # ----------------------------------------------------------------
        local_dem = features['local_dem']
        local_dem_flat = local_dem.view(local_dem.size(0), -1)  # Flatten to (batch_size, H*W)
        local_dem_out = self.local_dem_fc(local_dem_flat)
        local_dem_out = self.local_dem_bn(local_dem_out)
        # Non-climate branches always use ReLU
        local_dem_out = F.relu(local_dem_out)
        # Second dense stage for local DEM branch
        local_dem_out = self.local_dem_fc2(local_dem_out)
        local_dem_out = self.local_dem_bn2(local_dem_out)
        local_dem_out = F.relu(local_dem_out)
        
        # ----------------------------------------------------------------
        # Regional DEM Branch Processing
        # ----------------------------------------------------------------
        regional_dem = features['regional_dem']
        regional_dem_flat = regional_dem.view(regional_dem.size(0), -1)  # Flatten to (batch_size, H*W)
        regional_dem_out = self.regional_dem_fc(regional_dem_flat)
        regional_dem_out = self.regional_dem_bn(regional_dem_out)
        regional_dem_out = F.relu(regional_dem_out)
        # Second dense stage for regional DEM branch
        regional_dem_out = self.regional_dem_fc2(regional_dem_out)
        regional_dem_out = self.regional_dem_bn2(regional_dem_out)
        regional_dem_out = F.relu(regional_dem_out)
        
        # ----------------------------------------------------------------
        # Temporal Branch Processing
        # ----------------------------------------------------------------
        temporal = features['temporal']
        month_out = self.month_fc(temporal)
        month_out = self.month_bn(month_out)
        month_out = F.relu(month_out)
        
        # ----------------------------------------------------------------
        # Feature Fusion and Dense Processing
        # ----------------------------------------------------------------
        # Concatenate all features
        combined = torch.cat(tensors=[climate_out, local_dem_out, regional_dem_out, month_out], dim=1)
        
        # First dense layer
        x = self.fc1(combined)
        x = self.bn1(x)
        # Dense head uses ReLU as requested
        x = F.relu(x)

        # Store for potential residual connection
        residual = x if self.use_residual else None
        
        # Second dense layer
        x = self.fc2(x)
        x = self.bn2(x)
        
        # Add residual connection if enabled and dimensions match
        if self.use_residual and residual is not None:
            x = x + residual
        
        x = F.relu(x)
        x = self.dropout2(x)
        
        # ----------------------------------------------------------------
        # Output Layer and Activation
        # ----------------------------------------------------------------
        output = self.output(x)
        
        # Apply output activation if specified
        if self.output_activation == 'relu':
            output = F.relu(input=output)
        elif self.output_activation == 'softplus':
            output = F.softplus(input=output)
        # No activation for None case
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================
# Model Factory Function
# ================================================================

def create_model_from_hyperparams(hyperparams: Dict, metadata: Dict) -> LANDModel:
    """
    Create a LAND model from hyperparameters and data metadata.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        metadata: Dictionary of data metadata
        
    Returns:
        Initialized LAND model
    """
    # Validate required hyperparameters are present
    required_keys = [
        'climate_units', 'local_dem_units', 'regional_dem_units',
        'na', 'nb', 'dropout_rate', 'l2_reg', 'use_residual',
        'output_activation', 'climate_processing'
        # 'climate_activation' is optional; defaults to 'relu'
    ]
    missing = [k for k in required_keys if k not in hyperparams]
    if missing:
        raise ValueError(f"Missing required hyperparameters: {missing}")

    # Backward compatibility: allow either 'temporal_units' or legacy 'month_units'
    temporal_units = hyperparams.get('temporal_units', hyperparams.get('month_units'))
    if temporal_units is None:
        raise ValueError("Missing required hyperparameter: 'temporal_units' (or legacy 'month_units')")

    model = LANDModel(
        climate_units=hyperparams['climate_units'],
        local_dem_units=hyperparams['local_dem_units'],
        regional_dem_units=hyperparams['regional_dem_units'],
        temporal_units=temporal_units,
        na=hyperparams['na'],
        nb=hyperparams['nb'],
        dropout_rate=hyperparams['dropout_rate'],
        l2_reg=hyperparams['l2_reg'],
        use_residual=hyperparams['use_residual'],
        climate_activation=hyperparams.get('climate_activation', 'relu'),
        output_activation=hyperparams['output_activation'],
        climate_processing=hyperparams['climate_processing'],
        climate_shape=metadata['climate_shape'],
        local_dem_shape=metadata['local_dem_shape'],
        regional_dem_shape=metadata['regional_dem_shape'],
        num_temporal_encodings=metadata['num_temporal_encodings'],
    )
    
    return model


if __name__ == "__main__":
    print("LANDModel module loaded successfully.")
    print("Import this module to use LANDModel and create_model_from_hyperparams().")
