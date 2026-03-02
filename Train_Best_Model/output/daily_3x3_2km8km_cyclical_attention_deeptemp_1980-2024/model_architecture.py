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
import math


# ================================================================
# Attention Mechanisms
# ================================================================

class SpatialSelfAttention(nn.Module):
    """
    Spatial self-attention for 2D feature maps.
    Processes spatial relationships in climate data (3x3 grid).
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super(SpatialSelfAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Attended features of same shape
        """
        batch_size, channels, height, width = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Transpose for attention computation
        q = q.transpose(-2, -1)  # (batch, heads, hw, head_dim)
        k = k.transpose(-2, -1)  # (batch, heads, hw, head_dim)
        v = v.transpose(-2, -1)  # (batch, heads, hw, head_dim)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, hw, hw)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, hw, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch, hw, heads, head_dim)
        out = out.view(batch_size, height, width, channels)
        out = out.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        
        # Final projection
        out = self.out(out)
        
        return out + x  # Residual connection


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for flattened feature vectors.
    Processes relationships between all climate features.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Attended features of same shape
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)
        
        # Final projection
        out = self.out_proj(out)
        
        return out + x  # Residual connection


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
                 num_temporal_encodings: int = 12,
                 use_spatial_attention: bool = False,
                 use_multihead_attention: bool = False,
                 attention_heads: int = 4,
                 attention_dropout: float = 0.1,
                 temporal_depth: int = 2,
                 temporal_dropout: float = 0.1):
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
            num_temporal_encodings: Number of temporal encodings (2 for day_cyc, 12 for month_onehot)
            use_spatial_attention: Use spatial self-attention on climate 3x3 grid
            use_multihead_attention: Use multi-head attention on flattened climate features
            attention_heads: Number of attention heads for spatial/multihead attention
            attention_dropout: Dropout rate for spatial/multihead attention
            temporal_depth: Number of layers in temporal branch (1, 2, or 3)
            temporal_dropout: Dropout rate for temporal branch
        """
        super(LANDModel, self).__init__()
        
        self.climate_shape = climate_shape
        self.local_dem_shape = local_dem_shape
        self.regional_dem_shape = regional_dem_shape
        self.num_temporal_encodings = num_temporal_encodings
        self.use_residual = use_residual and (na == nb)  # Only use residual if dimensions match
        self.output_activation = output_activation
        self.climate_processing = climate_processing
        self.use_spatial_attention = use_spatial_attention
        self.use_multihead_attention = use_multihead_attention
        self.temporal_depth = temporal_depth
        
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
        # Attention mechanisms for climate branch
        if use_spatial_attention and climate_processing == 'conv2d':
            # Spatial attention operates on the 2D feature maps before flattening
            # Ensure num_heads divides the number of channels
            spatial_heads = attention_heads
            num_channels = climate_shape[0]
            if num_channels % spatial_heads != 0:
                # Find all divisors of num_channels that are <= attention_heads
                valid_heads = [h for h in range(1, attention_heads + 1) 
                              if num_channels % h == 0]
                if valid_heads:
                    spatial_heads = max(valid_heads)
                else:
                    # Fallback: find any divisor
                    spatial_heads = max([h for h in range(1, num_channels + 1) 
                                        if num_channels % h == 0])
                print(f"Warning: Adjusted spatial attention heads from {attention_heads} to {spatial_heads} "
                      f"(must divide {num_channels} channels)")
            self.spatial_attention = SpatialSelfAttention(
                channels=climate_shape[0], 
                num_heads=spatial_heads
            )
        
        if use_multihead_attention:
            # Multi-head attention operates on flattened features
            # Ensure num_heads divides climate_units
            mha_heads = attention_heads
            if self._climate_units_out % mha_heads != 0:
                # Find all divisors of climate_units that are <= attention_heads
                valid_heads = [h for h in range(1, attention_heads + 1) 
                              if self._climate_units_out % h == 0]
                if valid_heads:
                    mha_heads = max(valid_heads)
                else:
                    # Fallback: find any divisor
                    mha_heads = max([h for h in range(1, self._climate_units_out + 1) 
                                    if self._climate_units_out % h == 0])
                print(f"Warning: Adjusted multi-head attention heads from {attention_heads} to {mha_heads} "
                      f"(must divide {self._climate_units_out} features)")
            self.multihead_attention = MultiHeadAttention(
                embed_dim=self._climate_units_out,
                num_heads=mha_heads,
                dropout=attention_dropout
            )
        
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
        # Month/Temporal Branch Architecture (Deeper MLP)
        # ----------------------------------------------------------------
        # Layer 1 (always present)
        self.month_fc1 = nn.Linear(in_features=num_temporal_encodings, out_features=temporal_units)
        self.month_bn1 = nn.BatchNorm1d(num_features=temporal_units)
        
        # Layer 2 (if temporal_depth >= 2)
        if temporal_depth >= 2:
            self.month_fc2 = nn.Linear(in_features=temporal_units, out_features=temporal_units)
            self.month_bn2 = nn.BatchNorm1d(num_features=temporal_units)
            self.month_dropout2 = nn.Dropout(temporal_dropout)
        
        # Layer 3 (if temporal_depth >= 3)
        if temporal_depth >= 3:
            self.month_fc3 = nn.Linear(in_features=temporal_units, out_features=temporal_units)
            self.month_bn3 = nn.BatchNorm1d(num_features=temporal_units)
            self.month_dropout3 = nn.Dropout(temporal_dropout)
        
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
        elif activation == 'softplus':
            return F.softplus
        else:
            raise ValueError(f"Unknown climate_activation: {activation}. Use 'relu', 'softplus', or 'none'.")
    
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
        
        # Apply spatial self-attention if enabled (before conv/flatten)
        if self.use_spatial_attention and self.climate_processing == 'conv2d':
            climate = self.spatial_attention(climate)
        
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
        
        # Apply multi-head attention if enabled (on flattened features)
        if self.use_multihead_attention:
            # Add sequence dimension for attention
            climate_out_seq = climate_out.unsqueeze(1)  # (batch, 1, features)
            climate_out_seq = self.multihead_attention(climate_out_seq)
            climate_out = climate_out_seq.squeeze(1)  # (batch, features)
        
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
        # Temporal Branch Processing (Deeper MLP)
        # ----------------------------------------------------------------
        temporal = features['temporal']
        
        # Layer 1 (always present)
        month_out = self.month_fc1(temporal)
        month_out = self.month_bn1(month_out)
        month_out = F.relu(month_out)
        
        # Layer 2 (if temporal_depth >= 2)
        if self.temporal_depth >= 2:
            month_out = self.month_fc2(month_out)
            month_out = self.month_bn2(month_out)
            month_out = F.relu(month_out)
            month_out = self.month_dropout2(month_out)
        
        # Layer 3 (if temporal_depth >= 3)
        if self.temporal_depth >= 3:
            month_out = self.month_fc3(month_out)
            month_out = self.month_bn3(month_out)
            month_out = F.relu(month_out)
            month_out = self.month_dropout3(month_out)
        
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
        # Attention mechanism parameters (optional, default to False)
        use_spatial_attention=hyperparams.get('use_spatial_attention', False),
        use_multihead_attention=hyperparams.get('use_multihead_attention', False),
        attention_heads=hyperparams.get('attention_heads', 4),
        attention_dropout=hyperparams.get('attention_dropout', 0.1),
        # Temporal branch parameters
        temporal_depth=hyperparams.get('temporal_depth', 2),
        temporal_dropout=hyperparams.get('temporal_dropout', 0.1),
    )
    
    return model


if __name__ == "__main__":
    print("LANDModel module loaded successfully.")
    print("Import this module to use LANDModel and create_model_from_hyperparams().")
