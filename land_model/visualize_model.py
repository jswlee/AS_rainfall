#!/usr/bin/env python3
"""
Visualize the LAND-inspired model architecture.
This script creates and saves a visualization of the model architecture
used in the rainfall prediction project.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from pathlib import Path

# Define a build_model function here to avoid dependency issues
def build_model(data_metadata, hyperparams):
    """
    Build the LAND-inspired model for rainfall prediction.
    
    Parameters
    ----------
    data_metadata : dict
        Dictionary containing metadata about the input data
    hyperparams : dict
        Dictionary of hyperparameters
        
    Returns
    -------
    tf.keras.Model
        Compiled LAND model
    """
    # Extract hyperparameters
    na = hyperparams.get('na', 512)
    nb = hyperparams.get('nb', 384)
    dropout_rate = hyperparams.get('dropout_rate', 0.1)
    l2_reg = hyperparams.get('l2_reg', 1e-6)
    learning_rate = hyperparams.get('learning_rate', 0.01)
    weight_decay = hyperparams.get('weight_decay', 1e-7)
    local_dem_units = hyperparams.get('local_dem_units', 64)
    regional_dem_units = hyperparams.get('regional_dem_units', 32)
    month_units = hyperparams.get('month_units', 16)
    climate_units = hyperparams.get('climate_units', 384)
    use_residual = hyperparams.get('use_residual', False)
    activation = hyperparams.get('activation', 'relu')
    
    # Create input layers
    climate_input = layers.Input(shape=data_metadata['climate_shape'], name='climate')
    local_dem_input = layers.Input(shape=data_metadata['local_dem_shape'], name='local_dem')
    regional_dem_input = layers.Input(shape=data_metadata['regional_dem_shape'], name='regional_dem')
    month_input = layers.Input(shape=(data_metadata['num_month_encodings'],), name='month')
    
    # Process climate data
    climate_flat = layers.Reshape((data_metadata['climate_shape'][0],))(climate_input)
    climate = layers.Dense(
        climate_units, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='climate_dense'
    )(climate_flat)
    climate = layers.BatchNormalization()(climate)
    
    # Process local DEM
    local_dem = layers.Flatten()(local_dem_input)
    local_dem = layers.Dense(
        local_dem_units, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='local_dem_dense'
    )(local_dem)
    local_dem = layers.BatchNormalization()(local_dem)
    
    # Process regional DEM
    regional_dem = layers.Flatten()(regional_dem_input)
    regional_dem = layers.Dense(
        regional_dem_units, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='regional_dem_dense'
    )(regional_dem)
    regional_dem = layers.BatchNormalization()(regional_dem)
    
    # Process month encoding
    month = layers.Dense(
        month_units, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='month_dense'
    )(month_input)
    month = layers.BatchNormalization()(month)
    
    # Concatenate all features
    concat = layers.Concatenate()([climate, local_dem, regional_dem, month])
    
    # First dense layer after concatenation
    x = layers.Dense(
        na, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_1'
    )(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Optional residual connection
    if use_residual:
        # Create a projection if dimensions don't match
        if na != concat.shape[-1]:
            residual = layers.Dense(na, name='residual_projection')(concat)
        else:
            residual = concat
        x = layers.Add()([x, residual])
    
    # Second dense layer
    x = layers.Dense(
        nb, 
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    output = layers.Dense(1, name='rainfall_prediction')(x)
    
    # Create model
    model = models.Model(
        inputs=[climate_input, local_dem_input, regional_dem_input, month_input],
        outputs=output,
        name='LAND_rainfall_model'
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def visualize_model_architecture(output_dir, hyperparams=None, show_shapes=True, show_layer_names=True, dpi=300):
    """
    Create and save visualizations of the LAND-inspired model architecture.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the visualization
    hyperparams : dict, optional
        Hyperparameters for the model
    show_shapes : bool, optional
        Whether to show shapes in the plot_model output
    show_layer_names : bool, optional
        Whether to show layer names in the plot_model output
    dpi : int, optional
        DPI for the matplotlib figure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy data metadata for model building
    data_metadata = {
        'climate_shape': (16, 1, 1),
        'local_dem_shape': (3, 3),
        'regional_dem_shape': (3, 3),
        'num_month_encodings': 12
    }
    
    # Default hyperparameters if none provided
    if hyperparams is None:
        hyperparams = {
            'na': 512,
            'nb': 384,
            'dropout_rate': 0.1,
            'l2_reg': 1e-06,
            'learning_rate': 0.01,
            'weight_decay': 1e-07,
            'local_dem_units': 64,
            'regional_dem_units': 32,
            'month_units': 16,
            'climate_units': 384,
            'use_residual': False,
            'activation': 'relu'
        }
    
    # Build the model
    model = build_model(data_metadata, hyperparams)
    
    # Save model summary to text file
    summary_path = os.path.join(output_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        # Redirect summary to file
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved to {summary_path}")
    
    # Save model visualization using plot_model
    plot_path = os.path.join(output_dir, 'model_architecture.png')
    plot_model(model, to_file=plot_path, show_shapes=show_shapes, 
               show_layer_names=show_layer_names, dpi=dpi)
    print(f"Model architecture visualization saved to {plot_path}")
    
    # Create a simplified diagram showing the main components
    create_simplified_diagram(output_dir, hyperparams)
    
    return model

def create_simplified_diagram(output_dir, hyperparams):
    """
    Create a simplified block diagram of the model architecture.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the visualization
    hyperparams : dict
        Hyperparameters for the model
    """
    # Create a figure
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Define block positions and sizes
    block_width = 0.2
    block_height = 0.1
    arrow_props = dict(arrowstyle='->', lw=1.5)
    
    # Input blocks
    climate_block = plt.Rectangle((0.1, 0.8), block_width, block_height, 
                                 fc='lightblue', ec='black', alpha=0.7)
    local_dem_block = plt.Rectangle((0.1, 0.6), block_width, block_height, 
                                   fc='lightgreen', ec='black', alpha=0.7)
    regional_dem_block = plt.Rectangle((0.1, 0.4), block_width, block_height, 
                                      fc='lightgreen', ec='black', alpha=0.7)
    month_block = plt.Rectangle((0.1, 0.2), block_width, block_height, 
                               fc='lightyellow', ec='black', alpha=0.7)
    
    # Processing blocks
    climate_proc = plt.Rectangle((0.4, 0.8), block_width, block_height, 
                                fc='royalblue', ec='black', alpha=0.7)
    local_dem_proc = plt.Rectangle((0.4, 0.6), block_width, block_height, 
                                  fc='forestgreen', ec='black', alpha=0.7)
    regional_dem_proc = plt.Rectangle((0.4, 0.4), block_width, block_height, 
                                     fc='forestgreen', ec='black', alpha=0.7)
    month_proc = plt.Rectangle((0.4, 0.2), block_width, block_height, 
                              fc='gold', ec='black', alpha=0.7)
    
    # Concatenation and dense layers
    concat_block = plt.Rectangle((0.7, 0.5), block_width, block_height, 
                                fc='lightgray', ec='black', alpha=0.7)
    dense1_block = plt.Rectangle((0.7, 0.3), block_width, block_height, 
                                fc='orange', ec='black', alpha=0.7)
    dense2_block = plt.Rectangle((0.7, 0.1), block_width, block_height, 
                                fc='orange', ec='black', alpha=0.7)
    
    # Output block
    output_block = plt.Rectangle((0.9, 0.1), block_width, block_height, 
                                fc='red', ec='black', alpha=0.7)
    
    # Add all blocks to the plot
    for block in [climate_block, local_dem_block, regional_dem_block, month_block,
                 climate_proc, local_dem_proc, regional_dem_proc, month_proc,
                 concat_block, dense1_block, dense2_block, output_block]:
        ax.add_patch(block)
    
    # Add arrows
    ax.annotate('', xy=(0.4, 0.85), xytext=(0.3, 0.85), arrowprops=arrow_props)
    ax.annotate('', xy=(0.4, 0.65), xytext=(0.3, 0.65), arrowprops=arrow_props)
    ax.annotate('', xy=(0.4, 0.45), xytext=(0.3, 0.45), arrowprops=arrow_props)
    ax.annotate('', xy=(0.4, 0.25), xytext=(0.3, 0.25), arrowprops=arrow_props)
    
    # Arrows to concatenation
    ax.annotate('', xy=(0.7, 0.55), xytext=(0.6, 0.8), arrowprops=arrow_props)
    ax.annotate('', xy=(0.7, 0.55), xytext=(0.6, 0.65), arrowprops=arrow_props)
    ax.annotate('', xy=(0.7, 0.55), xytext=(0.6, 0.45), arrowprops=arrow_props)
    ax.annotate('', xy=(0.7, 0.55), xytext=(0.6, 0.25), arrowprops=arrow_props)
    
    # Arrows for dense layers
    ax.annotate('', xy=(0.7, 0.3), xytext=(0.7, 0.45), arrowprops=arrow_props)
    ax.annotate('', xy=(0.7, 0.1), xytext=(0.7, 0.25), arrowprops=arrow_props)
    ax.annotate('', xy=(0.9, 0.15), xytext=(0.8, 0.15), arrowprops=arrow_props)
    
    # Add text labels
    ax.text(0.2, 0.85, 'Climate\nVariables', ha='center', va='center')
    ax.text(0.2, 0.65, 'Local DEM\n3x3', ha='center', va='center')
    ax.text(0.2, 0.45, 'Regional DEM\n3x3', ha='center', va='center')
    ax.text(0.2, 0.25, 'Month\nEncoding', ha='center', va='center')
    
    ax.text(0.5, 0.85, f'Dense\n{hyperparams["climate_units"]} units', ha='center', va='center')
    ax.text(0.5, 0.65, f'Dense\n{hyperparams["local_dem_units"]} units', ha='center', va='center')
    ax.text(0.5, 0.45, f'Dense\n{hyperparams["regional_dem_units"]} units', ha='center', va='center')
    ax.text(0.5, 0.25, f'Dense\n{hyperparams["month_units"]} units', ha='center', va='center')
    
    ax.text(0.8, 0.55, 'Concatenate', ha='center', va='center')
    ax.text(0.8, 0.35, f'Dense\n{hyperparams["na"]} units\nDropout {hyperparams["dropout_rate"]}', ha='center', va='center')
    ax.text(0.8, 0.15, f'Dense\n{hyperparams["nb"]} units\nDropout {hyperparams["dropout_rate"]}', ha='center', va='center')
    
    ax.text(1.0, 0.15, 'Rainfall\nPrediction', ha='center', va='center')
    
    # Set plot limits and remove axes
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    plt.title('LAND-inspired Model Architecture', fontsize=16)
    
    # Add legend for regularization
    legend_text = (f"Regularization:\n"
                  f"L2: {hyperparams['l2_reg']}\n"
                  f"Dropout: {hyperparams['dropout_rate']}\n"
                  f"Weight Decay: {hyperparams['weight_decay']}\n"
                  f"Activation: {hyperparams['activation']}")
    
    plt.figtext(0.02, 0.02, legend_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Save the figure
    diagram_path = os.path.join(output_dir, 'model_diagram.png')
    plt.savefig(diagram_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Simplified model diagram saved to {diagram_path}")

def load_hyperparameters(hyperparams_path):
    """
    Load hyperparameters from a text file.
    
    Parameters
    ----------
    hyperparams_path : str
        Path to the hyperparameters file
        
    Returns
    -------
    dict
        Dictionary of hyperparameters
    """
    hyperparams = {}
    with open(hyperparams_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    # Try to convert to float or int
                    value = float(value.strip())
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    # Keep as string if not a number
                    value = value.strip()
                    # Handle boolean values
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                hyperparams[key.strip()] = value
    return hyperparams

def main():
    parser = argparse.ArgumentParser(description='Visualize LAND-inspired model architecture')
    parser.add_argument('--output_dir', type=str, default='figures/model_architecture',
                        help='Directory to save visualizations')
    parser.add_argument('--hyperparams_path', type=str, 
                        default='land_model_extended_tuner_raw/best_hyperparameters.txt',
                        help='Path to hyperparameters file')
    parser.add_argument('--show_shapes', action='store_true', default=True,
                        help='Show shapes in the plot_model output')
    parser.add_argument('--show_layer_names', action='store_true', default=True,
                        help='Show layer names in the plot_model output')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for the matplotlib figure')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load hyperparameters if provided
    hyperparams = None
    if args.hyperparams_path and os.path.exists(args.hyperparams_path):
        print(f"Loading hyperparameters from {args.hyperparams_path}")
        hyperparams = load_hyperparameters(args.hyperparams_path)
        print(f"Loaded hyperparameters: {hyperparams}")
    
    # Visualize model architecture
    visualize_model_architecture(
        args.output_dir,
        hyperparams=hyperparams,
        show_shapes=args.show_shapes,
        show_layer_names=args.show_layer_names,
        dpi=args.dpi
    )

if __name__ == '__main__':
    main()