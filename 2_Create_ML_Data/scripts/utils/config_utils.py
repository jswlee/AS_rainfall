#!/usr/bin/env python3
"""
Configuration utilities for the AS_rainfall project.
Handles loading and parsing of configuration files.
"""
import os
import yaml
from pathlib import Path


def get_project_root():
    """Get the absolute path to the project root directory."""
    # Start from the current file and go up to find project root
    current_file = Path(__file__).resolve()
    # Go up three levels: utils -> scripts -> 2_Create_ML_Data -> PROJECT_ROOT
    return current_file.parent.parent.parent.parent


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to the config file. 
                                     If None, uses the default config.
    
    Returns:
        dict: Configuration dictionary with absolute paths.
    """
    project_root = get_project_root()
    
    if config_path is None:
        # Default config path
        config_path = os.path.join(project_root, '2_Create_ML_Data', 'config', 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute
    paths = config['paths']
    for key in paths:
        paths[key] = os.path.normpath(os.path.join(project_root, paths[key]))
    
    # Build final config dictionary in the same structure as the original CONFIG
    final_config = {
        'dem_path': paths['dem'],
        'climate_data_path': paths['climate_data'],
        'raw_climate_dir': paths['raw_climate'],
        'rainfall_dir': paths['rainfall'],
        'station_locations_path': paths['stations'],
        'output_dir': paths['output'],
        'grid_size': config['model']['grid_size'],
        'patch_sizes': config['model']['patch_sizes'],
        'km_per_cell': config['model']['km_per_cell']
    }
    
    return final_config


def parse_args():
    """
    Parse command-line arguments for the rainfall prediction pipeline.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the rainfall prediction pipeline')
    parser.add_argument('--config', type=str, help='Path to custom config YAML file')
    parser.add_argument('--dem-path', type=str, help='Path to DEM file')
    parser.add_argument('--climate-data-path', type=str, help='Path to processed climate data')
    parser.add_argument('--raw-climate-dir', type=str, help='Directory with raw climate data')
    parser.add_argument('--rainfall-dir', type=str, help='Directory with monthly rainfall data')
    parser.add_argument('--station-locations-path', type=str, help='Path to station locations CSV')
    parser.add_argument('--output-dir', type=str, help='Output directory for generated data')
    parser.add_argument('--grid-size', type=int, help='Grid size for data generation')
    
    # Advanced arguments for patch sizes and km_per_cell could be added as needed
    
    return parser.parse_args()


def merge_config_with_args(config, args):
    """
    Override config values with command-line arguments.
    
    Args:
        config (dict): Configuration dictionary.
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        dict: Updated configuration dictionary.
    """
    # Create a mapping between argument names and config keys
    arg_to_config = {
        'dem_path': 'dem_path',
        'climate_data_path': 'climate_data_path',
        'raw_climate_dir': 'raw_climate_dir',
        'rainfall_dir': 'rainfall_dir',
        'station_locations_path': 'station_locations_path',
        'output_dir': 'output_dir',
        'grid_size': 'grid_size'
    }
    
    # Convert args to dictionary and remove None values
    args_dict = vars(args)
    args_dict = {k: v for k, v in args_dict.items() if v is not None}
    
    # Update config with command-line arguments
    for arg_name, arg_value in args_dict.items():
        # Skip the config path argument
        if arg_name == 'config':
            continue
            
        # Convert hyphenated arg names to underscore format
        arg_key = arg_name.replace('-', '_')
        
        if arg_key in arg_to_config:
            config_key = arg_to_config[arg_key]
            config[config_key] = arg_value
    
    return config
