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
    # Go up three levels: utils -> scripts -> Create_ML_Data -> PROJECT_ROOT
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
        config_path = os.path.join(project_root, 'Create_ML_Data', 'config', 'config.yaml')
    
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
        'rainfall_dir': paths['rainfall_dir'],
        'station_locations_path': paths['stations'],
        'output_dir': paths['output_dir'],
        'grid_size': config['model']['grid_size'],
        'patch_sizes': config['model']['patch_sizes'],
        'km_per_cell': config['model']['km_per_cell']
    }
    
    return final_config


def create_config(config_overrides=None):
    """
    Create a configuration dictionary with optional overrides.
    This replaces the command-line argument functionality for Jupyter notebooks.
    
    Args:
        config_overrides (dict, optional): Dictionary of configuration overrides.
                                          Keys should match the keys in the config.
    
    Returns:
        dict: Final configuration dictionary with absolute paths.
    """
    # Load the base configuration
    config = load_config()
    
    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if key in config:
                config[key] = value
    
    return config


# Legacy functions kept for backward compatibility
def parse_args():
    """
    Parse command-line arguments for the rainfall prediction pipeline.
    This function is kept for backward compatibility.
    
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
    
    return parser.parse_args()


def merge_config_with_args(config, args):
    """
    Override config values with command-line arguments.
    This function is kept for backward compatibility.
    
    Args:
        config (dict): Configuration dictionary.
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        dict: Updated configuration dictionary.
    """
    # Override config with command-line arguments if provided
    if hasattr(args, 'dem_path') and args.dem_path:
        config['dem_path'] = args.dem_path
    
    if hasattr(args, 'climate_data_path') and args.climate_data_path:
        config['climate_data_path'] = args.climate_data_path
    
    if hasattr(args, 'raw_climate_dir') and args.raw_climate_dir:
        config['raw_climate_dir'] = args.raw_climate_dir
        
    if hasattr(args, 'rainfall_dir') and args.rainfall_dir:
        config['rainfall_dir'] = args.rainfall_dir
        
    if hasattr(args, 'station_locations_path') and args.station_locations_path:
        config['station_locations_path'] = args.station_locations_path
        
    if hasattr(args, 'output_dir') and args.output_dir:
        config['output_dir'] = args.output_dir
        
    if hasattr(args, 'grid_size') and args.grid_size:
        config['grid_size'] = args.grid_size
    
    return config
