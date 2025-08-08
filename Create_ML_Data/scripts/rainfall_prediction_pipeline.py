"""
Rainfall Prediction Pipeline

This script implements a complete pipeline for rainfall prediction:
1. Generate climate variables on a coarse 3x3 grid
2. Create a 5x5 grid with 25 evenly spaced points on the DEM
3. Create local (12km) and regional (60km) DEM patches around each point
4. Interpolate rainfall for each timestamp at each grid point
5. Interpolate climate variables to the 5x5 grid
6. Prepare data for deep learning with:
   - 16 interpolated climate variables
   - Month one-hot encoding
   - Local and regional DEM patches
   - Interpolated rainfall as labels
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
import logging

# Set up logging for progress tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Import the H5 to CSV conversion module
from Create_ML_Data.scripts.convert_h5_to_csv import extract_data_from_h5

# Import visualization utilities
from Create_ML_Data.scripts.utils.visualization import visualize_dem_patches, visualize_interpolated_rainfall

# Import the processors and utils
from Create_ML_Data.scripts.processors.dem_processor.processor import DEMProcessor
from Create_ML_Data.scripts.processors.climate_processor.processor import ClimateDataProcessor
from Create_ML_Data.scripts.processors.rainfall_processor.processor import RainfallProcessor
from Create_ML_Data.scripts.utils.data_generator import DataGenerator
from Create_ML_Data.scripts.utils.config_utils import load_config, parse_args, merge_config_with_args, create_config

# Load configuration from YAML file with optional overrides
def get_config(config_overrides=None):
    """
    Get configuration with optional overrides for notebook usage.
    
    Args:
        config_overrides (dict, optional): Dictionary of configuration overrides.
                                          Keys should match the keys in the config.
    
    Returns:
        dict: Final configuration dictionary with absolute paths.
    """
    if config_overrides is None:
        # For command-line usage (backward compatibility)
        args = parse_args()
        config = load_config(args.config if hasattr(args, 'config') else None)
        config = merge_config_with_args(config, args)
    else:
        # For notebook usage
        config = create_config(config_overrides)
    
    return config

def setup_environment(config):
    """
    Set up the environment for the pipeline.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        bool: True if setup was successful, False otherwise.
        bool: Whether climate data exists.
    """
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    # Create figures directory
    figures_dir = os.path.join(config['output_dir'], 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check if climate data exists (either processed file or raw files)
    processed_file_exists = os.path.exists(config['climate_data_path'])
    raw_files_exist = False
    
    # Check if raw climate data directory exists and has NetCDF files
    if os.path.exists(config['raw_climate_dir']):
        nc_files = [f for f in os.listdir(config['raw_climate_dir']) if f.endswith('.nc')]
        raw_files_exist = len(nc_files) > 0
        if raw_files_exist:
            print(f"Found {len(nc_files)} raw climate data files in {config['raw_climate_dir']}")
    
    # Set the flag based on whether either source of climate data exists
    climate_data_exists = processed_file_exists or raw_files_exist
    
    if not climate_data_exists:
        raise FileNotFoundError(
            f"No climate data found. Please ensure either:"
            f"\n1. Processed climate data exists at: {config['climate_data_path']}"
            f"\n2. OR Raw climate data files exist in: {config['raw_climate_dir']}"
            f"\n\nRaw climate data directory contents: {os.listdir(config['raw_climate_dir']) if os.path.exists(config['raw_climate_dir']) else 'Directory does not exist'}"
        )
    elif not processed_file_exists and raw_files_exist:
        print(f"Processed climate data not found at {config['climate_data_path']}, but raw files exist and will be processed.")
    elif processed_file_exists:
        print(f"Found existing processed climate data at: {config['climate_data_path']}")
    
    return True, climate_data_exists

def check_required_files(config):
    """Check if all required files exist.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        bool: True if all required files exist, False otherwise.
    """
    # Always required files (regardless of climate data)
    required_files = [
        config['dem_path'],
        config['rainfall_dir'],
        config['station_locations_path']
    ]
    
    # Check required files
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            return False
    
    print("All required files found.")
    return True

def process_dem(config):
    """Process DEM to create grid points and patches.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        dict: Dictionary containing grid points and DEM patches.
    """
    print("\nProcessing DEM...")
    
    # Initialize DEM processor
    dem_processor = DEMProcessor(config['dem_path'])
    
    # Generate grid points (5x5 grid = 25 points)
    grid_points = dem_processor.generate_grid_points(config['grid_size'])
    print(f"Generated {len(grid_points)} grid points")
    
    # Create local and regional patches for each grid point
    local_patches = []
    regional_patches = []
    
    for point in grid_points:
        local_patch = dem_processor.extract_patch(
            point, 
            patch_size=config['patch_sizes']['local'],
            km_per_cell=config['km_per_cell']['local']
        )
        
        regional_patch = dem_processor.extract_patch(
            point, 
            patch_size=config['patch_sizes']['regional'],
            km_per_cell=config['km_per_cell']['regional']
        )
        
        local_patches.append(local_patch)
        regional_patches.append(regional_patch)
    
    print(f"Created {len(local_patches)} local patches and {len(regional_patches)} regional patches")
    
    # Visualize all patches in a grid
    local_patch_km = config['patch_sizes']['local'] * config['km_per_cell']['local']
    regional_patch_km = config['patch_sizes']['regional'] * config['km_per_cell']['regional']
    
    dem_patches_path = visualize_dem_patches(
        local_patches, 
        regional_patches, 
        config['grid_size'],
        local_patch_km,
        regional_patch_km,
        config['output_dir']
    )
    print(f"Saved DEM patches visualization to {dem_patches_path}")
    
    return {
        'grid_points': grid_points,
        'local_patches': local_patches,
        'regional_patches': regional_patches
    }

def process_climate_data(config, climate_data_exists=True):
    """Process climate data and interpolate to grid points.
    
    Args:
        config (dict): Configuration dictionary.
        climate_data_exists (bool): Whether climate data exists.
        
    Returns:
        str or None: Path to processed climate data, or None if no data available.
    """
    if not climate_data_exists:
        print("Warning: No climate data available. Skipping climate data processing.")
        return None
    
    print("\nProcessing climate data...")
    existing_processed_data = config['climate_data_path']
    raw_climate_dir = config['raw_climate_dir']
    
    # Check if processed data already exists
    if os.path.exists(existing_processed_data):
        # Use existing processed data
        output_climate_path = os.path.join(config['output_dir'], 'processed_climate_data.nc')
        
        # Only copy if the file doesn't already exist in the output directory
        if not os.path.exists(output_climate_path) or os.path.getmtime(existing_processed_data) > os.path.getmtime(output_climate_path):
            print(f"Copying existing climate data to output directory: {output_climate_path}")
            import shutil
            shutil.copy2(existing_processed_data, output_climate_path)
        else:
            print(f"Using existing copy in output directory: {output_climate_path}")
            
        return output_climate_path
    else:
        # Process raw files
        print("Processing raw climate data files...")
        climate_processor = ClimateDataProcessor(data_dir=raw_climate_dir)
        
        # Process all climate variables
        for var_name in climate_processor.variable_configs.keys():
            print(f"Processing {var_name}...")
            climate_processor.process_variable(var_name)
        
        # Save processed climate data
        climate_data_path = os.path.join(config['output_dir'], 'processed_climate_data.nc')
        climate_processor.save_to_netcdf(climate_data_path)
        print(f"Saved processed climate data to {climate_data_path}")
        
        return climate_data_path

def process_rainfall_data(config, grid_points):
    """Process rainfall data and interpolate to grid points.
    
    Args:
        config (dict): Configuration dictionary.
        grid_points (list): List of grid points to interpolate rainfall to.
        
    Returns:
        tuple: Tuple containing interpolated rainfall data and available dates.
    """
    print("\nProcessing rainfall data...")
    
    # Initialize rainfall processor
    rainfall_processor = RainfallProcessor(
        monthly_rainfall_dir=config['rainfall_dir'],
        station_locations_path=config['station_locations_path']
    )
    
    # Get available dates
    available_dates = rainfall_processor.get_available_dates()
    print(f"Found {len(available_dates)} available dates for rainfall data")
    
    # Filter out dates with no valid data
    valid_dates = []
    for date_str in available_dates:
        rainfall_data = rainfall_processor.get_rainfall_for_date(date_str)
        values = np.array(rainfall_data.get('values', []), dtype=float)
        
        # Only keep if at least one station has a real value (including true zeros)
        if len(values) > 0 and not np.all(np.isnan(values)):
            valid_dates.append(date_str)
        else:
            print(f"WARNING: All stations missing/NaN for {date_str}, dropping month from dataset.")
    
    print(f"Using {len(valid_dates)} valid dates for spatiotemporal interpolation")
    
    # Use spatiotemporal GP interpolation for all dates at once
    print("Performing spatiotemporal GP interpolation...")
    interpolated_rainfall = rainfall_processor.interpolate_spatiotemporal(
        grid_points=grid_points,
        target_dates=valid_dates
    )
    
    print(f"Completed spatiotemporal interpolation for {len(interpolated_rainfall)} dates")
    
    # Visualize sample interpolated rainfall
    sample_date = available_dates[409]
    
    rainfall_plot_path = visualize_interpolated_rainfall(
        grid_points,
        interpolated_rainfall,
        sample_date,
        config['output_dir']
    )
    print(f"Saved rainfall interpolation visualization to {rainfall_plot_path}")
    
    return interpolated_rainfall, available_dates

def generate_training_data(config, dem_data, climate_data_path, rainfall_data, available_dates, climate_data_exists=True):
    """Generate training data for deep learning.
    
    Args:
        config (dict): Configuration dictionary.
        dem_data (dict): Dictionary containing DEM data (grid points and patches).
        climate_data_path (str): Path to climate data.
        rainfall_data (dict): Dictionary containing interpolated rainfall data.
        available_dates (list): List of available dates.
        climate_data_exists (bool): Whether climate data exists.
        
    Returns:
        str or None: Path to generated training data, or None if generation failed.
    """
    print("\nGenerating training data...")
    if not climate_data_exists or climate_data_path is None:
        print("Warning: Climate data is missing. Skipping training data generation.")
        return None
    print(f"Using climate data from: {climate_data_path}")
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(config['output_dir'], 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize data generator with both output and figures directories
    data_generator = DataGenerator(
        grid_points=dem_data['grid_points'],
        local_patches=dem_data['local_patches'],
        regional_patches=dem_data['regional_patches'],
        climate_data_path=climate_data_path,
        rainfall_data=rainfall_data,
        output_dir=config['output_dir'],
        figures_dir=figures_dir,
        grid_size=config['grid_size']  # Pass the grid size from configuration
    )
    
    # Find intersection of available dates between climate and rainfall data
    climate_dates = data_generator.available_dates
    rainfall_dates = list(rainfall_data.keys())
    
    common_dates = sorted(list(set(climate_dates).intersection(set(rainfall_dates))))
    print(f"Found {len(common_dates)} dates with both climate and rainfall data")
    
    if len(common_dates) == 0:
        print("ERROR: No common dates found between climate and rainfall data")
        print(f"Climate data dates: {climate_dates[:5]}... (total: {len(climate_dates)})")
        print(f"Rainfall data dates: {rainfall_dates[:5]}... (total: {len(rainfall_dates)})")
        return None
    
    # Generate data for common dates
    all_data = []
    
    for date_str in common_dates:
        print(f"Generating data for {date_str}...")
        try:
            data = data_generator.generate_data_for_date(date_str)
            all_data.append(data)
            print(f"  Successfully generated data for {date_str}")
        except Exception as e:
            print(f"  Error generating data for {date_str}: {e}")
    
    # Save generated data
    if all_data:
        h5_path = data_generator.save_data(all_data)
        print(f"Saved generated data to {h5_path}")
        
        # Visualize sample data
        try:
            data_generator.visualize_sample(all_data[409], common_dates[409])
            print(f"Sample visualization complete for {common_dates[409]}")
        except Exception as e:
            print(f"Error visualizing sample: {e}")
        
        return h5_path
    else:
        print("No data was generated. Check the error messages above.")
        return None

def run_pipeline(config_overrides=None, redirect_output=True):
    """
    Main function to run the entire rainfall prediction pipeline.
    
    Args:
        config_overrides (dict, optional): Dictionary of configuration overrides.
        redirect_output (bool): Whether to redirect output to a log file.
        
    Returns:
        dict: Dictionary containing the results of the pipeline run.
    """
    # Get configuration
    config = get_config(config_overrides)
    
    print("\n" + "="*80)
    print("RAINFALL PREDICTION PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {config['output_dir']}")
    print("Configuration:")
    for key, value in config.items():
        if key not in ['patch_sizes', 'km_per_cell']:
            print(f"  {key}: {value}")
    print(f"  patch_sizes: local={config['patch_sizes']['local']}, regional={config['patch_sizes']['regional']}")
    print(f"  km_per_cell: local={config['km_per_cell']['local']}, regional={config['km_per_cell']['regional']}")
    print("="*80 + "\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Redirect output to log file if requested
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None
    
    if redirect_output:
        log_path = os.path.join(config['output_dir'], "pipeline_output.log")
        log_file = open(log_path, "w")
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Output redirected to {log_path}")
    
    # Define total steps for progress tracking
    total_steps = 6
    current_step = 0
    
    def report_progress(step_name):
        nonlocal current_step
        current_step += 1
        # Create progress message
        progress_msg = f"PROGRESS: {current_step}/{total_steps} - {step_name}"
        
        # Print to stdout with flush
        print(progress_msg, flush=True)
        sys.stdout.flush()
        
        # Also log the message
        logging.info(progress_msg)
        
        # Write to a special progress file that can be monitored
        progress_file = os.path.join(config['output_dir'], 'progress.log')
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")
    
    try:
        # Step 1: Setup environment
        report_progress("Setting up environment")
        setup_success, climate_data_exists = setup_environment(config)
        if not setup_success or not check_required_files(config):
            return {"success": False, "error": "Environment setup failed"}
            
        # Step 2: Process DEM
        report_progress("Processing DEM data")
        dem_data = process_dem(config)
        
        # Step 3: Process climate data
        report_progress("Processing climate data")
        climate_data_path = process_climate_data(config, climate_data_exists)
        
        # Step 4: Process rainfall data
        report_progress("Processing rainfall data")
        rainfall_data, available_dates = process_rainfall_data(config, dem_data['grid_points'])
        
        # Step 5: Generate training data
        report_progress("Generating training data")
        # Generate training data only if climate data exists
        if climate_data_exists and climate_data_path is not None:
            training_data_path = generate_training_data(
                config,
                dem_data,
                climate_data_path,
                rainfall_data,
                available_dates,
                climate_data_exists
            )
            if training_data_path:
                print(f"\nPipeline complete! Training data saved to {training_data_path}")
                print(f"Features: 16 climate variables, month encoding, local and regional DEM patches")
                print(f"Labels: Interpolated rainfall")
                
                # Step 6: Convert H5 to CSV
                report_progress("Converting H5 data to CSV format")
                output_dir = os.path.join(config['output_dir'], 'csv_data')
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    # Extract data from H5 file and filter out zero rainfall entries
                    features_df, targets_df, metadata_df = extract_data_from_h5(
                        training_data_path, 
                        filter_zero_rainfall=True  # Only keep entries with non-zero rainfall
                    )
                    
                    # Save to CSV
                    features_path = os.path.join(output_dir, 'features.csv')
                    targets_path = os.path.join(output_dir, 'targets.csv')
                    metadata_path = os.path.join(output_dir, 'metadata.csv')
                    
                    print(f"Saving features to {features_path}...")
                    features_df.to_csv(features_path, index=False)
                    
                    print(f"Saving targets to {targets_path}...")
                    targets_df.to_csv(targets_path, index=False)
                    
                    print(f"Saving metadata to {metadata_path}...")
                    metadata_df.to_csv(metadata_path, index=False)
                    
                    print(f"\nConversion complete. CSV files saved to {output_dir}")
                    print(f"Total samples: {len(targets_df)}")
                    print(f"Features shape: {features_df.shape}")
                    
                    return {
                        "success": True,
                        "training_data_path": training_data_path,
                        "csv_dir": output_dir,
                        "features_path": features_path,
                        "targets_path": targets_path,
                        "metadata_path": metadata_path,
                        "samples": len(targets_df)
                    }
                except Exception as e:
                    print(f"Error converting H5 to CSV: {e}")
                    return {"success": True, "training_data_path": training_data_path, "error_csv": str(e)}
            else:
                print("\nPipeline failed to generate training data. Check the error messages above.")
                return {"success": False, "error": "Failed to generate training data"}
        else:
            print("\nPipeline skipped training data generation due to missing climate data.")
            return {"success": False, "error": "Missing climate data"}
    
    finally:
        # Restore stdout and stderr if they were redirected
        if redirect_output and log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
            print(f"Pipeline log saved to {os.path.join(config['output_dir'], 'pipeline_output.log')}")



def get_config(config_overrides=None):
    """
    Get configuration for the pipeline, with optional overrides.
    
    Args:
        config_overrides (dict, optional): Dictionary of configuration overrides.
        
    Returns:
        dict: Configuration dictionary.
    """
    # For CLI usage, parse arguments and merge with config
    if config_overrides is None and len(sys.argv) > 1:
        args = parse_args()
        config = load_config(args.config)
        config = merge_config_with_args(config, args)
        return config
    
    # For programmatic usage, use create_config with overrides
    return create_config(config_overrides)


if __name__ == "__main__":
    # When run as a script, use command-line arguments
    run_pipeline()
