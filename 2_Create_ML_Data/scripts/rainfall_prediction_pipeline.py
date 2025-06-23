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
import matplotlib.pyplot as plt
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
from convert_h5_to_csv import extract_data_from_h5

# Define script and project directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))

# Add scripts directory to Python path for imports
sys.path.append(os.path.join(PROJECT_ROOT, '2_Create_ML_Data', 'scripts'))

# Import the processors and utils
from processors.dem_processor.processor import DEMProcessor
from processors.climate_processor.processor import ClimateDataProcessor
from processors.rainfall_processor.processor import RainfallProcessor
from utils.data_generator import DataGenerator
from utils.config_utils import load_config, parse_args, merge_config_with_args

# Load configuration from YAML file and/or command-line arguments
def get_config():
    # Parse command-line arguments
    args = parse_args()
    
    # Load config from file (default or specified by --config)
    config = load_config(args.config if hasattr(args, 'config') else None)
    
    # Override config with command-line arguments
    config = merge_config_with_args(config, args)
    
    return config

# Get configuration
CONFIG = get_config()

# Create output directories
os.makedirs(CONFIG['output_dir'], exist_ok=True)
# Create figures directory
figures_dir = os.path.join(os.path.dirname(CONFIG['output_dir']), 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Check if climate data exists (either processed file or raw files)
processed_file_exists = os.path.exists(CONFIG['climate_data_path'])
raw_files_exist = False

# Check if raw climate data directory exists and has NetCDF files
if os.path.exists(CONFIG['raw_climate_dir']):
    nc_files = [f for f in os.listdir(CONFIG['raw_climate_dir']) if f.endswith('.nc')]
    raw_files_exist = len(nc_files) > 0
    if raw_files_exist:
        print(f"Found {len(nc_files)} raw climate data files in {CONFIG['raw_climate_dir']}")

# Set the flag based on whether either source of climate data exists
CLIMATE_DATA_EXISTS = processed_file_exists or raw_files_exist

if not CLIMATE_DATA_EXISTS:
    print("Warning: Neither processed climate data nor raw climate files found. Climate-dependent steps will be skipped.")
elif not processed_file_exists and raw_files_exist:
    print(f"Processed climate data not found at {CONFIG['climate_data_path']}, but raw files exist and will be processed.")
elif processed_file_exists:
    print(f"Found existing processed climate data at: {CONFIG['climate_data_path']}")

# Redirect all stdout and stderr to a log file in the output directory
log_path = os.path.join(CONFIG['output_dir'], "pipeline_output.log")
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

def setup_environment():
    """Create necessary directories and check for required files."""
    # Always required files (regardless of climate data)
    required_files = [
        CONFIG['dem_path'],
        CONFIG['rainfall_dir'],
        CONFIG['station_locations_path']
    ]
    
    # Check required files
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            return False
    
    # Climate data is already checked at initialization, and CLIMATE_DATA_EXISTS is set
    # We don't need to check it again here
    
    print("All required files found.")
    return True

def process_dem():
    """Process DEM to create grid points and patches."""
    print("\nProcessing DEM...")
    
    # Initialize DEM processor
    dem_processor = DEMProcessor(CONFIG['dem_path'])
    
    # Generate grid points (5x5 grid = 25 points)
    grid_points = dem_processor.generate_grid_points(CONFIG['grid_size'])
    print(f"Generated {len(grid_points)} grid points")
    
    # Create local and regional patches for each grid point
    local_patches = []
    regional_patches = []
    
    for point in grid_points:
        local_patch = dem_processor.extract_patch(
            point, 
            patch_size=CONFIG['patch_sizes']['local'],
            km_per_cell=CONFIG['km_per_cell']['local']  # 12km total (3x3 grid with 4km per cell)
        )
        
        regional_patch = dem_processor.extract_patch(
            point, 
            patch_size=CONFIG['patch_sizes']['regional'],
            km_per_cell=CONFIG['km_per_cell']['regional']  # ~60km total (3x3 grid with ~20km per cell)
        )
        
        local_patches.append(local_patch)
        regional_patches.append(regional_patch)
    
    print(f"Created {len(local_patches)} local patches and {len(regional_patches)} regional patches")
    
    # Visualize all patches in a grid
    plt.figure(figsize=(14, 6))
    
    # Arrange local patches in a grid
    local_patches_array = np.array(local_patches)
    n_patches, local_patch_height, local_patch_width = local_patches_array.shape
    grid_rows, grid_cols = CONFIG['grid_size'], CONFIG['grid_size']  # Use the configured grid size
    
    # Create empty grid for local patches
    local_grid = np.zeros((grid_rows * local_patch_height, grid_cols * local_patch_width))
    
    # Place local patches in grid
    for i in range(min(n_patches, grid_rows * grid_cols)):
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate position in the grid
        row_start = row * local_patch_height
        row_end = (row + 1) * local_patch_height
        col_start = col * local_patch_width
        col_end = (col + 1) * local_patch_width
        
        # Place the patch
        local_grid[row_start:row_end, col_start:col_end] = local_patches_array[i]
    
    # Do the same for regional patches - but with their own dimensions
    regional_patches_array = np.array(regional_patches)
    _, regional_patch_height, regional_patch_width = regional_patches_array.shape
    
    # Create empty grid for regional patches with the regional patch dimensions
    regional_grid = np.zeros((grid_rows * regional_patch_height, grid_cols * regional_patch_width))
    
    # Place regional patches in grid
    for i in range(min(n_patches, grid_rows * grid_cols)):
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate position in the grid using regional patch dimensions
        row_start = row * regional_patch_height
        row_end = (row + 1) * regional_patch_height
        col_start = col * regional_patch_width
        col_end = (col + 1) * regional_patch_width
        
        # Place the patch
        regional_grid[row_start:row_end, col_start:col_end] = regional_patches_array[i]
    
    # Plot local patches grid
    plt.subplot(121)
    plt.imshow(local_grid, cmap='terrain', origin='lower')
    local_patch_km = CONFIG['patch_sizes']['local'] * CONFIG['km_per_cell']['local']
    plt.title(f'Local DEM Patches ({CONFIG["grid_size"]}x{CONFIG["grid_size"]} grid, {local_patch_km}km each)')
    plt.colorbar()
    
    # Plot regional patches grid
    plt.subplot(122)
    plt.imshow(regional_grid, cmap='terrain', origin='lower')
    regional_patch_km = CONFIG['patch_sizes']['regional'] * CONFIG['km_per_cell']['regional']
    plt.title(f'Regional DEM Patches ({CONFIG["grid_size"]}x{CONFIG["grid_size"]} grid, {regional_patch_km}km each)')
    plt.colorbar()
    
    plt.tight_layout()
    dem_patches_path = os.path.join(figures_dir, 'dem_patches.png')
    plt.savefig(dem_patches_path)
    print(f"Saved DEM patches visualization to {dem_patches_path}")
    
    return {
        'grid_points': grid_points,
        'local_patches': local_patches,
        'regional_patches': regional_patches
    }

def process_climate_data():
    """Process climate data and interpolate to grid points."""
    if not CLIMATE_DATA_EXISTS:
        print("Warning: No climate data available. Skipping climate data processing.")
        return None
    
    print("\nProcessing climate data...")
    existing_processed_data = CONFIG['climate_data_path']
    raw_climate_dir = CONFIG['raw_climate_dir']
    
    # We already know if raw files exist from the initial check
    if os.path.exists(existing_processed_data):
        # Use existing processed data
        output_climate_path = os.path.join(CONFIG['output_dir'], 'processed_climate_data.nc')
        
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
        climate_data_path = os.path.join(CONFIG['output_dir'], 'processed_climate_data.nc')
        climate_processor.save_to_netcdf(climate_data_path)
        print(f"Saved processed climate data to {climate_data_path}")
        
        return climate_data_path

def process_rainfall_data(grid_points):
    """Process rainfall data and interpolate to grid points."""
    print("\nProcessing rainfall data...")
    
    # Initialize rainfall processor
    rainfall_processor = RainfallProcessor(
        rainfall_dir=CONFIG['rainfall_dir'],
        station_locations_path=CONFIG['station_locations_path']
    )
    
    # Get available dates
    available_dates = rainfall_processor.get_available_dates()
    print(f"Found {len(available_dates)} available dates for rainfall data")
    
    # Interpolate rainfall for each date and grid point
    interpolated_rainfall = {}
    skipped_dates = []
    
    for date_str in available_dates:
        print(f"Interpolating rainfall for {date_str}...")
        
        # Get rainfall data for this date
        rainfall_data = rainfall_processor.get_rainfall_for_date(date_str)
        
        # If all stations are missing or have NaN for this month, skip it entirely
        values = np.array(rainfall_data.get('values', []), dtype=float)
        # Only keep if at least one station has a real value (including true zeros)
        if len(values) == 0 or np.all(np.isnan(values)):
            print(f"WARNING: All stations missing/NaN for {date_str}, dropping month from dataset.")
            skipped_dates.append(date_str)
            continue
        # Remove any stations that have NaN values (optional, or could be handled in interpolation)
        valid_idx = ~np.isnan(values)
        rainfall_data['stations'] = list(np.array(rainfall_data['stations'])[valid_idx])
        rainfall_data['locations'] = list(np.array(rainfall_data['locations'])[valid_idx])
        rainfall_data['values'] = list(values[valid_idx])
        if len(rainfall_data['stations']) == 0:
            print(f"WARNING: After removing NaN stations, no valid rainfall for {date_str}, dropping month.")
            skipped_dates.append(date_str)
            continue
            
        # Use Gaussian Process interpolation by default for all years
        # This handles sparse data much better than previous methods
        grid_rainfall = rainfall_processor.interpolate_to_grid(
            rainfall_data, 
            grid_points,
            method='gp'
        )
        
        # Verify we have valid rainfall values
        if np.all(grid_rainfall == 0.0):
            # If all values are zero, check if the original data had non-zero values
            if not np.all(np.array(rainfall_data['values']) == 0):
                # Try again with RBF interpolation as fallback
                print(f"Original data had non-zero values, but GP interpolation produced all zeros.")
                print(f"Trying RBF interpolation as fallback...")
                grid_rainfall = rainfall_processor.interpolate_to_grid(
                    rainfall_data, 
                    grid_points,
                    method='rbf'
                )
                
                # If RBF still fails, try IDW as a last resort
                if np.all(grid_rainfall == 0.0) and not np.all(np.array(rainfall_data['values']) == 0):
                    print(f"RBF interpolation also failed. Trying IDW as last resort...")
                    grid_rainfall = rainfall_processor.interpolate_to_grid(
                        rainfall_data, 
                        grid_points,
                        method='idw'
                    )
                
        interpolated_rainfall[date_str] = grid_rainfall
    
    print(f"Interpolated rainfall for {len(interpolated_rainfall)} dates")
    
    # Visualize sample interpolated rainfall
    sample_date = available_dates[500]
    sample_rainfall = interpolated_rainfall[sample_date]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        [p[0] for p in grid_points],
        [p[1] for p in grid_points],
        c=sample_rainfall,
        cmap='Blues',
        s=100
    )
    plt.colorbar(label='Rainfall (mm)')
    plt.title(f'Interpolated Rainfall for {sample_date}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    rainfall_plot_path = os.path.join(figures_dir, f'interpolated_rainfall_{sample_date}.png')
    plt.savefig(rainfall_plot_path)
    print(f"Saved rainfall interpolation visualization to {rainfall_plot_path}")
    
    return interpolated_rainfall, available_dates

def generate_training_data(dem_data, climate_data_path, rainfall_data, available_dates):
    """Generate training data for deep learning."""
    print("\nGenerating training data...")
    if not CLIMATE_DATA_EXISTS or climate_data_path is None:
        print("Warning: Climate data is missing. Skipping training data generation.")
        return None
    print(f"Using climate data from: {climate_data_path}")
    
    # Initialize data generator with both output and figures directories
    data_generator = DataGenerator(
        grid_points=dem_data['grid_points'],
        local_patches=dem_data['local_patches'],
        regional_patches=dem_data['regional_patches'],
        climate_data_path=climate_data_path,
        rainfall_data=rainfall_data,
        output_dir=CONFIG['output_dir'],
        figures_dir=figures_dir,
        grid_size=CONFIG['grid_size']  # Pass the grid size from configuration
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
            data_generator.visualize_sample(all_data[500], common_dates[500])
            print(f"Sample visualization complete for {common_dates[500]}")
        except Exception as e:
            print(f"Error visualizing sample: {e}")
        
        return h5_path
    else:
        print("No data was generated. Check the error messages above.")
        return None

def main():
    """Main function to run the entire pipeline."""
    print("\n" + "="*80)
    print("RAINFALL PREDICTION PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print("Configuration:")
    for key, value in CONFIG.items():
        if key not in ['patch_sizes', 'km_per_cell']:
            print(f"  {key}: {value}")
    print(f"  patch_sizes: local={CONFIG['patch_sizes']['local']}, regional={CONFIG['patch_sizes']['regional']}")
    print(f"  km_per_cell: local={CONFIG['km_per_cell']['local']}, regional={CONFIG['km_per_cell']['regional']}")
    print("="*80 + "\n")
    
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
        progress_file = os.path.join(CONFIG['output_dir'], 'progress.log')
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, 'a') as f:
            f.write(f"{progress_msg}\n")
    
    # Step 1: Setup environment
    report_progress("Setting up environment")
    if not setup_environment():
        return
        
    # Step 2: Process DEM
    report_progress("Processing DEM data")
    dem_data = process_dem()
    
    # Step 3: Process climate data
    report_progress("Processing climate data")
    climate_data_path = process_climate_data()
    
    # Step 4: Process rainfall data
    report_progress("Processing rainfall data")
    rainfall_data, available_dates = process_rainfall_data(dem_data['grid_points'])
    
    # Step 5: Generate training data
    report_progress("Generating training data")
    # Generate training data only if climate data exists
    if CLIMATE_DATA_EXISTS and climate_data_path is not None:
        training_data_path = generate_training_data(
            dem_data,
            climate_data_path,
            rainfall_data,
            available_dates
        )
        if training_data_path:
            print(f"\nPipeline complete! Training data saved to {training_data_path}")
            print(f"Features: 16 climate variables, month encoding, local and regional DEM patches")
            print(f"Labels: Interpolated rainfall")
            
            # Step 6: Convert H5 to CSV
            report_progress("Converting H5 data to CSV format")
            output_dir = os.path.join(CONFIG['output_dir'], 'csv_data')
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
            except Exception as e:
                print(f"Error converting H5 to CSV: {e}")
        else:
            print("\nPipeline failed to generate training data. Check the error messages above.")
    else:
        print("\nPipeline skipped training data generation due to missing climate data.")


if __name__ == "__main__":
    main()
