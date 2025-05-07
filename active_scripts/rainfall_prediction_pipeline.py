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

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the new directory structure
from src.processors.dem_processor.processor import DEMProcessor
from src.processors.climate_processor.processor import ClimateDataProcessor
from src.processors.rainfall_processor.processor import RainfallProcessor
from src.utils.data_generator import DataGenerator

# Configuration
CONFIG = {
    'dem_path': 'data/DEM/DEM_Tut1.tif',
    'climate_data_path': 'processed_data/AS_climate_var_ds_updated.nc',
    'rainfall_dir': 'processed_data/mon_rainfall',
    'station_locations_path': 'data/as_raingage_list2.csv',
    'output_dir': 'output',
    'grid_size': 5,  # 5x5 grid = 25 points
    'patch_sizes': {
        'local': 3,    # 3x3 grid for local patch (12km)
        'regional': 3  # 3x3 grid for regional patch (60km)
    },
    'km_per_cell': {
        'local': 4,    # 4km per cell for local patch (12km total)
        'regional': 20  # 20km per cell for regional patch (60km total)
    }
}

def setup_environment():
    """Create necessary directories and check for required files."""
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Check if required files exist
    required_files = [
        CONFIG['dem_path'],
        CONFIG['climate_data_path'],
        CONFIG['rainfall_dir'],
        CONFIG['station_locations_path']
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            return False
    
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
    
    # Visualize a sample patch
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(local_patches[0], cmap='terrain')
    plt.title('Sample Local Patch (12km)')
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(regional_patches[0], cmap='terrain')
    plt.title('Sample Regional Patch (60km)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/dem_patches.png")
    
    return {
        'grid_points': grid_points,
        'local_patches': local_patches,
        'regional_patches': regional_patches
    }

def process_climate_data():
    """Process climate data and interpolate to grid points."""
    print("\nProcessing climate data...")
    
    # Initialize climate processor
    climate_processor = ClimateDataProcessor()
    
    # Process all climate variables
    for var_name in climate_processor.variable_configs.keys():
        print(f"Processing {var_name}...")
        climate_processor.process_variable(var_name)
    
    # Save processed climate data
    climate_data_path = f"{CONFIG['output_dir']}/processed_climate_data.nc"
    climate_processor.save_to_netcdf(climate_data_path)
    print(f"Saved processed climate data to {climate_data_path}")
    
    # Return the path to the processed climate data
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
    
    for date_str in available_dates:
        print(f"Interpolating rainfall for {date_str}...")
        
        # Get rainfall data for this date
        rainfall_data = rainfall_processor.get_rainfall_for_date(date_str)
        
        # Interpolate to grid points
        grid_rainfall = rainfall_processor.interpolate_to_grid(
            rainfall_data, 
            grid_points
        )
        
        interpolated_rainfall[date_str] = grid_rainfall
    
    print(f"Interpolated rainfall for {len(interpolated_rainfall)} dates")
    
    # Visualize sample interpolated rainfall
    sample_date = available_dates[0]
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
    plt.savefig(f"{CONFIG['output_dir']}/interpolated_rainfall_{sample_date}.png")
    
    return interpolated_rainfall, available_dates

def generate_training_data(dem_data, climate_data_path, rainfall_data, available_dates):
    """Generate training data for deep learning."""
    print("\nGenerating training data...")
    
    # Initialize data generator
    data_generator = DataGenerator(
        grid_points=dem_data['grid_points'],
        local_patches=dem_data['local_patches'],
        regional_patches=dem_data['regional_patches'],
        climate_data_path=climate_data_path,
        rainfall_data=rainfall_data,
        output_dir=CONFIG['output_dir']
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
            data_generator.visualize_sample(all_data[0], common_dates[0])
            print(f"Sample visualization complete for {common_dates[0]}")
        except Exception as e:
            print(f"Error visualizing sample: {e}")
        
        return h5_path
    else:
        print("No data was generated. Check the error messages above.")
        return None

def main():
    """Main function to run the entire pipeline."""
    print("Starting Rainfall Prediction Pipeline...")
    
    # Setup environment
    if not setup_environment():
        return
    
    # Process DEM
    dem_data = process_dem()
    
    # Process climate data
    climate_data_path = process_climate_data()
    
    # Process rainfall data
    rainfall_data, available_dates = process_rainfall_data(dem_data['grid_points'])
    
    # Generate training data
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
    else:
        print("\nPipeline failed to generate training data. Check the error messages above.")

if __name__ == "__main__":
    main()
