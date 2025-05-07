"""
Script to regenerate the climate dataset with the updated precipitable water file.
"""
import os
import sys
import numpy as np
import xarray as xr

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processors.climate_processor.processor import ClimateProcessor
from src.processors.climate_processor.config import DEFAULT_VARIABLE_CONFIGS

# Output file path
output_file = 'processed_data/AS_climate_var_ds_updated.nc'

# Initialize the climate processor
print("Initializing climate data processor...")
processor = ClimateProcessor()

# Process all variables
print("\nProcessing climate variables...")
for var_name in DEFAULT_VARIABLE_CONFIGS.keys():
    print(f"Processing {var_name}...")
    success = processor.process_variable(var_name)
    print(f"  Success: {success}")
    
    # Check for NaN values in the processed data
    if success and var_name in processor.climate_data:
        data = processor.climate_data[var_name].values
        nan_count = np.isnan(data).sum()
        total_count = data.size
        nan_percentage = (nan_count / total_count) * 100
        print(f"  NaN values: {nan_count}/{total_count} ({nan_percentage:.2f}%)")

# Validate grid consistency
print("\nValidating grid consistency...")
is_consistent = processor.validate_grid_consistency()
print(f"Grid consistency: {is_consistent}")

# Save the dataset
print(f"\nSaving dataset to {output_file}...")
processor.save_to_netcdf(output_file)
print("Dataset saved successfully!")

# Print summary of variables in the processor
print("\nVariables in the processor:")
for var_name, data_array in processor.climate_data.items():
    print(f"- {var_name}")
    
    # Check for NaN values
    data = data_array.values
    nan_count = np.isnan(data).sum()
    total_count = data.size
    nan_percentage = (nan_count / total_count) * 100
    print(f"  NaN values: {nan_count}/{total_count} ({nan_percentage:.2f}%)")

print("\nClimate data regeneration complete!")
