"""
Daily Data Concatenation Script

This script concatenates daily climate variable files into consolidated NetCDF files
following the same naming convention as the monthly data, but with 'day' instead of 'mon'.
"""

import os
import glob
import numpy as np
import xarray as xr
from pathlib import Path
import ML_Data_Preprocessing.config as config

# Mapping from daily folder names to variable names in VARIABLE_MAPPING
DAILY_FOLDER_TO_VARIABLE = {
    'gh_1980-1984': 'Geopotential Height',
    'mslp_1980-1984': 'Sea Level Pressure', 
    't2m_1980-1984': 'Air 2m',
    'temp_1980-1984': 'Air',
    'omg1980-1984': 'Omega',
    'pwat_1980-1984': 'Precipitable Water',
    'shum_1980-1984': 'Specific Humidity'
}

def get_output_filename(variable_name, time_interval='daily', statistic='mean'):
    """
    Get the output filename following the same convention as monthly data.
    
    Parameters
    ----------
    variable_name : str
        Variable name from VARIABLE_MAPPING
    time_interval : str
        Time interval ('daily' or 'monthly')
    statistic : str
        Statistic type ('mean')
    
    Returns
    -------
    str
        Output filename
    """
    if variable_name not in config.VARIABLE_MAPPING:
        raise ValueError(f"Unknown variable: {variable_name}")
    
    var_code = config.VARIABLE_MAPPING[variable_name]
    time_code = config.TIME_INTERVAL_MAPPING[time_interval]
    stat_code = config.STATISTIC_MAPPING[statistic]
    
    return f"{var_code}.{time_code}.{stat_code}.nc"

def concatenate_daily_files(input_dir, output_dir, folder_name, variable_name, batch_size=5):
    """
    Concatenate all NetCDF files in a daily folder into a single file using a memory-efficient
    batch processing approach.
    
    Parameters
    ----------
    input_dir : str
        Input directory containing daily folders
    output_dir : str
        Output directory for concatenated files
    folder_name : str
        Name of the daily folder (e.g., 'gh_1980-1984')
    variable_name : str
        Variable name from VARIABLE_MAPPING
    batch_size : int, optional
        Number of files to process in each batch, default is 5
    """
    folder_path = os.path.join(input_dir, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist")
        return False
    
    # Find all NetCDF files in the folder
    nc_files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))
    
    if not nc_files:
        print(f"Warning: No NetCDF files found in {folder_path}")
        return False
    
    total_files = len(nc_files)
    print(f"Processing {total_files} files for {variable_name} in batches of {batch_size}...")
    
    # Create output filename
    output_filename = get_output_filename(variable_name, 'daily', 'mean')
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in batches
    temp_files = []
    try:
        # Process files in batches and save intermediate results
        for batch_idx in range(0, total_files, batch_size):
            batch_end = min(batch_idx + batch_size, total_files)
            batch_files = nc_files[batch_idx:batch_end]
            
            print(f"  Processing batch {batch_idx//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}: files {batch_idx+1}-{batch_end} of {total_files}")
            
            # Open batch of files
            datasets = []
            for file_path in batch_files:
                try:
                    ds = xr.open_dataset(file_path)
                    datasets.append(ds)
                except Exception as e:
                    print(f"    Warning: Could not open {os.path.basename(file_path)}: {e}")
                    continue
            
            if not datasets:
                print(f"    Warning: No valid datasets in this batch, skipping...")
                continue
            
            # Concatenate batch along time dimension
            batch_ds = xr.concat(datasets, dim='time')
            
            # Sort by time
            batch_ds = batch_ds.sortby('time')
            
            # Save intermediate batch result
            temp_file = f"{output_path}.batch{batch_idx//batch_size + 1}.temp.nc"
            batch_ds.to_netcdf(temp_file)
            temp_files.append(temp_file)
            
            # Close datasets to free memory
            for ds in datasets:
                ds.close()
            batch_ds.close()
            
            # Force garbage collection
            import gc
            gc.collect()
        
        if not temp_files:
            print(f"Error: No batches were successfully processed for {variable_name}")
            return False
        
        print(f"Merging {len(temp_files)} batch files...")
        temp_datasets = []
        for merge_idx, temp_file in enumerate(temp_files, 1):
            basename = os.path.basename(temp_file)
            print(
                f"  [{merge_idx}/{len(temp_files)}] Loading batch file '{basename}'",
                flush=True,
            )
            ds = xr.open_dataset(temp_file)
            temp_datasets.append(ds)
        
        print("  Concatenating loaded batches...", flush=True)
        final_ds = xr.concat(temp_datasets, dim='time')
        
        print("  Sorting concatenated dataset by time...", flush=True)
        final_ds = final_ds.sortby('time')
        
        print("  Writing final NetCDF file...", flush=True)
        final_ds.to_netcdf(output_path)

        # Get time range for reporting
        time_start = final_ds.time.values[0]
        time_end = final_ds.time.values[-1]
        # Close datasets and remove temporary files
        for ds in temp_datasets:
            ds.close()
        final_ds.close()
        
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"  Removed temporary file: {os.path.basename(temp_file)}")
            except Exception as e:
                print(f"  Warning: Could not remove temporary file {temp_file}: {e}")
        
        print(f"Successfully created {output_path}")
        print(f"  Time range: {time_start} to {time_end}")
        
        return True
    except Exception as e:
        print(f"Error processing {variable_name}: {e}")
        
        # Clean up any temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
                
        return False

def main(variables=None, batch_size=5):
    """
    Main function to concatenate daily climate variable files.
    
    Parameters
    ----------
    variables : list, optional
        List of variable names to process. If None, process all variables.
    batch_size : int, optional
        Number of files to process in each batch.
    """
    input_dir = "raw_data/climate_variables_daily"
    output_dir = config.REANALYSIS_DIR_DAILY
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size} files per batch")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Filter variables if specified
    if variables:
        # Create a filtered dictionary of folders to process
        folders_to_process = {}
        for folder, var in DAILY_FOLDER_TO_VARIABLE.items():
            if var in variables or folder in variables:
                folders_to_process[folder] = var
        
        if not folders_to_process:
            print(f"Error: No matching variables found for {variables}")
            print(f"Available variables: {list(DAILY_FOLDER_TO_VARIABLE.values())}")
            print(f"Available folders: {list(DAILY_FOLDER_TO_VARIABLE.keys())}")
            return
    else:
        # Process all variables
        folders_to_process = DAILY_FOLDER_TO_VARIABLE
    
    # Process each daily folder
    successful = 0
    total = len(folders_to_process)
    
    print(f"Will process {total} variables: {list(folders_to_process.values())}")
    print()
    
    for i, (folder_name, variable_name) in enumerate(folders_to_process.items(), 1):
        print(f"[{i}/{total}] Processing {folder_name} -> {variable_name}")
        
        if concatenate_daily_files(input_dir, output_dir, folder_name, variable_name, batch_size):
            successful += 1
        
        print()
    
    print(f"Concatenation complete: {successful}/{total} variables processed successfully")
    
    if successful > 0:
        print(f"\nDaily data files created in: {output_dir}")
        print("You can now use these files with the existing pipeline by setting:")
        print("  feature_builder = ReanalysisFeatureBuilder(time_interval=\"daily\")")
        print("  # or")
        print("  feature_builder = ReanalysisFeatureBuilder(reanalysis_dir=config.REANALYSIS_DIR_DAILY)")
    else:
        print("\nNo variables were successfully processed.")
        print("Check the error messages above and try again with a smaller batch size.")
        print("Example: python concatenate_daily_data.py --batch-size 3 --variables 'Geopotential Height'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Concatenate daily climate variable files into consolidated NetCDF files")
    parser.add_argument(
        "--variables", 
        nargs="+", 
        help="Specific variables or folders to process (e.g., 'Geopotential Height' or 'gh_1980-1984')"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=3,
        help="Number of files to process in each batch (default: 3)"
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List available variables and folders without processing"
    )
    
    args = parser.parse_args()
    
    if args.list_variables:
        print("Available variables and folders:")
        print("\nVariable Names:")
        for var in sorted(DAILY_FOLDER_TO_VARIABLE.values()):
            print(f"  - {var}")
        print("\nFolder Names:")
        for folder in sorted(DAILY_FOLDER_TO_VARIABLE.keys()):
            print(f"  - {folder}")
    else:
        main(variables=args.variables, batch_size=args.batch_size)
