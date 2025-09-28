"""
Concatenation Script for American Samoa Subset Data

This script is a modified version of concatenate_daily_data.py that works with
the subset NetCDF files created by subset_netcdf_files.py.
"""

import os
import sys
from ML_Data_Preprocessing.concatenate_daily_data import (
    concatenate_daily_files, DAILY_FOLDER_TO_VARIABLE
)

def main(variables=None, batch_size=5):
    """
    Main function to concatenate subset NetCDF files.
    
    Parameters
    ----------
    variables : list, optional
        List of variable names to process. If None, process all variables.
    batch_size : int, optional
        Number of files to process in each batch.
    """
    input_dir = "raw_data/climate_variables_daily_as_subset"
    output_dir = "raw_data/climate_variables_daily_processed_as_subset"
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size} files per batch")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        print("Please run subset_netcdf_files.py first to create the subset files.")
        return False
    
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
            return False
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
        print("  feature_builder = ReanalysisFeatureBuilder(reanalysis_dir=\"raw_data/climate_variables_daily_processed_as_subset\")")
        return True
    else:
        print("\nNo variables were successfully processed.")
        print("Check the error messages above and try again with a smaller batch size.")
        print("Example: python concatenate_subset_data.py --batch-size 3 --variables 'Geopotential Height'")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Concatenate subset NetCDF files into consolidated NetCDF files")
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
        success = main(args.variables, args.batch_size)
        sys.exit(0 if success else 1)
