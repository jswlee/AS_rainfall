"""
Combined Subsetting and Concatenation Script for Climate Data

This script performs a two-step process for specified climate variables:
1.  Subset: It finds daily NetCDF files, cuts them down to a specific 
    geographic region (e.g., American Samoa), and saves them temporarily.
2.  Concatenate: It combines these smaller daily files into a single, 
    time-series NetCDF file for each variable, renaming them according to a 
    predefined convention from the config file.
"""
import xarray as xr
import argparse
from pathlib import Path
import config  # Import directly from your config.py file

# --- Configuration Dictionaries ---

# This dictionary remains here because it was not found in the provided config.py
# It maps the raw data folder names to their human-readable variable names.
DAILY_FOLDER_TO_VARIABLE = {
    'gh_1980-1984': 'Geopotential Height',
    'mslp_1980-1984': 'Sea Level Pressure',
    't2m_1980-1984': 'Air 2m',
    'temp_1980-1984': 'Air',
    'omg1980-1984': 'Omega',
    'pwat_1980-1984': 'Precipitable Water',
    'shum_1980-1984': 'Specific Humidity',
    'uwnd_1980-1984': 'Zonal Wind',
    'vwnd_1980-1984': 'Meridional Wind',
    'tskn_1980-1984': 'Skin Temperature',
    'ptmp_1980-1984': 'Potential Temperature',
}

def process_variable(
    folder_name: str, 
    variable_name: str,
    input_base_dir: Path, 
    subset_base_dir: Path, 
    output_base_dir: Path, 
    lat_slice: slice, 
    lon_slice: slice
):
    """Handles the full subset-and-concatenate process for one variable."""
    
    # 1. --- SUBSETTING ---
    print(f"\n--- Step 1: Subsetting {variable_name} files from '{folder_name}' ---")
    source_dir = input_base_dir / folder_name
    subset_dir = subset_base_dir / folder_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print(f"  ✗ Warning: Source folder not found at '{source_dir}'. Skipping.")
        return False

    nc_files = sorted(list(source_dir.glob("*.nc")))
    if not nc_files:
        print(f"  ✗ Warning: No .nc files found in '{source_dir}'. Skipping.")
        return False
        
    print(f"  Found {len(nc_files)} files. Creating subsets in '{subset_dir}'...")
    for input_file in nc_files:
        output_file = subset_dir / f"{input_file.stem}_subset.nc"
        try:
            with xr.open_dataset(input_file) as ds:
                subset = ds.sel(latitude=lat_slice, longitude=lon_slice)
                subset.to_netcdf(output_file)
        except Exception as e:
            print(f"    ✗ Failed to subset {input_file.name}: {e}")
            continue
    print("  ✓ Subsetting complete.")

    # 2. --- CONCATENATION & RENAMING ---
    print(f"\n--- Step 2: Concatenating {variable_name} and renaming ---")
    subset_files = sorted(list(subset_dir.glob("*_subset.nc")))
    if not subset_files:
        print("  ✗ No subset files found to concatenate. Skipping.")
        return False

    # Use the dictionaries to construct the final output filename
    # VARIABLE_MAPPING is now sourced from your config file.
    short_name = config.VARIABLE_MAPPING.get(variable_name)
    if not short_name:
        print(f"  ✗ Warning: No short name found for '{variable_name}' in config.VARIABLE_MAPPING. Using folder name for output.")
        output_filename = f"{folder_name}_concatenated.nc"
    else:
        output_filename = f"{short_name}.day.mean.nc"

    output_file = output_base_dir / output_filename
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Combining {len(subset_files)} files into '{output_file}'...")
    try:
        with xr.open_mfdataset(subset_files, combine='by_coords') as ds:
            ds.to_netcdf(output_file)
        print("  ✓ Concatenation and renaming complete.")
        return True
    except Exception as e:
        print(f"  ✗ Failed to concatenate files: {e}")
        return False

def main():
    """Main function to parse arguments and process variables."""
    parser = argparse.ArgumentParser(
        description="A two-step script to subset and concatenate climate NetCDF data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Directory arguments
    parser.add_argument("--input_dir", type=Path, default="raw_data/climate_variables_daily",
                        help="Base directory containing the original variable subfolders.")
    parser.add_argument("--subset_dir", type=Path, default="raw_data/climate_variables_daily_as_subset",
                        help="Intermediate directory to save temporary subset files.")
    parser.add_argument("--output_dir", type=Path, default="raw_data/climate_variables_daily_concatenated",
                        help="Final directory to save the concatenated and renamed NetCDF files.")
    # Subsetting arguments
    parser.add_argument("--lat_max", type=float, default=-10.0, help="Maximum latitude for the subset.")
    parser.add_argument("--lat_min", type=float, default=-20.0, help="Minimum latitude for the subset.")
    parser.add_argument("--lon_min", type=float, default=-180.0, help="Minimum longitude for the subset.")
    parser.add_argument("--lon_max", type=float, default=-160.0, help="Maximum longitude for the subset.")
    # Control arguments
    parser.add_argument("--variables", nargs="+", help="Specific variables (folder names) to process (e.g., 'gh_1980-1984').")
    
    args = parser.parse_args()

    # Determine which folders to process based on user input
    if args.variables:
        folders_to_process = {k: v for k, v in DAILY_FOLDER_TO_VARIABLE.items() if k in args.variables}
    else:
        folders_to_process = DAILY_FOLDER_TO_VARIABLE

    if not folders_to_process:
        print("Error: None of the specified variables were found. Check folder names.")
        return

    print(f"Processing {len(folders_to_process)} of {len(DAILY_FOLDER_TO_VARIABLE)} total variables.\n")

    # Define the geographic slices for xarray selection
    lat_slice = slice(args.lat_max, args.lat_min)
    lon_slice = slice(args.lon_min, args.lon_max)
    
    successful_vars = 0
    # Loop through the items to get both folder and variable name
    for folder_name, variable_name in folders_to_process.items():
        if process_variable(
            folder_name, variable_name, args.input_dir, args.subset_dir, 
            args.output_dir, lat_slice, lon_slice
        ):
            successful_vars += 1

    print(f"\n========================================================")
    print(f"Finished: {successful_vars} / {len(folders_to_process)} variables processed successfully.")
    print(f"Final concatenated files are in: '{args.output_dir}'")
    print(f"========================================================")

if __name__ == "__main__":
    main()