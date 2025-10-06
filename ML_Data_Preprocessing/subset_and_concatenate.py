import xarray as xr
import argparse
from pathlib import Path
import ML_Data_Preprocessing.config as config

def process_variable(folder_name, variable_name, input_dir, subset_dir, output_dir, lat_slice, lon_slice):
    """Subset and concatenate files for one variable."""
    source_dir = input_dir / folder_name
    temp_dir = subset_dir / folder_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine final output path up front; if present, skip entire variable
    short_name = config.VARIABLE_MAPPING.get(variable_name, folder_name)
    final_out = output_dir / f"{short_name}.day.mean.nc"
    if final_out.exists():
        print(f"Skipping {variable_name}: final output already exists -> {final_out}")
        return True
    
    # Subset files
    nc_files = sorted(source_dir.glob("*.nc"))
    if not nc_files:
        print(f"No files found in {source_dir}")
        return False
    
    print(f"Subsetting {len(nc_files)} files for {variable_name}...")
    for input_file in nc_files:
        output_file = temp_dir / f"{input_file.stem}_subset.nc"
        # Skip per-file work if subset already exists
        if output_file.exists():
            continue
        with xr.open_dataset(input_file) as ds:
            subset = ds.sel(latitude=lat_slice, longitude=lon_slice)
            subset.to_netcdf(output_file)
    
    # Concatenate
    subset_files = sorted(temp_dir.glob("*_subset.nc"))
    output_file = final_out
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not subset_files:
        print(f"No subset files found for {variable_name}; skipping concatenation.")
        return False
    
    print(f"Concatenating to {output_file.name}...")
    with xr.open_mfdataset(subset_files, combine='by_coords') as ds:
        ds.to_netcdf(output_file)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Subset and concatenate climate data.")
    parser.add_argument('--input_dir', type=Path, default='raw_data/climate_variables_daily')
    parser.add_argument('--subset_dir', type=Path, default=Path('./tmp_subsets_daily'))
    parser.add_argument('--output_dir', type=Path, default=Path('./raw_data/climate_variables_daily_processed'))
    parser.add_argument('--min_lat', type=float, default=-10.0)
    parser.add_argument('--max_lat', type=float, default=-20.0)
    parser.add_argument('--min_lon', type=float, default=-180.0)
    parser.add_argument('--max_lon', type=float, default=-160.0)
    parser.add_argument('--variables', nargs='*', help='Specific folders to process')
    args = parser.parse_args()

    folders = {f: v for f, v in config.DAILY_FOLDER_TO_VARIABLE.items() 
               if not args.variables or f in args.variables}
    
    lat_slice = slice(args.min_lat, args.max_lat)
    lon_slice = slice(args.min_lon, args.max_lon)

    successful_vars = 0
    for folder_name, variable_name in folders.items():
        if process_variable(folder_name, variable_name, args.input_dir, 
                        args.subset_dir, args.output_dir, lat_slice, lon_slice):
            successful_vars += 1

    print(f"Finished: {successful_vars} / {len(folders)} variables processed successfully.")
    print(f"Final concatenated files are in: '{args.output_dir}'")

if __name__ == '__main__':
    main()