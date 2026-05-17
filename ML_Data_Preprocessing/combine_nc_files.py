#!/usr/bin/env python3
"""
Simple script to combine NetCDF files from two directories.
Concatenates files with matching names along the time dimension.
"""

import xarray as xr
from pathlib import Path
import argparse
import numpy as np


def combine_matching_files(dir1: Path, dir2: Path, output_dir: Path):
    """
    Combine matching .nc files from two directories.
    
    Args:
        dir1: First directory with .nc files
        dir2: Second directory with .nc files
        output_dir: Directory to save combined files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .nc files from first directory
    files1 = {f.name: f for f in dir1.glob("*.nc")}
    files2 = {f.name: f for f in dir2.glob("*.nc")}
    
    # Find matching files
    matching_files = set(files1.keys()) & set(files2.keys())
    
    if not matching_files:
        print("No matching files found!")
        return
    
    print(f"Found {len(matching_files)} matching files to combine:")
    for filename in sorted(matching_files):
        print(f"  - {filename}")
    
    # Combine each matching file
    for filename in sorted(matching_files):
        file1 = files1[filename]
        file2 = files2[filename]
        output_file = output_dir / filename
        
        # Skip if already exists
        if output_file.exists():
            print(f"Skipping {filename} (already exists)")
            continue
        
        print(f"Combining {filename}...")
        try:
            # Explicitly open each file and concatenate along a time-like dimension
            with xr.open_dataset(file1) as ds1, xr.open_dataset(file2) as ds2:
                # Prefer 'valid_time' if present, otherwise fall back to 'time'
                time_dims1 = [d for d in ds1.dims if d in ("valid_time", "time")]
                time_dims2 = [d for d in ds2.dims if d in ("valid_time", "time")]

                if not time_dims1 or not time_dims2 or time_dims1[0] != time_dims2[0]:
                    raise ValueError(
                        f"Cannot determine common time dimension for {filename}: "
                        f"dims1={list(ds1.dims)}, dims2={list(ds2.dims)}"
                    )

                time_dim = time_dims1[0]

                ds = xr.concat([ds1, ds2], dim=time_dim, join='inner')

                # Sort pressure_level descending if present (match original order)
                if 'pressure_level' in ds.dims:
                    ds = ds.sortby('pressure_level', ascending=False)

                # Sort along the time dimension to enforce monotonic order
                if time_dim in ds.coords:
                    ds = ds.sortby(time_dim)

                    # Drop duplicate time indices, if any, to ensure strictly monotonic
                    coord_values = ds[time_dim].values
                    _, unique_indices = np.unique(coord_values, return_index=True)
                    unique_indices = np.sort(unique_indices)
                    ds = ds.isel({time_dim: unique_indices})

                ds.to_netcdf(output_file)

            print(f"  ✓ Saved to {output_file.name}")

        except Exception as e:
            print(f"  ✗ Error combining {filename}: {e}")
    
    print(f"\nDone! Combined files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine matching NetCDF files from two directories"
    )
    parser.add_argument(
        '--dir1',
        type=Path,
        default=Path('raw_data/AS/climate_variables_daily_FULL_1980-2024'),
        help='First directory with .nc files'
    )
    parser.add_argument(
        '--dir2',
        type=Path,
        default=Path('raw_data/AS/2010-2014'),
        help='Second directory with .nc files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('raw_data/AS/climate_variables_daily_1980-2024_updated'),
        help='Output directory for combined files'
    )
    
    args = parser.parse_args()
    
    # Validate directories exist
    if not args.dir1.exists():
        print(f"Error: Directory not found: {args.dir1}")
        return
    if not args.dir2.exists():
        print(f"Error: Directory not found: {args.dir2}")
        return
    
    print(f"Combining files from:")
    print(f"  Dir 1: {args.dir1}")
    print(f"  Dir 2: {args.dir2}")
    print(f"  Output: {args.output}")
    print()
    
    combine_matching_files(args.dir1, args.dir2, args.output)


if __name__ == '__main__':
    main()
