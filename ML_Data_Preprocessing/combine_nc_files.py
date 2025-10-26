#!/usr/bin/env python3
"""
Simple script to combine NetCDF files from two directories.
Concatenates files with matching names along the time dimension.
"""

import xarray as xr
from pathlib import Path
import argparse


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
            # Open via multi-file dataset and let xarray align by coordinates
            # This mirrors: with xr.open_mfdataset(subset_files, combine='by_coords') as ds: ds.to_netcdf(...)
            with xr.open_mfdataset([file1, file2], combine='by_coords') as ds:
                # Sort by time to ensure chronological order if 'time' exists
                if 'time' in ds.coords:
                    ds = ds.sortby('time')
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
        default=Path('raw_data/climate_variables_daily_processed'),
        help='First directory with .nc files'
    )
    parser.add_argument(
        '--dir2',
        type=Path,
        default=Path('raw_data/climate_variables_daily_1994-1994_concatenated'),
        help='Second directory with .nc files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('raw_data/climate_variables_daily_FULL'),
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
