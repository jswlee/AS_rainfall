#!/usr/bin/env python3
"""
Process raw rainfall data into monthly aggregates.

This script reads raw rainfall data CSV files and aggregates them into monthly totals,
which are then saved as CSV files in the output directory.
"""

import pandas as pd
import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def process_rainfall_directory(input_dir, output_dir="processed_data/mon_rainfall"):
    """
    Process all CSV files in the input directory and create monthly aggregates.
    
    Parameters
    ----------
    input_dir : str
        Path to directory containing raw rainfall CSV files
    output_dir : str, optional
        Path to output directory for monthly aggregated files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing rainfall data from {input_dir} to {output_dir}")
    
    # Count files for progress reporting
    files = list(input_path.glob("*.csv"))
    total_files = len(files)
    processed_files = 0
    
    for file in files:
        try:
            df = pd.read_csv(file)
            
            # Convert datetime column
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Extract year-month period
            df['year_month'] = df['datetime'].dt.to_period('M')
            grouped = df.groupby('year_month')
            
            # Define aggregation function
            def sum_or_nan(series):
                return pd.NA if series.isna().any() else series.sum()
            
            # Apply only to the 'precip_in' column to avoid deprecation warning
            monthly_precip = grouped['precip_in'].apply(sum_or_nan).reset_index(name='monthly_total_precip_in')
            
            # Convert period to string for easier handling
            monthly_precip['year_month'] = monthly_precip['year_month'].astype(str)
            
            # Save to output file
            output_file = output_path / f"{file.stem}_monthly.csv"
            monthly_precip.to_csv(output_file, index=False)
            
            processed_files += 1
            print(f"Processed: {file.name} -> {output_file.name} [{processed_files}/{total_files}]")
            
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
    
    print(f"Completed processing {processed_files} out of {total_files} files.")

def main():
    parser = argparse.ArgumentParser(description='Process raw rainfall data into monthly aggregates')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw rainfall CSV files')
    parser.add_argument('--output_dir', type=str, default='processed_data/mon_rainfall',
                        help='Directory to save monthly aggregated files')
    
    args = parser.parse_args()
    
    process_rainfall_directory(args.input_dir, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
