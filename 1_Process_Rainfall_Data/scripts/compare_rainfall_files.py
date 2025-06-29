#!/usr/bin/env python3
"""
Compare rainfall files between raw_data/rainfall and raw_data/rainfall_added.
Compares files based on matching datetime values and saves differences to a log file.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Directories
RAINFALL_DIR = '/Users/jlee/Desktop/github/AS_rainfall/raw_data/rainfall'
RAINFALL_ADDED_DIR = '/Users/jlee/Desktop/github/AS_rainfall/raw_data/rainfall_added'
LOG_FILE = '/Users/jlee/Desktop/github/AS_rainfall/rainfall_comparison.log'

def setup_logging():
    """Set up logging to file."""
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return open(LOG_FILE, 'w')

def parse_date(date_str):
    """Parse a date string and return it in a consistent format (MM/DD/YYYY)."""
    try:
        # Handle different date formats (with/without leading zeros)
        parts = date_str.split('/')
        if len(parts) == 3:
            month = int(parts[0])
            day = int(parts[1])
            year = parts[2]
            return f"{month:02d}/{day:02d}/{year}"
    except:
        pass
    return date_str  # Return as-is if parsing fails

def load_rainfall_data(file_path):
    """Load rainfall data from a CSV file with date normalization."""
    try:
        df = pd.read_csv(file_path, header=0, names=['index', 'datetime', 'precip_in'])
        df = df[['datetime', 'precip_in']]
        # Normalize date format
        df['datetime'] = df['datetime'].apply(parse_date)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def log_message(log_file, message, print_console=True):
    """Log a message to file and optionally print to console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    log_file.write(log_entry)
    if print_console:
        print(message)

def compare_files():
    """Compare files between the two directories based on datetime."""
    log_file = setup_logging()
    
    try:
        log_message(log_file, "Starting rainfall data comparison", print_console=True)
        
        # Find all CSV files in both directories
        original_files = set(f.name for f in Path(RAINFALL_DIR).glob('*.csv'))
        added_files = set(f.name for f in Path(RAINFALL_ADDED_DIR).glob('*.csv'))
        
        # Find files only in original and only in added
        only_in_original = original_files - added_files
        only_in_added = added_files - original_files
        
        # Log files only in original
        if only_in_original:
            log_message(log_file, "\nFiles only in original directory:")
            for f in sorted(only_in_original):
                log_message(log_file, f"  - {f}")
        
        # Log files only in added
        if only_in_added:
            log_message(log_file, "\nFiles only in added directory:")
            for f in sorted(only_in_added):
                log_message(log_file, f"  - {f}")
        
        # Find common files
        common_files = original_files.intersection(added_files)
        log_message(log_file, f"\nFound {len(common_files)} common files to compare\n")

        for filename in common_files:
            log_message(log_file, f"\nComparing {filename}:", print_console=True)
            
            try:
                # Read original file
                orig_path = os.path.join(RAINFALL_DIR, filename)
                df_orig = load_rainfall_data(orig_path)
                
                # Read new file
                added_path = os.path.join(RAINFALL_ADDED_DIR, filename)
                df_added = load_rainfall_data(added_path)
                
                # Find common datetimes (dates are already normalized in load_rainfall_data)
                common_dates = set(df_orig['datetime']).intersection(set(df_added['datetime']))
                
                # Log the count of common dates
                log_message(log_file, f"  Found {len(common_dates)} common dates between files", print_console=True)
                
                if not common_dates:
                    log_message(log_file, "  No common dates found between files", print_console=True)
                    continue
                
                # Filter dataframes to only include common dates
                df_orig_common = df_orig[df_orig['datetime'].isin(common_dates)]
                df_added_common = df_added[df_added['datetime'].isin(common_dates)]
                
                # Merge dataframes on datetime for comparison
                merged = pd.merge(
                    df_orig_common, 
                    df_added_common, 
                    on='datetime', 
                    suffixes=('_orig', '_added')
                )
                
                # Find rows with different values and calculate factor
                merged['factor'] = merged['precip_in_orig'].astype(float) / merged['precip_in_added'].astype(float)
                # Only keep rows where factor is significantly different from 1.00 (more than 0.5% difference)
                diff_rows = merged[abs(merged['factor'] - 1) > 0.005]  # 0.5% threshold for significant difference
                
                if len(diff_rows) == 0:
                    log_message(log_file, "  All common dates have matching values", print_console=True)
                else:
                    log_message(log_file, f"  Found {len(diff_rows)} dates with different values", print_console=True)
                    
                    # Log detailed differences
                    log_message(log_file, "  First 10 differences (if any):", print_console=True)
                    for _, row in diff_rows.head(10).iterrows():
                        log_message(
                            log_file,
                            f"    {row['datetime']}: original={row['precip_in_orig']}, added={row['precip_in_added']}, factor={row['factor']:.2f}x",
                            print_console=True
                        )
                    
                    # Save full differences to log
                    log_message(log_file, "\n  Full difference summary:", print_console=False)
                    log_message(log_file, "  Date,Original,Added,Factor", print_console=False)
                    for _, row in diff_rows.iterrows():
                        log_message(
                            log_file,
                            f"  {row['datetime']},{row['precip_in_orig']},{row['precip_in_added']},{row['factor']:.2f}x",
                            print_console=False
                        )
                
                # Check for dates only in original
                orig_only = set(df_orig['datetime']) - set(df_added['datetime'])
                if orig_only:
                    log_message(log_file, f"  Found {len(orig_only)} dates only in original file", print_console=True)
                
                # Check for dates only in added
                added_only = set(df_added['datetime']) - set(df_orig['datetime'])
                if added_only:
                    log_message(log_file, f"  Found {len(added_only)} dates only in added file", print_console=True)
                
            except Exception as e:
                log_message(log_file, f"  Error processing {filename}: {str(e)}", print_console=True)
        
        log_message(log_file, "\nComparison complete!", print_console=True)
        
    finally:
        log_file.close()
        print(f"\nDetailed comparison log saved to: {LOG_FILE}")

if __name__ == "__main__":
    compare_files()
