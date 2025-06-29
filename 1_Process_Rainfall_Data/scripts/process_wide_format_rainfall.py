#!/usr/bin/env python3
"""
Process wide-format rainfall CSV file into individual station files.

This script reads a wide-format CSV file where each column represents a day's rainfall
for multiple stations, and converts it to individual station files in the format:
,datetime,precip_in

Usage:
    python process_wide_format_rainfall.py
"""

import os
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

# Constants
INPUT_FILE = '/Users/jlee/Desktop/github/AS_rainfall/raw_data/daily_wide_4302025.csv'
OUTPUT_DIR = '/Users/jlee/Desktop/github/AS_rainfall/raw_data/rainfall_added'
MM_TO_INCH = 0.0393701  # Conversion factor from mm to inches

def ensure_output_dir():
    """Ensure the output directory exists."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def process_wide_format_csv():
    """Process the wide-format CSV file into individual station files."""
    print(f"Reading input file: {INPUT_FILE}")
    
    # Read the wide-format CSV file
    df = pd.read_csv(INPUT_FILE)
    
    # Create a dictionary to store station data
    stations = {}
    
    # Get the date columns (all columns after source_freq)
    date_columns = df.columns[df.columns.get_loc('source_freq') + 1:]
    
    # Process each row (station)
    for _, row in df.iterrows():
        station_name = row['station_name']
        source_unit = row['source_unit']
        
        print(f"Processing station: {station_name}")
        
        # Create a list to store the station's data
        station_data = []
        
        # Get station's start and end years
        start_yr = int(row['start_yr']) if not pd.isna(row['start_yr']) else 1900
        end_yr = int(row['end_yr']) if not pd.isna(row['end_yr']) else 2100
        
        # Process each date column
        for date_col in date_columns:
            # Parse the date to check if it's within the station's date range
            try:
                date_obj = datetime.strptime(date_col, '%m/%d/%Y')
                year = date_obj.year
                
                # Skip dates outside the station's year range
                if year < start_yr or year > end_yr:
                    continue
                    
                formatted_date = date_obj.strftime('%m/%d/%Y')
            except ValueError:
                # If date parsing fails, use the original format and include it
                formatted_date = date_col
            
            value = row[date_col]
            
            # Convert mm to inches if needed
            if not pd.isna(value) and source_unit == 'mm':
                value = value * MM_TO_INCH
            
            # Add the data point (keeping NA as NA)
            station_data.append({
                'datetime': formatted_date,
                'precip_in': 'NA' if pd.isna(value) else value
            })
        
        # Store the station's data
        stations[station_name] = station_data
    
    # Write each station's data to a separate file
    for station_name, data in stations.items():
        if not data:  # Skip stations with no data
            print(f"Skipping {station_name} - no valid data")
            continue
        
        # Create the output file path
        output_file = os.path.join(OUTPUT_DIR, f"{station_name}.csv")
        
        print(f"Writing {len(data)} records to {output_file}")
        
        # Write the data to a CSV file
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['', 'datetime', 'precip_in'])
            
            for i, record in enumerate(data, 1):
                writer.writerow([i, record['datetime'], record['precip_in']])
    
    print("Processing complete!")

if __name__ == "__main__":
    ensure_output_dir()
    process_wide_format_csv()
