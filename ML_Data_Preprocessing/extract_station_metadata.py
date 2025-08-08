"""
Station Metadata Extraction Module

This module handles the extraction and cleaning of station metadata from the station locations CSV file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from . import config


def get_station_metadata(file_path):
    """
    Load, clean, and return station metadata as a dictionary keyed by station_name.

    - Loads CSV from file_path
    - Renames ['Station','LAT','LONG'] -> ['station_name','latitude','longitude']
    - Drops rows with missing key fields
    - Casts latitude/longitude to float
    - Deduplicates station_name keeping first occurrence

    Args:
        file_path (str): Path to the station metadata CSV file.

    Returns:
        dict: { station_name: { 'latitude': float, 'longitude': float, ...extra columns } }
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded station metadata from {file_path}")
        print(f"Found {len(df)} stations")
    except Exception as e:
        print(f"Error loading station metadata: {e}")
        return {}

    required_columns = ['Station', 'LAT', 'LONG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return {}

    # Standardize names
    df = df.rename(columns={'Station': 'station_name', 'LAT': 'latitude', 'LONG': 'longitude'})

    # Drop rows with missing key fields
    missing_values = df[['station_name', 'latitude', 'longitude']].isna().sum()
    if missing_values.sum() > 0:
        print(f"Warning: Found missing values in key columns:\n{missing_values}")
        df = df.dropna(subset=['station_name', 'latitude', 'longitude'])
        print(f"Dropped rows with missing values. {len(df)} stations remaining.")

    # Cast types
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)

    # Deduplicate
    dup = df['station_name'].duplicated()
    if dup.any():
        print(f"Warning: Found {dup.sum()} duplicate station names. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['station_name'])

    # Build dict
    station_dict = {}
    for _, row in df.iterrows():
        station_name = row['station_name']
        station_dict[station_name] = {
            'latitude': row['latitude'],
            'longitude': row['longitude']
        }
        for col in df.columns:
            if col not in ['station_name', 'latitude', 'longitude']:
                station_dict[station_name][col] = row[col]

    return station_dict


def main():
    """
    Main function to demonstrate the module's functionality.
    """
    print("Extracting station metadata...")
    file_path = config.STATION_METADATA_PATH
    station_dict = get_station_metadata(file_path)
    print(f"Created dictionary with {len(station_dict)} stations")
    if station_dict:
        sample_station = next(iter(station_dict))
        print(f"Sample station data for '{sample_station}': {station_dict[sample_station]}")
    return station_dict


if __name__ == "__main__":
    main()
