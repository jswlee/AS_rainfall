"""
Rainfall Processor Module

This module handles processing of rainfall data and interpolation to grid points.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

class RainfallProcessor:
    """
    A class to process rainfall data and interpolate it to grid points.
    """
    
    def __init__(self, rainfall_dir, station_locations_path):
        """
        Initialize the RainfallProcessor.
        
        Parameters
        ----------
        rainfall_dir : str
            Directory containing processed monthly rainfall data
        station_locations_path : str
            Path to CSV file with rainfall station locations
        """
        self.rainfall_dir = Path(rainfall_dir)
        self.station_locations_path = Path(station_locations_path)
        self._load_station_locations()
        self._load_rainfall_data()
    
    def _load_station_locations(self):
        """Load rainfall station locations from CSV file."""
        try:
            # Load station locations
            df = pd.read_csv(self.station_locations_path)
            
            # Check for expected columns and rename if needed
            if all(col in df.columns for col in ['Station', 'LAT', 'LONG']):
                # Rename columns to standardized names
                column_mapping = {
                    'Station': 'station_name',
                    'LAT': 'latitude',
                    'LONG': 'longitude'
                }
                df = df.rename(columns=column_mapping)
                print(f"Mapped columns from {list(column_mapping.keys())} to {list(column_mapping.values())}")
            
            # Ensure required columns exist
            required_cols = ['station_name', 'latitude', 'longitude']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Station locations file must contain columns: {required_cols}")
            
            # Create a mapping of normalized station names to actual station names
            # This will help match station names from different sources
            df['normalized_name'] = df['station_name'].apply(self._normalize_station_name)
            
            self.station_locations = df
            print(f"Loaded locations for {len(df)} stations")
            
            # Create a mapping from normalized names to original names
            self.station_name_map = dict(zip(df['normalized_name'], df['station_name']))
            print(f"Created mapping for {len(self.station_name_map)} normalized station names")
        
        except Exception as e:
            print(f"Error loading station locations: {e}")
            self.station_locations = pd.DataFrame(columns=['station_name', 'latitude', 'longitude', 'normalized_name'])
            self.station_name_map = {}
    
    def _normalize_station_name(self, name):
        """Normalize station name for better matching."""
        if not isinstance(name, str):
            return ""
        
        # Convert to lowercase, remove special characters and spaces
        normalized = name.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        # Remove common suffixes that might differ between datasets
        for suffix in ['_uh', '_wrcc', '_usgs', '_monthly']:
            normalized = normalized.replace(suffix, '')
        
        return normalized
    
    def _load_rainfall_data(self):
        """Load all rainfall data from processed monthly files."""
        self.rainfall_data = {}
        station_data = {}
        
        try:
            # Load all monthly rainfall files
            rainfall_files = list(self.rainfall_dir.glob("*_monthly.csv"))
            if not rainfall_files:
                print(f"WARNING: No rainfall files found in {self.rainfall_dir}")
                return
            
            print(f"Found {len(rainfall_files)} rainfall files")
            
            # Create a mapping of normalized file names to actual file names
            file_name_map = {}
            for file in rainfall_files:
                station_name = file.stem.replace('_monthly', '')
                normalized_name = self._normalize_station_name(station_name)
                file_name_map[normalized_name] = file
            
            # Print the available normalized file names for debugging
            print(f"Available normalized file names: {sorted(list(file_name_map.keys()))}")
            
            # Print the available normalized station names from the station locations
            available_normalized_stations = self.station_locations['normalized_name'].tolist()
            print(f"Available normalized station names: {sorted(available_normalized_stations)}")
            
            # Find the intersection of available files and stations
            common_stations = set(file_name_map.keys()).intersection(set(available_normalized_stations))
            print(f"Found {len(common_stations)} stations with both location and rainfall data")
            
            if not common_stations:
                print("WARNING: No stations with both location and rainfall data found")
                print("This will result in zero rainfall values in the dataset")
                return
            
            # Process only files that have matching station locations
            for normalized_name in common_stations:
                file = file_name_map[normalized_name]
                station_name = self.station_name_map.get(normalized_name, file.stem.replace('_monthly', ''))
                
                df = pd.read_csv(file)
                # Convert year_month to string format 'YYYY-MM'
                if 'year_month' in df.columns:
                    # Ensure year_month is in the correct format
                    df['date'] = df['year_month'].astype(str)
                else:
                    # If using a different column name, adjust accordingly
                    date_col = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()][0]
                    df['date'] = df[date_col].astype(str)
                
                # Get the rainfall column
                rainfall_col = [col for col in df.columns if 'precip' in col.lower() or 'rain' in col.lower()][0]
                
                # Store data by station
                station_data[station_name] = {
                    'dates': df['date'].tolist(),
                    'rainfall': df[rainfall_col].tolist(),
                    'normalized_name': normalized_name
                }
                
                print(f"Loaded rainfall data for station {station_name} ({len(df)} records)")
            
            # Organize data by date
            all_dates = set()
            for station in station_data.values():
                all_dates.update(station['dates'])
            
            for date in sorted(all_dates):
                self.rainfall_data[date] = {
                    'stations': [],
                    'locations': [],
                    'values': []
                }
            
            # Populate data for each date
            for station_name, data in station_data.items():
                # Get station location
                normalized_name = data['normalized_name']
                station_loc = self.station_locations[
                    self.station_locations['normalized_name'] == normalized_name
                ]
                
                if len(station_loc) == 0:
                    print(f"Warning: No location found for station {station_name} (normalized: {normalized_name})")
                    continue
                
                lon = station_loc['longitude'].values[0]
                lat = station_loc['latitude'].values[0]
                
                for i, date in enumerate(data['dates']):
                    if date in self.rainfall_data:
                        # Check if rainfall value is valid
                        rainfall = data['rainfall'][i]
                        if pd.notna(rainfall) and rainfall >= 0:
                            # The data is already in inches, so we'll keep it that way
                            # and not perform any conversion
                            
                            self.rainfall_data[date]['stations'].append(station_name)
                            self.rainfall_data[date]['locations'].append((lon, lat))
                            self.rainfall_data[date]['values'].append(rainfall)
            
            # Count stations with data
            stations_with_data = set()
            dates_with_data = 0
            for date, date_data in self.rainfall_data.items():
                stations_with_data.update(date_data['stations'])
                if date_data['stations']:
                    dates_with_data += 1
            
            print(f"Loaded rainfall data for {len(stations_with_data)} stations across {dates_with_data} dates")
            print(f"Available stations in rainfall data: {sorted(list(stations_with_data))}")
            
            # Check if we have any dates with rainfall data
            if dates_with_data == 0:
                print("WARNING: No dates with rainfall data found")
                print("This will result in zero rainfall values in the dataset")
        
        except Exception as e:
            print(f"Error loading rainfall data: {e}")
            import traceback
            traceback.print_exc()

    def get_available_dates(self):
        """
        Get list of available dates with rainfall data.
        
        Returns
        -------
        list
            List of date strings in format 'YYYY-MM'
        """
        return sorted(list(self.rainfall_data.keys()))
    
    def get_rainfall_for_date(self, date_str):
        """
        Get rainfall data for a specific date.
        
        Parameters
        ----------
        date_str : str
            Date string in format 'YYYY-MM'
        
        Returns
        -------
        dict
            Dictionary with stations, locations, and rainfall values
        """
        if date_str in self.rainfall_data:
            return self.rainfall_data[date_str]
        else:
            print(f"No rainfall data available for {date_str}")
            return {'stations': [], 'locations': [], 'values': []}
    
    def interpolate_to_grid(self, rainfall_data, grid_points, method='rbf'):
        """
        Interpolate rainfall data to grid points.
        
        Parameters
        ----------
        rainfall_data : dict
            Dictionary with stations, locations, and rainfall values
        grid_points : list
            List of (lon, lat) coordinates for grid points
        method : str, optional
            Interpolation method ('rbf' or 'idw')
        
        Returns
        -------
        numpy.ndarray
            Array of interpolated rainfall values for grid points
        """
        # Check if we have any data points
        if len(rainfall_data['locations']) == 0:
            print("No rainfall data points available for interpolation")
            # Return zeros instead of NaN to avoid issues in visualizations and models
            return np.zeros(len(grid_points))
        
        # Extract coordinates and values
        lons = [loc[0] for loc in rainfall_data['locations']]
        lats = [loc[1] for loc in rainfall_data['locations']]
        values = rainfall_data['values']
        
        # Check for NaN values in the input data
        if any(np.isnan(v) for v in values):
            print("WARNING: Input rainfall data contains NaN values. Replacing with zeros.")
            values = [0.0 if np.isnan(v) else v for v in values]
        
        # For cases with only 1 or 2 stations, use IDW regardless of specified method
        if len(rainfall_data['locations']) < 3:
            print(f"Only {len(rainfall_data['locations'])} rainfall data points available, using nearest neighbor or IDW")
            
            if len(rainfall_data['locations']) == 1:
                # With only one station, use the same value for all grid points
                # This is essentially nearest neighbor interpolation
                print("Using nearest neighbor interpolation with single station")
                return np.full(len(grid_points), values[0])
            
            else:  # 2 stations
                # With two stations, use IDW
                print("Using IDW interpolation with two stations")
                method = 'idw'
        
        if method == 'rbf':
            # Use Radial Basis Function interpolation (requires at least 3 points)
            try:
                rbf = Rbf(lons, lats, values, function='multiquadric', epsilon=2)
                
                # Interpolate to grid points
                grid_lons = [p[0] for p in grid_points]
                grid_lats = [p[1] for p in grid_points]
                
                interpolated = rbf(grid_lons, grid_lats)
                
                # Check for NaN or infinite values in the interpolation result
                if np.any(~np.isfinite(interpolated)):
                    print("WARNING: RBF interpolation produced NaN or infinite values. Falling back to IDW.")
                    method = 'idw'
                else:
                    # Ensure non-negative rainfall
                    interpolated = np.maximum(interpolated, 0)
                    return interpolated
            
            except Exception as e:
                print(f"Error in RBF interpolation: {e}, falling back to IDW")
                method = 'idw'  # Fall back to IDW if RBF fails
        
        if method == 'idw':
            # Use Inverse Distance Weighting interpolation
            interpolated = np.zeros(len(grid_points))
            
            for i, point in enumerate(grid_points):
                # Calculate distances to all stations
                distances = np.sqrt(
                    (np.array(lons) - point[0])**2 + 
                    (np.array(lats) - point[1])**2
                )
                
                # Handle zero distances (exact matches)
                zero_dist_idx = np.where(distances == 0)[0]
                if len(zero_dist_idx) > 0:
                    # Use the exact value if point matches a station
                    interpolated[i] = values[zero_dist_idx[0]]
                else:
                    # Check for very small distances to avoid division by zero
                    min_distance = 1e-10
                    distances = np.maximum(distances, min_distance)
                    
                    # Calculate weights as inverse of squared distance
                    weights = 1.0 / (distances**2)
                    
                    # Normalize weights
                    weights = weights / np.sum(weights)
                    
                    # Calculate weighted average
                    interpolated[i] = np.sum(weights * np.array(values))
            
            # Final check for NaN or infinite values
            if np.any(~np.isfinite(interpolated)):
                print("WARNING: IDW interpolation produced NaN or infinite values. Replacing with zeros.")
                interpolated = np.nan_to_num(interpolated, nan=0.0, posinf=0.0, neginf=0.0)
            
            return interpolated
        
        else:
            print(f"Unknown interpolation method: {method}, using IDW instead")
            # Recursively call with IDW method
            return self.interpolate_to_grid(rainfall_data, grid_points, method='idw')

    def visualize_rainfall(self, date_str, grid_points=None, interpolated=None, output_path=None):
        """
        Visualize rainfall data for a specific date.
        
        Parameters
        ----------
        date_str : str
            Date string in format 'YYYY-MM'
        grid_points : list, optional
            List of (lon, lat) coordinates for grid points
        interpolated : numpy.ndarray, optional
            Array of interpolated rainfall values for grid points
        output_path : str, optional
            Path to save the visualization
        """
        if date_str not in self.rainfall_data:
            print(f"No rainfall data available for {date_str}")
            return
        
        rainfall_data = self.rainfall_data[date_str]
        
        plt.figure(figsize=(10, 8))
        
        # Plot station data
        lons = [loc[0] for loc in rainfall_data['locations']]
        lats = [loc[1] for loc in rainfall_data['locations']]
        values = rainfall_data['values']
        
        scatter = plt.scatter(lons, lats, c=values, cmap='Blues', 
                             s=100, edgecolor='black', label='Stations')
        
        # Plot grid points and interpolated values if provided
        if grid_points is not None and interpolated is not None:
            grid_lons = [p[0] for p in grid_points]
            grid_lats = [p[1] for p in grid_points]
            
            plt.scatter(grid_lons, grid_lats, c=interpolated, cmap='Blues',
                       marker='s', s=50, edgecolor='red', label='Grid Points')
        
        plt.colorbar(label='Rainfall (mm)')
        plt.title(f'Rainfall for {date_str}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved rainfall visualization to {output_path}")
        else:
            plt.show()
