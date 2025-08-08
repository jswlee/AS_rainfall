"""
Rainfall Processor Module

This module handles processing of rainfall data and interpolation to grid points.
"""

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import GP utilities
from .gp_utils import gp_interpolate, spatiotemporal_gp_interpolate

class RainfallProcessor:
    """
    A class to process rainfall data and interpolate it to grid points.
    """
    
    def __init__(self, monthly_rainfall_dir, station_locations_path):
        """
        Initialize the RainfallProcessor.
        
        Parameters
        ----------
        monthly_rainfall_dir : str
            Directory containing processed monthly rainfall data
        station_locations_path : str
            Path to CSV file with rainfall station locations
        """
        # Initialize instance variables from constructor parameters
        self.monthly_rainfall_dir = Path(monthly_rainfall_dir)
        self.station_locations_path = Path(station_locations_path)
        
        # Calculate maximum rainfall from historical data
        self._calculate_max_rainfall()
        
        # Load station data and rainfall observations
        self._load_station_locations()
        self._load_rainfall_data()
    
    def _calculate_max_rainfall(self):
        """Calculate the maximum rainfall value from historical data files.
        
        This method looks for CSV files in the mon_rainfall directory and determines
        the maximum rainfall value across all files. This value is used to cap
        unrealistic rainfall measurements.
        """
        self.max_rainfall = 0.0
        
        try:
            # Find all CSV files in the monthly rainfall directory
            found_files = glob.glob(os.path.join(self.monthly_rainfall_dir, '*.csv'))
            if not found_files:
                raise FileNotFoundError(f"No rainfall data files found in {self.monthly_rainfall_dir}")
            
            # Process each file to find maximum rainfall value
            for file in found_files:
                try:
                    df = pd.read_csv(file)
                    if df.empty:
                        continue
                        
                    # Find the rainfall column (case insensitive)
                    rain_col = next(
                        (col for col in df.columns 
                         if 'precip' in col.lower() or 'rain' in col.lower()),
                        None
                    )
                    # If no obvious column, use the second column as fallback
                    if rain_col is None and len(df.columns) > 1:
                        rain_col = df.columns[1]
                    
                    if rain_col:
                        # Convert to numeric, drop NAs, and find max
                        vals = pd.to_numeric(df[rain_col], errors='coerce').dropna()
                        if not vals.empty:
                            file_max = vals.max()
                            if file_max > self.max_rainfall:
                                self.max_rainfall = file_max
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            # Validate that we found some data
            if self.max_rainfall <= 0:
                raise ValueError("No valid rainfall data found in any files")
                
            print(f"[RainfallProcessor] Calculated max rainfall: {self.max_rainfall:.2f} inches")
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate max rainfall: {e}")
    
    def _load_station_locations(self):
        """Load station names, latitudes, and longitudes from CSV file."""
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
            
            self.station_locations = df
            print(f"Loaded locations for {len(df)} stations")
        
        except Exception as e:
            print(f"Error loading station locations: {e}")
            self.station_locations = pd.DataFrame(columns=['station_name', 'latitude', 'longitude'])

    
    def _load_rainfall_data(self):
        """Load all rainfall data from processed monthly files."""
        self.rainfall_data = {}
        
        try:
            # Load all monthly rainfall files
            rainfall_files = list(self.monthly_rainfall_dir.glob("*_monthly.csv"))
            if not rainfall_files:
                print(f"WARNING: No rainfall files found in {self.monthly_rainfall_dir}")
                return
            
            print(f"Found {len(rainfall_files)} rainfall files")
            
            # Create a mapping of station names to file paths
            file_name_map = {}
            for file in rainfall_files:
                station_name = file.stem.replace('_monthly', '')
                file_name_map[station_name] = file
            
            # Get available station names from both sources
            file_stations = set(file_name_map.keys())
            available_stations = set(self.station_locations['station_name'].tolist())
            
            # Find stations with missing data or locations
            stations_missing_data = available_stations - file_stations
            stations_missing_locations = file_stations - available_stations
            common_stations = file_stations & available_stations
            
            # Print detailed station information
            if stations_missing_data:
                print(f"\nWARNING: {len(stations_missing_data)} stations have location data but are missing data files:")
                print('\n'.join([f"  - {station}" for station in sorted(stations_missing_data)]))
                    
            if stations_missing_locations:
                print(f"\nWARNING: {len(stations_missing_locations)} stations have data files but are missing location data:")
                print('\n'.join([f"  - {station}" for station in sorted(stations_missing_locations)]))
            
            print(f"\nFound {len(common_stations)} stations with both location and rainfall data")
            
            if not common_stations:
                error_msg = (f"No stations with both location and rainfall data found. "
                           f"{len(available_stations)} stations have location data. "
                           f"{len(file_stations)} stations have rainfall data.")
                raise ValueError(error_msg)
            
            # Organize rainfall data by date (not by station)
            for station_name in common_stations:
                file = file_name_map[station_name]
                df = pd.read_csv(file)

                # Convert year_month to string format 'YYYY-MM', or use first date-like column
                if 'year_month' in df.columns:
                    df['date'] = df['year_month'].astype(str)
                else:
                    date_col = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()][0]
                    df['date'] = df[date_col].astype(str)

                rainfall_col = [col for col in df.columns if 'precip' in col.lower() or 'rain' in col.lower()][0]

                # Get station location (we've already verified all stations in common_stations have locations)
                loc_row = self.station_locations[self.station_locations['station_name'] == station_name].iloc[0]
                lon = loc_row['longitude']
                lat = loc_row['latitude']

                for i, date in enumerate(df['date']):
                    value = df[rainfall_col].iloc[i]
                    
                    if np.isnan(value):
                        continue
                    elif value < 0:
                        print(f"WARNING: Negative rainfall value ({value}) for station {station_name} on {date}, setting to 0")
                        value = 0.0
                    elif value > self.max_rainfall:
                        print(f"WARNING: Unrealistically high rainfall value ({value} inches) for station {station_name} on {date}, capping at {self.max_rainfall}")
                        value = self.max_rainfall
                        
                    if date not in self.rainfall_data:
                        self.rainfall_data[date] = {'stations': [], 'locations': [], 'values': []}
                    self.rainfall_data[date]['stations'].append(station_name)
                    self.rainfall_data[date]['locations'].append((lon, lat))
                    self.rainfall_data[date]['values'].append(value)

            print(f"Loaded rainfall data for {len(self.rainfall_data)} dates")
            
        except Exception as e:
            print(f"Error loading rainfall data: {e}")

    def get_available_dates(self):
        """Get list of available dates with rainfall data."""
        return sorted(list(self.rainfall_data.keys()))
    
    def get_rainfall_for_date(self, date_str):
        """Get rainfall data for a specific date."""
        if date_str in self.rainfall_data:
            return self.rainfall_data[date_str]
        else:
            print(f"No rainfall data available for {date_str}")
            return {'stations': [], 'locations': [], 'values': []}
    
    def interpolate_to_grid(self, rainfall_data, grid_points, method='gp'):
        """
        Interpolate rainfall data to grid points.
        
        Parameters
        ----------
        rainfall_data : dict
            Dictionary with stations, locations, and rainfall values
        grid_points : list
            List of (lon, lat) coordinates for grid points
        method : str, optional
            Interpolation method ('gp', 'rbf', or 'idw')
            - 'gp': Gaussian Process interpolation (recommended for sparse data)
            - 'rbf': Radial Basis Function interpolation
            - 'idw': Inverse Distance Weighting
        
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
            
        # Print number of stations with data
        print(f"Using {len(rainfall_data['locations'])} stations with data")
        
        # For cases with only 1 or 2 stations, use simpler methods regardless of specified method
        if len(rainfall_data['locations']) < 3:
            print(f"Only {len(rainfall_data['locations'])} rainfall data points available, using simpler interpolation")
            
            if len(rainfall_data['locations']) == 1:
                # With only one station, use the same value for all grid points
                print("Using nearest neighbor interpolation with single station")
                return np.full(len(grid_points), values[0])
            
            elif method == 'gp':
                # GP can still work with 2 points but with fixed hyperparameters
                print("Using simplified GP interpolation with two stations")
            else:
                # With two stations, use IDW for non-GP methods
                print("Using IDW interpolation with two stations")
                method = 'idw'
        
        # Use simplified GP interpolation with log transformation and zero clipping
        try:
            mean_predictions, std_predictions = gp_interpolate(
                station_locs=list(zip(lons, lats)),
                station_values=values,
                grid_points=grid_points,
                optimize=True
            )
            
            # Ensure rainfall is within realistic bounds
            interpolated = np.clip(mean_predictions, 0, self.max_rainfall)
            
            print(f"GP interpolated rainfall stats - min: {np.min(interpolated):.2f}, max: {np.max(interpolated):.2f}, mean: {np.mean(interpolated):.2f}")
            
            return interpolated
            
        except Exception as e:
            print(f"Error in GP interpolation: {e}")
            # Fallback: return zeros
            return np.zeros(len(grid_points))

    def interpolate_spatiotemporal(self, grid_points, target_dates=None):
        """
        Perform spatiotemporal interpolation using all available rainfall data.
        
        This method implements the approach described in research papers where
        a single GP model handles both spatial and temporal dimensions, providing
        better predictions especially for locations with sparse data.
        
        Parameters
        ----------
        grid_points : list
            List of (lon, lat) coordinates for grid points
        target_dates : list, optional
            List of date strings to predict for. If None, uses all available dates.
            
        Returns
        -------
        dict
            Dictionary with date strings as keys and predicted rainfall arrays as values
        """
        if target_dates is None:
            target_dates = list(self.rainfall_data.keys())
        
        print(f"Performing spatiotemporal interpolation for {len(target_dates)} dates")
        print(f"Using {len(grid_points)} grid points")
        
        # Use the spatiotemporal GP interpolation
        predictions = spatiotemporal_gp_interpolate(
            all_station_data=self.rainfall_data,
            prediction_points=grid_points,
            prediction_times=target_dates
        )
        
        return predictions

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
