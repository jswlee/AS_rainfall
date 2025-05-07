"""
Data Generator Module

This module handles the generation of training data for the rainfall prediction model,
combining DEM patches, climate variables, and rainfall data.
"""

import os
import sys
import numpy as np
import h5py
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

class DataGenerator:
    """
    Generator for rainfall prediction training data.
    
    This class combines:
    1. DEM patches (local and regional)
    2. Climate variables
    3. Month encoding
    4. Rainfall data
    
    to create a complete dataset for deep learning.
    """
    
    def __init__(self, grid_points=None, local_patches=None, regional_patches=None,
                 climate_data_path=None, rainfall_data=None, output_dir='output'):
        """
        Initialize the data generator.
        
        Parameters
        ----------
        grid_points : list, optional
            List of (lon, lat) coordinates for grid points
        local_patches : list, optional
            List of local DEM patches (3x3 grid, 12km)
        regional_patches : list, optional
            List of regional DEM patches (9x9 grid, 60km)
        climate_data_path : str, optional
            Path to NetCDF file with climate data
        rainfall_data : dict, optional
            Dictionary with interpolated rainfall data by date
        output_dir : str, optional
            Directory to save output files
        """
        self.grid_points = grid_points
        self.local_patches = local_patches
        self.regional_patches = regional_patches
        self.climate_data_path = climate_data_path
        self.rainfall_data = rainfall_data
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load climate data if path is provided
        if climate_data_path:
            self._load_climate_data()
    
    def _load_climate_data(self):
        """Load climate data from NetCDF file."""
        try:
            self.climate_ds = xr.open_dataset(self.climate_data_path)
            self.climate_vars = list(self.climate_ds.data_vars)
            print(f"Loaded climate data with variables: {self.climate_vars}")
            
            # Get time values
            self.climate_times = self.climate_ds.time.values
            
            # Convert to string format 'YYYY-MM'
            self.available_dates = []
            for t in self.climate_times:
                # Handle different time formats
                try:
                    # For numpy datetime64 objects
                    date_str = str(t)[:7]  # Extract YYYY-MM part
                    self.available_dates.append(date_str)
                except:
                    # For other formats, try a more general approach
                    date_str = str(t)
                    if len(date_str) >= 7:
                        self.available_dates.append(date_str)
            
            print(f"Climate data available for {len(self.available_dates)} dates")
            if self.available_dates:
                print(f"Sample dates: {self.available_dates[:5]}...")
        
        except Exception as e:
            print(f"Error loading climate data: {e}")
            import traceback
            traceback.print_exc()
            self.climate_ds = None
            self.climate_vars = []
            self.available_dates = []

    def _get_month_encoding(self, date_str):
        """
        Create one-hot encoding for month.
        
        Parameters
        ----------
        date_str : str
            Date string in format 'YYYY-MM'
        
        Returns
        -------
        numpy.ndarray
            One-hot encoding vector for month (12 elements)
        """
        # Parse month from date string
        month = int(date_str.split('-')[1])
        
        # Create one-hot encoding
        encoding = np.zeros(12)
        encoding[month - 1] = 1.0
        
        return encoding
    
    def _interpolate_climate_for_date(self, date_str):
        """
        Interpolate climate variables to grid points for a specific date.
        
        Parameters
        ----------
        date_str : str
            Date string in format 'YYYY-MM'
        
        Returns
        -------
        dict
            Dictionary of interpolated climate variables
        """
        if not self.climate_ds or date_str not in self.available_dates:
            print(f"Climate data not available for {date_str}")
            return {}
        
        # Find index of date in climate data
        date_idx = self.available_dates.index(date_str)
        
        # Extract climate data for this date
        climate_data = {}
        
        for var_name in self.climate_vars:
            # Extract variable data
            var_data = self.climate_ds[var_name].isel(time=date_idx).values
            
            # Interpolate to grid points
            interpolated = self._interpolate_to_grid(
                var_data,
                self.climate_ds.lon.values,
                self.climate_ds.lat.values,
                self.grid_points
            )
            
            climate_data[var_name] = interpolated
        
        return climate_data
    
    def _interpolate_to_grid(self, data, lons, lats, grid_points):
        """
        Interpolate 2D data to grid points.
        
        Parameters
        ----------
        data : numpy.ndarray
            2D data array
        lons : numpy.ndarray
            Longitude values
        lats : numpy.ndarray
            Latitude values
        grid_points : list
            List of (lon, lat) coordinates for grid points
        
        Returns
        -------
        numpy.ndarray
            Interpolated values at grid points
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (lats, lons),
            data,
            bounds_error=False,
            fill_value=None
        )
        
        # Prepare grid points for interpolation
        points = np.array([(p[1], p[0]) for p in grid_points])
        
        # Interpolate
        interpolated = interpolator(points)
        
        return interpolated
    
    def generate_data_for_date(self, date_str):
        """
        Generate complete data for a specific date.
        
        Parameters
        ----------
        date_str : str
            Date string in format 'YYYY-MM'
        
        Returns
        -------
        dict
            Dictionary with all data components
        """
        # Check if date is available
        if date_str not in self.available_dates:
            raise ValueError(f"Date {date_str} not available in climate data")
        
        if date_str not in self.rainfall_data:
            raise ValueError(f"Date {date_str} not available in rainfall data")
        
        # Get month encoding
        month_one_hot = self._get_month_encoding(date_str)
        
        # Get interpolated climate data
        climate_vars = self._interpolate_climate_for_date(date_str)
        
        # Get rainfall data
        rainfall = self.rainfall_data[date_str]
        
        # Combine all data
        data = {
            'grid_points': np.array(self.grid_points),
            'local_patches': np.array(self.local_patches),
            'regional_patches': np.array(self.regional_patches),
            'month_one_hot': month_one_hot,
            'climate_vars': climate_vars,
            'rainfall': rainfall,
            'date': date_str
        }
        
        return data
    
    def save_data(self, data_list):
        """
        Save generated data to HDF5 file.
        
        Parameters
        ----------
        data_list : list
            List of data dictionaries for different dates
        
        Returns
        -------
        str
            Path to saved HDF5 file
        """
        # Create output file path
        output_path = self.output_dir / 'rainfall_prediction_data.h5'
        
        # Save data to HDF5 file
        with h5py.File(output_path, 'w') as h5_file:
            # Add metadata
            metadata = h5_file.create_group('metadata')
            metadata.attrs['creation_date'] = np.bytes_(datetime.now().isoformat())
            metadata.attrs['num_dates'] = len(data_list)
            metadata.attrs['num_grid_points'] = len(self.grid_points)
            metadata.attrs['grid_size'] = int(np.sqrt(len(self.grid_points)))
            metadata.attrs['local_patch_size'] = self.local_patches[0].shape[0]
            metadata.attrs['regional_patch_size'] = self.regional_patches[0].shape[0]
            
            # Add data for each date
            for i, data in enumerate(data_list):
                date_str = data['date']
                group_name = f"date_{i:03d}_{date_str}"
                date_group = h5_file.create_group(group_name)
                
                # Add grid points
                date_group.create_dataset('grid_points', data=data['grid_points'])
                
                # Add DEM patches
                date_group.create_dataset('local_patches', data=data['local_patches'])
                date_group.create_dataset('regional_patches', data=data['regional_patches'])
                
                # Add month encoding
                date_group.create_dataset('month_one_hot', data=data['month_one_hot'])
                
                # Add climate variables
                climate_group = date_group.create_group('climate_vars')
                for var_name, var_data in data['climate_vars'].items():
                    climate_group.create_dataset(var_name, data=var_data)
                
                # Add rainfall
                date_group.create_dataset('rainfall', data=data['rainfall'])
        
        print(f"Data saved to {output_path}")
        return str(output_path)
    
    def visualize_sample(self, data, date_str, output_path=None):
        """
        Visualize a sample of the generated data.
        
        Parameters
        ----------
        data : dict
            Data dictionary for a specific date
        date_str : str
            Date string in format 'YYYY-MM'
        output_path : str, optional
            Path to save the visualization
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Plot local patch
        im1 = axes[0, 0].imshow(data['local_patches'][0], cmap='terrain')
        axes[0, 0].set_title('Local DEM Patch (12km)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Plot regional patch
        im2 = axes[0, 1].imshow(data['regional_patches'][0], cmap='terrain')
        axes[0, 1].set_title('Regional DEM Patch (60km)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. Plot month encoding
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_one_hot = data['month_one_hot']
        month_idx = np.argmax(month_one_hot)
        
        axes[1, 0].bar(range(len(month_one_hot)), month_one_hot)
        axes[1, 0].set_xticks(range(len(month_one_hot)))
        axes[1, 0].set_xticklabels(month_names, rotation=45)
        axes[1, 0].set_title(f'Month: {month_names[month_idx]}')
        
        # 4. Plot rainfall
        im3 = axes[1, 1].imshow(data['rainfall'].reshape(5, 5), cmap='Blues')
        axes[1, 1].set_title('Interpolated Rainfall')
        plt.colorbar(im3, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved visualization to {output_path}")
        else:
            # Save to default location
            output_path = self.output_dir / f"sample_visualization_{date_str}.png"
            plt.savefig(output_path)
            print(f"Saved visualization to {output_path}")
    
    def visualize_climate_variables(self, data, date_str, output_path=None):
        """
        Visualize climate variables for a specific date.
        
        Parameters
        ----------
        data : dict
            Data dictionary for a specific date
        date_str : str
            Date string in format 'YYYY-MM'
        output_path : str, optional
            Path to save the visualization
        """
        # Get climate variables
        climate_vars = data['climate_vars']
        
        # Determine grid layout
        n_vars = len(climate_vars)
        n_cols = 4
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
        axes = axes.flatten()
        
        # Plot each climate variable
        for i, (var_name, var_data) in enumerate(climate_vars.items()):
            if i < len(axes):
                # Reshape to grid
                grid_size = int(np.sqrt(len(var_data)))
                grid_data = var_data.reshape(grid_size, grid_size)
                
                # Plot
                im = axes[i].imshow(grid_data, cmap='viridis')
                axes[i].set_title(var_name)
                plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved climate variables visualization to {output_path}")
        else:
            # Save to default location
            output_path = self.output_dir / f"climate_vars_{date_str}.png"
            plt.savefig(output_path)
            print(f"Saved climate variables visualization to {output_path}")
