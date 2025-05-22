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
                 climate_data_path=None, rainfall_data=None, output_dir='output',
                 figures_dir=None):
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
        figures_dir : str, optional
            Directory to save figure visualizations
        """
        self.grid_points = grid_points
        self.local_patches = local_patches
        self.regional_patches = regional_patches
        self.climate_data_path = climate_data_path
        self.rainfall_data = rainfall_data
        self.output_dir = Path(output_dir)
        
        # Set up figures directory
        if figures_dir is None:
            self.figures_dir = self.output_dir / 'figures'
        else:
            self.figures_dir = Path(figures_dir)
        
        # Create output and figures directories
        self.output_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True, parents=True)
        
        # Load climate data if path is provided
        if climate_data_path:
            self._load_climate_data()
    
    def _load_climate_data(self):
        """Load climate data from NetCDF file and interpolate to match grid points."""
        try:
            print(f"Loading climate data from: {self.climate_data_path}")
            print(f"File exists: {os.path.exists(self.climate_data_path)}")
            
            # Load the dataset with more explicit error handling
            self.climate_ds = xr.open_dataset(self.climate_data_path)
            
            # Print basic information for debugging
            print(f"Dataset dimensions: {self.climate_ds.dims}")
            print(f"Dataset coordinates: {list(self.climate_ds.coords)}")
            print(f"Climate data grid - Lats: {self.climate_ds.lat.values}")
            print(f"Climate data grid - Lons: {self.climate_ds.lon.values}")
            
            # Get data variables
            self.climate_vars = list(self.climate_ds.data_vars)
            print(f"Loaded climate data with variables: {self.climate_vars}")
            
            # Check if 'time' is in coordinates
            if 'time' not in self.climate_ds.coords:
                print("WARNING: 'time' coordinate not found in dataset!")
                print(f"Available coordinates: {list(self.climate_ds.coords)}")
                # Try to find a suitable time-like coordinate
                time_candidates = [coord for coord in self.climate_ds.coords 
                                 if any(t in coord.lower() for t in ['time', 'date', 'month', 'year'])]
                if time_candidates:
                    time_coord = time_candidates[0]
                    print(f"Using '{time_coord}' as time coordinate instead")
                    self.climate_times = self.climate_ds[time_coord].values
                else:
                    raise ValueError("No suitable time coordinate found in climate data")
            else:
                # Get time values
                self.climate_times = self.climate_ds.time.values
                print(f"Found {len(self.climate_times)} time points in climate data")
                
            # Note: We'll interpolate climate data to grid points when extracting data for each date
            
            # Convert to string format 'YYYY-MM'
            self.available_dates = []
            for t in self.climate_times:
                # Handle different time formats
                try:
                    # For numpy datetime64 objects
                    date_str = str(t)[:7]  # Extract YYYY-MM part
                    self.available_dates.append(date_str)
                except Exception as e:
                    print(f"Warning: Error converting time value {t}: {e}")
                    # For other formats, try a more general approach
                    date_str = str(t)
                    if len(date_str) >= 7:
                        self.available_dates.append(date_str)
            
            print(f"Extracted {len(self.available_dates)} date strings from climate data")
            if self.available_dates:
                print(f"Sample dates: {self.available_dates[:5]}...")
            
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
        
        # Print debugging information
        if not hasattr(self, '_debug_interpolation_printed'):
            print(f"Interpolating from climate grid to DEM grid:")
            print(f"  Climate grid shape: {data.shape}")
            print(f"  Climate grid lons: {lons}")
            print(f"  Climate grid lats: {lats}")
            print(f"  Target grid points: {len(grid_points)} points")
            print(f"  Sample target points: {grid_points[:3]}")
            self._debug_interpolation_printed = True
        
        # Handle longitude format differences (0-360 vs -180 to 180)
        points = []
        for p in grid_points:
            lon, lat = p
            # Convert longitude to 0-360 format if needed
            if lon < 0:
                lon_360 = lon + 360
            else:
                lon_360 = lon
            points.append((lat, lon_360))
        points = np.array(points)
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (lats, lons),
            data,
            bounds_error=False,
            fill_value=None
        )
        
        # Interpolate
        try:
            interpolated = interpolator(points)
            return interpolated
        except Exception as e:
            print(f"Error during interpolation: {e}")
            print(f"Points shape: {points.shape}")
            print(f"Points min/max: {np.min(points, axis=0)}, {np.max(points, axis=0)}")
            print(f"Grid bounds - Lons: {np.min(lons)}-{np.max(lons)}, Lats: {np.min(lats)}-{np.max(lats)}")
            # Return zeros as fallback
            return np.zeros(len(grid_points))
    
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
        # Create figure with subplots (3x2 grid for 5 visualizations)
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Plot all local DEM patches in a 5x5 grid
        local_patches = np.array(data['local_patches'])
        local_grid = self._arrange_patches_in_grid(local_patches, 5, 5)
        im1 = axes[0, 0].imshow(local_grid, cmap='terrain', origin='lower')
        axes[0, 0].set_title('Local DEM Patches (5x5 grid, 6km each)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Plot all regional DEM patches in a 5x5 grid
        regional_patches = np.array(data['regional_patches'])
        regional_grid = self._arrange_patches_in_grid(regional_patches, 5, 5)
        im2 = axes[0, 1].imshow(regional_grid, cmap='terrain', origin='lower')
        axes[0, 1].set_title('Regional DEM Patches (5x5 grid, 24km each)')
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
        
        # 5. Plot a sample climate variable (air_2m is usually available)
        climate_vars = data['climate_vars']
        if 'air_2m' in climate_vars:
            # Use air_2m as it's a commonly available variable
            var_name = 'air_2m'
        else:
            # Otherwise use the first available climate variable
            var_name = list(climate_vars.keys())[0] if climate_vars else None
            
        if var_name:
            var_data = climate_vars[var_name]
            # Reshape to grid (5x5)
            grid_size = int(np.sqrt(len(var_data)))
            grid_data = var_data.reshape(grid_size, grid_size)
            
            im4 = axes[2, 0].imshow(grid_data, cmap='viridis')
            axes[2, 0].set_title(f'Climate Variable: {var_name}')
            plt.colorbar(im4, ax=axes[2, 0])
        else:
            axes[2, 0].text(0.5, 0.5, 'No climate data available', 
                           horizontalalignment='center', verticalalignment='center')
            axes[2, 0].set_title('Climate Variable')
            
        # Hide the last unused subplot
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        
        if output_path is None:
            # Save to figures directory with date in filename
            output_path = self.figures_dir / f"sample_visualization_{date_str}.png"
            
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {output_path}")
            
    def _arrange_patches_in_grid(self, patches, grid_rows, grid_cols):
        """
        Arrange multiple patches into a single grid visualization.
        
        Parameters
        ----------
        patches : numpy.ndarray
            Array of patches, shape (n_patches, patch_height, patch_width)
        grid_rows : int
            Number of rows in the grid
        grid_cols : int
            Number of columns in the grid
            
        Returns
        -------
        numpy.ndarray
            Combined grid of all patches
        """
        # Get patch dimensions
        n_patches, patch_height, patch_width = patches.shape
        
        # Create empty grid
        grid = np.zeros((grid_rows * patch_height, grid_cols * patch_width))
        
        # Place patches in grid
        for i in range(min(n_patches, grid_rows * grid_cols)):
            row = i // grid_cols
            col = i % grid_cols
            
            # Calculate position in the grid
            row_start = row * patch_height
            row_end = (row + 1) * patch_height
            col_start = col * patch_width
            col_end = (col + 1) * patch_width
            
            # Place the patch
            grid[row_start:row_end, col_start:col_end] = patches[i]
            
        return grid
    
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
