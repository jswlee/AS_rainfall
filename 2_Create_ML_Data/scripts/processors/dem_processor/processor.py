"""
DEM Processor Module

This module handles processing of Digital Elevation Model (DEM) data,
including generating grid points and extracting patches.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from math import radians, cos, sin, asin, sqrt, atan2
import matplotlib.pyplot as plt

class DEMProcessor:
    """
    Processor for Digital Elevation Model (DEM) data.
    
    This class handles:
    1. Loading DEM data from GeoTIFF files
    2. Generating evenly spaced grid points
    3. Extracting local and regional patches around grid points
    """
    
    def __init__(self, dem_path):
        """
        Initialize the DEM processor.
        
        Parameters
        ----------
        dem_path : str
            Path to the DEM GeoTIFF file
        """
        self.dem_path = dem_path
        self._load_dem()
    
    def _load_dem(self):
        """Load the DEM data from the GeoTIFF file."""
        with rasterio.open(self.dem_path) as src:
            self.dem_data = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            
            # Get pixel size in degrees
            self.pixel_size_x = abs(self.transform[0])
            self.pixel_size_y = abs(self.transform[4])
            
            # Store dimensions
            self.height, self.width = self.dem_data.shape
            
            # Calculate approximate meters per degree at the center of the DEM
            center_lon = (self.bounds.left + self.bounds.right) / 2
            center_lat = (self.bounds.bottom + self.bounds.top) / 2
            self.meters_per_degree_lon = self._haversine(center_lon, center_lat, 
                                                       center_lon + 1, center_lat)
            self.meters_per_degree_lat = self._haversine(center_lon, center_lat, 
                                                        center_lon, center_lat + 1)
        
        print(f"Loaded DEM with dimensions: {self.width}x{self.height}")
        print(f"Approximate meters per degree - Lon: {self.meters_per_degree_lon:.2f}, "
              f"Lat: {self.meters_per_degree_lat:.2f}")
    
    def _haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth specified in decimal degrees.
        """
        # Convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        
        # Radius of earth in kilometers
        r = 6371.0
        return c * r * 1000  # Convert to meters
        print(f"Bounds: {self.bounds}")
        print(f"Pixel size: {self.pixel_size_x}x{self.pixel_size_y} meters")
    
    def generate_grid_points(self, grid_size):
        """
        Generate evenly spaced grid points across the DEM.
        
        Parameters
        ----------
        grid_size : int
            Size of the grid (e.g., 5 for a 5x5 grid)
        
        Returns
        -------
        list
            List of (lon, lat) coordinates for grid points
        """
        # Calculate step sizes
        lon_step = (self.bounds.right - self.bounds.left) / (grid_size - 1)
        lat_step = (self.bounds.top - self.bounds.bottom) / (grid_size - 1)
        
        # Generate grid points
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                lon = self.bounds.left + j * lon_step
                lat = self.bounds.bottom + i * lat_step
                grid_points.append((lon, lat))
        
        return grid_points
    
    def extract_patch(self, point, patch_size, km_per_cell):
        """
        Extract a patch of DEM data around a grid point.
        Used to construct local and regional patches for rainfall prediction.
        
        Parameters
        ----------
        point : tuple
            (lon, lat) coordinates of the center point
        patch_size : int
            Size of the patch (e.g., 3 for a 3x3 patch)
        km_per_cell : float
            Kilometers per cell in the patch
        
        Returns
        -------
        numpy.ndarray
            Patch of DEM data
        """
        lon, lat = point
        
        # Debug information
        print(f"Extracting patch at ({lon}, {lat}) with size {patch_size}x{patch_size}, {km_per_cell} km per cell")
        
        # Check if the point is within the DEM bounds
        if (lon < self.bounds.left or lon > self.bounds.right or 
            lat < self.bounds.bottom or lat > self.bounds.top):
            print(f"WARNING: Point ({lon}, {lat}) is outside DEM bounds")
            # Instead of returning zeros, return a small portion of the DEM near the edge
            row = 0 if lat < self.bounds.bottom else (self.height-1 if lat > self.bounds.top else None)
            col = 0 if lon < self.bounds.left else (self.width-1 if lon > self.bounds.right else None)
            
            # Use the center of the DEM if completely outside bounds
            if row is None and col is None:
                row, col = self.height // 2, self.width // 2
            elif row is None:
                row = self.height // 2
            elif col is None:
                col = self.width // 2
                
            # Extract a small patch from the edge
            patch_half_size = min(100, self.height // 2, self.width // 2)
            row_start = max(0, row - patch_half_size)
            row_end = min(self.height, row + patch_half_size)
            col_start = max(0, col - patch_half_size)
            col_end = min(self.width, col + patch_half_size)
            
            # Extract and resize
            edge_patch = self.dem_data[row_start:row_end, col_start:col_end].copy()
            
            # Handle extreme values
            edge_patch = self._clean_dem_data(edge_patch)
            
            # Simple resize to target patch size
            from scipy.ndimage import zoom
            zoom_factor = (patch_size / edge_patch.shape[0], patch_size / edge_patch.shape[1])
            return zoom(edge_patch, zoom_factor, order=1)
        
        # Convert center point to row, col
        row, col = rowcol(self.transform, lon, lat)
        print(f"Center point converted to row={row}, col={col}")
        
        # Convert center point to lat/lon for accurate distance calculations
        center_lon, center_lat = xy(self.transform, row, col, offset='center')
        
        # Calculate meters per pixel in x and y directions at this location
        meters_per_pixel_x = (self._haversine(center_lon, center_lat, 
                                           center_lon + self.pixel_size_x, center_lat))
        meters_per_pixel_y = (self._haversine(center_lon, center_lat, 
                                           center_lon, center_lat + self.pixel_size_y))
        
        # Calculate total patch size in meters (convert km to m)
        patch_size_meters = km_per_cell * 1000
        
        # Calculate how many pixels we need for the desired physical size
        # Add 1 to ensure we have at least the requested size
        patch_width_pixels_x = int((patch_size * patch_size_meters) / meters_per_pixel_x) + 1
        patch_width_pixels_y = int((patch_size * patch_size_meters) / meters_per_pixel_y) + 1
        
        # Ensure the patch has at least patch_size pixels in each dimension
        patch_width_pixels_x = max(patch_width_pixels_x, patch_size)
        patch_width_pixels_y = max(patch_width_pixels_y, patch_size)
        
        # Ensure we have an odd number of pixels to maintain symmetry
        if patch_width_pixels_x % 2 == 0:
            patch_width_pixels_x += 1
        if patch_width_pixels_y % 2 == 0:
            patch_width_pixels_y += 1
        
        print(f"Patch width in pixels: {patch_width_pixels_x} x {patch_width_pixels_y}")
        
        # Calculate patch boundaries
        half_width_x = patch_width_pixels_x // 2
        half_width_y = patch_width_pixels_y // 2
        
        row_start = max(0, row - half_width_y)
        row_end = min(self.height, row + half_width_y)
        col_start = max(0, col - half_width_x)
        col_end = min(self.width, col + half_width_x)
        
        print(f"Patch boundaries: rows {row_start}:{row_end}, cols {col_start}:{col_end}")
        
        # Check if we have a valid patch size
        if row_end <= row_start or col_end <= col_start:
            print(f"WARNING: Invalid patch boundaries")
            return np.zeros((patch_size, patch_size))
        
        # Extract patch
        patch = self.dem_data[row_start:row_end, col_start:col_end].copy()
        print(f"Extracted patch shape: {patch.shape}")
        
        # Clean the DEM data (handle extreme values)
        patch = self._clean_dem_data(patch)
        
        # Check if patch is too small
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            print(f"WARNING: Extracted patch is smaller than requested size. Padding with edge values.")
            # Pad the patch to at least patch_size
            from numpy.lib.pad import pad
            pad_width = (
                (max(0, (patch_size - patch.shape[0]) // 2), max(0, (patch_size - patch.shape[0] + 1) // 2)),
                (max(0, (patch_size - patch.shape[1]) // 2), max(0, (patch_size - patch.shape[1] + 1) // 2))
            )
            patch = pad(patch, pad_width, mode='edge')
        
        # Resize to the exact patch_size using interpolation
        from scipy.ndimage import zoom
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            zoom_factor = (patch_size / patch.shape[0], patch_size / patch.shape[1])
            patch = zoom(patch, zoom_factor, order=1)
        
        # Final check for NaN or extreme values
        patch = self._clean_dem_data(patch)
        
        print(f"Final patch shape: {patch.shape}")
        print(f"Patch min: {np.nanmin(patch)}, max: {np.nanmax(patch)}, mean: {np.nanmean(patch)}")
        
        return patch
    
    def _clean_dem_data(self, data, nodata_values=None, impute_strategy="mean"):
        """
        Clean DEM data by handling NaN, infinite, and NoData/extreme values robustly.

        Parameters
        ----------
        data : numpy.ndarray
            DEM data to clean
        nodata_values : list or None
            List of known NoData values (e.g., [-9999, -32768, -3.4e38]). If None, uses defaults.
        impute_strategy : str
            Strategy for filling NoData/extreme values: "mean", "median", or "zero"

        Returns
        -------
        numpy.ndarray
            Cleaned DEM data
        """
        import warnings
        cleaned = data.copy()

        # 1. Identify NoData/extreme values
        if nodata_values is None:
            nodata_values = [-9999, -32768, -3.4028235e+38, -1e10]
        mask_nodata = np.isin(cleaned, nodata_values)
        # Also treat very large negative values as NoData
        mask_extreme = cleaned < -1e6
        # Combine masks
        mask = mask_nodata | mask_extreme | ~np.isfinite(cleaned)

        n_masked = np.sum(mask)
        if n_masked > 0:
            print(f"[DEM CLEAN] Found {n_masked} NoData/extreme/invalid values in DEM patch.")
        
        # 2. If all values are invalid, replace with zeros and warn
        if np.all(mask):
            warnings.warn("All DEM values are invalid/extreme. Replacing with zeros.")
            return np.zeros_like(cleaned)

        # 3. Impute missing/extreme values
        valid = cleaned[~mask]
        if impute_strategy == "mean":
            fill_value = np.mean(valid)
        elif impute_strategy == "median":
            fill_value = np.median(valid)
        elif impute_strategy == "zero":
            fill_value = 0.0
        else:
            raise ValueError(f"Unknown impute_strategy: {impute_strategy}")
        
        cleaned[mask] = fill_value
        
        # 4. Final check for any remaining invalids
        if np.any(~np.isfinite(cleaned)):
            warnings.warn("DEM patch still contains NaN or infinite values after cleaning. Setting to fill_value.")
            cleaned = np.nan_to_num(cleaned, nan=fill_value, posinf=fill_value, neginf=fill_value)
        
        return cleaned
    
    def visualize_dem(self, output_path=None):
        """
        Visualize the DEM data.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the visualization
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.dem_data, cmap='terrain')
        plt.colorbar(label='Elevation (m)')
        plt.title('Digital Elevation Model')
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved DEM visualization to {output_path}")
        else:
            plt.show()
    
    def visualize_grid_points(self, grid_points, output_path=None):
        """
        Visualize grid points on the DEM.
        
        Parameters
        ----------
        grid_points : list
            List of (lon, lat) coordinates for grid points
        output_path : str, optional
            Path to save the visualization
        """
        # Convert grid points to row, col
        rows, cols = [], []
        for lon, lat in grid_points:
            row, col = rowcol(self.transform, lon, lat)
            rows.append(row)
            cols.append(col)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(self.dem_data, cmap='terrain')
        plt.scatter(cols, rows, c='red', s=50, marker='x')
        plt.colorbar(label='Elevation (m)')
        plt.title('Grid Points on DEM')
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved grid points visualization to {output_path}")
        else:
            plt.show()
