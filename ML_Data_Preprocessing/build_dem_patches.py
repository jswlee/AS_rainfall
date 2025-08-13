"""
DEM Patch Builder Module

This module handles the extraction of Digital Elevation Model (DEM) patches
at station locations for both local and regional scales.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from scipy.ndimage import zoom

from . import config
from .utils import haversine, visualize_patches, discover_station_months, visualize_grid
from .extract_station_metadata import get_station_metadata


class DEMPatchBuilder:
    """
    Class for building DEM patches at station locations.
    
    This class handles:
    1. Loading DEM data from GeoTIFF files
    2. Extracting local and regional patches at station locations
    3. Standardizing patches for ML input
    """
    
    def __init__(self, dem_path=None):
        """
        Initialize the DEM patch builder.
        
        Args:
            dem_path (str, optional): Path to the DEM GeoTIFF file.
                If None, uses the path from the config.
        """
        if dem_path is None:
            dem_path = config.DEM_PATH
        
        self.dem_path = dem_path
        self.dem_data = None
        self.transform = None
        self.crs = None
        self.bounds = None
        self.height = None
        self.width = None
        self.pixel_size_x = None
        self.pixel_size_y = None
        
        # Load the DEM data
        self._load_dem()
    
    def _load_dem(self):
        """Load the DEM data from the GeoTIFF file."""
        try:
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
                
                print(f"Successfully loaded DEM from {self.dem_path}")
                print(f"DEM shape: {self.dem_data.shape}")
                print(f"Bounds: {self.bounds}")
                print(f"Pixel size: {self.pixel_size_x}° x {self.pixel_size_y}°")
        except Exception as e:
            print(f"Error loading DEM: {e}")
    
    def extract_patch(self, lon, lat, patch_size, km_per_cell):
        """
        Extract a patch of Digital Elevation Model (DEM) data centered around a rainfall station location.
        
        This method extracts topographic context around each rainfall station by creating a square patch
        of elevation data centered on the station's coordinates. The patch captures the surrounding terrain
        that may influence rainfall patterns at the station. Two types of patches are typically extracted:
        - Local patches: Smaller area with higher resolution to capture fine topographic details
        - Regional patches: Larger area with lower resolution to capture broader topographic context
        
        These DEM patches will later be aligned with reanalysis features and rainfall data by station-year-month
        to create comprehensive training examples for machine learning models.
        
        Args:
            lon (float): Longitude of the rainfall station location
            lat (float): Latitude of the rainfall station location
            patch_size (int): Size of the output patch in grid cells (e.g., 3 for a 3x3 patch)
            km_per_cell (float): Physical size in kilometers that each output cell should represent
            
        Returns:
            numpy.ndarray: Patch of DEM elevation data with shape (patch_size, patch_size) centered
                          on the rainfall station location
        """
        if self.dem_data is None:
            print("DEM data not loaded. Cannot extract patch.")
            return np.zeros((patch_size, patch_size))
        
        # Check if the rainfall station location is within the DEM bounds
        if (lon < self.bounds.left or lon > self.bounds.right or 
            lat < self.bounds.bottom or lat > self.bounds.top):
            print(f"Warning: Rainfall station at ({lon}, {lat}) is outside DEM bounds")
            # Return zeros for stations outside bounds
            return np.zeros((patch_size, patch_size))
        
        # Convert rainfall station coordinates to DEM raster row and column indices
        row, col = rowcol(self.transform, lon, lat)
        
        # Convert row/col indices back to precise lat/lon coordinates for accurate distance calculations
        # This accounts for any rounding that occurred during the initial conversion
        center_lon, center_lat = xy(self.transform, row, col, offset='center')
        
        # Calculate physical size (meters) of each DEM pixel at this rainfall station's latitude
        # This varies with latitude due to Earth's curvature, so we calculate it specifically for this location
        meters_per_pixel_x = haversine(center_lon, center_lat, 
                                     center_lon + self.pixel_size_x, center_lat) * 1000
        meters_per_pixel_y = haversine(center_lon, center_lat, 
                                     center_lon, center_lat + self.pixel_size_y) * 1000
        
        # Convert the requested physical size (km_per_cell) to meters
        patch_size_meters = km_per_cell * 1000
        
        # Calculate how many DEM pixels we need to cover the requested physical area
        # Each output cell should represent km_per_cell kilometers, and we need patch_size of them
        # Add 1 to ensure we have at least the requested coverage
        patch_width_pixels_x = int((patch_size * patch_size_meters) / meters_per_pixel_x) + 1
        patch_width_pixels_y = int((patch_size * patch_size_meters) / meters_per_pixel_y) + 1
        
        # Ensure the patch has at least patch_size pixels in each dimension
        # This guarantees we have enough data to create the final output patch
        patch_width_pixels_x = max(patch_width_pixels_x, patch_size)
        patch_width_pixels_y = max(patch_width_pixels_y, patch_size)
        
        # Ensure we have an odd number of pixels to maintain symmetry around the rainfall station
        # This keeps the station at the exact center of the patch
        if patch_width_pixels_x % 2 == 0:
            patch_width_pixels_x += 1
        if patch_width_pixels_y % 2 == 0:
            patch_width_pixels_y += 1
        
        # Calculate the boundaries of the DEM patch centered on the rainfall station
        half_width_x = patch_width_pixels_x // 2
        half_width_y = patch_width_pixels_y // 2
        
        # Ensure patch boundaries stay within the DEM raster limits
        row_start = max(0, row - half_width_y)
        row_end = min(self.height, row + half_width_y + 1)
        col_start = max(0, col - half_width_x)
        col_end = min(self.width, col + half_width_x + 1)
        
        # Check if we have a valid patch size after boundary adjustments
        if row_end <= row_start or col_end <= col_start:
            print(f"Warning: Invalid patch boundaries for rainfall station at ({lon}, {lat})")
            return np.zeros((patch_size, patch_size))
        
        # Extract the DEM elevation data patch centered on the rainfall station
        patch = self.dem_data[row_start:row_end, col_start:col_end].copy()
        
        # Clean the DEM data by handling NoData values, NaNs, and extreme elevation values
        # This ensures the patch contains valid elevation data for machine learning
        patch = self._clean_dem_data(patch)
        
        # Check if the extracted patch is too small (can happen near DEM boundaries)
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            # Pad the patch to at least patch_size by extending edge values
            # This maintains the elevation profile while reaching the required dimensions
            from numpy.lib.pad import pad
            pad_width = (
                (max(0, (patch_size - patch.shape[0]) // 2), max(0, (patch_size - patch.shape[0] + 1) // 2)),
                (max(0, (patch_size - patch.shape[1]) // 2), max(0, (patch_size - patch.shape[1] + 1) // 2))
            )
            patch = pad(patch, pad_width, mode='edge')
        
        # Resize to the exact requested patch_size using bilinear interpolation
        # This ensures all patches have consistent dimensions regardless of source resolution
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            # zoom_factor is a tuple of scale multipliers per axis: (rows_scale, cols_scale)
            # - rows_scale = desired_rows / current_rows
            # - cols_scale = desired_cols / current_cols
            # Using per-axis factors preserves the aspect ratio of the output grid (square patch_size x patch_size)
            # even when the extracted patch is slightly rectangular due to padding/cropping at DEM edges.
            # scipy.ndimage.zoom applies these factors independently to height (axis 0) and width (axis 1).
            zoom_factor = (patch_size / patch.shape[0], patch_size / patch.shape[1])
            # Save the pre-resize patch for alternative averaging comparison
            pre_resize = patch
            # Bilinear interpolation result
            patch_interpolated = zoom(pre_resize, zoom_factor, order=1)  # order=1 is bilinear interpolation
            print(f"Interpolated DEM (shape {patch_interpolated.shape}):\n{patch_interpolated}")

            # Alternative (commented out): block-wise average downsampling via reshape
            # This computes the mean elevation within non-overlapping blocks so that each
            # output cell is the average of its corresponding input block, which can be
            # preferable if you want area-averaged values rather than interpolated ones.
            # Note: This requires that patch.shape is exactly (patch_size * bh, patch_size * bw)
            # for some integers bh and bw. If not, you'd need to pad/crop first.
            
            # Compute block-mean from the pre-resize patch using edge padding to ensure divisibility
            # We need the pre-resize array shape to be exactly divisible by patch_size in both dims
            # so it can be reshaped into (patch_size, bh, patch_size, bw) blocks. The expressions
            # (-h) % patch_size and (-w) % patch_size give the minimal non-negative padding needed
            # to reach the next multiples of patch_size without changing the center location.
            h, w = pre_resize.shape
            pad_h = (-h) % patch_size  # minimal rows to add so (h + pad_h) % patch_size == 0
            pad_w = (-w) % patch_size  # minimal cols to add so (w + pad_w) % patch_size == 0
            if pad_h or pad_w:
                # Split padding as symmetrically as possible around the center so the station
                # remains centered in the patch. Any odd remainder is placed on the bottom/right.
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                # Edge padding replicates border values, avoiding artificial gradients or zeros that
                # would bias the block averages near edges.
                patch_padded = np.pad(pre_resize, ((top, bottom), (left, right)), mode='edge')
                print(f"Applied edge padding for block-mean: top={top}, bottom={bottom}, left={left}, right={right}")
            else:
                patch_padded = pre_resize
            # Now divisible; compute block sizes
            bh = patch_padded.shape[0] // patch_size  # rows per output cell (block height)
            bw = patch_padded.shape[1] // patch_size  # cols per output cell (block width)
            # Reshape to (out_rows, bh, out_cols, bw), where each output cell corresponds to a
            # contiguous bh x bw block in the padded array; then average over the block axes (1, 3).
            patch_block = patch_padded.reshape(patch_size, bh, patch_size, bw)
            patch_avg = patch_block.mean(axis=(1, 3))
            patch_avg = self._clean_dem_data(patch_avg)
            print(f"Block-mean averaged DEM (shape {patch_avg.shape}):\n{patch_avg}")

            # Continue with the interpolated patch as the function output path
            patch = patch_interpolated
        
        # Final check for NaN or extreme values
        patch = self._clean_dem_data(patch)
        
        return patch
    
    def _clean_dem_data(self, data, nodata_values=None, impute_strategy="mean"):
        """
        Clean DEM data by handling NaN, infinite, and NoData/extreme values.
        
        Args:
            data (numpy.ndarray): DEM data to clean
            nodata_values (list, optional): List of known NoData values
            impute_strategy (str): Strategy for filling NoData values
            
        Returns:
            numpy.ndarray: Cleaned DEM data
        """
        cleaned = data.copy()
        
        # Identify NoData/extreme values
        if nodata_values is None:
            nodata_values = [-9999, -32768, -3.4028235e+38, -1e10]
        mask_nodata = np.isin(cleaned, nodata_values)
        
        # Also treat very large negative values as NoData
        mask_extreme = cleaned < -1e6
        
        # Combine masks
        mask = mask_nodata | mask_extreme | ~np.isfinite(cleaned)
        
        n_masked = np.sum(mask)

        # Optional print statement for debugging
        # if n_masked > 0:
            # print(f"Found {n_masked} NoData/extreme/invalid values in DEM patch.")
        
        # If all values are invalid, replace with zeros
        if np.all(mask):
            print("Warning: All DEM values are invalid/extreme. Replacing with zeros.")
            return np.zeros_like(cleaned)
        
        # Impute missing/extreme values
        valid = cleaned[~mask]
        if impute_strategy == "mean":
            fill_value = np.mean(valid)
        elif impute_strategy == "median":
            fill_value = np.median(valid)
        elif impute_strategy == "zero":
            fill_value = 0.0
        else:
            fill_value = 0.0
        
        cleaned[mask] = fill_value
        
        # Final check for any remaining invalids
        if np.any(~np.isfinite(cleaned)):
            print("Warning: DEM patch still contains NaN or infinite values after cleaning.")
            cleaned = np.nan_to_num(cleaned, nan=fill_value, posinf=fill_value, neginf=fill_value)
        
        return cleaned
    
    def build_patches_for_stations(self, station_metadata=None):
        """
        Build Digital Elevation Model (DEM) patches centered on all rainfall station coordinates.
        
        This method processes each rainfall station's geographic coordinates to extract topographic
        context at two different spatial scales. These DEM patches capture the terrain surrounding
        each station that may influence local rainfall patterns:
        
        - Local patches: Smaller area with finer detail immediately around the station location
          (typically 3x3 grid with 2km per cell = 6km x 6km area)
        - Regional patches: Larger area showing broader topographic context around the station
          (typically 3x3 grid with 8km per cell = 24km x 24km area)
        
        These multi-scale DEM patches will later be aligned with reanalysis features and rainfall data
        by station-year-month to create comprehensive training examples for machine learning models.
        The patches provide critical topographic context that helps models understand how elevation
        and terrain features influence rainfall patterns at each station location.
        
        Args:
            station_metadata (dict, optional): Dictionary of rainfall station metadata with format
                {station_name: {'latitude': float, 'longitude': float, ...}}
                If None, loads from the default station metadata file in config.
                
        Returns:
            dict: Dictionary with station names as keys and DEM patches as values
                 {station_name: {'local': local_patch_array, 'regional': regional_patch_array}}
        """
        if station_metadata is None:
            # Load station metadata
            station_metadata = get_station_metadata(config.STATION_METADATA_PATH)
        
        if not station_metadata:
            print("No station metadata available. Cannot build DEM patches.")
            return {}
        
        # Get patch configurations from config for both local and regional scales
        # Local: Smaller area with finer detail (e.g., 3x3 grid with 2km per cell = 6km x 6km area)
        # Regional: Larger area showing broader context (e.g., 3x3 grid with 8km per cell = 24km x 24km area)
        local_config = config.DEM_PATCH_CONFIG['local']
        regional_config = config.DEM_PATCH_CONFIG['regional']
        
        # Build DEM patches for each rainfall station
        dem_patches = {}
        for station_name, metadata in station_metadata.items():
            print(f"Building DEM patches for station: {station_name}")
            
            # Extract local-scale DEM patch centered on the rainfall station
            # This captures fine-grained elevation details in the immediate vicinity
            local_patch = self.extract_patch(
                metadata['longitude'],  # Rainfall station longitude
                metadata['latitude'],   # Rainfall station latitude
                local_config['patch_size'],  # Number of cells in output grid (e.g., 3 for 3x3)
                local_config['km_per_cell']  # Physical size each cell represents (e.g., 2km)
            )
            
            # Extract regional-scale DEM patch centered on the same rainfall station
            # This captures broader topographic context around the station
            regional_patch = self.extract_patch(
                metadata['longitude'],  # Same rainfall station longitude
                metadata['latitude'],   # Same rainfall station latitude
                regional_config['patch_size'],  # Number of cells in output grid (e.g., 3 for 3x3)
                regional_config['km_per_cell']  # Larger physical size per cell (e.g., 8km)
            )
            
            # Store both local and regional DEM patches for this rainfall station
            # These will be used later for machine learning model input features
            dem_patches[station_name] = {
                'local': local_patch,     # Fine-grained elevation details (smaller area)
                'regional': regional_patch # Broader topographic context (larger area)
            }
        
        return dem_patches

    def export_all_dem_npz_standardized(self, dem_patches, station_months_map, out_dir,
                                        filename: str = "dem_patches_all_standardized.npz"):
        """
        Export Digital Elevation Model (DEM) patches aligned to all rainfall station-year-month combinations.
        
        This method standardizes DEM data in two ways to prepare it for machine learning:
        1. Min-max scaling: Normalizes elevation values to [0,1] range based on global min/max
        2. Standard deviation scaling: Divides by global standard deviation
        
        Since topography (DEM) is static for each rainfall station location, we:
        1. First compute global statistics across all station patches
        2. Scale each station's local/regional patch using these global statistics
        3. Replicate the standardized patches for each (year, month) combination where that station has rainfall data

        Args:
            dem_patches (dict): Dictionary of DEM patches by station name {station_name: {'local': array, 'regional': array}}
            station_months_map (dict): Mapping of station names to (year, month) pairs where rainfall data exists
            out_dir (str): Directory to save the output NPZ file
            filename (str): Name of the output NPZ file
            
        Returns:
            str or None: Path to the saved NPZ file, or None if no data to export

        Saves NPZ file containing:
          - dem_local_minmax:    (N, H_l, W_l)  # Min-max normalized local patches
          - dem_regional_minmax: (N, H_r, W_r)  # Min-max normalized regional patches
          - dem_local_divstd:    (N, H_l, W_l)  # Local patches divided by global std
          - dem_regional_divstd: (N, H_r, W_r)  # Regional patches divided by global std
          - stations, years, months: (N,)        # Corresponding metadata arrays
          - dem_local_min, dem_local_max, dem_regional_min, dem_regional_max: floats  # Global statistics
          - dem_local_std, dem_regional_std: floats  # Global standard deviations
          - local_patch_size, regional_patch_size: ints  # Patch dimensions
        """
        os.makedirs(out_dir, exist_ok=True)
        if not dem_patches:
            print("No DEM patches provided.")
            return None

        # Compute global statistics across all rainfall station DEM patches for standardization
        # This ensures consistent scaling across all stations for machine learning
        local_vals = []
        regional_vals = []
        
        # Collect all elevation values from all stations' patches
        for p in dem_patches.values():
            local_vals.append(np.asarray(p['local']).ravel())  # Flatten local patches
            regional_vals.append(np.asarray(p['regional']).ravel())  # Flatten regional patches
        
        # Concatenate all values to compute global statistics
        local_vals = np.concatenate(local_vals)  # All local patch values
        regional_vals = np.concatenate(regional_vals)  # All regional patch values

        # Calculate global min/max for min-max normalization
        lmin, lmax = np.nanmin(local_vals), np.nanmax(local_vals)  # Local patch min/max
        rmin, rmax = np.nanmin(regional_vals), np.nanmax(regional_vals)  # Regional patch min/max
        
        # Calculate global standard deviations for standardization
        lstd = np.nanstd(local_vals)  # Local patch standard deviation
        rstd = np.nanstd(regional_vals)  # Regional patch standard deviation
        
        # Calculate denominators for min-max scaling, with safety checks
        lden = (lmax - lmin) if (lmax - lmin) != 0 else 1.0  # Avoid division by zero
        rden = (rmax - rmin) if (rmax - rmin) != 0 else 1.0  # Avoid division by zero
        
        # Safe standard deviations for division, avoiding division by zero
        lstd_safe = lstd if lstd not in (0.0, np.float32(0.0)) else 1.0
        rstd_safe = rstd if rstd not in (0.0, np.float32(0.0)) else 1.0

        # Build arrays by replicating standardized DEM patches for each station-year-month combination
        # Since topography is static, we create one standardized patch per station and replicate it
        # for each time period (year, month) where that station has rainfall data
        entries_local = []           # Will hold min-max normalized local patches
        entries_regional = []        # Will hold min-max normalized regional patches
        entries_local_divstd = []    # Will hold std-normalized local patches
        entries_regional_divstd = [] # Will hold std-normalized regional patches
        stations = []                # Will hold corresponding station names
        years = []                   # Will hold corresponding years
        months = []                  # Will hold corresponding months
        
        # Process each rainfall station
        for station_name, pairs in station_months_map.items():
            # Skip stations that don't have DEM patches
            if station_name not in dem_patches:
                continue
                
            # Get raw DEM patches for this rainfall station
            raw_local = np.asarray(dem_patches[station_name]['local'], dtype=np.float32)
            raw_reg = np.asarray(dem_patches[station_name]['regional'], dtype=np.float32)
            
            # Apply min-max normalization to scale elevation values to [0,1] range
            loc_scaled = (raw_local - lmin) / lden  # (value - min) / (max - min)
            reg_scaled = (raw_reg - rmin) / rden    # (value - min) / (max - min)
            
            # Apply standard deviation normalization
            loc_divstd = raw_local / lstd_safe  # value / global_std
            reg_divstd = raw_reg / rstd_safe    # value / global_std
            
            # Replicate this station's DEM patches for each year-month combination
            # where this station has rainfall data
            for (y, m) in pairs:  # Each (year, month) pair for this station
                # Add a batch dimension and append to our lists
                entries_local.append(loc_scaled[np.newaxis, ...])  # Add batch dimension
                entries_regional.append(reg_scaled[np.newaxis, ...])  # Add batch dimension
                entries_local_divstd.append(loc_divstd[np.newaxis, ...])  # Add batch dimension
                entries_regional_divstd.append(reg_divstd[np.newaxis, ...])  # Add batch dimension
                
                # Store corresponding metadata
                stations.append(station_name)
                years.append(int(y))
                months.append(int(m))

        # Check if we have any data to export
        if len(entries_local) == 0:
            print("No standardized DEM entries to export.")
            return None

        # Concatenate all patches into final numpy arrays
        # Each array will have shape (N, patch_height, patch_width) where N is the number of station-month pairs
        local_arr = np.concatenate(entries_local, axis=0).astype(np.float32)  # Min-max normalized local patches
        regional_arr = np.concatenate(entries_regional, axis=0).astype(np.float32)  # Min-max normalized regional patches
        local_arr_divstd = np.concatenate(entries_local_divstd, axis=0).astype(np.float32)  # Std-normalized local patches
        regional_arr_divstd = np.concatenate(entries_regional_divstd, axis=0).astype(np.float32)  # Std-normalized regional patches
        
        # Convert metadata lists to numpy arrays
        stations_arr = np.array(stations, dtype=object)  # Station names corresponding to each patch
        years_arr = np.array(years, dtype=np.int32)      # Years corresponding to each patch
        months_arr = np.array(months, dtype=np.int32)    # Months corresponding to each patch

        # Create the output file path
        save_path = os.path.join(out_dir, filename)
        
        # Save all arrays and metadata to a compressed NPZ file
        # This file will be used later in the ML pipeline to provide DEM features
        np.savez_compressed(
            save_path,
            # Standardized DEM patches for each station-month pair
            dem_local_minmax=local_arr,          # Local patches with min-max scaling
            dem_regional_minmax=regional_arr,    # Regional patches with min-max scaling
            dem_local_divstd=local_arr_divstd,   # Local patches divided by standard deviation
            dem_regional_divstd=regional_arr_divstd,  # Regional patches divided by standard deviation
            
            # Metadata arrays identifying each patch
            stations=stations_arr,  # Station names
            years=years_arr,        # Years
            months=months_arr,      # Months
            
            # Global statistics used for standardization (useful for inverse transforms)
            dem_local_min=np.array(lmin, dtype=np.float32),      # Global minimum for local patches
            dem_local_max=np.array(lmax, dtype=np.float32),      # Global maximum for local patches
            dem_regional_min=np.array(rmin, dtype=np.float32),   # Global minimum for regional patches
            dem_regional_max=np.array(rmax, dtype=np.float32),   # Global maximum for regional patches
            dem_local_std=np.array(lstd, dtype=np.float32),      # Global std for local patches
            dem_regional_std=np.array(rstd, dtype=np.float32),   # Global std for regional patches
            
            # Configuration parameters
            local_patch_size=np.array(config.DEM_PATCH_CONFIG['local']['patch_size']),         # Size of local patches
            regional_patch_size=np.array(config.DEM_PATCH_CONFIG['regional']['patch_size'])   # Size of regional patches
        )
        return save_path

    def visualize_patches(self, dem_patches, station_name, output_dir=None):
        """
        Visualize Digital Elevation Model (DEM) patches for a specific rainfall station.
        
        Creates a side-by-side visualization of both local and regional DEM patches
        centered on the specified rainfall station. This helps to visually inspect
        the topographic context around the station at different spatial scales.
        
        Args:
            dem_patches (dict): Dictionary with station names as keys and DEM patches as values
                                in the format {station_name: {'local': array, 'regional': array}}
            station_name (str): Name of the rainfall station to visualize
            output_dir (str, optional): Directory to save visualizations. If None, only displays
                                       the visualization without saving.
        
        Returns:
            None: Displays and/or saves the visualization
        """
        # Check if the requested rainfall station exists in our DEM patches dictionary
        if station_name not in dem_patches:
            print(f"Station '{station_name}' not found in DEM patches.")
            return
            
        # Extract both local and regional DEM patches for this rainfall station
        # Local patch: Fine-grained elevation details in immediate vicinity
        # Regional patch: Broader topographic context around the station
        patches = [dem_patches[station_name]['local'], dem_patches[station_name]['regional']]
        titles = [f"{station_name} - Local DEM", f"{station_name} - Regional DEM"]
            
        # Determine where to save the visualization if output_dir is provided
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
            save_path = os.path.join(output_dir, f"{station_name}_dem_patches.png")
        
        # Use the shared grid visualizer for consistent visualization across the project
        # The 'terrain' colormap is appropriate for elevation data
        visualize_grid(patches, titles=titles, cmap='terrain', suptitle=None, save_path=save_path)


def main():
    """
    Main function to demonstrate the module's functionality for extracting DEM patches
    around rainfall station locations.
    
    This function performs the following steps:
    1. Initializes the DEM patch builder
    2. Loads rainfall station metadata (containing coordinates)
    3. Builds local and regional DEM patches for all stations
    4. Visualizes patches for a sample station
    5. Exports standardized patches for machine learning
    """
    print("Building DEM patches around rainfall station locations...")
        
    # Initialize DEM patch builder with the configured DEM file
    dem_builder = DEMPatchBuilder()
        
    # Load rainfall station metadata containing lat/lon coordinates
    station_metadata = get_station_metadata(config.STATION_METADATA_PATH)
        
    # Build local and regional DEM patches centered on each rainfall station
    dem_patches = dem_builder.build_patches_for_stations()
    if not dem_patches:
        print("No DEM patches built. Exiting.")
        return

    # Discover which station-month combinations have available rainfall data
    # This ensures we only create DEM patches for time periods where we have rainfall labels
    station_months_map = discover_station_months(station_metadata)

    # Export standardized DEM patches for each station-month combination
    # This creates a single NPZ file with all patches standardized using global statistics
    # The resulting file will be used as input features for the machine learning model
    out_dir = os.path.join(str(config.OUTPUT_DIR), "dem_npz")
    save_path = dem_builder.export_all_dem_npz_standardized(
        dem_patches,                           # DEM patches for all stations
        station_months_map,                    # Map of station to (year, month) pairs
        out_dir,                              # Output directory
        filename="dem_patches_all_standardized.npz"  # Output filename
    )
    if save_path:
        print(f"Saved standardized DEM patches for all rainfall stations to {save_path}")

    # Visualize DEM patches for a sample rainfall station to verify the extraction worked correctly
    try:
        if save_path:
            # Load the standardized DEM patches from the saved NPZ file
            npz = np.load(save_path, allow_pickle=True)
            
            # Extract station names and corresponding DEM patches
            stations = npz['stations']  # Array of station names
            dem_local_minmax = npz['dem_local_minmax']  # Local DEM patches (min-max normalized)
            dem_regional_minmax = npz['dem_regional_minmax']  # Regional DEM patches (min-max normalized)
            
            # Create a visualization for the first station in the dataset
            if len(stations) > 0:
                idx = 0  # Use the first station as a sample
                sample_station = str(stations[idx])  # Get the station name
                
                # Create a patches dictionary in the format expected by visualize_patches
                sample_patches = {
                    sample_station: {
                        'local': dem_local_minmax[idx],      # Local patch for this station
                        'regional': dem_regional_minmax[idx],  # Regional patch for this station
                    }
                }
                
                # Save the visualization to the figures directory
                output_dir = os.path.join(config.OUTPUT_DIR, 'figures')
                dem_builder.visualize_patches(sample_patches, sample_station, output_dir)
                print(f"Visualization for station '{sample_station}' saved to {output_dir}")
    except Exception as e:
        print(f"Visualization of DEM patches skipped due to error: {e}")

    return None


if __name__ == "__main__":
    main()
