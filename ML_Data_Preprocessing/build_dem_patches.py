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
        Extract a patch of DEM data around a point.
        
        Args:
            lon (float): Longitude of the center point
            lat (float): Latitude of the center point
            patch_size (int): Size of the patch (e.g., 3 for a 3x3 patch)
            km_per_cell (float): Kilometers per cell in the patch
            
        Returns:
            numpy.ndarray: Patch of DEM data with shape (patch_size, patch_size)
        """
        if self.dem_data is None:
            print("DEM data not loaded. Cannot extract patch.")
            return np.zeros((patch_size, patch_size))
        
        # Check if the point is within the DEM bounds
        if (lon < self.bounds.left or lon > self.bounds.right or 
            lat < self.bounds.bottom or lat > self.bounds.top):
            print(f"Warning: Point ({lon}, {lat}) is outside DEM bounds")
            # Return zeros for points outside bounds
            return np.zeros((patch_size, patch_size))
        
        # Convert center point to row, col
        row, col = rowcol(self.transform, lon, lat)
        
        # Convert center point to lat/lon for accurate distance calculations
        center_lon, center_lat = xy(self.transform, row, col, offset='center')
        
        # Calculate meters per pixel in x and y directions at this location
        meters_per_pixel_x = haversine(center_lon, center_lat, 
                                     center_lon + self.pixel_size_x, center_lat) * 1000
        meters_per_pixel_y = haversine(center_lon, center_lat, 
                                     center_lon, center_lat + self.pixel_size_y) * 1000
        
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
        
        # Calculate patch boundaries
        half_width_x = patch_width_pixels_x // 2
        half_width_y = patch_width_pixels_y // 2
        
        row_start = max(0, row - half_width_y)
        row_end = min(self.height, row + half_width_y + 1)
        col_start = max(0, col - half_width_x)
        col_end = min(self.width, col + half_width_x + 1)
        
        # Check if we have a valid patch size
        if row_end <= row_start or col_end <= col_start:
            print(f"Warning: Invalid patch boundaries")
            return np.zeros((patch_size, patch_size))
        
        # Extract patch
        patch = self.dem_data[row_start:row_end, col_start:col_end].copy()
        
        # Clean the DEM data (handle extreme values)
        patch = self._clean_dem_data(patch)
        
        # Check if patch is too small
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            # Pad the patch to at least patch_size
            from numpy.lib.pad import pad
            pad_width = (
                (max(0, (patch_size - patch.shape[0]) // 2), max(0, (patch_size - patch.shape[0] + 1) // 2)),
                (max(0, (patch_size - patch.shape[1]) // 2), max(0, (patch_size - patch.shape[1] + 1) // 2))
            )
            patch = pad(patch, pad_width, mode='edge')
        
        # Resize to the exact patch_size using interpolation
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            zoom_factor = (patch_size / patch.shape[0], patch_size / patch.shape[1])
            patch = zoom(patch, zoom_factor, order=1)
        
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
        Build DEM patches for all stations.
        
        Args:
            station_metadata (dict, optional): Dictionary of station metadata.
                
        Returns:
            dict: Dictionary with station names as keys and DEM patches as values
        """
        if station_metadata is None:
            # Load station metadata
            station_metadata = get_station_metadata(config.STATION_METADATA_PATH)
        
        if not station_metadata:
            print("No station metadata available. Cannot build DEM patches.")
            return {}
        
        # Get patch configurations from config
        local_config = config.DEM_PATCH_CONFIG['local']
        regional_config = config.DEM_PATCH_CONFIG['regional']
        
        # Build patches for each station
        dem_patches = {}
        for station_name, metadata in station_metadata.items():
            print(f"Building DEM patches for station: {station_name}")
            
            # Extract local patch
            local_patch = self.extract_patch(
                metadata['longitude'], 
                metadata['latitude'],
                local_config['patch_size'],
                local_config['km_per_cell']
            )
            
            # Extract regional patch
            regional_patch = self.extract_patch(
                metadata['longitude'], 
                metadata['latitude'],
                regional_config['patch_size'],
                regional_config['km_per_cell']
            )
            
            # Store patches
            dem_patches[station_name] = {
                'local': local_patch,
                'regional': regional_patch
            }
        
        return dem_patches

    


    def export_all_dem_npz_standardized(self, dem_patches, station_months_map, out_dir,
                                        filename: str = "dem_patches_all_standardized.npz"):
        """
        Export DEM patches aligned to all station-year-month combos, with DEM min–max
        standardized globally per patch type to [0,1]. Since DEM is static per station,
        we first compute global min/max across raw station patches, scale each station's
        local/regional patch, then replicate per (year, month).

        Saves:
          - dem_local_minmax:    (N, H_l, W_l)
          - dem_regional_minmax: (N, H_r, W_r)
          - dem_local_divstd:    (N, H_l, W_l)  # raw / global_std
          - dem_regional_divstd: (N, H_r, W_r)  # raw / global_std
          - stations, years, months: (N,)
          - dem_local_min, dem_local_max, dem_regional_min, dem_regional_max: floats
          - dem_local_std, dem_regional_std: floats
          - local_patch_size, regional_patch_size: ints
        """
        os.makedirs(out_dir, exist_ok=True)
        if not dem_patches:
            print("No DEM patches provided.")
            return None

        # Compute global min/max and std over raw per-station patches
        local_vals = []
        regional_vals = []
        for p in dem_patches.values():
            local_vals.append(np.asarray(p['local']).ravel())
            regional_vals.append(np.asarray(p['regional']).ravel())
        local_vals = np.concatenate(local_vals)
        regional_vals = np.concatenate(regional_vals)

        lmin, lmax = np.nanmin(local_vals), np.nanmax(local_vals)
        rmin, rmax = np.nanmin(regional_vals), np.nanmax(regional_vals)
        lstd = np.nanstd(local_vals)
        rstd = np.nanstd(regional_vals)
        lden = (lmax - lmin) if (lmax - lmin) != 0 else 1.0
        rden = (rmax - rmin) if (rmax - rmin) != 0 else 1.0
        lstd_safe = lstd if lstd not in (0.0, np.float32(0.0)) else 1.0 # To avoid division by zero
        rstd_safe = rstd if rstd not in (0.0, np.float32(0.0)) else 1.0 # To avoid division by zero

        # Build arrays by replicating standardized patches
        entries_local = []
        entries_regional = []
        entries_local_divstd = []
        entries_regional_divstd = []
        stations = []
        years = []
        months = []
        for station_name, pairs in station_months_map.items():
            if station_name not in dem_patches:
                continue
            raw_local = np.asarray(dem_patches[station_name]['local'], dtype=np.float32)
            raw_reg = np.asarray(dem_patches[station_name]['regional'], dtype=np.float32)
            loc_scaled = (raw_local - lmin) / lden
            reg_scaled = (raw_reg - rmin) / rden
            loc_divstd = raw_local / lstd_safe
            reg_divstd = raw_reg / rstd_safe
            for (y, m) in pairs:
                entries_local.append(loc_scaled[np.newaxis, ...])
                entries_regional.append(reg_scaled[np.newaxis, ...])
                entries_local_divstd.append(loc_divstd[np.newaxis, ...])
                entries_regional_divstd.append(reg_divstd[np.newaxis, ...])
                stations.append(station_name)
                years.append(int(y))
                months.append(int(m))

        if len(entries_local) == 0:
            print("No standardized DEM entries to export.")
            return None

        local_arr = np.concatenate(entries_local, axis=0).astype(np.float32)
        regional_arr = np.concatenate(entries_regional, axis=0).astype(np.float32)
        local_arr_divstd = np.concatenate(entries_local_divstd, axis=0).astype(np.float32)
        regional_arr_divstd = np.concatenate(entries_regional_divstd, axis=0).astype(np.float32)
        stations_arr = np.array(stations, dtype=object)
        years_arr = np.array(years, dtype=np.int32)
        months_arr = np.array(months, dtype=np.int32)

        save_path = os.path.join(out_dir, filename)
        np.savez_compressed(
            save_path,
            dem_local_minmax=local_arr,
            dem_regional_minmax=regional_arr,
            dem_local_divstd=local_arr_divstd,
            dem_regional_divstd=regional_arr_divstd,
            stations=stations_arr,
            years=years_arr,
            months=months_arr,
            dem_local_min=np.array(lmin, dtype=np.float32),
            dem_local_max=np.array(lmax, dtype=np.float32),
            dem_regional_min=np.array(rmin, dtype=np.float32),
            dem_regional_max=np.array(rmax, dtype=np.float32),
            dem_local_std=np.array(lstd, dtype=np.float32),
            dem_regional_std=np.array(rstd, dtype=np.float32),
            local_patch_size=np.array(config.DEM_PATCH_CONFIG['local']['patch_size']),
            regional_patch_size=np.array(config.DEM_PATCH_CONFIG['regional']['patch_size'])
        )
        return save_path

    def visualize_patches(self, dem_patches, station_name, output_dir=None):
        """
        Visualize DEM patches for a station.
        
        Args:
            dem_patches (dict): Dictionary with station names as keys and DEM patches as values
            station_name (str): Name of the station to visualize
            output_dir (str, optional): Directory to save visualizations
        """
        if station_name not in dem_patches:
            print(f"Station '{station_name}' not found in DEM patches.")
            return
            
        patches = [dem_patches[station_name]['local'], dem_patches[station_name]['regional']]
        titles = [f"{station_name} - Local DEM", f"{station_name} - Regional DEM"]
            
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{station_name}_dem_patches.png")
        
        # Use the shared grid visualizer for consistent behavior
        visualize_grid(patches, titles=titles, cmap='terrain', suptitle=None, save_path=save_path)


def main():
    """
    Main function to demonstrate the module's functionality.
    """
    print("Building DEM patches...")
        
    # Initialize DEM patch builder
    dem_builder = DEMPatchBuilder()
        
    # Load station metadata
    station_metadata = get_station_metadata(config.STATION_METADATA_PATH)
        
    # Build patches for all stations
    dem_patches = dem_builder.build_patches_for_stations()
    if not dem_patches:
        print("No DEM patches built. Exiting.")
        return

    # Discover station-months consistent with rainfall availability
    station_months_map = discover_station_months(station_metadata)

    # Export standardized DEM NPZ (global min–max) for each station-month combo
    out_dir = os.path.join(str(config.OUTPUT_DIR), "dem_npz")
    save_path = dem_builder.export_all_dem_npz_standardized(
        dem_patches, station_months_map, out_dir,
        filename="dem_patches_all_standardized.npz"
    )
    if save_path:
        print(f"Saved standardized DEM NPZ to {save_path}")

    # Visualize patches for a sample station using saved min–max arrays
    try:
        if save_path:
            npz = np.load(save_path, allow_pickle=True)
            stations = npz['stations']
            dem_local_minmax = npz['dem_local_minmax']
            dem_regional_minmax = npz['dem_regional_minmax']
            if len(stations) > 0:
                idx = 0
                sample_station = str(stations[idx])
                sample_patches = {
                    sample_station: {
                        'local': dem_local_minmax[idx],
                        'regional': dem_regional_minmax[idx],
                    }
                }
                output_dir = os.path.join(config.OUTPUT_DIR, 'figures')
                dem_builder.visualize_patches(sample_patches, sample_station, output_dir)
    except Exception as e:
        print(f"Visualization skipped due to error: {e}")

    return None


if __name__ == "__main__":
    main()
