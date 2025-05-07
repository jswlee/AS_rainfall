"""
Data Operations Module

This module provides functions for processing climate data.
"""

import numpy as np
import xarray as xr

def interpolate_to_target_grid(data_array, target_lats, target_lons):
    """
    Interpolate a DataArray to the target grid.
    
    Parameters
    ----------
    data_array : xarray.DataArray
        DataArray to interpolate
    target_lats : numpy.ndarray
        Target latitude grid for interpolation
    target_lons : numpy.ndarray
        Target longitude grid for interpolation
    
    Returns
    -------
    xarray.DataArray
        Interpolated DataArray
    """
    return data_array.interp(
        lat=target_lats,
        lon=target_lons,
        method="linear"
    )

def compute_difference(dataset, var_key, level1, level2, lon_slice, lat_slice):
    """
    Compute the difference between two pressure levels for a variable.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing the variable
    var_key : str
        Variable name in the dataset
    level1 : int
        First pressure level
    level2 : int
        Second pressure level
    lon_slice : slice
        Longitude slice for data selection
    lat_slice : slice
        Latitude slice for data selection
    
    Returns
    -------
    xarray.DataArray
        Difference between the two levels
    """
    data1 = dataset.sel(level=level1, lon=lon_slice, lat=lat_slice)[var_key]
    data2 = dataset.sel(level=level2, lon=lon_slice, lat=lat_slice)[var_key]
    return data1 - data2

def compute_product(data1, data2):
    """
    Compute the product of two DataArrays.
    
    Parameters
    ----------
    data1 : xarray.DataArray
        First DataArray
    data2 : xarray.DataArray
        Second DataArray
    
    Returns
    -------
    xarray.DataArray
        Product of the two DataArrays
    """
    return data1 * data2

def select_data(dataset, var_key, level=None, lon_slice=None, lat_slice=None):
    """
    Select data from a dataset with optional level selection.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing the variable
    var_key : str
        Variable name in the dataset
    level : int, optional
        Pressure level for selection
    lon_slice : slice, optional
        Longitude slice for data selection
    lat_slice : slice, optional
        Latitude slice for data selection
    
    Returns
    -------
    xarray.DataArray
        Selected data
    """
    selection_kwargs = {}
    if level is not None:
        selection_kwargs['level'] = level
    if lon_slice is not None:
        selection_kwargs['lon'] = lon_slice
    if lat_slice is not None:
        selection_kwargs['lat'] = lat_slice
    
    return dataset.sel(**selection_kwargs)[var_key]

def validate_grid_consistency(climate_data):
    """
    Check that all variables are on the same grid.
    
    Parameters
    ----------
    climate_data : dict
        Dictionary of climate data variables
    
    Returns
    -------
    bool
        True if all variables have consistent lat/lon grids, False otherwise
    """
    if not climate_data:
        print("No climate data to validate")
        return False
    
    # Get the first variable's coordinates
    first_var = list(climate_data.values())[0]
    ref_lats = first_var.lat.values
    ref_lons = first_var.lon.values
    
    # Check all other variables
    for var_name, data in climate_data.items():
        if not np.array_equal(data.lat.values, ref_lats):
            print(f"Latitude mismatch for {var_name}")
            return False
        if not np.array_equal(data.lon.values, ref_lons):
            print(f"Longitude mismatch for {var_name}")
            return False
    
    print("All variables have consistent lat/lon grids")
    return True

def save_to_netcdf(climate_data, output_path):
    """
    Save the processed climate data to a NetCDF file.
    
    Parameters
    ----------
    climate_data : dict
        Dictionary of climate data variables
    output_path : str
        Path to save the NetCDF file
    """
    # Create a dataset from the climate data dictionary with compat='override' to handle conflicting coordinates
    try:
        # First try to create the dataset normally
        ds = xr.Dataset(climate_data)
    except xr.structure.merge.MergeError as e:
        print(f"Warning: {e}")
        print("Attempting to create dataset with compat='override'...")
        
        # Create an empty dataset
        ds = xr.Dataset()
        
        # Add each variable individually
        for var_name, data_array in climate_data.items():
            # Drop conflicting coordinates if needed
            if 'level' in data_array.coords and any('level' in v.coords for v in climate_data.values() if v is not data_array):
                data_array = data_array.reset_coords('level', drop=True)
            
            # Add to dataset
            ds[var_name] = data_array
    
    # Save to NetCDF
    ds.to_netcdf(output_path)
    print(f"Saved climate data to {output_path}")

def get_data_array(climate_data):
    """
    Convert the dataset into a DataArray with a 'variable' dimension.
    
    Parameters
    ----------
    climate_data : dict
        Dictionary of climate data variables
    
    Returns
    -------
    xarray.DataArray
        DataArray with dimensions (time, lat, lon, variable)
    """
    # Create a list of DataArrays with a new 'variable' dimension
    data_arrays = []
    for var_name, data in climate_data.items():
        # Add a new dimension for the variable name
        da = data.expand_dims(variable=[var_name])
        data_arrays.append(da)
    
    # Concatenate along the 'variable' dimension
    combined = xr.concat(data_arrays, dim='variable')
    
    return combined
