import xarray as xr
import numpy as np
import pandas as pd

# Paths to NetCDF files
a_path = "/Users/jlee/Desktop/github/AS_rainfall/2_Create_ML_Data/output/processed_climate_data.nc"
b_path = "/Users/jlee/Desktop/github/AS_rainfall/2_Create_ML_Data/output_old/processed_climate_data.nc"

# Load datasets
da = xr.open_dataset(a_path)
db = xr.open_dataset(b_path)

# Get variable names in both files
vars_a = set(da.data_vars)
vars_b = set(db.data_vars)
common_vars = vars_a & vars_b
only_in_a = vars_a - vars_b
only_in_b = vars_b - vars_a

results = []

for var in sorted(common_vars):
    arr_a = da[var]
    arr_b = db[var]
    
    # Check shape
    if arr_a.shape != arr_b.shape:
        results.append({
            "variable": var,
            "status": "shape_mismatch",
            "shape_a": arr_a.shape,
            "shape_b": arr_b.shape,
            "mean_abs_diff": None,
            "max_abs_diff": None
        })
        continue
    
    # Compute differences (flatten to avoid shape issues)
    diff = np.abs(arr_a.values - arr_b.values)
    mean_abs_diff = np.nanmean(diff)
    max_abs_diff = np.nanmax(diff)
    results.append({
        "variable": var,
        "status": "ok",
        "shape_a": arr_a.shape,
        "shape_b": arr_b.shape,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff
    })

for var in sorted(only_in_a):
    results.append({
        "variable": var,
        "status": "only_in_a",
        "shape_a": da[var].shape,
        "shape_b": None,
        "mean_abs_diff": None,
        "max_abs_diff": None
    })
for var in sorted(only_in_b):
    results.append({
        "variable": var,
        "status": "only_in_b",
        "shape_a": None,
        "shape_b": db[var].shape,
        "mean_abs_diff": None,
        "max_abs_diff": None
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("compare_netcdf_results.csv", index=False)

print("\nComparison complete. Results saved to compare_netcdf_results.csv.")
