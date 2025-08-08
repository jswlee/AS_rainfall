"""
Utility to print a single assembled datapoint for quick inspection.

It loads:
- Reanalysis NPZ: output/reanalysis_npz/reanalysis_features_all_standardized.npz
- DEM NPZ:        output/dem_npz/dem_patches_all.npz
- Extras NPZ:     output/assembled_npz/training_extras.npz

By default, it prints the first datapoint (index 0). You can select a
specific datapoint by --index, or by a key (--station, --year, --month).

Usage examples:
  python3 -m ML_Data_Preprocessing.print_single_datapoint
  python3 -m ML_Data_Preprocessing.print_single_datapoint --index 10
  python3 -m ML_Data_Preprocessing.print_single_datapoint --station aasu_UH --year 2005 --month 7
"""

import os
import sys
import numpy as np

from . import config

def tuple_index_map(stations, years, months):
    return { (str(s), int(y), int(m)): i for i, (s, y, m) in enumerate(zip(stations.tolist(), years.tolist(), months.tolist())) }

def summarize(arr):
    arr = np.asarray(arr)
    return dict(shape=arr.shape, dtype=str(arr.dtype), min=float(np.nanmin(arr)), max=float(np.nanmax(arr)), mean=float(np.nanmean(arr)))

def main():
    # Path to combined, already-standardized dataset
    full_path = os.path.join(str(config.OUTPUT_DIR), 'assembled_npz', 'full_training_data.npz')
    if not os.path.exists(full_path):
        print(f'ERROR: Missing combined dataset: {full_path}')
        sys.exit(1)

    data = np.load(full_path, allow_pickle=True)
    idx = 0
    re_st = data['stations']
    re_yr = data['years']
    re_mo = data['months']
    key = (str(re_st[idx]), int(re_yr[idx]), int(re_mo[idx]))

    # Fields
    mo_onehot = data['month_onehot'][idx] if 'month_onehot' in data.files else None
    rain_in = data['rainfall_in'][idx] if 'rainfall_in' in data.files else None
    # DEM (already standardized via min-max in assembly)
    dem_local = None
    dem_regional = None
    if 'dem_local_minmax' in data.files:
        dem_local = data['dem_local_minmax'][idx]
    elif 'dem_local_std' in data.files:
        dem_local = data['dem_local_std'][idx]
    if 'dem_regional_minmax' in data.files:
        dem_regional = data['dem_regional_minmax'][idx]
    elif 'dem_regional_std' in data.files:
        dem_regional = data['dem_regional_std'][idx]
    # Reanalysis
    features = data['reanalysis_patches'] if 'reanalysis_patches' in data.files else None
    var_names = data['variables'].tolist() if 'variables' in data.files else None
    re_vec = features[idx] if features is not None else None

    # Print
    print('--- Datapoint ---')
    print(f'Index:   {idx}')
    print(f'Station: {key[0]}')
    print(f'Year:    {key[1]}')
    print(f'Month:   {key[2]}')
    if rain_in is not None:
        print(f'Rainfall (in): {rain_in}')
    if mo_onehot is not None:
        print(f'Month one-hot (12): {mo_onehot}')

    if dem_local is not None:
        s_local = summarize(dem_local)
        print(f'Local DEM (standardized):  {s_local}')
        print(f'Local DEM[0]: {dem_local}\n')

    if dem_regional is not None:
        s_regional = summarize(dem_regional)
        print(f'Regional DEM (standardized):  {s_regional}')
        print(f'Regional DEM[0]: {dem_regional}\n')

    if re_vec is not None:
        s = summarize(re_vec)
        print(f'Reanalysis features summary: {s}\n')
        if var_names is not None:
            print(f'Variables: {var_names}')
            # quick sample of first-cell across first 5 variables
            if re_vec.shape[0] >= 1:
                first_cells = [float(re_vec[i,0,0]) for i in range(min(5, re_vec.shape[0]))]
                print(f'Sample first-cell per variable (first 5): {first_cells}')
        print('\nReanalysis full matrix (V,H,W)=' + str(re_vec.shape) + ':')
        print(re_vec)


if __name__ == '__main__':
    main()
