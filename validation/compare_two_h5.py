import h5py
import numpy as np
import pandas as pd

file_a = "output_new/rainfall_prediction_data.h5"
file_b = "output_new_2/rainfall_prediction_data.h5"

def get_all_datasets(h5file):
    datasets = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj.shape
    h5file.visititems(visitor)
    return datasets

with h5py.File(file_a, 'r') as fa, h5py.File(file_b, 'r') as fb:
    datasets_a = get_all_datasets(fa)
    datasets_b = get_all_datasets(fb)
    keys_a = set(datasets_a.keys())
    keys_b = set(datasets_b.keys())
    common_keys = keys_a & keys_b
    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a

    results = []
    for key in sorted(common_keys):
        arr_a = fa[key][...]
        arr_b = fb[key][...]
        if arr_a.shape != arr_b.shape:
            results.append({
                "dataset": key,
                "status": "shape_mismatch",
                "shape_a": arr_a.shape,
                "shape_b": arr_b.shape,
                "mean_abs_diff": None,
                "max_abs_diff": None
            })
            continue
        diff = np.abs(arr_a - arr_b)
        mean_abs_diff = np.nanmean(diff)
        max_abs_diff = np.nanmax(diff)
        results.append({
            "dataset": key,
            "status": "ok",
            "shape_a": arr_a.shape,
            "shape_b": arr_b.shape,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff
        })
    for key in sorted(only_in_a):
        results.append({
            "dataset": key,
            "status": "only_in_a",
            "shape_a": datasets_a[key],
            "shape_b": None,
            "mean_abs_diff": None,
            "max_abs_diff": None
        })
    for key in sorted(only_in_b):
        results.append({
            "dataset": key,
            "status": "only_in_b",
            "shape_a": None,
            "shape_b": datasets_b[key],
            "mean_abs_diff": None,
            "max_abs_diff": None
        })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("compare_h5_results.csv", index=False)
print("\nComparison complete. Results saved to compare_h5_results.csv.")
