#!/usr/bin/env python3
"""
Compare NPZ files after aligning by station-date keys.
This handles different ordering of station-date pairs.
"""

import numpy as np
import sys

# File paths
optimized_path = "/Users/jlee/Desktop/github/AS_rainfall/ML_Data_Preprocessing/output/reanalysis_npz/reanalysis_features_all_standardized_monthly.npz"
monthly_path = "/Users/jlee/Desktop/github/AS_rainfall/ML_Data_Preprocessing/output/reanalysis_npz_monthly_1/reanalysis_features_all_standardized.npz"

print("=" * 80)
print("ALIGNED NPZ FILE COMPARISON")
print("=" * 80)
print(f"\nOptimized: {optimized_path}")
print(f"Monthly:   {monthly_path}")
print()

# Load both files
print("Loading files...")
opt = np.load(optimized_path, allow_pickle=True)
mon = np.load(monthly_path, allow_pickle=True)
print("✓ Files loaded successfully\n")

# Create station-date keys
opt_keys = [(s, y, m) for s, y, m in zip(opt['stations'], opt['years'], opt['months'])]
mon_keys = [(s, y, m) for s, y, m in zip(mon['stations'], mon['years'], mon['months'])]

print(f"Number of station-date pairs:")
print(f"  Optimized: {len(opt_keys)}")
print(f"  Monthly:   {len(mon_keys)}")

# Check if same pairs exist
opt_set = set(opt_keys)
mon_set = set(mon_keys)

if opt_set != mon_set:
    print("\n❌ ERROR: Files contain different station-date pairs")
    print(f"  Only in optimized: {len(opt_set - mon_set)} pairs")
    print(f"  Only in monthly:   {len(mon_set - opt_set)} pairs")
    sys.exit(1)

print("✓ Same station-date pairs exist\n")

# Create mapping from monthly to optimized
print("Creating alignment mapping...")
mon_to_opt = {}
for opt_idx, key in enumerate(opt_keys):
    mon_to_opt[key] = opt_idx

# Reorder monthly data to match optimized
print("Reordering monthly data to match optimized order...")
mon_patches_aligned = np.zeros_like(opt['patches'])

for mon_idx, key in enumerate(mon_keys):
    opt_idx = mon_to_opt[key]
    mon_patches_aligned[opt_idx] = mon['patches'][mon_idx]

print("✓ Data aligned\n")

# Now compare patch values
print("=" * 80)
print("PATCH VALUES COMPARISON (ALIGNED)")
print("=" * 80)

opt_patches = opt['patches']
opt_vars = opt['variables']

# Check if arrays are identical
if np.array_equal(opt_patches, mon_patches_aligned):
    print("\n✅ IDENTICAL: All patch values are exactly the same!")
    print("\nThe optimized script produces IDENTICAL results to the monthly script.")
    print("The only difference was the ordering of station-date pairs.")
else:
    print("\n❌ DIFFERENT: Patch values differ even after alignment")
    
    # Calculate differences
    diff = opt_patches - mon_patches_aligned
    abs_diff = np.abs(diff)
    
    print(f"\nDifference statistics:")
    print(f"  Mean absolute difference: {np.mean(abs_diff):.6e}")
    print(f"  Max absolute difference:  {np.max(abs_diff):.6e}")
    print(f"  Min absolute difference:  {np.min(abs_diff):.6e}")
    print(f"  Std of differences:       {np.std(diff):.6e}")
    
    # Count how many values differ
    num_different = np.sum(~np.isclose(opt_patches, mon_patches_aligned, rtol=1e-5, atol=1e-8))
    total_values = opt_patches.size
    pct_different = 100 * num_different / total_values
    print(f"  Number of differing values: {num_different:,} / {total_values:,} ({pct_different:.2f}%)")
    
    # Check if differences are numerical precision only
    if np.allclose(opt_patches, mon_patches_aligned, rtol=1e-5, atol=1e-8):
        print("\n✓ Values are identical within numerical precision (rtol=1e-5, atol=1e-8)")
    else:
        print("\n❌ Differences exceed numerical precision thresholds")
        
        # Find locations with largest differences
        print(f"\nLocations with largest differences:")
        flat_idx = np.argsort(abs_diff.flatten())[-10:][::-1]  # Top 10 differences
        
        for rank, idx in enumerate(flat_idx, 1):
            n, v, h, w = np.unravel_index(idx, opt_patches.shape)
            station, year, month = opt_keys[n]
            var_name = opt_vars[v]
            opt_val = opt_patches[n, v, h, w]
            mon_val = mon_patches_aligned[n, v, h, w]
            diff_val = opt_val - mon_val
            
            print(f"\n  #{rank}: Difference = {diff_val:.6f}")
            print(f"      Location: station={station}, {year}-{month:02d}, var={var_name}, patch[{h},{w}]")
            print(f"      Optimized: {opt_val:.6f}")
            print(f"      Monthly:   {mon_val:.6f}")
        
        # Analyze differences by variable
        print(f"\nDifferences by variable:")
        for v, var_name in enumerate(opt_vars):
            var_diff = abs_diff[:, v, :, :]
            mean_diff = np.mean(var_diff)
            max_diff = np.max(var_diff)
            num_diff = np.sum(~np.isclose(opt_patches[:, v, :, :], mon_patches_aligned[:, v, :, :], rtol=1e-5, atol=1e-8))
            total = opt_patches[:, v, :, :].size
            pct = 100 * num_diff / total
            
            if num_diff > 0:
                print(f"  {var_name:25s}: mean_diff={mean_diff:.6e}, max_diff={max_diff:.6e}, "
                      f"{num_diff:5d}/{total:5d} differ ({pct:5.2f}%)")
        
        # Analyze differences by spatial location
        print(f"\nSpatial pattern of differences:")
        for h in range(opt_patches.shape[2]):
            for w in range(opt_patches.shape[3]):
                loc_diff = abs_diff[:, :, h, w]
                mean_diff = np.mean(loc_diff)
                max_diff = np.max(loc_diff)
                num_diff = np.sum(~np.isclose(opt_patches[:, :, h, w], mon_patches_aligned[:, :, h, w], rtol=1e-5, atol=1e-8))
                total = opt_patches[:, :, h, w].size
                pct = 100 * num_diff / total
                
                if num_diff > 0:
                    print(f"  Patch[{h},{w}]: mean_diff={mean_diff:.6e}, max_diff={max_diff:.6e}, "
                          f"{num_diff:5d}/{total:5d} differ ({pct:5.2f}%)")

# Sample comparison - show a few examples
print("\n" + "=" * 80)
print("SAMPLE COMPARISONS")
print("=" * 80)

# Pick a few random indices
np.random.seed(42)
sample_indices = np.random.choice(len(opt_keys), min(5, len(opt_keys)), replace=False)

for idx in sample_indices:
    station, year, month = opt_keys[idx]
    print(f"\nStation: {station}, Date: {year}-{month:02d}")
    print(f"Variable: {opt_vars[0]} (first variable)")
    print(f"Optimized patch:\n{opt_patches[idx, 0, :, :]}")
    print(f"Monthly patch:\n{mon_patches_aligned[idx, 0, :, :]}")
    if np.array_equal(opt_patches[idx, 0, :, :], mon_patches_aligned[idx, 0, :, :]):
        print("✓ IDENTICAL")
    else:
        diff = opt_patches[idx, 0, :, :] - mon_patches_aligned[idx, 0, :, :]
        print(f"Difference:\n{diff}")
        print(f"Max diff: {np.max(np.abs(diff)):.6e}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
