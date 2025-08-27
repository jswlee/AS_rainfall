"""
Extract preprocessing statistics from training data for API inference.
Run this script to generate preprocessing.json for the API.
"""
import os
import json
import numpy as np
import sys
sys.path.append('..')

def extract_preprocessing_stats():
    """Extract normalization stats from the training dataset."""
    
    # Load the training data
    npz_path = '../ML_Data_Preprocessing/output/assembled_npz/full_training_data.npz'
    print(f"Loading data from {npz_path}...")
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract the arrays we need for preprocessing
    dem_local = data['dem_local_minmax']  # Shape: (2032, 3, 3)
    dem_regional = data['dem_regional_minmax']  # Shape: (2032, 3, 3)
    climate = data['reanalysis_patches']  # Shape: (2032, 16, 3, 3)
    
    # Calculate statistics
    stats = {
        # DEM statistics (for min-max normalization)
        "dem_local_min": float(np.min(dem_local)),
        "dem_local_max": float(np.max(dem_local)),
        "dem_regional_min": float(np.min(dem_regional)),
        "dem_regional_max": float(np.max(dem_regional)),
        
        # Climate statistics (for standardization if needed)
        "climate_mean": float(np.mean(climate)),
        "climate_std": float(np.std(climate)),
        "climate_min": float(np.min(climate)),
        "climate_max": float(np.max(climate)),
        
        # Target statistics (for reference)
        "rainfall_mean": float(np.mean(data['rainfall_mm'])),
        "rainfall_std": float(np.std(data['rainfall_mm'])),
        "rainfall_min": float(np.min(data['rainfall_mm'])),
        "rainfall_max": float(np.max(data['rainfall_mm'])),
    }
    
    print("Extracted preprocessing statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save to JSON
    output_path = 'preprocessing.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nPreprocessing statistics saved to {output_path}")
    return stats

if __name__ == "__main__":
    extract_preprocessing_stats()
