# Daily Data Processing Guide

This guide explains how to process daily-scale climate data in addition to the existing monthly-scale processing.

## Overview

The pipeline now supports both monthly and daily climate data processing:
- **Monthly data**: Existing workflow using `raw_data/climate_variables_monthly/`
- **Daily data**: New workflow using `raw_data/climate_variables_daily_processed/`

## Step 1: Concatenate Daily Files

First, concatenate the individual daily files into consolidated NetCDF files:

```bash
cd ML_Data_Preprocessing
python concatenate_daily_data.py
```

This script will:
1. Read files from `raw_data/climate_variables_daily/` folders
2. Concatenate them by variable following the naming convention
3. Save consolidated files to `raw_data/climate_variables_daily_processed/`

**Expected output files:**
- `hgt.day.mean.nc` (from `gh_1980-1984/` folder)
- `slp.day.mean.nc` (from `mslp_1980-1984/` folder)  
- `air.2m.day.mean.nc` (from `t2m_1980-1984/` folder)
- `air.day.mean.nc` (from `temp_1980-1984/` folder)
- `omega.day.mean.nc` (from `omg1980-1984/` folder)
- `pr_wtr.eatm.day.mean.nc` (from `pwat_1980-1984/` folder)
- `shum.day.mean.nc` (from `shum_1980-1984/` folder)

## Step 2: Process Daily Features

Process daily climate features using the updated pipeline:

```bash
# Process daily features
python build_reanalysis_features.py daily

# Or process monthly features (default)
python build_reanalysis_features.py monthly
```

## Step 3: Use in Your Code

### For Daily Processing:
```python
from ML_Data_Preprocessing.build_reanalysis_features import ReanalysisFeatureBuilder
import ML_Data_Preprocessing.config as config

# Create daily feature builder
feature_builder = ReanalysisFeatureBuilder(time_interval="daily")

# Or explicitly specify the directory
feature_builder = ReanalysisFeatureBuilder(reanalysis_dir=config.REANALYSIS_DIR_DAILY)
```

### For Monthly Processing (unchanged):
```python
# Default behavior (monthly)
feature_builder = ReanalysisFeatureBuilder()

# Or explicitly specify monthly
feature_builder = ReanalysisFeatureBuilder(time_interval="monthly")
```

## Data Volume Considerations

Daily data will significantly increase your dataset size:
- **Monthly data**: ~12 samples per station per year
- **Daily data**: ~365 samples per station per year (~30x increase)
- **Actual increase**: Even larger due to filtering out months with missing rainfall days

## Configuration Changes

The configuration now supports both scales:

```python
# New configuration variables
REANALYSIS_DIR_MONTHLY = "raw_data/climate_variables_monthly"
REANALYSIS_DIR_DAILY = "raw_data/climate_variables_daily_processed"

# Backward compatibility
REANALYSIS_DIR = REANALYSIS_DIR_MONTHLY  # defaults to monthly
```

## Output Structure

Daily processing creates separate output directories:
- Monthly: `ML_Data_Preprocessing/output/reanalysis_npz_monthly/`
- Daily: `ML_Data_Preprocessing/output/reanalysis_npz_daily/`

## Variable Mapping

The daily folders are mapped to existing variables as follows:

| Daily Folder | Variable Name | Output File |
|--------------|---------------|-------------|
| `gh_1980-1984` | Geopotential Height | `hgt.day.mean.nc` |
| `mslp_1980-1984` | Sea Level Pressure | `slp.day.mean.nc` |
| `t2m_1980-1984` | Air 2m | `air.2m.day.mean.nc` |
| `temp_1980-1984` | Air | `air.day.mean.nc` |
| `omg1980-1984` | Omega | `omega.day.mean.nc` |
| `pwat_1980-1984` | Precipitable Water | `pr_wtr.eatm.day.mean.nc` |
| `shum_1980-1984` | Specific Humidity | `shum.day.mean.nc` |

## Next Steps

After processing daily features, you can:
1. Update your training pipeline to handle the increased data volume
2. Modify your model architecture if needed for daily predictions
3. Consider temporal modeling approaches (e.g., LSTM, attention) for daily sequences
4. Update your evaluation metrics to work with daily predictions
