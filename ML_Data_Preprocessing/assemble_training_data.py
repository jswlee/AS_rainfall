"""
Training Data Assembler Module

This module handles the assembly of training data for the ML pipeline,
combining rainfall labels, one-hot encoded months, DEM patches, and reanalysis features.
"""

import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

from . import config
from .utils import filter_outliers

def month_one_hot(month: int) -> np.ndarray:
    """Return a 12-dim one-hot vector for month in [1..12]."""
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month: {month}")
    v = np.zeros(12, dtype=np.float32)
    v[month - 1] = 1.0
    return v


class TrainingDataAssembler:
    """
    Class for assembling training data for the ML pipeline.
    
    This class handles:
    1. Loading station rainfall data
    2. Combining with DEM patches and reanalysis features
    3. Normalizing rainfall labels
    4. Saving the assembled dataset
    """
    
    def __init__(self, time_interval='monthly', rainfall_dir=None, output_dir=None):
        """
        Initialize the training data assembler.
        
        Args:
            time_interval (str): 'monthly' or 'daily'
            rainfall_dir (str, optional): Directory containing station rainfall CSV files.
            output_dir (str, optional): Directory to save the assembled dataset.
        """
        self.time_interval = time_interval
        
        if rainfall_dir is None:
            if time_interval == 'monthly':
                rainfall_dir = config.MONTHLY_RAINFALL_DATA_DIR
            elif time_interval == 'daily':
                rainfall_dir = config.DAILY_RAINFALL_DATA_DIR
            else:
                raise ValueError(f"Unsupported time_interval: {time_interval}")
        
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        self.rainfall_dir = rainfall_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.rainfall_data = {}
        self.rainfall_mean = None
        self.rainfall_std = None
    
    def load_station_rainfall(self, station_name):
        """
        Load rainfall data for a specific station.
        
        Args:
            station_name (str): Name of the station
            
        Returns:
            pandas.DataFrame: DataFrame containing rainfall data, or None if not found
        """
        # Construct the file path
        file_path = os.path.join(self.rainfall_dir, f"{station_name}_monthly.csv")
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            print(f"Loaded rainfall data for station {station_name} from {file_path}")
            
            # Check if the required columns exist
            required_columns = ['Year', 'Month', 'Rainfall']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Error: Missing required columns for station {station_name}: {missing_columns}")
                return None
            
            return df
        except Exception as e:
            print(f"Error loading rainfall data for station {station_name}: {e}")
            return None

    def _build_rainfall_lookup(self, time_interval=None):
        """
        Builds a unified rainfall lookup from daily and monthly CSVs.

        The key format is determined by the data's granularity:
        - Daily:   (station, year, month, day) -> rainfall_inches
        - Monthly: (station, year, month) -> rainfall_inches

        It supports two CSV schemas by checking column headers:
        1. Daily:   ['datetime', 'precip_in']
        2. Monthly: ['year_month', 'monthly_total_precip_in']
        """
        if time_interval is None:
            time_interval = self.time_interval
        
        lookup = {}
        files_processed = 0
        
        # A single glob finds all potential candidate files.
        for csv_path in glob.glob(os.path.join(str(self.rainfall_dir), '*.csv')):
            try:
                basename = os.path.basename(csv_path)
                # Read CSV - daily has unnamed index, monthly has year_month as first column
                df = pd.read_csv(csv_path, na_values=['', 'NA'], keep_default_na=True)
                
                # Handle unnamed index column for daily files
                if df.columns[0].startswith('Unnamed:'):
                    df = df.set_index(df.columns[0])
                
                cols = set(df.columns)

                # --- Schema 1: Daily ---
                if time_interval == 'daily':
                    if 'datetime' not in cols or 'precip_in' not in cols:
                        continue
                    
                    station = basename.replace('.csv', '')
                    df_valid = df.dropna(subset=['datetime', 'precip_in']).copy()
                    
                    if len(df_valid) == 0:
                        continue
                    
                    df_valid['datetime'] = pd.to_datetime(df_valid['datetime'], format='%m/%d/%Y')
                    
                    # Use vectorized operations instead of iterrows() for performance
                    keys = zip(
                        [station] * len(df_valid),
                        df_valid['datetime'].dt.year,
                        df_valid['datetime'].dt.month,
                        df_valid['datetime'].dt.day
                    )
                    values = df_valid['precip_in'].astype(float)
                    lookup.update(dict(zip(keys, values)))
                    files_processed += 1

                # --- Schema 2: Monthly ---
                elif time_interval == 'monthly':
                    if 'year_month' not in cols or 'monthly_total_precip_in' not in cols:
                        continue
                    
                    station = basename.replace('_monthly.csv', '')
                    df_valid = df.dropna(subset=['year_month', 'monthly_total_precip_in']).copy()
                    
                    if len(df_valid) == 0:
                        continue
                    
                    ym = df_valid['year_month'].astype(str).str.split('-', expand=True)
                    
                    # Ensure the split operation resulted in at least two columns
                    if ym.shape[1] >= 2:
                        keys = zip(
                            [station] * len(df_valid),
                            ym[0].astype(int),
                            ym[1].astype(int)
                        )
                        values = df_valid['monthly_total_precip_in'].astype(float)
                        lookup.update(dict(zip(keys, values)))
                        files_processed += 1

            except Exception as e:
                # A single, consistent error handler
                print(f"Warning: failed to parse rainfall CSV {csv_path}: {e}")
        
        print(f"Loaded rainfall data from {files_processed} CSV files, {len(lookup)} total records")
        return lookup

    def assemble_from_precomputed(self,
                                  dem_npz_path: str = None,
                                  reanalysis_npz_path: str = None,
                                  out_dir: str = None,
                                  out_filename: str = None):
        """
        Assemble a single, ready-to-train NPZ file by combining DEM patches (extracted around rainfall station coordinates),
        reanalysis features, and rainfall data for machine learning.
        
        This method aligns three key data sources:
        1. DEM patches: Topographic elevation data extracted around each rainfall station location at two scales
           (local for fine detail and regional for broader context)
        2. Reanalysis features: Atmospheric variables from climate reanalysis data
        3. Rainfall data: Monthly rainfall measurements from station records
        
        The resulting NPZ file contains the following arrays:
        
        Metadata arrays:
          - stations, years, months: [N] - Station names, years, and months for each data point
          - month_onehot:           [N, 12] - One-hot encoded month indicators
          
        Rainfall data (labels for ML model):
          - rainfall_mm:            [N] - Monthly rainfall values (inches -> mm, min-max normalized to [0,1])
          - rainfall_mm_divstd:     [N] - Monthly rainfall values (inches -> mm, divided by global std)
          - rainfall_mm_min, rainfall_mm_max: floats - Global min/max values for rainfall (in mm)
          - rainfall_mm_std:        float - Global standard deviation for rainfall (in mm)
          
        DEM patches (extracted around rainfall station coordinates):
          - dem_local_minmax:       [N, H_l, W_l] - Local DEM patches scaled to [0,1] range
          - dem_regional_minmax:    [N, H_r, W_r] - Regional DEM patches scaled to [0,1] range
          - dem_local_divstd:       [N, H_l, W_l] - Local DEM patches divided by global std
          - dem_regional_divstd:    [N, H_r, W_r] - Regional DEM patches divided by global std
          
        DEM global statistics (for denormalization if needed):
          - dem_local_min, dem_local_max: floats - Global min/max values for local DEM patches
          - dem_regional_min, dem_regional_max: floats - Global min/max values for regional DEM patches
          - dem_local_std, dem_regional_std: floats - Global standard deviations for DEM patches
          
        Reanalysis data:
          - reanalysis_patches:     [N, V, H, W] - Standardized reanalysis patches
          - variables:              [V] - Names of reanalysis variables

        Notes:
          - The reanalysis NPZ defines the primary index. DEM patches and rainfall are aligned to it.
          - DEM patches are extracted around rainfall station coordinates using their longitude and latitude.
          - DEM min-max scaling is computed globally across all available DEM patches (separately for local and regional).
          - Each station-year-month combination has corresponding DEM patches, reanalysis features, and rainfall values.
        """
        if dem_npz_path is None:
            # Select DEM NPZ filename based on granularity to match build_dem_patches.py outputs
            dem_fname = 'dem_patches_all_standardized_monthly.npz' if self.time_interval == 'monthly' else 'dem_patches_all_standardized_daily.npz'
            dem_npz_path = os.path.join(str(config.OUTPUT_DIR), 'dem_npz', dem_fname)
        if reanalysis_npz_path is None:
            reanalysis_npz_path = os.path.join(str(config.OUTPUT_DIR), 'reanalysis_npz', 
                                               f'reanalysis_features_all_standardized_{self.time_interval}.npz')
        if out_dir is None:
            out_dir = os.path.join(str(config.OUTPUT_DIR), 'assembled_npz')
        if out_filename is None:
            out_filename = f'full_training_data_{self.time_interval}.npz'
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(reanalysis_npz_path):
            print(f"ERROR: Missing reanalysis NPZ at {reanalysis_npz_path}. Run build_reanalysis_features first.")
            return None
        if not os.path.exists(dem_npz_path):
            print(f"ERROR: Missing DEM NPZ at {dem_npz_path}. Run build_dem_patches first.")
            return None

        # Load NPZs
        re_npz = np.load(reanalysis_npz_path, allow_pickle=True)
        dem_npz = np.load(dem_npz_path, allow_pickle=True)

        re_stations = re_npz['stations']
        re_years = re_npz['years']
        re_months = re_npz['months']
        re_days = re_npz.get('days', None)

        dem_stations = dem_npz['stations']
        dem_years = dem_npz['years']
        dem_months = dem_npz['months']

        # Build index sets
        if self.time_interval == 'daily' and re_days is not None:
            re_keys = [(str(s), int(y), int(m), int(d)) for s, y, m, d in 
                      zip(re_stations.tolist(), re_years.tolist(), re_months.tolist(), re_days.tolist())]
            dem_keys = {(str(s), int(y), int(m)) for s, y, m in 
                       zip(dem_stations.tolist(), dem_years.tolist(), dem_months.tolist())}
        else:
            re_keys = [(str(s), int(y), int(m)) for s, y, m in 
                      zip(re_stations.tolist(), re_years.tolist(), re_months.tolist())]
            dem_keys = {(str(s), int(y), int(m)) for s, y, m in 
                       zip(dem_stations.tolist(), dem_years.tolist(), dem_months.tolist())}

        # Check for missing DEM data
        if self.time_interval == 'daily':
            missing_in_dem = [k for k in re_keys if (k[0], k[1], k[2]) not in dem_keys]
        else:
            missing_in_dem = [k for k in re_keys if k not in dem_keys]
        if missing_in_dem:
            print(f"Warning: {len(missing_in_dem)} reanalysis tuples missing in DEM NPZ.")

        # Month one-hot
        month_onehot = np.stack([month_one_hot(int(m)) for m in re_months.tolist()], axis=0).astype(np.float32)

        # For daily data, also compute cyclical day-of-year encoding (sin, cos)
        day_cyc = None
        if self.time_interval == 'daily' and re_days is not None:
            # Compute day-of-year for each (year, month, day)
            try:
                doy = pd.to_datetime({
                    'year': re_years.astype(int),
                    'month': re_months.astype(int),
                    'day': re_days.astype(int)
                }).dayofyear.values
            except Exception:
                # Fallback via constructing strings if needed
                date_strs = pd.Series(re_years.astype(int)).astype(str) + '-' + \
                            pd.Series(re_months.astype(int)).astype(str) + '-' + \
                            pd.Series(re_days.astype(int)).astype(str)
                doy = pd.to_datetime(date_strs).dt.dayofyear.values

            angle = 2.0 * np.pi * (doy / 365.25)
            day_sin = np.sin(angle).astype(np.float32)
            day_cos = np.cos(angle).astype(np.float32)
            day_cyc = np.stack([day_sin, day_cos], axis=1)  # [N, 2]

        # Rainfall mapping from CSVs
        rainfall_lookup = self._build_rainfall_lookup()
        
        rainfall_in = []
        missing_rain = 0
        for k in re_keys:
            if k in rainfall_lookup:
                rainfall_in.append(float(rainfall_lookup[k]))
            else:
                rainfall_in.append(np.nan)
                missing_rain += 1
        if missing_rain > 0:
            print(f"Warning: rainfall missing for {missing_rain} of {len(re_keys)} tuples.")
        rainfall_in = np.asarray(rainfall_in, dtype=np.float32)
        # Convert inches -> millimeters (raw), then compute both min–max and divstd variants
        rainfall_mm_raw = rainfall_in * 25.4
        rmin = np.nanmin(rainfall_mm_raw)
        rmax = np.nanmax(rainfall_mm_raw)
        print(f"Rainfall min: {rmin}, max: {rmax}")
        rden = (rmax - rmin) if (rmax - rmin) not in (0.0, np.float32(0.0)) else 1.0
        rstd = np.nanstd(rainfall_mm_raw)
        rstd_safe = rstd if rstd not in (0.0, np.float32(0.0)) else 1.0
        rainfall_mm = (rainfall_mm_raw - rmin) / rden
        rainfall_mm_divstd = rainfall_mm_raw / rstd_safe

        # Load and align standardized DEM patches (extracted around rainfall station locations)
        # These patches represent the topographic context around each rainfall station
        # and are important features for the machine learning model
        
        # Get the min-max normalized DEM patches (scaled to [0,1] range)
        dem_local_npz = dem_npz['dem_local_minmax']      # Local patches (finer detail, smaller area)
        dem_regional_npz = dem_npz['dem_regional_minmax']  # Regional patches (broader context, larger area)
        
        # Get the standard deviation normalized DEM patches if available
        dem_local_divstd_npz = dem_npz['dem_local_divstd'] if 'dem_local_divstd' in dem_npz.files else None
        dem_regional_divstd_npz = dem_npz['dem_regional_divstd'] if 'dem_regional_divstd' in dem_npz.files else None
        
        # Initialize lists to store aligned DEM patches
        # We need to align DEM patches with the reanalysis data index
        local_list = []           # For local min-max normalized patches
        regional_list = []        # For regional min-max normalized patches
        local_divstd_list = []    # For local std-normalized patches
        regional_divstd_list = []  # For regional std-normalized patches
        # Build DEM index mapping
        dem_idx_map = {(str(s), int(y), int(m)): i for i, (s, y, m) in 
                      enumerate(zip(dem_stations.tolist(), dem_years.tolist(), dem_months.tolist()))}
        
        for key in re_keys:
            # For daily data, DEM uses (station, year, month) key (excluding day)
            # because topography doesn't change day-to-day - we reuse the same
            # monthly DEM patch for all days within that month
            dem_key = (key[0], key[1], key[2]) if self.time_interval == 'daily' else key
            
            if dem_key in dem_idx_map:
                di = dem_idx_map[dem_key]
                local_list.append(dem_local_npz[di])
                regional_list.append(dem_regional_npz[di])
                
                if dem_local_divstd_npz is not None and dem_regional_divstd_npz is not None:
                    local_divstd_list.append(dem_local_divstd_npz[di])
                    regional_divstd_list.append(dem_regional_divstd_npz[di])
            else:
                lshape = dem_local_npz[0].shape
                rshape = dem_regional_npz[0].shape
                local_list.append(np.full(lshape, np.nan, dtype=np.float32))
                regional_list.append(np.full(rshape, np.nan, dtype=np.float32))
                
                if dem_local_divstd_npz is not None and dem_regional_divstd_npz is not None:
                    local_divstd_list.append(np.full(dem_local_divstd_npz[0].shape, np.nan, dtype=np.float32))
                    regional_divstd_list.append(np.full(dem_regional_divstd_npz[0].shape, np.nan, dtype=np.float32))
        # Convert lists of DEM patches to numpy arrays for the final dataset
        # These arrays contain the topographic context around each rainfall station
        # and will be used as input features for the machine learning model
        dem_local_minmax = np.asarray(local_list, dtype=np.float32)  # Local patches (min-max normalized)
        dem_regional_minmax = np.asarray(regional_list, dtype=np.float32)  # Regional patches (min-max normalized)
        
        # Convert std-normalized patches if available
        dem_local_divstd = np.asarray(local_divstd_list, dtype=np.float32) if local_divstd_list else None
        dem_regional_divstd = np.asarray(regional_divstd_list, dtype=np.float32) if regional_divstd_list else None

        # Extract the global statistics used for DEM standardization
        # These are needed if we want to convert back to original elevation values
        # or ensure consistent scaling across different datasets
        l_min = float(dem_npz['dem_local_min'])    # Global minimum for local patches
        l_max = float(dem_npz['dem_local_max'])    # Global maximum for local patches
        r_min = float(dem_npz['dem_regional_min'])  # Global minimum for regional patches
        r_max = float(dem_npz['dem_regional_max'])  # Global maximum for regional patches
        l_std = float(dem_npz['dem_local_std']) if 'dem_local_std' in dem_npz.files else np.float32(1.0)  # Global std for local
        r_std = float(dem_npz['dem_regional_std']) if 'dem_regional_std' in dem_npz.files else np.float32(1.0)  # Global std for regional

        # Reanalysis patches (already standardized in builder save)
        re_features = re_npz['patches'] if 'patches' in re_npz.files else None
        re_variables = re_npz['variables'] if 'variables' in re_npz.files else None

        # Save all data to a single compressed NPZ file
        # This file will contain:
        # 1. Metadata (stations, years, months)
        # 2. Month one-hot encodings
        # 3. Rainfall data (both min-max and std-normalized)
        # 4. DEM patches extracted around rainfall station coordinates (both local and regional)
        # 5. Reanalysis features
        # 6. Global statistics for denormalization
        out_path = os.path.join(out_dir, out_filename)
        save_data = {
            'stations': re_stations,
            'years': re_years,
            'months': re_months,
            'month_onehot': month_onehot,
            'rainfall_mm': rainfall_mm,
            'rainfall_mm_divstd': rainfall_mm_divstd,
            'rainfall_mm_std': np.array(rstd, dtype=np.float32),
            'rainfall_mm_min': np.array(rmin, dtype=np.float32),
            'rainfall_mm_max': np.array(rmax, dtype=np.float32),
            'dem_local_minmax': dem_local_minmax,
            'dem_regional_minmax': dem_regional_minmax,
            'dem_local_divstd': dem_local_divstd if dem_local_divstd is not None else np.array([]),
            'dem_regional_divstd': dem_regional_divstd if dem_regional_divstd is not None else np.array([]),
            'dem_local_min': np.array(l_min, dtype=np.float32),
            'dem_local_max': np.array(l_max, dtype=np.float32),
            'dem_regional_min': np.array(r_min, dtype=np.float32),
            'dem_regional_max': np.array(r_max, dtype=np.float32),
            'dem_local_std': np.array(l_std, dtype=np.float32),
            'dem_regional_std': np.array(r_std, dtype=np.float32),
            'reanalysis_patches': re_features if re_features is not None else np.array([]),
            'variables': re_variables if re_variables is not None else np.array([]),
        }
        
        if self.time_interval == 'daily' and re_days is not None:
            save_data['days'] = re_days
            if day_cyc is not None:
                save_data['day_cyc'] = day_cyc.astype(np.float32)
        
        np.savez_compressed(out_path, **save_data)
        print(f"Saved full training NPZ to {out_path}")
        return out_path
    
    def load_all_station_rainfall(self, station_metadata):
        """
        Load rainfall data for all stations.
        
        Args:
            station_metadata (dict): Dictionary of station metadata
            
        Returns:
            dict: Dictionary with station names as keys and rainfall DataFrames as values
        """
        rainfall_data = {}
        
        for station_name in station_metadata:
            df = self.load_station_rainfall(station_name)
            if df is not None:
                rainfall_data[station_name] = df
        
        print(f"Loaded rainfall data for {len(rainfall_data)} stations")
        self.rainfall_data = rainfall_data
        return rainfall_data
    
    def normalize_rainfall(self, rainfall_data, outlier_threshold=None):
        """
        Normalize rainfall data by filtering outliers and standardizing.
        
        Args:
            rainfall_data (dict): Dictionary with station rainfall data
            outlier_threshold (float, optional): Threshold for outlier detection in standard deviations.
                If None, uses the value from the config.
                
        Returns:
            dict: Dictionary with normalized rainfall data
        """
        if outlier_threshold is None:
            outlier_threshold = config.RAINFALL_OUTLIER_THRESHOLD
        
        # Collect all rainfall values
        all_rainfall = []
        for station_df in rainfall_data.values():
            all_rainfall.extend(station_df['Rainfall'].values)
        
        all_rainfall = np.array(all_rainfall)
        
        # Filter outliers
        filtered_rainfall = filter_outliers(all_rainfall, threshold=outlier_threshold)
        
        # Calculate mean and standard deviation
        rainfall_mean = np.mean(filtered_rainfall)
        rainfall_std = np.std(filtered_rainfall)
        
        print(f"Rainfall statistics - Mean: {rainfall_mean:.2f} inches, Std: {rainfall_std:.2f} inches")
        print(f"Filtered {len(all_rainfall) - len(filtered_rainfall)} outliers out of {len(all_rainfall)} values")
        
        # Store statistics for later use
        self.rainfall_mean = rainfall_mean
        self.rainfall_std = rainfall_std
        
        # Normalize rainfall data
        normalized_data = {}
        for station_name, df in rainfall_data.items():
            normalized_df = df.copy()
            normalized_df['Rainfall_Normalized'] = (df['Rainfall'] - rainfall_mean) / rainfall_std
            normalized_data[station_name] = normalized_df
        
        return normalized_data
    
    def assemble_training_examples(self, normalized_rainfall, dem_patches, reanalysis_features):
        """
        Assemble training examples by combining rainfall data, DEM patches extracted around rainfall station coordinates,
        and reanalysis features into a single DataFrame for machine learning.
        
        This method creates a tabular dataset where each row represents a specific station-year-month combination
        and includes:
        1. Metadata (station name, year, month)
        2. Rainfall values (both original and normalized)
        3. One-hot encoded month indicators
        4. Flattened DEM patches (both local and regional) extracted around the rainfall station's coordinates
        5. Flattened reanalysis features
        
        The resulting DataFrame can be used directly for traditional machine learning models that expect
        tabular data, while the NPZ format is more suitable for deep learning models that can work with
        the original 2D patch structure.
        
        Args:
            normalized_rainfall (dict): Dictionary with normalized rainfall data keyed by station name
                Each value is a DataFrame with columns: Year, Month, Rainfall, Rainfall_Normalized
            dem_patches (dict): Dictionary with DEM patches extracted around rainfall station coordinates
                Format: {station_name: {'local': local_patch_array, 'regional': regional_patch_array}}
                Where local_patch_array captures fine-grained elevation details around the station
                and regional_patch_array captures broader topographic context
            reanalysis_features (dict): Dictionary with reanalysis features
                Format: {station_name: {(year, month): {var_name: feature_array, ...}, ...}}
            
        Returns:
            pandas.DataFrame: DataFrame containing assembled training examples with all features flattened
                into columns, suitable for traditional machine learning models
        """
        # Lists to store data for the final DataFrame
        data_rows = []
        
        # Process each station
        for station_name in normalized_rainfall:
            print(f"Assembling training examples for station: {station_name}")
            
            # Skip if DEM patches or reanalysis features are not available
            if station_name not in dem_patches or station_name not in reanalysis_features:
                print(f"Skipping station {station_name} - missing DEM patches or reanalysis features")
                continue
            
            # Get rainfall data
            rainfall_df = normalized_rainfall[station_name]
            
            # Get DEM patches that were extracted around this rainfall station's coordinates
            # These patches represent the topographic context around the station location
            local_dem = dem_patches[station_name]['local']      # Fine-grained elevation details (smaller area, higher resolution)
            regional_dem = dem_patches[station_name]['regional']  # Broader topographic context (larger area, lower resolution)
            
            # Process each year and month
            for _, row in rainfall_df.iterrows():
                year = int(row['Year'])
                month = int(row['Month'])
                rainfall = row['Rainfall']  # Original rainfall in inches
                rainfall_normalized = row['Rainfall_Normalized']  # Normalized rainfall
                
                # Skip if reanalysis features are not available for this time
                if (year, month) not in reanalysis_features[station_name]:
                    continue
                
                # Get reanalysis features
                time_features = reanalysis_features[station_name][(year, month)]
                
                # Create one-hot encoding for month
                month_onehot = month_one_hot(month)
                
                # Create a row dictionary
                row_dict = {
                    'station_name': station_name,
                    'year': year,
                    'month': month,
                    'rainfall': rainfall,  # Original rainfall in inches
                    'rainfall_normalized': rainfall_normalized,  # Normalized rainfall
                }
                
                # Add one-hot encoded month
                for i in range(12):
                    row_dict[f'month_onehot_{i+1}'] = month_onehot[i]
                
                # Add flattened DEM patches extracted around the rainfall station's coordinates
                # We convert the 2D patches into a series of 1D features for tabular machine learning
                # Local patches capture fine-grained elevation details immediately surrounding the station
                for i in range(local_dem.shape[0]):
                    for j in range(local_dem.shape[1]):
                        row_dict[f'dem_local_{i}_{j}'] = local_dem[i, j]
                
                # Regional patches capture broader topographic context around the station
                # This helps the model understand larger-scale terrain features that may influence rainfall
                for i in range(regional_dem.shape[0]):
                    for j in range(regional_dem.shape[1]):
                        row_dict[f'dem_regional_{i}_{j}'] = regional_dem[i, j]
                
                # Add flattened reanalysis features
                for var_name, patch in time_features.items():
                    for i in range(patch.shape[0]):
                        for j in range(patch.shape[1]):
                            row_dict[f'reanalysis_{var_name}_{i}_{j}'] = patch[i, j]
                
                # Add to the list of rows
                data_rows.append(row_dict)
        
        # Create DataFrame from the list of rows
        df = pd.DataFrame(data_rows)
        
        print(f"Assembled {len(df)} training examples")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {len(df.columns)}")

       # Save metadata about the dataset
        metadata_path = os.path.join(os.path.dirname(output_path), 'dataset_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Dataset created on: {pd.Timestamp.now()}\n")
            f.write(f"Number of examples: {len(df)}\n")
            f.write(f"Number of features: {len(df.columns) - 1}\n")  # Excluding the target variable
            f.write(f"Rainfall statistics - Mean: {self.rainfall_mean:.2f} inches, Std: {self.rainfall_std:.2f} inches\n")
            f.write(f"Columns: {', '.join(df.columns)}\n")
        
        print(f"Dataset metadata saved to {metadata_path}")
        
        return df
        
    
    def visualize_dataset(self, df, output_dir=None):
        """
        Visualize the assembled dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame containing assembled training examples
            output_dir (str, optional): Directory to save visualizations.
                If None, uses the figures subdirectory of the output directory.
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'figures')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize rainfall distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['rainfall'], bins=50, alpha=0.7)
        plt.title('Rainfall Distribution (mm)')
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'rainfall_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize normalized rainfall distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['rainfall_normalized'], bins=50, alpha=0.7)
        plt.title('Normalized Rainfall Distribution')
        plt.xlabel('Normalized Rainfall')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'normalized_rainfall_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize rainfall by month
        monthly_rainfall = df.groupby('month')['rainfall'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.bar(monthly_rainfall['month'], monthly_rainfall['mean'], yerr=monthly_rainfall['std'], alpha=0.7)
        plt.title('Average Monthly Rainfall (mm)')
        plt.xlabel('Month')
        plt.ylabel('Average Rainfall (mm)')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'monthly_rainfall.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize rainfall by station
        station_rainfall = df.groupby('station_name')['rainfall'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(14, 8))
        plt.bar(range(len(station_rainfall)), station_rainfall['mean'], yerr=station_rainfall['std'], alpha=0.7)
        plt.title('Average Rainfall by Station (mm)')
        plt.xlabel('Station')
        plt.ylabel('Average Rainfall (mm)')
        plt.xticks(range(len(station_rainfall)), station_rainfall['station_name'], rotation=90)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'station_rainfall.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dataset visualizations saved to {output_dir}")


def main(time_interval='monthly'):
    print(f"Assembling {time_interval} training data...")
    assembler = TrainingDataAssembler(time_interval=time_interval)
    out_path = assembler.assemble_from_precomputed()
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Assemble training data.')
    parser.add_argument('time_interval', type=str, nargs='?', default='monthly',
                       choices=['monthly', 'daily'], help='Time interval to process')
    args = parser.parse_args()
    main(time_interval=args.time_interval)
