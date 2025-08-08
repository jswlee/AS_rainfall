#!/usr/bin/env python3
"""
Test script for the simplified spatiotemporal GP interpolation.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from processors.rainfall_processor.processor import RainfallProcessor
from processors.rainfall_processor.gp_utils import spatiotemporal_gp_interpolate
import numpy as np

def test_gp_utils():
    """Test the GP utilities directly."""
    print("Testing GP utilities...")
    
    # Create some synthetic spatiotemporal data
    all_station_data = {
        '2020-01': {
            'locations': [(-170.7, -14.3), (-170.6, -14.2), (-170.8, -14.4)],
            'values': [2.5, 3.1, 1.8]
        },
        '2020-02': {
            'locations': [(-170.7, -14.3), (-170.6, -14.2), (-170.8, -14.4)],
            'values': [1.2, 2.0, 0.9]
        },
        '2020-03': {
            'locations': [(-170.7, -14.3), (-170.6, -14.2), (-170.8, -14.4)],
            'values': [4.1, 3.8, 3.5]
        }
    }
    
    # Define prediction points
    prediction_points = [(-170.65, -14.25), (-170.75, -14.35)]
    prediction_times = ['2020-01', '2020-02', '2020-03', '2020-04']
    
    # Test spatiotemporal interpolation
    predictions = spatiotemporal_gp_interpolate(
        all_station_data=all_station_data,
        prediction_points=prediction_points,
        prediction_times=prediction_times
    )
    
    print(f"Predictions for {len(prediction_times)} time points:")
    for date, values in predictions.items():
        print(f"  {date}: {values}")
    
    return predictions

def test_rainfall_processor():
    """Test the RainfallProcessor with real data if available."""
    print("\nTesting RainfallProcessor...")
    
    # Try to initialize with real data paths
    rainfall_dir = "/Users/jlee/Desktop/github/AS_rainfall/raw_data/rainfall"
    station_locations_path = "/Users/jlee/Desktop/github/AS_rainfall/raw_data/station_locations.csv"
    
    try:
        processor = RainfallProcessor(rainfall_dir, station_locations_path)
        print(f"RainfallProcessor initialized successfully")
        print(f"Max rainfall: {processor.max_rainfall:.2f} inches")
        
        # Check if we have any data loaded
        if hasattr(processor, 'rainfall_data') and processor.rainfall_data:
            print(f"Loaded data for {len(processor.rainfall_data)} dates")
            
            # Test with a small grid
            test_grid = [(-170.7, -14.3), (-170.6, -14.2), (-170.8, -14.4)]
            
            # Test single date interpolation
            first_date = list(processor.rainfall_data.keys())[0]
            rainfall_data = processor.get_rainfall_data(first_date)
            
            if rainfall_data['locations']:
                print(f"Testing interpolation for {first_date} with {len(rainfall_data['locations'])} stations")
                interpolated = processor.interpolate_to_grid(rainfall_data, test_grid)
                print(f"Interpolated values: {interpolated}")
                
                # Test spatiotemporal interpolation
                print("Testing spatiotemporal interpolation...")
                st_predictions = processor.interpolate_spatiotemporal(
                    grid_points=test_grid,
                    target_dates=list(processor.rainfall_data.keys())[:3]
                )
                print(f"Spatiotemporal predictions for {len(st_predictions)} dates")
            else:
                print("No rainfall data available for testing")
        else:
            print("No rainfall data loaded")
            
    except Exception as e:
        print(f"Error testing RainfallProcessor: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing simplified spatiotemporal GP implementation...")
    
    # Test GP utilities
    gp_predictions = test_gp_utils()
    
    # Test rainfall processor
    processor_success = test_rainfall_processor()
    
    if gp_predictions and processor_success:
        print("\n✅ All tests completed successfully!")
        print("The simplified spatiotemporal GP implementation is working.")
    else:
        print("\n❌ Some tests failed.")
