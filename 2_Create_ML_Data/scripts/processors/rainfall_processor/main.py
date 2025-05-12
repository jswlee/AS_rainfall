import os
import sys
from pathlib import Path

# Add the scripts directory to the Python path
scripts_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(scripts_dir))

from processors.rainfall_processor.processor import RainfallProcessor

def main():
    # Default paths - user should update these
    station_locations_path = os.path.join(project_root, 'data', 'station_locations.csv')
    rainfall_dir = os.path.join(project_root, 'data', 'rainfall')
    output_dir = os.path.join(project_root, 'output', 'rainfall')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize RainfallProcessor
    processor = RainfallProcessor(
        station_locations_path=station_locations_path,
        rainfall_dir=rainfall_dir,
        output_dir=output_dir
    )
    
    # Load data
    processor.load_data()
    
    # Example: Interpolate and visualize rainfall for a specific month
    try:
        # Replace with an actual date from your data
        sample_date = processor.rainfall_data.keys()[0] if processor.rainfall_data else None
        
        if sample_date:
            # Interpolate rainfall
            grid_points = processor.create_grid_points()
            interpolated_rainfall = processor.interpolate_to_grid(
                processor.rainfall_data[sample_date], 
                grid_points
            )
            
            # Visualize rainfall
            output_path = os.path.join(output_dir, f'rainfall_{sample_date}.png')
            processor.visualize_rainfall(
                sample_date, 
                grid_points, 
                interpolated_rainfall, 
                output_path
            )
            print(f"Processed rainfall data for {sample_date}")
        else:
            print("No rainfall data available to process")
    
    except Exception as e:
        print(f"Error processing rainfall data: {e}")

if __name__ == '__main__':
    main()
