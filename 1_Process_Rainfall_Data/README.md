# Rainfall Data Processing

## Overview
This component processes raw rainfall data from daily measurements into monthly aggregates that can be used for model training. It's the first step in the rainfall prediction pipeline.

## Functionality
- Reads raw rainfall data from CSV files
- Aggregates daily rainfall measurements into monthly totals
- Handles missing data and performs basic data cleaning
- Outputs processed monthly rainfall data in CSV format

## Directory Structure
```
1_Process_Rainfall_Data/
├── scripts/
│   └── rainfall_daily_to_monthly.py  # Main script for processing
├── output/
│   └── monthly_rainfall/             # Contains processed monthly rainfall data
└── README.md                         # This file
```

## Usage
To process rainfall data, run:
```bash
cd 1_Process_Rainfall_Data/scripts
python rainfall_daily_to_monthly.py --input /path/to/raw/rainfall/data --output ../output/monthly_rainfall
```

## Input Data Format
The input data should be CSV files with daily rainfall measurements. Each file should represent a single station and include:
- A date column (in YYYY-MM-DD format)
- A rainfall measurement column (in inches)

## Output
The script generates monthly aggregated rainfall data in CSV format, stored in the `output/monthly_rainfall` directory. These files are used in subsequent steps of the pipeline.

## Dependencies
- pandas
- numpy
- pathlib

## Next Steps
After processing the rainfall data, proceed to step 2 (Create ML Data) to generate the machine learning datasets that combine rainfall data with climate variables and digital elevation model (DEM) data.
