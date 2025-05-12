# Climate Processor Package

A modular package for processing climate data from NetCDF files.

## Package Structure

The package is organized into the following modules:

- `__init__.py`: Package initialization and exports
- `config.py`: Configuration settings, variable mappings, and default configurations
- `file_utils.py`: Utilities for file path generation and dataset loading
- `data_ops.py`: Data processing operations (interpolation, calculations, etc.)
- `processor.py`: Main `ClimateDataProcessor` class that integrates all functionality

## Usage

```python
from climate_processor import ClimateDataProcessor

# Create a processor instance
processor = ClimateDataProcessor()

# Process specific variables
processor.process_variable("air_2m")
processor.process_variable("skin_temp")

# Or process by description
processor.process_by_description("sea level pressure")

# Process all variables
processor.process_all_variables()

# Save processed data
processor.save_to_netcdf("climate_data.nc")
```

## Extending the Package

### Adding New Variables

To add new variables, you can modify the `variable_configs` dictionary:

```python
processor = ClimateDataProcessor()

# Add a new variable configuration
processor.variable_configs["custom_temp_diff"] = {
    "description": "Custom temperature difference between 1000 and 850 hPa",
    "variable": "Air",
    "levels": [1000, 850],
    "operation": "diff"
}

# Process the new variable
processor.process_variable("custom_temp_diff")
```

### Custom File Paths

For variables with non-standard file paths, use the `custom_file` parameter:

```python
processor.variable_configs["special_var"] = {
    "description": "Special variable with custom file path",
    "variable": "Special Variable",
    "custom_file": "special_var.custom.nc"
}
```
