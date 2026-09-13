#!/usr/bin/env python3
"""
Comprehensive NetCDF audit script for reanalysis data.

This script performs a full audit of all NetCDF files in a directory,
including variable detection, pressure level analysis, time coverage,
and data quality checks.

Usage:
    python audit_nc_files.py /path/to/nc/directory
    python audit_nc_files.py /path/to/nc/directory --output audit_report.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import xarray as xr


def detect_time_dim(ds: xr.Dataset) -> str:
    """Detect the time dimension name."""
    for name in ("valid_time", "time"):
        if name in ds.dims or name in ds.coords:
            return name
    raise KeyError("No time dimension found")


def detect_lat_lon(ds: xr.Dataset) -> Tuple[str, str]:
    """Detect latitude and longitude dimension names."""
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    return lat_name, lon_name


def detect_level_dim(ds: xr.Dataset) -> Optional[str]:
    """Detect the vertical/pressure level dimension name."""
    for name in ("pressure_level", "level", "lev", "plev", "isobaricInhPa"):
        if name in ds.dims:
            return name
    return None


def analyze_nan_coverage(data: np.ndarray, var_name: str) -> Dict:
    """Analyze NaN coverage in a data array."""
    total_values = data.size
    nan_values = np.isnan(data).sum()
    valid_values = total_values - nan_values
    
    return {
        "total_values": total_values,
        "nan_values": nan_values,
        "valid_values": valid_values,
        "nan_percentage": (nan_values / total_values) * 100 if total_values > 0 else 100,
        "has_data": valid_values > 0
    }


def audit_single_file(nc_path: Path, expected_date_range: Tuple[pd.Timestamp, pd.Timestamp]) -> Dict:
    """Audit a single NetCDF file."""
    result = {
        "filename": nc_path.name,
        "file_path": str(nc_path),
        "file_size_mb": nc_path.stat().st_size / (1024 * 1024),
        "error": None,
        "variables": [],
        "time_coverage": {},
        "spatial_coverage": {},
        "pressure_levels": {},
        "data_quality": {}
    }
    
    try:
        ds = xr.open_dataset(nc_path)

        # Basic file info
        result["variables"] = list(ds.data_vars.keys())
        result["coordinates"] = list(ds.coords.keys())
        result["dimensions"] = dict(ds.sizes)
        
        # Time analysis
        try:
            time_dim = detect_time_dim(ds)
            times = pd.DatetimeIndex(ds[time_dim].values)
            
            result["time_coverage"] = {
                "time_dim": time_dim,
                "time_steps": len(times),
                "start_date": times[0],
                "end_date": times[-1],
                "expected_start": expected_date_range[0],
                "expected_end": expected_date_range[1],
                "expected_days": (expected_date_range[1] - expected_date_range[0]).days + 1,
                "missing_days": 0,
                "gaps": []
            }
            
            # Check for missing dates
            expected_dates = pd.date_range(expected_date_range[0], expected_date_range[1], freq='D')
            missing_dates = expected_dates.difference(times)
            result["time_coverage"]["missing_days"] = len(missing_dates)
            
            if len(missing_dates) > 0:
                result["time_coverage"]["first_missing"] = missing_dates[0]
                result["time_coverage"]["last_missing"] = missing_dates[-1]
                result["time_coverage"]["missing_percentage"] = (len(missing_dates) / len(expected_dates)) * 100
            
            # Check for gaps
            gaps = []
            for i in range(1, len(times)):
                gap_days = (times[i] - times[i-1]).days
                if gap_days > 1:
                    gaps.append({
                        "start": times[i-1],
                        "end": times[i],
                        "missing_days": gap_days - 1
                    })
            result["time_coverage"]["gaps"] = gaps
            
        except Exception as e:
            result["time_coverage"]["error"] = str(e)
        
        # Spatial analysis
        try:
            lat_name, lon_name = detect_lat_lon(ds)
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            
            result["spatial_coverage"] = {
                "lat_dim": lat_name,
                "lon_dim": lon_name,
                "lat_points": len(lats),
                "lon_points": len(lons),
                "lat_range": (float(lats.min()), float(lats.max())),
                "lon_range": (float(lons.min()), float(lons.max())),
                "total_grid_points": len(lats) * len(lons)
            }
        except Exception as e:
            result["spatial_coverage"]["error"] = str(e)
        
        # Pressure level analysis
        level_dim = detect_level_dim(ds)
        if level_dim:
            try:
                levels = ds[level_dim].values
                result["pressure_levels"] = {
                    "level_dim": level_dim,
                    "num_levels": len(levels),
                    "levels": levels.tolist(),
                    "level_units": ds[level_dim].attrs.get("units", "unknown")
                }
                
                # Analyze each pressure level for NaN coverage
                for var_name in ds.data_vars:
                    if level_dim in ds[var_name].dims:
                        var_data = ds[var_name].values
                        level_stats = {}
                        
                        for i, level in enumerate(levels):
                            if var_data.ndim >= 3:  # (time, level, lat, lon) or (level, lat, lon)
                                if var_data.ndim == 4:
                                    level_data = var_data[:, i, :, :]  # All time for this level
                                else:
                                    level_data = var_data[i, :, :]  # Single time for this level
                                
                                level_stats[f"level_{level}"] = analyze_nan_coverage(level_data, var_name)
                        
                        result["data_quality"][var_name] = level_stats
                        
            except Exception as e:
                result["pressure_levels"]["error"] = str(e)
        else:
            # No pressure levels - analyze surface variables
            for var_name in ds.data_vars:
                var_data = ds[var_name].values
                result["data_quality"][var_name] = analyze_nan_coverage(var_data, var_name)
        
        # Variable metadata
        result["variable_metadata"] = {}
        for var_name in ds.data_vars:
            var = ds[var_name]
            result["variable_metadata"][var_name] = {
                "dims": list(var.dims),
                "shape": var.shape,
                "dtype": str(var.dtype),
                "attributes": dict(var.attrs)
            }
        
        ds.close()
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def print_audit_summary(results: List[Dict], output_file: Optional[str] = None):
    """Print a comprehensive audit summary."""
    
    def write_line(text: str = ""):
        if text:
            print(text)
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
    
    # Clear output file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("")
    
    write_line("=" * 80)
    write_line("NETCDF AUDIT REPORT")
    write_line("=" * 80)
    write_line(f"Total files analyzed: {len(results)}")
    write_line(f"Files with errors: {sum(1 for r in results if r['error'])}")
    write_line()
    
    # File overview
    write_line("FILE OVERVIEW")
    write_line("-" * 40)
    for result in results:
        status = "❌ ERROR" if result["error"] else "✅ OK"
        write_line(f"{result['filename']:<25} {status:<10} {result['file_size_mb']:.1f} MB")
        if result["error"]:
            write_line(f"    Error: {result['error']}")
    write_line()
    
    # Time coverage comparison
    write_line("TIME COVERAGE ANALYSIS")
    write_line("-" * 40)
    
    # Get expected date range from the most complete file
    expected_ranges = []
    for result in results:
        if "time_coverage" in result and "expected_start" in result["time_coverage"]:
            expected_ranges.append((
                result["time_coverage"]["expected_start"],
                result["time_coverage"]["expected_end"]
            ))
    
    if expected_ranges:
        # Use the most common expected range
        from collections import Counter
        range_counter = Counter(expected_ranges)
        expected_range = range_counter.most_common(1)[0][0]
        
        write_line(f"Expected date range: {expected_range[0]} to {expected_range[1]}")
        write_line()
        
        for result in results:
            if "time_coverage" in result and "time_steps" in result["time_coverage"]:
                tc = result["time_coverage"]
                missing_pct = tc.get("missing_percentage", 0)
                
                write_line(f"{result['filename']:<25}")
                write_line(f"  Time steps: {tc['time_steps']:,} / {tc['expected_days']:,}")
                write_line(f"  Date range: {tc['start_date']} to {tc['end_date']}")
                write_line(f"  Missing: {tc['missing_days']:,} days ({missing_pct:.1f}%)")
                
                if tc["gaps"]:
                    write_line(f"  Gaps: {len(tc['gaps'])}")
                    for gap in tc["gaps"][:3]:  # Show first 3 gaps
                        write_line(f"    {gap['start']} to {gap['end']} ({gap['missing_days']} days)")
                
                write_line()
    
    # Variable comparison
    write_line("VARIABLE COMPARISON")
    write_line("-" * 40)
    
    # Collect all variables across files
    all_vars = set()
    var_file_mapping = {}
    for result in results:
        if not result["error"]:
            for var in result["variables"]:
                all_vars.add(var)
                if var not in var_file_mapping:
                    var_file_mapping[var] = []
                var_file_mapping[var].append(result["filename"])
    
    write_line(f"Total unique variables found: {len(all_vars)}")
    write_line()
    
    for var in sorted(all_vars):
        files_with_var = var_file_mapping[var]
        write_line(f"{var:<25} in {len(files_with_var)} files:")
        for filename in files_with_var:
            write_line(f"  - {filename}")
        write_line()
    
    # Pressure level analysis
    write_line("PRESSURE LEVEL ANALYSIS")
    write_line("-" * 40)
    
    pressure_info = {}
    for result in results:
        if not result["error"] and "pressure_levels" in result and "levels" in result["pressure_levels"]:
            levels = tuple(result["pressure_levels"]["levels"])
            if levels not in pressure_info:
                pressure_info[levels] = []
            pressure_info[levels].append(result["filename"])
    
    for levels, filenames in pressure_info.items():
        write_line(f"Levels {list(levels)} found in {len(filenames)} files:")
        for filename in filenames:
            write_line(f"  - {filename}")
        write_line()
    
    # Data quality issues
    write_line("DATA QUALITY ISSUES")
    write_line("-" * 40)
    
    for result in results:
        if not result["error"] and "data_quality" in result:
            quality_issues = []
            
            for var_name, quality_data in result["data_quality"].items():
                if isinstance(quality_data, dict):
                    # Check for high NaN percentages
                    if "nan_percentage" in quality_data and quality_data["nan_percentage"] > 50:
                        quality_issues.append(f"{var_name}: {quality_data['nan_percentage']:.1f}% NaN")
                    
                    # Check pressure level specific issues
                    for key, value in quality_data.items():
                        if key.startswith("level_") and isinstance(value, dict):
                            level = key.split("_")[1]
                            if value["nan_percentage"] > 50:
                                quality_issues.append(f"{var_name} level {level}: {value['nan_percentage']:.1f}% NaN")
            
            if quality_issues:
                write_line(f"{result['filename']}:")
                for issue in quality_issues:
                    write_line(f"  ⚠️  {issue}")
                write_line()
    
    # Recommendations
    write_line("RECOMMENDATIONS")
    write_line("-" * 40)
    
    # Check for inconsistent time coverage
    time_steps = set()
    for result in results:
        if "time_coverage" in result and "time_steps" in result["time_coverage"]:
            time_steps.add(result["time_coverage"]["time_steps"])
    
    if len(time_steps) > 1:
        write_line("⚠️  INCONSISTENT TIME COVERAGE DETECTED:")
        write_line("   Files have different numbers of time steps")
        write_line("   This will cause NaN values in derived variables")
        write_line("   Recommendation: Use the dataset with complete coverage")
        write_line()
    
    # Check for high NaN percentages
    high_nan_vars = []
    for result in results:
        if "data_quality" in result:
            for var_name, quality_data in result["data_quality"].items():
                if isinstance(quality_data, dict) and "nan_percentage" in quality_data:
                    if quality_data["nan_percentage"] > 80:
                        high_nan_vars.append(f"{result['filename']}:{var_name}")
    
    if high_nan_vars:
        write_line("⚠️  HIGH NaN COVERAGE DETECTED:")
        for var in high_nan_vars:
            write_line(f"   {var}")
        write_line("   Recommendation: Check data source or exclude problematic variables")
        write_line()
    
    write_line("Audit complete.")


def main():
    parser = argparse.ArgumentParser(description="Audit NetCDF files in a directory")
    parser.add_argument("directory", help="Directory containing NetCDF files")
    parser.add_argument("--output", help="Output file for audit report")
    parser.add_argument("--start-date", default="1980-01-01", help="Expected start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="Expected end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    nc_dir = Path(args.directory)
    if not nc_dir.exists():
        print(f"Error: Directory {nc_dir} does not exist")
        sys.exit(1)
    
    # Find all NetCDF files
    nc_files = list(nc_dir.glob("*.nc"))
    if not nc_files:
        print(f"No NetCDF files found in {nc_dir}")
        sys.exit(1)
    
    print(f"Found {len(nc_files)} NetCDF files")
    print(f"Auditing files from {args.start_date} to {args.end_date}")
    print()
    
    # Parse expected date range
    expected_start = pd.Timestamp(args.start_date)
    expected_end = pd.Timestamp(args.end_date)
    expected_range = (expected_start, expected_end)
    
    # Audit each file
    results = []
    for i, nc_file in enumerate(nc_files, 1):
        print(f"[{i}/{len(nc_files)}] Auditing {nc_file.name}...")
        result = audit_single_file(nc_file, expected_range)
        results.append(result)
    
    print()
    print("Generating audit report...")
    print()
    
    # Print comprehensive summary
    print_audit_summary(results, args.output)


if __name__ == "__main__":
    main()
