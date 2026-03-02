"""
Load raw rainfall CSVs and station metadata for American Samoa.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Daily_Modeling import config


def load_station_metadata(path: Optional[Path] = None) -> Dict[str, dict]:
    """Load station metadata CSV -> {station_name: {latitude, longitude, ...}}."""
    path = path or config.STATION_METADATA_PATH
    df = pd.read_csv(path)
    df = df.rename(columns={"Station": "station_name", "LAT": "latitude", "LONG": "longitude"})
    df = df.dropna(subset=["station_name", "latitude", "longitude"])
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df = df.drop_duplicates(subset=["station_name"])

    meta = {}
    for _, row in df.iterrows():
        name = row["station_name"]
        meta[name] = {col: row[col] for col in df.columns if col != "station_name"}
    return meta


def load_daily_rainfall(
    station_name: str,
    rainfall_dir: Optional[Path] = None,
    source_unit: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load a single station's daily rainfall CSV.

    The CSVs have columns: (index), datetime, precip_in  (or similar).
    Parses the datetime column to extract year/month/day and converts
    the precipitation value to millimetres.

    Returns DataFrame with columns [year, month, day, rainfall_mm] or None.
    """
    rainfall_dir = rainfall_dir or config.DAILY_RAINFALL_DIR
    csv_path = Path(rainfall_dir) / f"{station_name}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # --- Locate the datetime column ---
    dt_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("datetime", "date", "time", "dt"):
            dt_col = c
            break
    if dt_col is None:
        # Fall back: check if year/month/day columns already exist
        low = {c.strip().lower(): c for c in df.columns}
        if "year" in low and "month" in low and "day" in low:
            df = df.rename(columns={low["year"]: "year", low["month"]: "month", low["day"]: "day"})
        else:
            return None
    else:
        dt = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=False)
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day

    # --- Locate the precipitation column ---
    precip_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("precip_in", "precip", "precipitation", "rainfall",
                   "rain", "prcp", "precip_mm", "rainfall_mm"):
            precip_col = c
            break
    if precip_col is None:
        return None

    df["rainfall_mm"] = pd.to_numeric(df[precip_col], errors="coerce")

    # All files in rainfall_corrected_NEW are in inches -> convert to mm
    df["rainfall_mm"] = df["rainfall_mm"] * 25.4

    df = df[["year", "month", "day", "rainfall_mm"]].dropna()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["rainfall_mm"] = df["rainfall_mm"].astype(float)
    return df


def _get_source_unit(station_metadata: Dict[str, dict], station_name: str) -> Optional[str]:
    """Read the source_unit field from station metadata (e.g. 'in' or 'mm')."""
    meta = station_metadata.get(station_name, {})
    return str(meta.get("source_unit", "in")) if "source_unit" in meta else None


def load_all_station_rainfall(
    station_metadata: Optional[Dict[str, dict]] = None,
    rainfall_dir: Optional[Path] = None,
    min_days: int = 365,
) -> Dict[str, pd.DataFrame]:
    """Load daily rainfall for every station that has at least *min_days* records.

    Returns {station_name: DataFrame}.
    """
    if station_metadata is None:
        station_metadata = load_station_metadata()

    result = {}
    for name in sorted(station_metadata):
        df = load_daily_rainfall(name, rainfall_dir)
        if df is not None and len(df) >= min_days:
            result[name] = df
    print(f"Loaded daily rainfall for {len(result)}/{len(station_metadata)} stations "
          f"(>={min_days} days each)")
    return result


def discover_station_days(
    station_metadata: Dict[str, dict],
    rainfall_dir: Optional[Path] = None,
    start_date: str = "1980-01-01",
    end_date: str = "2024-12-31",
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Return {station_name: [(year, month, day), ...]} for all available days."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    result: Dict[str, List[Tuple[int, int, int]]] = {}

    for name in sorted(station_metadata):
        df = load_daily_rainfall(name, rainfall_dir)
        if df is None or df.empty:
            continue
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            continue
        tuples = list(zip(df["year"], df["month"], df["day"]))
        result[name] = tuples

    print(f"Discovered {sum(len(v) for v in result.values())} station-days "
          f"across {len(result)} stations")
    return result
