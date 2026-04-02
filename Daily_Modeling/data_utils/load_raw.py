"""
Load raw rainfall CSVs and station metadata for American Samoa.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    if "datetime" not in df.columns:
        raise ValueError(f"Expected 'datetime' column in {csv_path}")
    dt = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=False)
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day

    # --- Locate the precipitation column ---
    precip_col = None
    precip_unit = None
    if "precip_in" in df.columns:
        precip_col = "precip_in"
        precip_unit = "in"
    elif "precip_mm" in df.columns:
        precip_col = "precip_mm"
        precip_unit = "mm"
    else:
        raise ValueError(f"Expected 'precip_in' or 'precip_mm' column in {csv_path}")

    df["rainfall_mm"] = pd.to_numeric(df[precip_col], errors="coerce")

    # Convert to mm if necessary
    if precip_unit == "in":
        df["rainfall_mm"] = df["rainfall_mm"] * 25.4

    df = df[["year", "month", "day", "rainfall_mm"]].dropna()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["rainfall_mm"] = df["rainfall_mm"].astype(float)
    return df

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
