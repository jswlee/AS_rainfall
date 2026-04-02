import argparse
from pathlib import Path

import pandas as pd


def _detect_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Detect date, time, and rainfall columns.
    Returns: (date_col, time_col, rain_col)
    If time column not found, returns (date_col, None, rain_col).
    """
    # Date column
    date_candidates = ["date", "day", "datetime"]
    date_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in date_candidates:
            date_col = c
            break
    if date_col is None:
        # Fallback: first column that looks like a date string
        for c in df.columns:
            try:
                pd.to_datetime(df[c].iloc[0])
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        raise RuntimeError("Could not detect date column")

    # Time column
    time_col = None
    time_candidates = ["time", "hour"]
    for c in df.columns:
        cl = c.strip().lower()
        if cl in time_candidates:
            time_col = c
            break

    # Rainfall column
    rain_candidates = ["rainfall_mm", "precip_mm", "rainfall", "precip", "precip_in"]
    rain_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in rain_candidates:
            rain_col = c
            break
    if rain_col is None:
        raise RuntimeError("Could not detect rainfall column")

    return date_col, time_col, rain_col


def _parse_datetime(df: pd.DataFrame, date_col: str, time_col: str | None) -> pd.Series:
    """
    Parse date and optional time into a datetime Series.
    """
    if time_col is not None:
        # Combine date and time
        dt_str = df[date_col].astype(str) + " " + df[time_col].astype(str)
        # Try common formats
        for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return pd.to_datetime(dt_str, format=fmt)
            except Exception:
                continue
        # Fallback: let pandas infer
        return pd.to_datetime(dt_str, errors="coerce")
    else:
        # Date only
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(df[date_col], format=fmt)
            except Exception:
                continue
        return pd.to_datetime(df[date_col], errors="coerce")


def aggregate_to_daily(
    input_path: Path,
    interval: str,
    output_units: str,
    out_dir: Path,
) -> Path:
    """
    Aggregate sub-daily rainfall to daily totals.
    Returns path to the output file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise RuntimeError("Input file is empty")

    date_col, time_col, rain_col = _detect_columns(df)

    dt = _parse_datetime(df, date_col, time_col)
    df["datetime"] = dt
    if df["datetime"].isna().all():
        raise RuntimeError("Failed to parse datetime")

    # Ensure numeric rainfall
    rain_vals = pd.to_numeric(df[rain_col], errors="coerce")
    if rain_vals.isna().all():
        raise RuntimeError(f"Rainfall column '{rain_col}' is not numeric")
    df["rain_numeric"] = rain_vals

    # Convert units if needed
    if output_units.lower() == "inches" or output_units.lower().startswith("in"):
        if rain_col.lower().endswith("_mm"):
            df["precip_in"] = df["rain_numeric"] * 0.0393701
            out_col = "precip_in"
        else:
            # Assume already in inches
            df["precip_in"] = df["rain_numeric"]
            out_col = "precip_in"
    else:
        # Keep as mm
        if rain_col.lower().endswith("_in"):
            df["rainfall_mm"] = df["rain_numeric"] * 25.4
            out_col = "rainfall_mm"
        else:
            df["rainfall_mm"] = df["rain_numeric"]
            out_col = "rainfall_mm"

    # Group by date (ignore time)
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date")[out_col].sum().reset_index()
    daily["datetime"] = pd.to_datetime(daily["date"])

    # Sort and format output
    daily = daily.sort_values("datetime")
    daily["datetime"] = daily["datetime"].dt.strftime("%m/%d/%Y")
    # Keep only datetime and precip columns
    out_df = daily[["datetime", out_col]].copy()
    # Round mm to 1 decimal to avoid floating-point artifacts; keep inches unrounded
    if out_col == "rainfall_mm":
        out_df[out_col] = out_df[out_col].round(1)

    # Write output
    stem = input_path.stem.replace("_raw", "_daily")
    output_path = out_dir / f"{stem}_{output_units.lower()}.csv"
    out_df.to_csv(output_path, index=True, index_label=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate sub-daily rainfall to daily totals with optional unit conversion."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to raw rainfall CSV file (e.g., raw_data/rainfall_raw/aasufou90_raw.csv)",
    )
    parser.add_argument(
        "--interval",
        default="1hr",
        help="Input interval (e.g., 15min, 30min, 1hr, 3hr, 24hr). Only used for naming/logging; aggregation is always daily.",
    )
    parser.add_argument(
        "--output-units",
        choices=["mm", "inches"],
        default="mm",
        help="Output units (mm or inches). If inches, converts from mm assuming input is mm.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("raw_data/rainfall_aggregated"),
        help="Directory to write aggregated daily CSV files (default: raw_data/rainfall_aggregated)",
    )
    args = parser.parse_args()

    out_path = aggregate_to_daily(
        input_path=args.input_path,
        interval=args.interval,
        output_units=args.output_units,
        out_dir=args.out_dir,
    )
    print(f"Aggregated daily rainfall written to: {out_path}")


if __name__ == "__main__":
    main()
