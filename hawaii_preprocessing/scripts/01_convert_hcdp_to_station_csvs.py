"""
Convert HCDP daily station rainfall (wide, monthly) into per-station long CSVs
matching the format of `raw_data/AS/final_rainfall_per_station/*.csv`.

INPUT  : raw_data/HI/HCDP_data_daily/station_data/<YYYY>/<MM>/rainfall_new_day_statewide_partial_station_data_<YYYY>_<MM>.csv
         (rows = stations, columns = X<YYYY>.<MM>.<DD> daily rainfall in mm)

OUTPUT : raw_data/HI/final_rainfall_per_station/<station_id>.csv
         columns:  ,datetime,precip_mm
         (matches the AS layout; AS uses precip_in but the loader in
          Daily_Modeling/data_utils/load_raw.py accepts either column.)
         Also writes raw_data/HI/station_locations.csv with the same schema as
         raw_data/AS/station_locations.csv.

Run from the repo root:
    python hawaii_preprocessing/scripts/01_convert_hcdp_to_station_csvs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
HI_RAW_DIR = REPO_ROOT / "raw_data" / "HI" / "HCDP_data_daily" / "station_data"
HI_OUT_DIR = REPO_ROOT / "raw_data" / "HI" / "final_rainfall_per_station"
HI_STATION_META_PATH = REPO_ROOT / "raw_data" / "HI" / "station_locations.csv"

# AS metadata schema (column order to mirror)
AS_META_COLUMNS = [
    "Station", "Organization", "LAT", "LONG", "elev_ft", "elev_src",
    "start_yr", "end_yr", "range", "station_id", "source", "source_unit",
    "source_freq", "Island",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DATE_COL_RE = re.compile(r"^X(\d{4})\.(\d{2})\.(\d{2})$")


def _safe_id(skn: object) -> str:
    """Make a filesystem-safe station id, prefixed with HI_ to avoid clashes
    with AS station names (e.g. SKN '2.1' -> 'HI_2_1')."""
    s = str(skn).strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = s.strip("_")
    return f"HI_{s}" if s else "HI_unknown"


def _iter_monthly_csvs(root: Path):
    """Yield monthly CSV paths, sorted by year/month."""
    for year_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for month_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            for csv in sorted(month_dir.glob("*.csv")):
                yield csv


def _melt_one_month(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_rainfall_df, station_meta_df) for one monthly file.

    long_rainfall_df: columns [station_id, date, precip_mm]
    station_meta_df : columns [station_id, Station, LAT, LONG, elev_ft, Network, Island, station_id_raw]
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Identify date columns of the form X1990.01.01
    date_cols = [c for c in df.columns if _DATE_COL_RE.match(str(c))]
    if not date_cols:
        return pd.DataFrame(), pd.DataFrame()

    # Stable per-station ID
    if "SKN" not in df.columns:
        raise KeyError(f"Expected 'SKN' column in {csv_path}")
    df["station_id"] = df["SKN"].map(_safe_id)

    # --- Station metadata (one row per station) ---
    meta_cols = {
        "station_id": "station_id",
        "SKN": "station_id_raw",
        "Station.Name": "Station",
        "Network": "Network",
        "Island": "Island",
        "ELEV.m.": "elev_m",
        "LAT": "LAT",
        "LON": "LONG",  # rename to match AS
    }
    have = [c for c in meta_cols if c in df.columns]
    meta = df[have].rename(columns={c: meta_cols[c] for c in have}).copy()
    meta = meta.drop_duplicates(subset=["station_id"])

    # --- Rainfall: melt wide -> long ---
    island_map = df.set_index("station_id")["Island"] if "Island" in df.columns else None
    if island_map is not None:
        island_map = island_map[~island_map.index.duplicated(keep="first")]
    long_df = df.melt(
        id_vars=["station_id"],
        value_vars=date_cols,
        var_name="date_col",
        value_name="precip_mm",
    )
    parsed = long_df["date_col"].str.extract(_DATE_COL_RE)
    parsed.columns = ["year", "month", "day"]
    long_df["date"] = pd.to_datetime(parsed.astype(int).rename(
        columns={"year": "year", "month": "month", "day": "day"}
    ), errors="coerce")
    long_df = long_df.drop(columns=["date_col"])
    long_df["precip_mm"] = pd.to_numeric(long_df["precip_mm"], errors="coerce")
    if island_map is not None:
        long_df["Island"] = long_df["station_id"].map(island_map)

    return long_df, meta


def main() -> int:
    if not HI_RAW_DIR.exists():
        print(f"ERROR: {HI_RAW_DIR} does not exist", file=sys.stderr)
        return 1

    HI_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {HI_RAW_DIR} ...")
    monthly_files = list(_iter_monthly_csvs(HI_RAW_DIR))
    print(f"  Found {len(monthly_files)} monthly CSVs")

    long_chunks = []
    meta_chunks = []
    for i, csv_path in enumerate(monthly_files, 1):
        try:
            long_df, meta_df = _melt_one_month(csv_path)
        except Exception as e:
            print(f"  [WARN] skipping {csv_path.name}: {e}")
            continue
        if long_df.empty:
            continue
        long_chunks.append(long_df)
        meta_chunks.append(meta_df)
        if i % 50 == 0 or i == len(monthly_files):
            print(f"  {i}/{len(monthly_files)} processed")

    if not long_chunks:
        print("ERROR: no rainfall data was parsed", file=sys.stderr)
        return 1

    print("Concatenating monthly chunks ...")
    rainfall = pd.concat(long_chunks, ignore_index=True)
    metadata = pd.concat(meta_chunks, ignore_index=True).drop_duplicates(
        subset=["station_id"], keep="last"
    )

    # Sort + de-dupe (a date should appear once per station)
    rainfall = rainfall.dropna(subset=["date"]).sort_values(
        ["station_id", "date"]
    ).drop_duplicates(subset=["station_id", "date"], keep="last")

    # ----- Per-station CSVs -----
    print(f"Writing per-station CSVs to {HI_OUT_DIR} ...")
    written = 0
    station_year_ranges: dict[str, tuple[int, int]] = {}
    for sid, sdf in rainfall.groupby("station_id", sort=True):
        sdf = sdf.copy().reset_index(drop=True)
        out_df = pd.DataFrame({
            "datetime": sdf["date"].dt.strftime("%m/%d/%Y"),
            "precip_mm": sdf["precip_mm"],
            "Island": sdf["Island"] if "Island" in sdf.columns else np.nan,
        })
        out_df.index = np.arange(1, len(out_df) + 1)
        # Mirror AS file: leading unnamed index column
        out_df.to_csv(HI_OUT_DIR / f"{sid}.csv", index=True, index_label="")

        valid = sdf.dropna(subset=["precip_mm"])
        if not valid.empty:
            station_year_ranges[sid] = (
                int(valid["date"].dt.year.min()),
                int(valid["date"].dt.year.max()),
            )
        else:
            station_year_ranges[sid] = (
                int(sdf["date"].dt.year.min()),
                int(sdf["date"].dt.year.max()),
            )
        written += 1
    print(f"  Wrote {written} station CSVs")

    # ----- station_locations.csv (AS schema) -----
    print(f"Writing station metadata to {HI_STATION_META_PATH} ...")
    rows = []
    for _, m in metadata.iterrows():
        sid = m["station_id"]
        elev_m = pd.to_numeric(m.get("elev_m"), errors="coerce")
        elev_ft = float(elev_m) * 3.28084 if pd.notna(elev_m) else np.nan
        sy, ey = station_year_ranges.get(sid, (np.nan, np.nan))
        rng = (ey - sy + 1) if pd.notna(sy) and pd.notna(ey) else np.nan
        rows.append({
            "Station": sid,
            "Organization": m.get("Network", ""),
            "LAT": m.get("LAT"),
            "LONG": m.get("LONG"),
            "elev_ft": elev_ft,
            "elev_src": "HCDP",
            "start_yr": sy,
            "end_yr": ey,
            "range": rng,
            "station_id": m.get("station_id_raw", ""),
            "source": "HCDP",
            "source_unit": "mm",
            "source_freq": "daily",
            "Island": m.get("Island", ""),
        })
    meta_out = pd.DataFrame(rows, columns=AS_META_COLUMNS)
    meta_out = meta_out.dropna(subset=["LAT", "LONG"])
    HI_STATION_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta_out.to_csv(HI_STATION_META_PATH, index=False)
    print(f"  Wrote {len(meta_out)} stations to {HI_STATION_META_PATH.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
