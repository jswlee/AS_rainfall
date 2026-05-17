"""
Combine per-station rainfall CSVs and station metadata across regions
(AS + HI) into raw_data/aggregate/.

  raw_data/aggregate/final_rainfall_per_station/<station>.csv
  raw_data/aggregate/station_locations.csv

The HI CSVs/metadata must already exist (run
``hawaii_preprocessing/scripts/01_convert_hcdp_to_station_csvs.py`` first).

Run from repo root:
    python aggregate/scripts/02_combine_rainfall.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

AS_RAINFALL_DIR = REPO_ROOT / "raw_data" / "AS" / "final_rainfall_per_station"
HI_RAINFALL_DIR = REPO_ROOT / "raw_data" / "HI" / "final_rainfall_per_station"
AS_META = REPO_ROOT / "raw_data" / "AS" / "station_locations.csv"
HI_META = REPO_ROOT / "raw_data" / "HI" / "station_locations.csv"

OUT_DIR = REPO_ROOT / "raw_data" / "aggregate"
OUT_RAINFALL_DIR = OUT_DIR / "final_rainfall_per_station"
OUT_META = OUT_DIR / "station_locations.csv"


def _copy_csvs(src: Path, dst: Path, region: str) -> int:
    if not src.exists():
        print(f"  [WARN] {src} does not exist; skipping {region}")
        return 0
    n = 0
    for csv in sorted(src.glob("*.csv")):
        target = dst / csv.name
        shutil.copy2(csv, target)
        n += 1
    print(f"  Copied {n} CSVs from {region} ({src.name})")
    return n


def main() -> int:
    OUT_RAINFALL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Rainfall CSVs ----
    print("Copying per-station rainfall CSVs ...")
    _copy_csvs(AS_RAINFALL_DIR, OUT_RAINFALL_DIR, "AS")
    _copy_csvs(HI_RAINFALL_DIR, OUT_RAINFALL_DIR, "HI")

    # ---- Station metadata ----
    print("Combining station metadata ...")
    frames = []
    for label, path in [("AS", AS_META), ("HI", HI_META)]:
        if not path.exists():
            print(f"  [WARN] {path} not found; skipping {label}")
            continue
        df = pd.read_csv(path)
        df["region"] = label
        frames.append(df)
        print(f"  {label}: {len(df)} stations")

    if not frames:
        print("ERROR: no metadata files found", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Detect duplicate Station names across regions and warn
    dup = combined[combined.duplicated(subset=["Station"], keep=False)]
    if not dup.empty:
        print(f"  [WARN] {len(dup)} duplicate Station names across regions:")
        print(dup[["Station", "region"]].to_string(index=False))

    combined = combined.drop_duplicates(subset=["Station"], keep="first")
    combined.to_csv(OUT_META, index=False)
    print(f"  Wrote {len(combined)} station rows -> {OUT_META}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
