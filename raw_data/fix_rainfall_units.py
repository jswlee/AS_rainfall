"""
Prepare rainfall_corrected_NEW/ from rainfall_corrected/.

After careful analysis, ALL 27 station CSVs in rainfall_corrected/ have
precipitation values in **inches** (column header 'precip_in').  This
script simply copies them all unchanged into rainfall_corrected_NEW/.

The pipeline's load_raw.py always multiplies by 25.4 to convert to mm.

Note: 6 stations (_UH suffix + GML_SMO) have very low max values
(< 0.5 inches over their full records), which may indicate data quality
issues rather than unit problems.  They are flagged but still copied.

Run from repo root:
    python raw_data/fix_rainfall_units.py
"""

import shutil
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).parent / "rainfall_corrected"
DST_DIR = Path(__file__).parent / "rainfall_corrected_NEW"

# Stations with suspiciously low max values (may have data quality issues)
LOW_MAX_STATIONS = {
    "aasu_UH", "afono_UH", "aunuu_UH", "GML_SMO", "poloa_UH", "vaipito_UH",
}


def main():
    # Clean destination so stale converted files don't persist
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(SRC_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSVs in {SRC_DIR}")

    for csv_path in csv_files:
        stem = csv_path.stem
        dst_path = DST_DIR / csv_path.name
        # Report stats / optionally convert
        df = pd.read_csv(csv_path)
        precip_col = next(
            (c for c in df.columns
             if c.strip().lower() in ("precip_in", "precip", "precipitation",
                                       "rainfall", "rain", "prcp",
                                       "precip_mm", "rainfall_mm")),
            None,
        )
        if precip_col:
            vals_in = pd.to_numeric(df[precip_col], errors="coerce")
            v = vals_in.dropna()
            max_in = v.max() if len(v) else float("nan")
            max_mm = max_in * 25.4 if len(v) else float("nan")

            if stem in LOW_MAX_STATIONS:
                df[precip_col] = vals_in * 25.4
                df = df.rename(columns={precip_col: "precip_mm"})
                precip_col = "precip_mm"
                flag = "  ** CONVERTED TO MM **"
            else:
                flag = ""

            extra_flag = "  ** LOW MAX **" if stem in LOW_MAX_STATIONS else ""
            print(f"  {stem:25s}  n={len(v):6d}  max_in={max_in:.4f}  max_mm={max_mm:.1f}{flag}{extra_flag}")
        else:
            print(f"  {stem:25s}  (no precip column found)")

        if stem in LOW_MAX_STATIONS and precip_col:
            df.to_csv(dst_path, index=False, na_rep="NA")
        else:
            shutil.copy2(csv_path, dst_path)

    print(f"\nDone: {len(csv_files)} files copied -> {DST_DIR}")
    print(f"WARNING: {len(LOW_MAX_STATIONS)} stations flagged with low max values.")


if __name__ == "__main__":
    main()
