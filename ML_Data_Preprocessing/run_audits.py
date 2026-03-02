#!/usr/bin/env python3
"""Run lightweight, non-visual data audits.

This is a convenience entry point that runs:
- climate NetCDF audit (daily full 1980-2024)
- rainfall CSV coverage audit (daily + monthly)

Outputs go under ML_Data_Preprocessing/output/audits/.

No plots are generated.
"""

import os
import json
from typing import Dict

import numpy as np
import pandas as pd

import ML_Data_Preprocessing.config as config
from ML_Data_Preprocessing.audit_climate_data import main as audit_climate_main


def _audit_rainfall_coverage() -> Dict:
    out = {
        "monthly": {},
        "daily": {},
    }

    # Monthly rainfall coverage
    monthly_dir = str(config.MONTHLY_RAINFALL_DATA_DIR)
    if os.path.isdir(monthly_dir):
        files = [f for f in os.listdir(monthly_dir) if f.lower().endswith("_monthly.csv")]
        years_all = []
        missing_files = 0
        for fn in files:
            path = os.path.join(monthly_dir, fn)
            try:
                df = pd.read_csv(path)
                if 'year_month' not in df.columns:
                    continue
                ym = df['year_month'].astype(str).str.split('-', expand=True)
                if ym.shape[1] < 2:
                    continue
                years = pd.to_numeric(ym[0], errors='coerce').dropna().astype(int)
                years_all.extend(years.tolist())
            except Exception:
                missing_files += 1

        out['monthly'] = {
            "dir": monthly_dir,
            "n_files": len(files),
            "n_files_failed": int(missing_files),
            "year_min": int(np.min(years_all)) if years_all else None,
            "year_max": int(np.max(years_all)) if years_all else None,
        }
    else:
        out['monthly'] = {"dir": monthly_dir, "error": "dir_not_found"}

    # Daily rainfall coverage
    daily_dir = str(config.DAILY_RAINFALL_DATA_DIR)
    if os.path.isdir(daily_dir):
        files = [f for f in os.listdir(daily_dir) if f.lower().endswith(".csv")]
        years_all = []
        n_rows_total = 0
        n_valid_total = 0
        n_failed = 0

        for fn in files:
            path = os.path.join(daily_dir, fn)
            try:
                df = pd.read_csv(path)
                if 'datetime' not in df.columns:
                    continue
                dt = pd.to_datetime(df['datetime'], errors='coerce')
                years = dt.dt.year.dropna().astype(int)
                years_all.extend(years.tolist())

                n_rows_total += int(len(df))
                if 'precip_in' in df.columns:
                    valid = df['precip_in'].notna() & (pd.to_numeric(df['precip_in'], errors='coerce') >= 0)
                    n_valid_total += int(valid.sum())
            except Exception:
                n_failed += 1

        out['daily'] = {
            "dir": daily_dir,
            "n_files": len(files),
            "n_files_failed": int(n_failed),
            "year_min": int(np.min(years_all)) if years_all else None,
            "year_max": int(np.max(years_all)) if years_all else None,
            "n_rows_total": int(n_rows_total),
            "n_valid_precip_rows": int(n_valid_total),
        }
    else:
        out['daily'] = {"dir": daily_dir, "error": "dir_not_found"}

    return out


def main() -> int:
    out_dir = os.path.join(str(config.OUTPUT_DIR), "audits")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Climate audit (writes its own outputs)
    audit_climate_main()

    # 2) Rainfall coverage audit
    rainfall = _audit_rainfall_coverage()
    rainfall_path = os.path.join(out_dir, "rainfall_coverage_audit.json")
    with open(rainfall_path, "w") as f:
        json.dump(rainfall, f, indent=2)

    print(f"Wrote: {rainfall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
