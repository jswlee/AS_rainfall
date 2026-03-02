#!/usr/bin/env python3
"""Audit raw climate NetCDF inputs.

Produces a lightweight audit report for the raw daily climate inputs:
raw_data/climate_variables_daily_FULL_1980-2024/*.nc

Outputs:
- ML_Data_Preprocessing/output/audits/climate_audit_daily_full_1980-2024.csv
- ML_Data_Preprocessing/output/audits/climate_audit_daily_full_1980-2024.json
"""

import os
import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import xarray as xr

import ML_Data_Preprocessing.config as config


@dataclass
class FileAudit:
    filename: str
    variable_name: str
    time_dim: str
    n_time: int
    start_time: str
    end_time: str
    lat_dim: str
    lon_dim: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    dtype: str
    n_nans: int
    nan_fraction: float
    min: float
    max: float
    mean: float
    std: float


def _infer_main_data_var(ds: xr.Dataset) -> str:
    # Prefer first data var.
    if len(ds.data_vars) == 0:
        raise ValueError("No data_vars found")
    return list(ds.data_vars.keys())[0]


def _infer_time_dim(da: xr.DataArray) -> str:
    for cand in ("valid_time", "time"):
        if cand in da.dims:
            return cand
    raise ValueError(f"No supported time dim found in dims={da.dims}")


def _infer_lat_lon_dims(da: xr.DataArray) -> tuple[str, str]:
    if "latitude" in da.dims and "longitude" in da.dims:
        return "latitude", "longitude"
    if "lat" in da.dims and "lon" in da.dims:
        return "lat", "lon"
    raise ValueError(f"No supported lat/lon dims found in dims={da.dims}")


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def audit_file(path: str) -> FileAudit:
    with xr.open_dataset(path) as ds:
        var = _infer_main_data_var(ds)
        da = ds[var]

        # If there is a level/pressure dimension, select the first level for auditing.
        level_dim = None
        for cand in ("pressure_level", "level"):
            if cand in da.dims:
                level_dim = cand
                break
        if level_dim is not None:
            da = da.isel({level_dim: 0})

        time_dim = _infer_time_dim(da)
        lat_dim, lon_dim = _infer_lat_lon_dims(da)

        times = da[time_dim].values
        start_time = str(times[0]) if len(times) else ""
        end_time = str(times[-1]) if len(times) else ""

        lats = da[lat_dim].values
        lons = da[lon_dim].values

        # Sample statistics cheaply to avoid loading entire dataset into memory.
        # Compute stats on a coarse subsample (every ~30th timestep) for speed.
        stride = max(1, int(len(times) / 2000))  # cap to ~2000 time slices
        da_sub = da.isel({time_dim: slice(0, None, stride)})

        arr = da_sub.values
        arr = np.asarray(arr)

        n_total = int(arr.size)
        n_nans = int(np.isnan(arr).sum())
        nan_frac = float(n_nans / max(1, n_total))

        # nan-aware stats
        vmin = _safe_float(np.nanmin(arr))
        vmax = _safe_float(np.nanmax(arr))
        vmean = _safe_float(np.nanmean(arr))
        vstd = _safe_float(np.nanstd(arr))

        return FileAudit(
            filename=os.path.basename(path),
            variable_name=var,
            time_dim=time_dim,
            n_time=int(len(times)),
            start_time=start_time,
            end_time=end_time,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            lat_min=_safe_float(np.nanmin(lats)),
            lat_max=_safe_float(np.nanmax(lats)),
            lon_min=_safe_float(np.nanmin(lons)),
            lon_max=_safe_float(np.nanmax(lons)),
            dtype=str(arr.dtype),
            n_nans=n_nans,
            nan_fraction=nan_frac,
            min=vmin,
            max=vmax,
            mean=vmean,
            std=vstd,
        )


def _write_csv(rows: list[FileAudit], out_path: str) -> None:
    import csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def main() -> int:
    in_dir = config.REANALYSIS_DIR_DAILY
    if not os.path.isdir(in_dir):
        raise SystemExit(f"Input dir not found: {in_dir}")

    nc_files = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.lower().endswith(".nc")
    ]
    nc_files.sort()
    if not nc_files:
        raise SystemExit(f"No .nc files found in {in_dir}")

    out_dir = os.path.join(str(config.OUTPUT_DIR), "audits")
    out_csv = os.path.join(out_dir, "climate_audit_daily_full_1980-2024.csv")
    out_json = os.path.join(out_dir, "climate_audit_daily_full_1980-2024.json")

    rows: list[FileAudit] = []
    for p in nc_files:
        print(f"Auditing {os.path.basename(p)}")
        rows.append(audit_file(p))

    _write_csv(rows, out_csv)

    summary = {
        "input_dir": in_dir,
        "n_files": len(rows),
        "files": [asdict(r) for r in rows],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
