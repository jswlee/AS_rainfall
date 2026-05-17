"""
Combine NCEP/NCAR reanalysis NetCDF files for AS + HI into one set of files
under raw_data/aggregate/reanalysis_data/.

For every NetCDF file present in the AS reanalysis directory we look for a
file with the same name in the HI reanalysis directory.  The two datasets are
merged via ``xarray.combine_by_coords`` so that:
  - if their lat/lon grids are disjoint, a wider tiled grid is produced;
  - if they overlap (or are identical), duplicate coords are dropped and the
    AS values win on conflict.

Run from repo root:
    python aggregate/scripts/01_combine_reanalysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
AS_DIR = REPO_ROOT / "raw_data" / "AS" / "climate_variables_daily_1980-2024"
HI_DIR = REPO_ROOT / "raw_data" / "HI" / "hawaii_climate_variables_daily_1980-2024"
OUT_DIR = REPO_ROOT / "raw_data" / "aggregate" / "reanalysis_data"


def _detect_lat_lon(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    return lat_name, lon_name


def _detect_time(ds: xr.Dataset) -> str:
    for cand in ("valid_time", "time"):
        if cand in ds.dims:
            return cand
    raise KeyError(f"No time dimension found; have {list(ds.dims)}")


def _normalise_lon(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """Ensure longitudes share a common convention.  NCEP grids are typically
    0..360; if one dataset uses -180..180 we shift it to 0..360 to match."""
    lons = ds[lon_name].values
    if (lons < 0).any():
        ds = ds.assign_coords({lon_name: ((lons + 360) % 360)})
        ds = ds.sortby(lon_name)
    return ds


def _drop_dup_coord(ds: xr.Dataset, dim: str, label: str) -> xr.Dataset:
    """Keep only the first occurrence of each coordinate value along *dim*."""
    vals = ds[dim].values
    n = len(vals)
    # pandas handles datetime64/object/float uniformly
    import pandas as _pd
    mask = ~_pd.Index(vals).duplicated(keep="first")
    if mask.sum() != n:
        print(f"     [{label}] dropping {n - int(mask.sum())} duplicate {dim} entries")
        ds = ds.isel({dim: np.where(mask)[0]})
    return ds


def _combine_pair(as_path: Path, hi_path: Path, out_path: Path) -> None:
    """Combine AS and HI reanalysis files into a single NetCDF.

    Strategy
    --------
    1.  Intersect the two datasets along the time dimension (drops the extra
        boundary day in one file).  After this both datasets share an identical
        time coordinate.
    2.  Outer-merge in space (lat / lon).  AS and HI cover disjoint
        lat-boxes (e.g. -20..-10 and 15..25), so the combined grid has the
        union of both with NaNs in the unused quadrant.  Downstream nearest-
        grid lookup picks each station's home region, so the NaN holes are
        never accessed.
    """
    print(f"  -> {out_path.name}")
    as_ds = xr.open_dataset(as_path)
    hi_ds = xr.open_dataset(hi_path)

    lat_n, lon_n = _detect_lat_lon(as_ds)
    t_n = _detect_time(as_ds)

    as_ds = _normalise_lon(as_ds, lon_n)
    hi_ds = _normalise_lon(hi_ds, lon_n)

    # ---- De-duplicate coordinates (defensive: prior concats may have introduced dupes) ----
    for dim in (t_n, lat_n, lon_n):
        as_ds = _drop_dup_coord(as_ds, dim, "AS")
        hi_ds = _drop_dup_coord(hi_ds, dim, "HI")

    # ---- Info ----
    print(f"     AS  lat=[{as_ds[lat_n].values.min():.2f},{as_ds[lat_n].values.max():.2f}]"
          f" lon=[{as_ds[lon_n].values.min():.2f},{as_ds[lon_n].values.max():.2f}]"
          f"  sizes={dict(as_ds.sizes)}")
    print(f"     HI  lat=[{hi_ds[lat_n].values.min():.2f},{hi_ds[lat_n].values.max():.2f}]"
          f" lon=[{hi_ds[lon_n].values.min():.2f},{hi_ds[lon_n].values.max():.2f}]"
          f"  sizes={dict(hi_ds.sizes)}")

    # ---- 1. Time intersection ----
    as_times = as_ds[t_n].values
    hi_times = hi_ds[t_n].values
    common, _, _ = np.intersect1d(as_times, hi_times, return_indices=True)
    if len(common) == 0:
        raise RuntimeError("No overlapping timestamps between AS and HI files")
    print(f"     time: AS={len(as_times)} HI={len(hi_times)} -> common={len(common)}")
    as_ds = as_ds.sel({t_n: common})
    hi_ds = hi_ds.sel({t_n: common})

    # ---- 2. Manual spatial assembly (much faster than xr.merge outer join) ----
    # Build union lat / lon grids
    union_lats = np.union1d(as_ds[lat_n].values, hi_ds[lat_n].values)
    union_lons = np.union1d(as_ds[lon_n].values, hi_ds[lon_n].values)
    print(f"     union lat={len(union_lats)}  union lon={len(union_lons)}"
          f"  -> output shape ({len(common)}, {len(union_lats)}, {len(union_lons)})")

    lat_to_idx = {float(v): i for i, v in enumerate(union_lats)}
    lon_to_idx = {float(v): i for i, v in enumerate(union_lons)}

    def _idx(src_ds, dim, mapping):
        return np.array([mapping[float(v)] for v in src_ds[dim].values], dtype=np.int64)

    as_lat_i = _idx(as_ds, lat_n, lat_to_idx)
    as_lon_i = _idx(as_ds, lon_n, lon_to_idx)
    hi_lat_i = _idx(hi_ds, lat_n, lat_to_idx)
    hi_lon_i = _idx(hi_ds, lon_n, lon_to_idx)

    out_vars = {}
    coords = {t_n: as_ds[t_n], lat_n: union_lats, lon_n: union_lons}

    for vname in as_ds.data_vars:
        if vname not in hi_ds.data_vars:
            print(f"     [WARN] var '{vname}' missing in HI; skipping")
            continue
        as_da = as_ds[vname]
        hi_da = hi_ds[vname]
        dims = as_da.dims
        # We expect (t_n, [level,] lat_n, lon_n)
        if lat_n not in dims or lon_n not in dims or t_n not in dims:
            print(f"     [WARN] var '{vname}' has unexpected dims {dims}; skipping")
            continue

        # Determine optional middle (level) dim
        other_dims = [d for d in dims if d not in (t_n, lat_n, lon_n)]
        # Target shape: (T, [L,] LAT, LON)
        target_shape = [len(coords[t_n]), *(as_da.sizes[d] for d in other_dims),
                        len(union_lats), len(union_lons)]
        print(f"     building '{vname}' dims={dims} -> shape={tuple(target_shape)}")

        out = np.full(target_shape, np.nan, dtype=np.float32)

        # Reorder source arrays to (T, [L,] lat, lon)
        ordered = (t_n, *other_dims, lat_n, lon_n)
        as_arr = as_da.transpose(*ordered).values.astype(np.float32, copy=False)
        hi_arr = hi_da.transpose(*ordered).values.astype(np.float32, copy=False)

        # Place AS slab, then HI slab.  Latitudes are disjoint between
        # the two regions, so there is no overlap in (lat,lon) indices that
        # would cause one region to overwrite the other.
        out[..., as_lat_i[:, None], as_lon_i[None, :]] = as_arr
        out[..., hi_lat_i[:, None], hi_lon_i[None, :]] = hi_arr

        out_vars[vname] = xr.DataArray(
            out,
            dims=ordered,
            coords={d: (as_ds[d] if d in as_ds.coords else hi_ds[d]) for d in other_dims}
                    | {t_n: coords[t_n], lat_n: union_lats, lon_n: union_lons},
            attrs=dict(as_da.attrs),
            name=vname,
        )

    combined = xr.Dataset(out_vars, attrs={**hi_ds.attrs, **as_ds.attrs})
    combined = combined.sortby([lat_n, lon_n])

    print(f"     -> combined lat=[{combined[lat_n].values.min():.2f},{combined[lat_n].values.max():.2f}]"
          f" lon=[{combined[lon_n].values.min():.2f},{combined[lon_n].values.max():.2f}]"
          f"  sizes={dict(combined.sizes)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"     writing {out_path.name} ...")
    # No compression: NaN-heavy float32 arrays compress fine but zlib at this
    # size is the slowest step.  Skip it for speed; user can re-compress later.
    combined.to_netcdf(out_path)
    as_ds.close(); hi_ds.close(); combined.close()
    print(f"     done.")


def main() -> int:
    if not AS_DIR.exists():
        print(f"ERROR: missing {AS_DIR}", file=sys.stderr); return 1
    if not HI_DIR.exists():
        print(f"ERROR: missing {HI_DIR}", file=sys.stderr); return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    as_files = sorted(p for p in AS_DIR.iterdir() if p.suffix == ".nc")
    print(f"Found {len(as_files)} NetCDF files in {AS_DIR}")

    for as_path in as_files:
        hi_path = HI_DIR / as_path.name
        out_path = OUT_DIR / as_path.name
        if not hi_path.exists():
            print(f"  [WARN] no HI counterpart for {as_path.name}; skipping")
            continue
        if out_path.exists():
            print(f"  [SKIP] {out_path.name} already exists")
            continue
        try:
            _combine_pair(as_path, hi_path, out_path)
        except Exception as e:
            print(f"  [ERROR] {as_path.name}: {e}")

    print(f"Done. Wrote combined NetCDFs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
