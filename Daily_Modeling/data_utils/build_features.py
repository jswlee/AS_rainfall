"""
Build reanalysis patches and DEM patches for all station-day pairs.

This module wraps the same logic used in ML_Data_Preprocessing but is
self-contained so Daily_Modeling can live in its own repo.
"""

from pathlib import Path
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds

from Daily_Modeling import config

# ===================================================================
# Reanalysis feature building
# ===================================================================

def _get_nc_path(var_cfg: dict) -> Path:
    """Resolve the NetCDF file path for a variable config entry."""
    if "custom_file_daily" in var_cfg:
        return config.REANALYSIS_DIR / var_cfg["custom_file_daily"]
    base = config.VARIABLE_MAPPING[var_cfg["variable"]]
    return config.REANALYSIS_DIR / f"{base}.day.mean.nc"


def _detect_time_dim(ds: xr.Dataset) -> str:
    for name in ("valid_time", "time"):
        if name in ds.dims or name in ds.coords:
            return name
    raise KeyError("No time dimension found in dataset")


def _detect_lat_lon(ds: xr.Dataset) -> Tuple[str, str]:
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    return lat_name, lon_name


def _detect_level_dim(ds: xr.Dataset) -> Optional[str]:
    """Detect the vertical / pressure-level dimension name."""
    for name in ("pressure_level", "level", "lev", "plev"):
        if name in ds.dims:
            return name
    return None

def load_reanalysis_datasets() -> Dict[str, xr.Dataset]:
    """Load all unique daily reanalysis NetCDF files into memory.

    Keys are the *absolute file path string*, so each physical file is loaded
    only once even if multiple derived variables reference it.
    """
    loaded: Dict[str, xr.Dataset] = {}

    # Collect all unique NC paths (including primary files needed for multiply)
    nc_paths_needed: set = set()
    for _, cfg in config.DAILY_VARIABLE_CONFIGS.items():
        op = cfg.get("operation")
        if op == "divergence":
            # Fix E: divergence uses u_variable + v_variable, not 'variable'
            u_file = config.VARIABLE_MAPPING[cfg["u_variable"]]
            v_file = config.VARIABLE_MAPPING[cfg["v_variable"]]
            nc_paths_needed.add(str(config.REANALYSIS_DIR / f"{u_file}.day.mean.nc"))
            nc_paths_needed.add(str(config.REANALYSIS_DIR / f"{v_file}.day.mean.nc"))
        elif "depends_on" in cfg:
            # Multiply ops need the base file
            base_file_name = config.VARIABLE_MAPPING[cfg["variable"]]
            nc_paths_needed.add(str(config.REANALYSIS_DIR / f"{base_file_name}.day.mean.nc"))
        else:
            nc_paths_needed.add(str(_get_nc_path(cfg)))

    for nc_str in sorted(nc_paths_needed):
        nc = Path(nc_str)
        if not nc.exists():
            print(f"  WARNING: missing {nc}")
            continue
        ds = xr.open_dataset(nc)
        ds.load()
        loaded[nc_str] = ds
        print(f"  Loaded {nc.name} -> dims={dict(ds.sizes)}, vars={list(ds.data_vars)}")

    print(f"Loaded {len(loaded)} reanalysis files")
    return loaded


# -----------------------------------------------------------------------
# Pre-extracted numpy cube for fast indexing
# -----------------------------------------------------------------------
class _NumpyCube:
    """Pre-extract a (time, lat, lon) or (time, level, lat, lon) numpy array
    from an xarray Dataset for fast pure-numpy lookups."""

    def __init__(self, ds: xr.Dataset):
        tdim = _detect_time_dim(ds)
        lat_n, lon_n = _detect_lat_lon(ds)
        self.lats = ds[lat_n].values.astype(np.float64)
        self.lons = ds[lon_n].values.astype(np.float64)
        times = pd.DatetimeIndex(ds[tdim].values)
        self.date2idx = {t.date(): i for i, t in enumerate(times)}
        dvar = list(ds.data_vars)[0]
        lev_dim = _detect_level_dim(ds)
        if lev_dim is not None and lev_dim in ds[dvar].dims:
            self.levels = ds[lev_dim].values.astype(np.float64)
            self.data = ds[dvar].values.astype(np.float32)  # (T, L, lat, lon)
            self.has_levels = True
        else:
            self.levels = None
            self.data = ds[dvar].values.astype(np.float32)  # (T, lat, lon)
            self.has_levels = False

    def _level_idx(self, level: int) -> int:
        return int(np.argmin(np.abs(self.levels - level)))

    def get_field(self, dt, level: Optional[int] = None) -> Optional[np.ndarray]:
        """Return (lat, lon) 2-D array or None if date missing."""
        idx = self.date2idx.get(dt)
        if idx is None:
            return None
        if self.has_levels and level is not None:
            li = self._level_idx(level)
            return self.data[idx, li]  # (lat, lon)
        elif self.has_levels:
            return self.data[idx, 0]
        else:
            return self.data[idx]      # (lat, lon)


def build_reanalysis_patches(
    station_metadata: Dict[str, dict],
    station_days: Dict[str, List[Tuple[int, int, int]]],
    nc_cache: Dict[str, xr.Dataset],
    patch_size: int = config.REANALYSIS_PATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build (N, C, H, W) reanalysis patches for every station-day.

    Uses pre-extracted numpy arrays for fast indexing.
    Returns (patches, stations, years, months, days, var_names).
    """
    from datetime import date

    var_cfgs = config.DAILY_VARIABLE_CONFIGS
    var_order = config.DAILY_VARIABLE_NAMES

    # Pre-extract all NC files into NumpyCubes
    print("Pre-extracting numpy cubes ...")
    cubes: Dict[str, _NumpyCube] = {}
    for nc_str, ds in nc_cache.items():
        cubes[nc_str] = _NumpyCube(ds)
    print(f"  {len(cubes)} cubes ready")

    # Pre-compute station -> (lat_idx, lon_idx) per cube (nearest grid-point)
    station_grid: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for sname, smeta in station_metadata.items():
        lat, lon = smeta["latitude"], smeta["longitude"]
        grid = {}
        for nc_str, cube in cubes.items():
            ci = int(np.argmin(np.abs(cube.lats - lat)))
            cj = int(np.argmin(np.abs(cube.lons - lon)))
            grid[nc_str] = (ci, cj)
        station_grid[sname] = grid

    # Build a channel spec: for each derived variable, record what to do
    # so the inner loop avoids dict lookups and string comparisons
    channel_specs = []
    for vname in var_order:
        cfg = var_cfgs[vname]
        op = cfg.get("operation")
        if op == "diff":
            nc = str(_get_nc_path(cfg))
            channel_specs.append(("diff", nc, cfg["levels"][0], cfg["levels"][1]))
        elif op == "multiply":
            primary_nc = str(config.REANALYSIS_DIR / f"{config.VARIABLE_MAPPING[cfg['variable']]}.day.mean.nc")
            hum_cfg = var_cfgs[cfg["multiply_with"]]
            hum_nc = str(_get_nc_path(hum_cfg))
            lev = cfg.get("level")
            channel_specs.append(("multiply", primary_nc, hum_nc, lev))
        elif op == "divergence":
            # Fix E: horizontal wind divergence du/dx + dv/dy
            u_nc = str(config.REANALYSIS_DIR / f"{config.VARIABLE_MAPPING[cfg['u_variable']]}.day.mean.nc")
            v_nc = str(config.REANALYSIS_DIR / f"{config.VARIABLE_MAPPING[cfg['v_variable']]}.day.mean.nc")
            channel_specs.append(("divergence", u_nc, cfg["u_level"], v_nc, cfg["v_level"]))
        else:
            nc = str(_get_nc_path(cfg))
            lev = cfg.get("level")
            channel_specs.append(("simple", nc, lev))

    # Verify all needed cubes exist
    missing_cubes = set()
    for spec in channel_specs:
        if spec[0] == "diff":
            if spec[1] not in cubes:
                missing_cubes.add(spec[1])
        elif spec[0] == "multiply":
            if spec[1] not in cubes:
                missing_cubes.add(spec[1])
            if spec[2] not in cubes:
                missing_cubes.add(spec[2])
        elif spec[0] == "divergence":
            if spec[1] not in cubes:
                missing_cubes.add(spec[1])
            if spec[3] not in cubes:
                missing_cubes.add(spec[3])
        else:
            if spec[1] not in cubes:
                missing_cubes.add(spec[1])
    if missing_cubes:
        print(f"  WARNING: missing cubes for: {missing_cubes}")

    half = patch_size // 2
    n_channels = len(var_order)
    entries, meta_s, meta_y, meta_m, meta_d = [], [], [], [], []
    total_attempts = 0
    skipped = 0

    for station, days_list in sorted(station_days.items()):
        grids = station_grid[station]
        for (y, m, d) in days_list:
            total_attempts += 1
            dt = date(y, m, d)
            patch_buf = np.empty((n_channels, patch_size, patch_size), dtype=np.float32)
            ok = True

            for ci_ch, spec in enumerate(channel_specs):
                if spec[0] == "diff":
                    _, nc, lev0, lev1 = spec
                    cube = cubes.get(nc)
                    if cube is None:
                        ok = False; break
                    f0 = cube.get_field(dt, level=lev0)
                    f1 = cube.get_field(dt, level=lev1)
                    if f0 is None or f1 is None:
                        ok = False; break
                    field = f0 - f1
                    gi, gj = grids[nc]
                elif spec[0] == "multiply":
                    _, primary_nc, hum_nc, lev = spec
                    primary_cube = cubes.get(primary_nc)
                    hcube = cubes.get(hum_nc)
                    if primary_cube is None or hcube is None:
                        ok = False; break
                    wf = primary_cube.get_field(dt, level=lev)
                    hf = hcube.get_field(dt, level=lev)
                    if wf is None or hf is None:
                        ok = False; break
                    field = wf * hf
                    gi, gj = grids[primary_nc]
                elif spec[0] == "divergence":
                    # Fix E: du/dx + dv/dy via central finite differences on the grid
                    _, u_nc, u_lev, v_nc, v_lev = spec
                    ucube = cubes.get(u_nc)
                    vcube = cubes.get(v_nc)
                    if ucube is None or vcube is None:
                        ok = False; break
                    uf = ucube.get_field(dt, level=u_lev)
                    vf = vcube.get_field(dt, level=v_lev)
                    if uf is None or vf is None:
                        ok = False; break
                    # Central differences: pad edges with forward/backward difference
                    du_dx = np.gradient(uf, axis=1)   # d/d(col) ~ d/dx
                    dv_dy = np.gradient(vf, axis=0)   # d/d(row) ~ d/y (sign: N->S rows)
                    field = du_dx + dv_dy
                    gi, gj = grids[u_nc]
                else:
                    _, nc, lev = spec
                    cube = cubes.get(nc)
                    if cube is None:
                        ok = False; break
                    field = cube.get_field(dt, level=lev)
                    if field is None:
                        ok = False; break
                    gi, gj = grids[nc]

                # Extract spatial patch via numpy slicing
                i0 = max(gi - half, 0)
                j0 = max(gj - half, 0)
                p = field[i0:i0 + patch_size, j0:j0 + patch_size]
                if p.shape == (patch_size, patch_size):
                    patch_buf[ci_ch] = p
                else:
                    patch_buf[ci_ch] = np.nan
                    patch_buf[ci_ch, :p.shape[0], :p.shape[1]] = p

            if not ok:
                skipped += 1
                continue
            entries.append(patch_buf.copy())
            meta_s.append(station)
            meta_y.append(y)
            meta_m.append(m)
            meta_d.append(d)

            if total_attempts % 10000 == 0:
                print(f"  Processed {total_attempts} station-days, kept {len(entries)} ...")

    print(f"Built {len(entries)}/{total_attempts} reanalysis patches "
          f"({skipped} skipped)")

    patches = np.stack(entries, axis=0).astype(np.float32) if entries else np.empty((0,))
    return (
        patches,
        np.array(meta_s, dtype=object),
        np.array(meta_y, dtype=np.int32),
        np.array(meta_m, dtype=np.int32),
        np.array(meta_d, dtype=np.int32),
        var_order,
    )


# ===================================================================
# DEM patch building
# ===================================================================

def extract_dem_patch(
    dem_src,
    lon: float,
    lat: float,
    patch_size: int,
    km_per_cell: float,
) -> np.ndarray:
    """Extract a DEM patch from an open rasterio src.

    Each spatial cell covers a km_per_cell × km_per_cell area; the value stored
    is the mean of all valid pixels in that area (block averaging / coarsening).
    Cells that fall outside the DEM extent (ocean) are set to -1.0.

    For 3-band files (elevation, slope, aspect), aspect (degrees) is split into
    sin(aspect) and cos(aspect) after averaging, giving 4 output channels:
      band 0: elevation (m)
      band 1: slope (degrees)
      band 2: sin(aspect)
      band 3: cos(aspect)

    For single-band files, returns shape (1, patch_size, patch_size).
    """
    file_bands = dem_src.count
    # Output channels: aspect → 2 channels (sin, cos) for 3-band files
    n_out = file_bands + 1 if file_bands >= 3 else file_bands
    half = patch_size // 2

    if dem_src.crs is None:
        raise ValueError("DEM raster has no CRS; cannot transform station lon/lat")

    step_m = float(km_per_cell) * 1000.0

    # Raw patch for file bands; we'll expand aspect afterwards
    patch = np.full((file_bands, patch_size, patch_size), -1.0, dtype=np.float32)

    x0, y0 = transform("EPSG:4326", dem_src.crs, [lon], [lat])
    x0 = float(x0[0])
    y0 = float(y0[0])

    for pi in range(patch_size):
        for pj in range(patch_size):
            east_m = (pj - half) * step_m
            north_m = (half - pi) * step_m
            if bool(getattr(dem_src.crs, "is_geographic", False)):
                lat_scale = 111320.0
                lon_scale = max(111320.0 * math.cos(math.radians(float(lat))), 1e-6)
                center_lon = float(lon) + (east_m / lon_scale)
                center_lat = float(lat) + (north_m / lat_scale)
                x, y = transform("EPSG:4326", dem_src.crs, [center_lon], [center_lat])
                x = float(x[0])
                y = float(y[0])
                half_lon = 0.5 * step_m / lon_scale
                half_lat = 0.5 * step_m / lat_scale
                left, bottom = center_lon - half_lon, center_lat - half_lat
                right, top = center_lon + half_lon, center_lat + half_lat
                wx, wy = transform("EPSG:4326", dem_src.crs, [left, right], [bottom, top])
                xmin, xmax = float(min(wx)), float(max(wx))
                ymin, ymax = float(min(wy)), float(max(wy))
            else:
                x = x0 + east_m
                y = y0 + north_m
                xmin = x - 0.5 * step_m
                xmax = x + 0.5 * step_m
                ymin = y - 0.5 * step_m
                ymax = y + 0.5 * step_m
            try:
                r, c = dem_src.index(x, y)
                r, c = int(r), int(c)
                if 0 <= r < dem_src.height and 0 <= c < dem_src.width:
                    window = from_bounds(xmin, ymin, xmax, ymax, transform=dem_src.transform)
                    # Read all bands at once: shape (file_bands, H, W)
                    data_all = dem_src.read(window=window, boundless=True, masked=True)
                    for b in range(file_bands):
                        values = np.asarray(data_all[b].filled(np.nan), dtype=np.float32)
                        if dem_src.nodata is not None:
                            values = np.where(values == dem_src.nodata, np.nan, values)
                        # Band 0 (elevation): require >= 0 for valid land
                        # Band 1+ (slope, aspect): any finite value is valid
                        if b == 0:
                            values = np.where(np.isfinite(values) & (values >= 0), values, np.nan)
                        else:
                            values = np.where(np.isfinite(values), values, np.nan)
                        if np.isfinite(values).any():
                            patch[b, pi, pj] = float(np.nanmean(values))
                        else:
                            patch[b, pi, pj] = -1.0
            except Exception:
                pass

    if file_bands < 3:
        return patch  # single-band: return as-is (1, H, W)

    # Convert aspect (band 2, degrees) → sin + cos, keeping ocean sentinel intact
    aspect_deg = patch[2]  # (H, W)
    aspect_rad = np.deg2rad(aspect_deg)
    ocean_mask = aspect_deg <= -1.0

    sin_asp = np.where(ocean_mask, -1.0, np.sin(aspect_rad)).astype(np.float32)
    cos_asp = np.where(ocean_mask, -1.0, np.cos(aspect_rad)).astype(np.float32)

    # Stack: elev (0), slope (1), sin_aspect (2), cos_aspect (3)
    out = np.empty((4, patch_size, patch_size), dtype=np.float32)
    out[0] = patch[0]   # elevation
    out[1] = patch[1]   # slope
    out[2] = sin_asp
    out[3] = cos_asp
    return out


def build_dem_patches(
    station_metadata: Dict[str, dict],
    dem_path: Optional[Path] = None,
    local_cfg: Optional[dict] = None,
    regional_cfg: Optional[dict] = None,
) -> Dict[str, dict]:
    """Build local + regional DEM patches for each station.

    When ``dem_path`` is None the function uses ``config.get_dem_path_for_station``
    to select the correct DEM per station (e.g. AS vs HI in aggregate mode).
    Pass an explicit ``dem_path`` to force a single DEM for all stations.

    Args:
        dem_path:     Override DEM for all stations.  When None, each station
                      is routed to the DEM matching its region prefix.
        local_cfg:    dict with 'patch_size' and 'km_per_cell'.
                      Defaults to ``config.DEM_PATCH_CONFIG["local"]``.
        regional_cfg: same, defaults to ``config.DEM_PATCH_CONFIG["regional"]``.

    Returns {station: {"local": ndarray, "regional": ndarray}}.
    """
    if local_cfg is None:
        local_cfg = config.DEM_PATCH_CONFIG["local"]
    if regional_cfg is None:
        regional_cfg = config.DEM_PATCH_CONFIG["regional"]

    patches: Dict[str, dict] = {}

    if dem_path is not None:
        # Single explicit DEM for all stations (AS-only or HI-only runs)
        dem_groups: Dict[Path, list] = {Path(dem_path): sorted(station_metadata.items())}
    else:
        # Group stations by their region-specific DEM path
        dem_groups = {}
        for name, meta in sorted(station_metadata.items()):
            p = config.get_dem_path_for_station(name)
            dem_groups.setdefault(p, []).append((name, meta))

    for grp_path, station_items in dem_groups.items():
        with rasterio.open(str(grp_path)) as src:
            for name, meta in station_items:
                local = extract_dem_patch(
                    src, meta["longitude"], meta["latitude"],
                    local_cfg["patch_size"], local_cfg["km_per_cell"],
                )
                regional = extract_dem_patch(
                    src, meta["longitude"], meta["latitude"],
                    regional_cfg["patch_size"], regional_cfg["km_per_cell"],
                )
                patches[name] = {"local": local, "regional": regional}

    print(f"Built DEM patches for {len(patches)} stations  "
          f"local={local_cfg['patch_size']}x{local_cfg['patch_size']}@{local_cfg['km_per_cell']}km  "
          f"regional={regional_cfg['patch_size']}x{regional_cfg['patch_size']}@{regional_cfg['km_per_cell']}km")
    return patches
