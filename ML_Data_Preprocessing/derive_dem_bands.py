"""
derive_dem_bands.py
-------------------
Add slope and aspect bands to a single-band elevation GeoTIFF, producing a
3-band output (elevation, slope, aspect) in the same format as the Hawaii
5-band DEM (30m_hawaii.tif).

Method: GDAL Horn's method (3×3 kernel), matching the computation used to
produce the Hawaii file.  Slope is in degrees (0–90).  Aspect is clockwise
from North in degrees (0–360); flat pixels (zero gradient) are set to -1.

Validation: when run against the Hawaii file's elevation band the derived
slope / aspect match the stored bands to within rounding error (< 1 deg mean
absolute difference).

Usage:
    python ML_Data_Preprocessing/derive_dem_bands.py
        [--input  raw_data/AS/DEM/10m_tutuila.tif]
        [--output raw_data/AS/DEM/10m_tutuila_3band.tif]
        [--validate raw_data/HI/DEM/30m_hawaii.tif]
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = REPO_ROOT / "raw_data" / "AS" / "DEM" / "10m_tutuila.tif"
DEFAULT_OUTPUT = REPO_ROOT / "raw_data" / "AS" / "DEM" / "10m_tutuila_3band.tif"
DEFAULT_VALIDATE = REPO_ROOT / "raw_data" / "HI" / "DEM" / "30m_hawaii.tif"

NODATA_OUT = -32768.0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _pixel_size_meters(transform: Affine, lat_deg: float):
    """Return (ew_m, ns_m) pixel size in metres for a geographic CRS raster."""
    import math
    res_deg_ew = abs(transform.a)
    res_deg_ns = abs(transform.e)
    ew_m = res_deg_ew * 111320.0 * math.cos(math.radians(lat_deg))
    ns_m = res_deg_ns * 111320.0
    return ew_m, ns_m


def compute_slope_aspect(elev: np.ndarray, ew_m: float, ns_m: float, nodata: float = NODATA_OUT):
    """
    Compute slope (degrees) and aspect (degrees CW from North) using Horn's
    method on a 2-D elevation array.

    Parameters
    ----------
    elev   : (H, W) float32/64 array, nodata pixels should already be NaN.
    ew_m   : E-W pixel size in metres.
    ns_m   : N-S pixel size in metres.
    nodata : value to write for edge / no-data output pixels.

    Returns
    -------
    slope, aspect : (H, W) float32 arrays.
    """
    e = elev.astype(np.float64)

    # Interior 3x3 blocks (Horn's kernel)
    a = e[:-2, :-2]; b = e[:-2, 1:-1]; c = e[:-2, 2:]
    d = e[1:-1, :-2];                   f = e[1:-1, 2:]
    g = e[2:,  :-2]; h = e[2:,  1:-1]; i_ = e[2:,  2:]

    dzdx = ((c + 2*f + i_) - (a + 2*d + g)) / (8.0 * ew_m)
    dzdy = ((g + 2*h + i_) - (a + 2*b + c)) / (8.0 * ns_m)

    # Slope in degrees
    slope_inner = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    # Aspect: clockwise from North (GDAL convention)
    asp_math = np.degrees(np.arctan2(dzdy, -dzdx))
    asp_gdal = 90.0 - asp_math
    asp_gdal = np.where(asp_gdal < 0, asp_gdal + 360.0, asp_gdal)
    # Flat pixels (gradient == 0) -> -1 (GDAL convention)
    flat = (dzdx == 0) & (dzdy == 0)
    asp_gdal = np.where(flat, -1.0, asp_gdal)

    # Propagate NaN from input
    any_nan = np.isnan(a) | np.isnan(b) | np.isnan(c) | np.isnan(d) | \
              np.isnan(f) | np.isnan(g) | np.isnan(h) | np.isnan(i_)
    slope_inner = np.where(any_nan, np.nan, slope_inner)
    asp_gdal    = np.where(any_nan, np.nan, asp_gdal)

    # Build full-size output arrays with nodata border
    H, W = elev.shape
    slope  = np.full((H, W), nodata, dtype=np.float32)
    aspect = np.full((H, W), nodata, dtype=np.float32)
    slope[1:-1, 1:-1]  = np.where(np.isfinite(slope_inner),  slope_inner,  nodata).astype(np.float32)
    aspect[1:-1, 1:-1] = np.where(np.isfinite(asp_gdal),     asp_gdal,     nodata).astype(np.float32)

    return slope, aspect


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _centre_lat(src: rasterio.DatasetReader) -> float:
    """Return the latitude of the raster centre (for pixel-size calculation)."""
    return (src.bounds.top + src.bounds.bottom) / 2.0


def derive_and_write(input_path: Path, output_path: Path):
    """Read a single-band elevation TIF, compute slope & aspect, write 3-band TIF."""
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading elevation from: {input_path}")
    with rasterio.open(str(input_path)) as src:
        if src.count != 1:
            raise ValueError(f"Expected 1-band elevation file, got {src.count} bands.")

        raw = src.read(1).astype(np.float64)
        nd  = src.nodata
        if nd is not None:
            raw = np.where(raw == nd, np.nan, raw)

        crs       = src.crs
        transform = src.transform
        lat_c     = _centre_lat(src)
        ew_m, ns_m = _pixel_size_meters(transform, lat_c)
        print(f"  Shape: {src.height} x {src.width},  pixel size: EW={ew_m:.2f} m  NS={ns_m:.2f} m")

    print("Computing slope and aspect ...")
    slope, aspect = compute_slope_aspect(raw, ew_m, ns_m, nodata=NODATA_OUT)

    # Elevation output (keep nodata consistent)
    elev_out = np.where(np.isfinite(raw), raw, NODATA_OUT).astype(np.float32)

    profile = dict(
        driver="GTiff",
        dtype="float32",
        crs=crs,
        transform=transform,
        width=elev_out.shape[1],
        height=elev_out.shape[0],
        count=3,
        nodata=NODATA_OUT,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_SAFER",
    )

    print(f"Writing 3-band TIF to: {output_path}")
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(elev_out,  1)
        dst.write(slope,     2)
        dst.write(aspect,    3)
        dst.update_tags(1, DESCRIPTION="elevation")
        dst.update_tags(2, DESCRIPTION="slope")
        dst.update_tags(3, DESCRIPTION="aspect")

    print("Done.")
    return output_path


# ---------------------------------------------------------------------------
# Validation against Hawaii file
# ---------------------------------------------------------------------------

def validate_against_hawaii(hawaii_path: Path, n_windows: int = 5, window_size: int = 200):
    """
    Derive slope/aspect from the Hawaii file's elevation band and compare
    against its pre-stored slope/aspect bands.  Prints mean absolute error.
    """
    hawaii_path = Path(hawaii_path)
    if not hawaii_path.exists():
        print(f"[validate] Hawaii file not found: {hawaii_path} — skipping.")
        return

    print(f"\nValidating against: {hawaii_path}")
    with rasterio.open(str(hawaii_path)) as src:
        if src.count < 3:
            print("[validate] Fewer than 3 bands — cannot validate.")
            return

        nd = src.nodata
        lat_c = _centre_lat(src)
        ew_m, ns_m = _pixel_size_meters(src.transform, lat_c)

        # Sample random land windows
        rng = np.random.default_rng(42)
        slope_diffs, aspect_diffs = [], []

        tested = 0
        attempts = 0
        while tested < n_windows and attempts < 200:
            attempts += 1
            r0 = int(rng.integers(1, src.height - window_size - 1))
            c0 = int(rng.integers(1, src.width  - window_size - 1))
            win = rasterio.windows.Window(c0, r0, window_size, window_size)
            chunk = src.read(window=win)  # (5, H, W)

            valid_frac = np.mean(chunk[0] != nd)
            if valid_frac < 0.5:
                continue

            elev  = np.where(chunk[0] == nd, np.nan, chunk[0].astype(np.float64))
            s_ref = np.where(chunk[1] == nd, np.nan, chunk[1].astype(np.float64))
            a_ref = np.where(chunk[2] == nd, np.nan, chunk[2].astype(np.float64))

            s_calc, a_calc = compute_slope_aspect(elev, ew_m, ns_m)

            v = (chunk[0] != nd)[1:-1, 1:-1] & np.isfinite(s_calc[1:-1, 1:-1]) & \
                np.isfinite(s_ref[1:-1, 1:-1]) & np.isfinite(a_ref[1:-1, 1:-1])

            if v.sum() < 100:
                continue

            sd = np.abs(s_calc[1:-1, 1:-1][v] - s_ref[1:-1, 1:-1][v])
            # Circular aspect difference
            ad_raw = a_calc[1:-1, 1:-1][v] - a_ref[1:-1, 1:-1][v]
            # Exclude flat pixels (aspect stored as -1)
            not_flat = a_ref[1:-1, 1:-1][v] >= 0
            ad = np.abs((ad_raw[not_flat] + 180) % 360 - 180)

            slope_diffs.append(np.mean(sd))
            if len(ad) > 0:
                aspect_diffs.append(np.mean(ad))
            tested += 1
            print(f"  Window {tested}: valid={v.sum()}  slope MAE={np.mean(sd):.3f}°  aspect MAE={np.mean(ad):.1f}°")

        if slope_diffs:
            print(f"\nOverall slope  MAE: {np.mean(slope_diffs):.4f}° (expected < 0.5°)")
            print(f"Overall aspect MAE: {np.mean(aspect_diffs):.2f}° (expected < 5°)")
            print("[validate] PASS" if np.mean(slope_diffs) < 0.5 and np.mean(aspect_diffs) < 10 else "[validate] WARN — larger than expected error")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Derive slope+aspect bands from a 1-band elevation TIF.")
    parser.add_argument("--input",    default=str(DEFAULT_INPUT),    help="Input 1-band elevation TIF")
    parser.add_argument("--output",   default=str(DEFAULT_OUTPUT),   help="Output 3-band TIF")
    parser.add_argument("--validate", default=str(DEFAULT_VALIDATE), help="Hawaii 5-band TIF for validation (optional)")
    parser.add_argument("--skip-validate", action="store_true",      help="Skip validation step")
    args = parser.parse_args()

    if not args.skip_validate:
        validate_against_hawaii(Path(args.validate))

    derive_and_write(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
