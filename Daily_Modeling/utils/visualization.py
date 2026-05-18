"""
Visualization helpers: EDA plots, training curves, scatter plots, station maps.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns

import rasterio
from rasterio.warp import transform_bounds


def _save_and_close(fig, path, dpi: int = 150, message: Optional[str] = None):
    """Save a figure to *path* and close it.  Optionally print *message*."""
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    if message:
        print(message)
    plt.close(fig)


# ===================================================================
# EDA plots
# ===================================================================
def plot_rainfall_histograms(
    rain_mm: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
):
    """Histograms of raw rainfall per split (linear + log1p)."""
    fig, axes = plt.subplots(2, len(split_indices), figsize=(5 * len(split_indices), 8),
                             squeeze=False, sharey="row")
    bins_lin = np.linspace(0, np.nanquantile(rain_mm, 0.995), 60)
    bins_log = np.linspace(0, np.nanquantile(np.log1p(rain_mm), 0.995), 60)

    for col, (name, idx) in enumerate(split_indices.items()):
        y = rain_mm[idx]
        axes[0, col].hist(y, bins=bins_lin, alpha=0.8, color="steelblue")
        axes[0, col].set_title(f"{name} rainfall (mm)")
        axes[0, col].set_xlabel("mm")
        axes[0, col].grid(alpha=0.3)

        axes[1, col].hist(np.log1p(y), bins=bins_log, alpha=0.8, color="coral")
        axes[1, col].set_title(f"{name} log1p(rain)")
        axes[1, col].set_xlabel("log(1+mm)")
        axes[1, col].grid(alpha=0.3)

    axes[0, 0].set_ylabel("count")
    axes[1, 0].set_ylabel("count")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_multi_model_scatter(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    title: str = "Model Comparison Scatter",
    units: str = "mm",
    save_path: Optional[Path] = None,
):
    """Overlay multiple model prediction scatters on one axis."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colours = {"LAND": "steelblue", "Bernoulli-Gamma": "coral", "Site MLP": "seagreen"}
    lo = 0.0
    hi = 0.0
    for name, data in model_predictions.items():
        yt = np.asarray(data["y_true"]).ravel()
        yp = np.asarray(data["y_pred"]).ravel()
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[mask], yp[mask]
        if len(yt) == 0:
            continue
        ax.scatter(yt, yp, s=4, alpha=0.25, label=name, color=colours.get(name, None), rasterized=True)
        lo = min(lo, float(yt.min()), float(yp.min()))
        hi = max(hi, float(yt.max()), float(yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlabel(f"Observed ({units})")
    ax.set_ylabel(f"Predicted ({units})")
    ax.set_title(title)
    ax.legend(markerscale=4)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_split_year_counts(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """Plot sample counts per year for each split label."""
    pivot = df.groupby(["year", "split"]).size().unstack(fill_value=0)
    ax = pivot.plot(figsize=(14, 5), marker="o")
    ax.set_title("Samples per year by split")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sample count")
    fig = ax.figure
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_station_role_map(
    station_df: pd.DataFrame,
    station_meta: Dict[str, dict],
    save_path: Optional[Path] = None,
):
    """Plot station locations colored by assigned train/val/test role."""
    rows = []
    for _, row in station_df.iterrows():
        st = row["station"]
        meta = station_meta.get(st)
        if meta is None:
            continue
        rows.append(
            {
                "station": st,
                "role": row["role"],
                "lat": float(meta["latitude"]),
                "lon": float(meta["longitude"]),
            }
        )
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return None

    colors = {"train": "#4c72b0", "val": "#55a868", "test": "#c44e52"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for role, g in rdf.groupby("role"):
        ax.scatter(g["lon"], g["lat"], s=55, label=role, color=colors.get(role, "gray"), alpha=0.9)
        for _, r in g.iterrows():
            ax.text(r["lon"], r["lat"], str(r["station"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station roles for split assignment")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path, dpi=180)
    return fig


def save_optuna_visualizations(study, out_dir: Path) -> None:
    """Save common Optuna matplotlib visualizations."""
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
        )

        figures = {
            "hp_importance.png": plot_param_importances(study),
            "optimization_history.png": plot_optimization_history(study),
            "slice_plots.png": plot_slice(study),
        }

        for filename, fig in figures.items():
            target_fig = fig.figure if hasattr(fig, "figure") else fig
            _save_and_close(target_fig, Path(out_dir) / filename, message=f"  Saved {filename}")
    except Exception as e:
        print(f"  WARNING: Could not generate tuning visuals: {e}")


def save_top_trials_plots(
    all_df: pd.DataFrame,
    out_dir: Path,
    title_suffix: str = "",
    top_k: int = 10,
) -> Optional[pd.DataFrame]:
    """Save top-k trials table and hyperparameter distribution plots."""
    if all_df.empty:
        return None

    top_trials = all_df.nsmallest(top_k, "value").reset_index(drop=True)
    top_trials.to_csv(Path(out_dir) / "top10_trials.csv", index=False)
    print(f"\n--- Top {min(top_k, len(top_trials))} Trials{title_suffix} ---")
    print(top_trials.to_string(index=False))

    hp_cols = [c for c in top_trials.columns if c not in ("trial", "value")]
    if not hp_cols:
        return top_trials

    n_cols = min(4, len(hp_cols))
    n_rows = (len(hp_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).ravel()
    colours = plt.cm.viridis_r(np.linspace(0.1, 0.9, len(top_trials)))

    for i, col in enumerate(hp_cols):
        ax = axes[i]
        vals = top_trials[col].dropna()
        try:
            numeric_vals = vals.astype(float).values
            ax.violinplot(numeric_vals, positions=[0], showmedians=True)
            jitter = np.random.RandomState(0).uniform(-0.05, 0.05, len(numeric_vals))
            for j, (v, c) in enumerate(zip(numeric_vals, colours)):
                ax.scatter(jitter[j], v, color=c, s=40, zorder=3)
            ax.set_xticks([])
        except (ValueError, TypeError):
            vc = vals.value_counts()
            ax.bar(range(len(vc)), vc.values, tick_label=vc.index.tolist(), color="steelblue")
            ax.tick_params(axis="x", rotation=30)
        ax.set_title(col, fontsize=9)

    for j in range(len(hp_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"HP Distribution — Top {top_k} Trials{title_suffix}", fontsize=11, y=1.01)
    plt.tight_layout()
    _save_and_close(fig, Path(out_dir) / "hp_distribution_top10.png", message="  Saved hp_distribution_top10.png")
    return top_trials


def plot_split_heatmap(
    stations: np.ndarray,
    years: np.ndarray,
    station_groups: Dict[str, str],
    train_years: Tuple[int, int],
    val_years: Tuple[int, int],
    test_years: Tuple[int, int],
    save_path: Optional[Path] = None,
    title: str = "Spatiotemporal Split",
):
    unique_stations = sorted(set(str(s) for s in stations))
    yr_int = years.astype(int)
    unique_years = sorted(set(yr_int))
    s2i = {s: i for i, s in enumerate(unique_stations)}
    y2j = {y: j for j, y in enumerate(unique_years)}

    grid = np.zeros((len(unique_stations), len(unique_years)), dtype=int)
    for k in range(len(stations)):
        si = s2i[str(stations[k])]
        yj = y2j[int(yr_int[k])]
        role = station_groups.get(str(stations[k]), "train")
        yr_val = int(yr_int[k])

        in_train_yr = train_years[0] <= yr_val <= train_years[1]
        in_val_yr = val_years[0] <= yr_val <= val_years[1]
        in_test_yr = test_years[0] <= yr_val <= test_years[1]

        if role == "train" and in_train_yr:
            grid[si, yj] = 1
        elif role == "val" and in_val_yr:
            grid[si, yj] = 2
        elif role == "test" and in_test_yr:
            grid[si, yj] = 3
        elif role == "train" and in_val_yr:
            grid[si, yj] = 4
        elif role == "train" and in_test_yr:
            grid[si, yj] = 5
        elif grid[si, yj] == 0:
            grid[si, yj] = 6

    colours = ["white", "#4c72b0", "#55a868", "#c44e52", "#b5cf6b", "#f4a460", "#cccccc"]
    labels = ["No data", "Train", "Val spatial", "Test spatial", "Val temporal", "Test temporal", "Unused"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(14, len(unique_years) * 0.22), max(5, len(unique_stations) * 0.35)))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(unique_stations)))
    ax.set_yticklabels(unique_stations, fontsize=7)
    step = max(1, len(unique_years) // 15)
    ax.set_xticks(range(0, len(unique_years), step))
    ax.set_xticklabels([unique_years[i] for i in range(0, len(unique_years), step)], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Station")
    ax.set_title(title)
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=True)

    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path, message=f"  Split heatmap saved to {save_path}")
    return fig


def plot_station_proportional_split_daily_raster(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    save_path: Optional[Path] = None,
    title: str = "Site Model Station-Proportional Split (per-day)",
):
    import datetime as dt

    unique_stations = sorted(set(str(s) for s in stations))
    idx_all = np.arange(len(stations))
    xs, ys, cs = [], [], []
    c_train, c_val, c_test, c_none = "#4c72b0", "#55a868", "#c44e52", "white"

    for si, st in enumerate(unique_stations):
        mask = np.array([str(s) == st for s in stations])
        idx = idx_all[mask]
        if len(idx) == 0:
            continue
        yr = years[idx].astype(int)
        mo = months[idx].astype(int)
        dy = days[idx].astype(int)
        order = np.lexsort((dy, mo, yr))
        sorted_idx = idx[order]
        n = len(sorted_idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        split_map = [
            (sorted_idx[:n_train], c_train),
            (sorted_idx[n_train:n_train + n_val], c_val),
            (sorted_idx[n_train + n_val:], c_test),
        ]
        for subset, colour in split_map:
            for k in subset:
                xs.append(mdates.date2num(dt.date(int(years[k]), int(months[k]), int(days[k]))))
                ys.append(si)
                cs.append(colour)

    fig, ax = plt.subplots(figsize=(16, max(5, len(unique_stations) * 0.35)))
    if xs:
        ax.scatter(xs, ys, c=cs, marker="s", s=6, linewidths=0)
    ax.set_yticks(range(len(unique_stations)))
    ax.set_yticklabels(unique_stations, fontsize=7)
    ax.set_ylim(-0.5, len(unique_stations) - 0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("Station")
    ax.set_title(title)
    patches = [
        mpatches.Patch(color=c_none, label="No data"),
        mpatches.Patch(color=c_train, label="Train"),
        mpatches.Patch(color=c_val, label="Val"),
        mpatches.Patch(color=c_test, label="Test"),
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=True)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path, dpi=200, message=f"  Daily split raster saved to {save_path}")
    return fig


def plot_station_proportional_cv_folds_heatmap(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    cv_folds: int,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    save_path: Optional[Path] = None,
    title: str = "Site Model CV Folds (expanding-window)",
):
    if cv_folds <= 1:
        return None

    unique_stations = sorted(set(str(s) for s in stations))
    yr_int = years.astype(int)
    unique_years = sorted(set(yr_int))
    y2j = {y: j for j, y in enumerate(unique_years)}

    def _sorted_station_idx(st: str):
        mask = np.array([str(s) == st for s in stations])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        yr = years[idx].astype(int)
        mo = months[idx].astype(int)
        dy = days[idx].astype(int)
        order = np.lexsort((dy, mo, yr))
        return idx[order]

    def _expanding_folds(indices_sorted: np.ndarray, k: int):
        n = len(indices_sorted)
        val_size = max(n // (k + 1), 1)
        folds = []
        for i in range(1, k + 1):
            tr_end = i * val_size
            va_end = min((i + 1) * val_size, n)
            if tr_end >= n or tr_end >= va_end:
                break
            folds.append((indices_sorted[:tr_end], indices_sorted[tr_end:va_end]))
        return folds

    n_rows = len(unique_stations) * cv_folds
    grid = np.zeros((n_rows, len(unique_years)), dtype=int)
    row_labels = []

    for si, st in enumerate(unique_stations):
        sorted_idx = _sorted_station_idx(st)
        for f in range(1, cv_folds + 1):
            row_labels.append(f"{st}  [fold {f}]")
        if len(sorted_idx) == 0:
            continue
        n = len(sorted_idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        trainval_sorted = sorted_idx[: n_train + n_val]
        folds = _expanding_folds(trainval_sorted, cv_folds)
        for f_idx, (tr, va) in enumerate(folds, start=1):
            r = si * cv_folds + (f_idx - 1)
            for k in tr:
                grid[r, y2j[int(yr_int[k])]] = max(grid[r, y2j[int(yr_int[k])]], 1)
            for k in va:
                grid[r, y2j[int(yr_int[k])]] = 2

    colours = ["white", "#4c72b0", "#55a868"]
    labels = ["No data", "Train", "Val"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(max(14, len(unique_years) * 0.22), max(7, n_rows * 0.18)))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6)
    step = max(1, len(unique_years) // 15)
    ax.set_xticks(range(0, len(unique_years), step))
    ax.set_xticklabels([unique_years[i] for i in range(0, len(unique_years), step)], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Station × Fold")
    ax.set_title(title)
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=True)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path, message=f"  CV fold heatmap saved to {save_path}")
    return fig


def plot_station_proportional_split_heatmap(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    save_path: Optional[Path] = None,
    title: str = "Site Model Station-Proportional Split",
):
    unique_stations = sorted(set(str(s) for s in stations))
    yr_int = years.astype(int)
    unique_years = sorted(set(yr_int))
    s2i = {s: i for i, s in enumerate(unique_stations)}
    y2j = {y: j for j, y in enumerate(unique_years)}

    grid = np.zeros((len(unique_stations), len(unique_years)), dtype=int)
    has_train = np.zeros_like(grid, dtype=bool)
    has_val = np.zeros_like(grid, dtype=bool)
    has_test = np.zeros_like(grid, dtype=bool)
    idx_all = np.arange(len(stations))

    for st in unique_stations:
        mask = np.array([str(s) == st for s in stations])
        idx = idx_all[mask]
        if len(idx) == 0:
            continue
        yr = years[idx].astype(int)
        mo = months[idx].astype(int)
        dy = days[idx].astype(int)
        date_order = np.lexsort((dy, mo, yr))
        sorted_idx = idx[date_order]
        n = len(sorted_idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        tr = set(sorted_idx[:n_train].tolist())
        va = set(sorted_idx[n_train:n_train + n_val].tolist())
        te = set(sorted_idx[n_train + n_val:].tolist())

        for k in sorted_idx:
            si = s2i[str(stations[k])]
            yj = y2j[int(yr_int[k])]
            if k in tr:
                has_train[si, yj] = True
            elif k in va:
                has_val[si, yj] = True
            elif k in te:
                has_test[si, yj] = True

    grid[has_test] = 3
    grid[has_val] = 2
    grid[has_train] = 1

    colours = ["white", "#4c72b0", "#55a868", "#c44e52"]
    labels = ["No data", "Train", "Val", "Test"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(max(14, len(unique_years) * 0.22), max(5, len(unique_stations) * 0.35)))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(unique_stations)))
    ax.set_yticklabels(unique_stations, fontsize=7)
    step = max(1, len(unique_years) // 15)
    ax.set_xticks(range(0, len(unique_years), step))
    ax.set_xticklabels([unique_years[i] for i in range(0, len(unique_years), step)], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Station")
    ax.set_title(title)
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=True)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path, message=f"  Split heatmap saved to {save_path}")
    return fig


def plot_stations_on_dem_raster(
    dem_path: Path,
    station_metadata: dict,
    station_groups: Optional[Dict[str, str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Stations over DEM",
):
    """Plot the full DEM raster with correct georeferenced lon/lat extent and overlay stations.

    This is intended as a sanity check for CRS / lat-lon mismatch issues.

    Notes:
    - Station coordinates are assumed to be lon/lat in EPSG:4326.
    - DEM is read in its native CRS and bounds are transformed to EPSG:4326.
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(dem_path)

    role_colours = {"train": "steelblue", "val": "orange", "test": "crimson"}

    with rasterio.open(str(dem_path)) as src:
        if src.crs is None:
            raise ValueError("DEM raster has no CRS; cannot transform to lon/lat")

        # Read DEM and convert nodata to NaN for plotting
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem = np.where(dem == float(src.nodata), np.nan, dem)
        dem = np.where(np.isfinite(dem), dem, np.nan)

        # DEM bounds in lon/lat
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Robust colour scaling so the ocean (0) doesn't dominate
    finite = dem[np.isfinite(dem)]
    if finite.size > 0:
        vmin, vmax = np.nanpercentile(finite, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(
        dem,
        extent=[west, east, south, north],
        origin="upper",
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        alpha=0.95,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Elevation (m)")

    # Overlay stations
    for sname, info in station_metadata.items():
        lat = info.get("latitude", info.get("lat", None))
        lon = info.get("longitude", info.get("lon", None))
        if lat is None or lon is None:
            continue
        role = station_groups.get(sname, "train") if station_groups else "train"
        colour = role_colours.get(role, "black")
        ax.scatter(lon, lat, s=30, c=colour, edgecolors="black", linewidths=0.4, zorder=5)
        ax.annotate(sname, (lon, lat), fontsize=6, xytext=(3, 3), textcoords="offset points", zorder=6)

    # Legend
    if station_groups:
        from matplotlib.lines import Line2D
        present_roles = sorted(set(station_groups.values()))
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=role_colours[r],
                   markeredgecolor="black", markersize=7, label=r)
            for r in ("train", "val", "test") if r in present_roles
        ]
        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", "box")
    plt.tight_layout()

    if save_path:
        _save_and_close(fig, save_path, dpi=180)
    return fig


def plot_monthly_seasonality(
    rain_mm: np.ndarray,
    months: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
):
    """Mean rainfall by month across splits."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, idx in split_indices.items():
        import pandas as pd
        df = pd.DataFrame({"month": months[idx], "rain": rain_mm[idx]})
        m = df.groupby("month")["rain"].mean().reindex(range(1, 13))
        ax.plot(m.index, m.values, marker="o", label=name)
    ax.set_xticks(range(1, 13))
    ax.set_title("Mean daily rainfall by month (mm)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean mm")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_station_sample_counts(
    stations: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
):
    """Bar chart of sample counts per station per split."""
    import pandas as pd
    rows = []
    for name, idx in split_indices.items():
        for st in np.unique(stations[idx]):
            rows.append({"station": str(st), "split": name, "count": int((stations[idx] == st).sum())})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot = df.pivot_table(index="station", columns="split", values="count", fill_value=0)
    pivot.plot.bar(ax=ax, stacked=True)
    ax.set_title("Samples per station by split")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_per_station_histograms(
    rain_mm: np.ndarray,
    stations: np.ndarray,
    save_dir: Optional[Path] = None,
):
    """One histogram per station, saved as individual PNGs."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    unique = np.unique(stations)
    for st in unique:
        mask = stations == st
        y = rain_mm[mask]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(y, bins=50, alpha=0.8, color="steelblue")
        ax.set_title(f"{st} - daily rainfall (N={len(y)})")
        ax.set_xlabel("mm")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_dir:
            _save_and_close(fig, save_dir / f"hist_{st}.png", dpi=100)
        else:
            plt.close(fig)


# ===================================================================
# Training plots
# ===================================================================

def plot_training_history(
    history: Dict[str, list],
    title: str = "Training History",
    save_path: Optional[Path] = None,
):
    """Plot train vs val loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


# ===================================================================
# Result plots
# ===================================================================

def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Observed",
    units: str = "mm",
    save_path: Optional[Path] = None,
):
    """Scatter plot with 1:1 line and density colouring."""
    fig, ax = plt.subplots(figsize=(6, 6))
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if save_path:
            _save_and_close(fig, save_path)
        return fig

    ax.scatter(yt, yp, s=4, alpha=0.3, rasterized=True)
    lo = min(yt.min(), yp.min(), 0)
    hi = max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1")
    ax.set_xlabel(f"Observed ({units})")
    ax.set_ylabel(f"Predicted ({units})")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_model_comparison_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
):
    """Render a metrics comparison table as a matplotlib figure."""
    import pandas as pd
    df = pd.DataFrame(results).T
    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(df))))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(df.columns))))
    ax.set_title("Model Comparison", fontsize=12, pad=20)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig

# ===================================================================
# Dataset inspection / audit plots
# ===================================================================

def plot_sample_dem_patches(
    local_dem: np.ndarray,
    regional_dem: np.ndarray,
    stations: np.ndarray,
    sample_indices: Optional[Sequence[int]] = None,
    n_samples: int = 6,
    save_path: Optional[Path] = None,
):
    """Visualise local and regional DEM patches for a handful of samples.

    Supports both single-band (S, H, W) and multi-band (S, n_bands, H, W) arrays.
    For multi-band arrays, each band gets its own pair of rows (local + regional).

    Picks *n_samples* evenly spaced across unique stations if *sample_indices*
    is not given.
    """
    if sample_indices is None:
        unique_st = np.unique(stations)
        pick_st = unique_st[np.linspace(0, len(unique_st) - 1, min(n_samples, len(unique_st)), dtype=int)]
        sample_indices = [int(np.where(stations == s)[0][0]) for s in pick_st]

    # Normalise to always be (S, n_bands, H, W)
    if local_dem.ndim == 3:
        local_dem = local_dem[:, np.newaxis]
        regional_dem = regional_dem[:, np.newaxis]

    n_bands = local_dem.shape[1]
    band_names = ["Elevation (m)", "Slope (°)", "sin(Aspect)", "cos(Aspect)"]
    band_cmaps = ["terrain", "YlOrRd", "RdBu", "RdBu"]
    # Row layout: for each band, 2 rows (local on top, regional below)
    n_row_groups = n_bands   # each group = 1 local row + 1 regional row
    n_rows = n_row_groups * 2
    n_cols = len(sample_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), squeeze=False)

    for col, idx in enumerate(sample_indices):
        st = str(stations[idx])
        for b in range(n_bands):
            local_row = b * 2
            reg_row = b * 2 + 1
            bname = band_names[b] if b < len(band_names) else f"Band {b}"
            cmap = band_cmaps[b] if b < len(band_cmaps) else "viridis"

            ld_raw = local_dem[idx, b]
            rd_raw = regional_dem[idx, b]

            # Mask ocean sentinel (-1) only for elevation band; other bands use any finite value
            if b == 0:
                ld_plot = np.where(ld_raw <= -1, np.nan, ld_raw)
                rd_plot = np.where(rd_raw <= -1, np.nan, rd_raw)
            else:
                ld_plot = np.where(ld_raw <= -1, np.nan, ld_raw)
                rd_plot = np.where(rd_raw <= -1, np.nan, rd_raw)

            def _vrange(arr):
                valid = arr[np.isfinite(arr)]
                if len(valid) == 0:
                    return 0.0, 1.0
                vmin, vmax = np.nanpercentile(valid, [2, 98])
                if vmin == vmax:
                    vmin -= 1.0; vmax += 1.0
                return float(vmin), float(vmax)

            ld_vmin, ld_vmax = _vrange(ld_plot)
            rd_vmin, rd_vmax = _vrange(rd_plot)

            ld_valid_n = int(np.isfinite(ld_plot).sum())
            rd_valid_n = int(np.isfinite(rd_plot).sum())

            im0 = axes[local_row, col].imshow(ld_plot, cmap=cmap, interpolation="nearest",
                                               vmin=ld_vmin, vmax=ld_vmax)
            axes[local_row, col].set_title(
                f"Local {bname}\n{st} [#{idx}]\n{ld_valid_n}/{ld_raw.size} valid",
                fontsize=7,
            )
            plt.colorbar(im0, ax=axes[local_row, col], fraction=0.046, pad=0.04)

            im1 = axes[reg_row, col].imshow(rd_plot, cmap=cmap, interpolation="nearest",
                                             vmin=rd_vmin, vmax=rd_vmax)
            axes[reg_row, col].set_title(
                f"Regional {bname}\n{st} [#{idx}]\n{rd_valid_n}/{rd_raw.size} valid",
                fontsize=7,
            )
            plt.colorbar(im1, ax=axes[reg_row, col], fraction=0.046, pad=0.04)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    band_label = f"{n_bands}-band (elev, slope, sin_asp, cos_asp)" if n_bands == 4 else f"{n_bands}-band"
    fig.suptitle(f"Sample DEM Patches ({band_label})", fontsize=12)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_sample_reanalysis_patches(
    reanalysis: np.ndarray,
    variable_names: Sequence[str],
    stations: np.ndarray,
    sample_idx: int = 0,
    save_path: Optional[Path] = None,
):
    """Plot all reanalysis channels for a single sample as a grid of 3x3 heatmaps."""
    n_vars = reanalysis.shape[1]
    ncols = 5
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows), squeeze=False)
    st = str(stations[sample_idx])

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i < n_vars:
            patch = reanalysis[sample_idx, i]
            vname = variable_names[i] if i < len(variable_names) else f"ch{i}"
            im = ax.imshow(patch, cmap="coolwarm", interpolation="nearest")
            ax.set_title(f"{vname}\n[{patch.min():.1f}, {patch.max():.1f}]", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Reanalysis patch - station {st}  (sample #{sample_idx})", fontsize=11)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_nan_audit(
    arrays: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
):
    """Bar chart showing NaN fraction per feature group."""
    names, fracs, counts, totals = [], [], [], []
    for name, arr in arrays.items():
        flat = arr.astype(np.float32).ravel()
        n_nan = int(np.isnan(flat).sum())
        total = len(flat)
        names.append(name)
        counts.append(n_nan)
        totals.append(total)
        fracs.append(100.0 * n_nan / total if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(names, fracs, color="salmon")
    for bar, cnt, tot in zip(bars, counts, totals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,} / {tot:,}", va="center", fontsize=8)
    ax.set_xlabel("NaN fraction (%)")
    ax.set_title("NaN Audit - Raw Features")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_feature_distributions(
    arrays: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    tag: str = "raw",
):
    """Histograms of each feature group's value distribution (ignoring NaN)."""
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for col, (name, arr) in enumerate(arrays.items()):
        ax = axes[0][col]
        flat = arr.astype(np.float32).ravel()
        valid = flat[np.isfinite(flat)]
        if len(valid) == 0:
            ax.text(0.5, 0.5, "All NaN", ha="center", va="center", transform=ax.transAxes)
        else:
            lo, hi = np.percentile(valid, [0.5, 99.5])
            ax.hist(valid, bins=80, range=(lo, hi), alpha=0.8, color="steelblue")
            ax.axvline(valid.mean(), color="red", ls="--", lw=1, label=f"mean={valid.mean():.2f}")
            ax.legend(fontsize=7)
        ax.set_title(f"{name} ({tag})", fontsize=9)
        ax.set_xlabel("value")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("count")
    fig.suptitle(f"Feature Distributions - {tag}", fontsize=12)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_reanalysis_channel_distributions(
    reanalysis: np.ndarray,
    variable_names: Sequence[str],
    save_path: Optional[Path] = None,
    tag: str = "raw",
):
    """Per-channel histograms for all reanalysis variables."""
    n_vars = reanalysis.shape[1]
    ncols = 5
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i < n_vars:
            vals = reanalysis[:, i].ravel()
            valid = vals[np.isfinite(vals)]
            vname = variable_names[i] if i < len(variable_names) else f"ch{i}"
            if len(valid) > 0:
                lo, hi = np.percentile(valid, [0.5, 99.5])
                ax.hist(valid, bins=60, range=(lo, hi), alpha=0.8, color="teal")
                ax.set_title(f"{vname}\nmean={valid.mean():.2f} std={valid.std():.2f}", fontsize=7)
            else:
                ax.text(0.5, 0.5, "All NaN", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(vname, fontsize=7)
        else:
            ax.axis("off")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Reanalysis Channel Distributions - {tag}", fontsize=11)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_reanalysis_correlation(
    reanalysis: np.ndarray,
    variable_names: Sequence[str],
    save_path: Optional[Path] = None,
    tag: str = "raw",
):
    """Correlation heatmap across reanalysis channels (sampled for speed)."""
    n = reanalysis.shape[0]
    idx = np.random.RandomState(42).choice(n, min(n, 5000), replace=False)
    flat = reanalysis[idx].reshape(len(idx), reanalysis.shape[1], -1).mean(axis=-1)
    vnames = [variable_names[i] if i < len(variable_names) else f"ch{i}" for i in range(flat.shape[1])]
    corr = np.corrcoef(flat.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(vnames)))
    ax.set_xticklabels(vnames, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(vnames)))
    ax.set_yticklabels(vnames, fontsize=7)
    for i in range(len(vnames)):
        for j in range(len(vnames)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr[i, j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Reanalysis Channel Correlation - {tag}", fontsize=11)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_normalization_comparison(
    raw_arrays: Dict[str, np.ndarray],
    norm_arrays: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
):
    """Side-by-side histograms of raw vs normalised features."""
    keys = list(raw_arrays.keys())
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)
    for col, key in enumerate(keys):
        for row, (arr, label) in enumerate([(raw_arrays[key], "raw"), (norm_arrays[key], "normalised")]):
            ax = axes[row][col]
            flat = arr.astype(np.float32).ravel()
            valid = flat[np.isfinite(flat)]
            if len(valid) > 0:
                lo, hi = np.percentile(valid, [0.5, 99.5])
                ax.hist(valid, bins=80, range=(lo, hi), alpha=0.8,
                        color="steelblue" if label == "raw" else "coral")
                ax.axvline(valid.mean(), color="black", ls="--", lw=1, label=f"mean={valid.mean():.3f}")
                ax.axvline(valid.mean() + valid.std(), color="gray", ls=":", lw=1, label=f"std={valid.std():.3f}")
                ax.axvline(valid.mean() - valid.std(), color="gray", ls=":", lw=1)
                ax.legend(fontsize=6)
            ax.set_title(f"{key} ({label})", fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle("Feature Distributions - Raw vs Normalised", fontsize=12)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_per_station_dem_summary(
    local_dem: np.ndarray,
    regional_dem: np.ndarray,
    stations: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Box plots of DEM elevation per station (centre pixel)."""
    import pandas as pd
    rows = []
    for st in np.unique(stations):
        mask = stations == st
        # Centre pixel of the elevation band
        if local_dem.ndim == 4:   # (S, n_bands, H, W) — use band 0 (elevation)
            H, W = local_dem.shape[2], local_dem.shape[3]
            ld_center = local_dem[mask, 0, H // 2, W // 2]
            H2, W2 = regional_dem.shape[2], regional_dem.shape[3]
            rd_center = regional_dem[mask, 0, H2 // 2, W2 // 2]
        elif local_dem.ndim == 3:  # (S, H, W)
            H, W = local_dem.shape[1], local_dem.shape[2]
            ld_center = local_dem[mask, H // 2, W // 2]
            H2, W2 = regional_dem.shape[1], regional_dem.shape[2]
            rd_center = regional_dem[mask, H2 // 2, W2 // 2]
        else:
            ld_center = local_dem[mask].ravel()
            rd_center = regional_dem[mask].ravel()
        ld_val = float(np.nanmean(ld_center))
        rd_val = float(np.nanmean(rd_center))
        rows.append({"station": str(st), "local_dem_centre": ld_val, "regional_dem_centre": rd_val})
    df = pd.DataFrame(rows).sort_values("local_dem_centre")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, colour in zip(axes, ["local_dem_centre", "regional_dem_centre"], ["teal", "coral"]):
        ax.barh(df["station"], df[col], color=colour, alpha=0.8)
        ax.set_xlabel("Elevation (m)")
        ax.set_title(col.replace("_", " ").title())
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Mean Centre-Pixel DEM by Station", fontsize=12)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


# ===================================================================
# Enhanced EDA plots (spatial, correlation, autocorrelation, lay-person)
# ===================================================================

def plot_station_map(
    station_metadata: dict,
    station_groups: Optional[Dict[str, str]] = None,
    save_path: Optional[Path] = None,
):
    """Map of American Samoa showing station locations coloured by train/val/test role.

    *station_metadata*: {name: {latitude, longitude, ...}}
    *station_groups*:   {name: 'train'|'val'|'test'} (optional colour coding)
    """
    role_colours = {"train": "steelblue", "val": "orange", "test": "crimson"}
    fig, ax = plt.subplots(figsize=(12, 7))

    for sname, info in station_metadata.items():
        lat = info.get("latitude", info.get("lat", None))
        lon = info.get("longitude", info.get("lon", None))
        if lat is None or lon is None:
            continue
        role = station_groups.get(sname, "train") if station_groups else "train"
        colour = role_colours.get(role, "gray")
        ax.scatter(lon, lat, c=colour, s=80, edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(sname, (lon, lat), fontsize=6, xytext=(4, 4),
                    textcoords="offset points", zorder=6)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                       markersize=8, label=r) for r, c in role_colours.items()
               if station_groups is None or r in set(station_groups.values())]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station Locations — American Samoa", fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_reanalysis_rainfall_correlation(
    climate: np.ndarray,
    rain_mm: np.ndarray,
    variable_names: Sequence[str],
    save_path: Optional[Path] = None,
):
    """Bar chart of Pearson correlation between each reanalysis channel mean and rainfall."""
    # Spatially average each channel: (N, C, H, W) -> (N, C)
    if climate.ndim == 4:
        ch_means = np.nanmean(climate, axis=(2, 3))
    else:
        ch_means = climate

    n_vars = ch_means.shape[1]
    corrs = []
    for ci in range(n_vars):
        valid = np.isfinite(ch_means[:, ci]) & np.isfinite(rain_mm)
        if valid.sum() < 10:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(ch_means[valid, ci], rain_mm[valid])[0, 1]))

    vnames = [variable_names[i] if i < len(variable_names) else f"ch{i}" for i in range(n_vars)]
    order = np.argsort(np.abs(corrs))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#e74c3c" if c > 0 else "#3498db" for c in np.array(corrs)[order]]
    ax.barh([vnames[i] for i in order], [corrs[i] for i in order], color=colours, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson Correlation with Rainfall")
    ax.set_title("Reanalysis Variable — Rainfall Correlation\n(spatially-averaged channel means)", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_temporal_autocorrelation(
    rain_mm: np.ndarray,
    stations: np.ndarray,
    max_lag: int = 14,
    save_path: Optional[Path] = None,
):
    """Per-station lag-1…max_lag autocorrelation of daily rainfall.

    Shows how much temporal dependence exists — relevant for deciding
    whether to add temporal context to models.
    """
    import pandas as pd
    unique_st = np.unique(stations)
    all_acf = []
    for st in unique_st:
        mask = stations == st
        y = rain_mm[mask]
        if len(y) < max_lag + 20:
            continue
        acf_vals = []
        for lag in range(1, max_lag + 1):
            c = np.corrcoef(y[:-lag], y[lag:])[0, 1]
            acf_vals.append(c if np.isfinite(c) else 0.0)
        all_acf.append(acf_vals)

    if not all_acf:
        return None

    acf_arr = np.array(all_acf)  # (n_stations, max_lag)
    lags = np.arange(1, max_lag + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: individual station ACFs
    ax = axes[0]
    for i, st in enumerate(unique_st[:len(all_acf)]):
        ax.plot(lags, acf_arr[i], alpha=0.4, lw=1, color="steelblue")
    ax.plot(lags, acf_arr.mean(axis=0), color="red", lw=2.5, label="Mean across stations")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Daily Rainfall Autocorrelation by Station")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: heatmap
    ax = axes[1]
    im = ax.imshow(acf_arr, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.5,
                   interpolation="nearest")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Station index")
    ax.set_xticks(range(max_lag))
    ax.set_xticklabels(lags)
    ax.set_title("ACF Heatmap (stations × lags)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_rainfall_exceedance(
    rain_mm: np.ndarray,
    split_indices: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[Path] = None,
):
    """Exceedance probability curve: P(rainfall > x) vs x.

    Intuitive for lay people — shows how often extreme events occur.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    if split_indices is None:
        split_indices = {"all": np.arange(len(rain_mm))}

    for name, idx in split_indices.items():
        y = np.sort(rain_mm[idx])[::-1]
        y = y[y > 0]  # only rainy days
        prob = np.arange(1, len(y) + 1) / len(y)
        ax.semilogy(y, prob, lw=1.5, label=f"{name} (N={len(y):,})")

    ax.set_xlabel("Daily Rainfall (mm)")
    ax.set_ylabel("Exceedance Probability P(rain > x)")
    ax.set_title("Rainfall Exceedance Curve\n(rainy days only — how often do extreme events occur?)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_dry_wet_spells(
    rain_mm: np.ndarray,
    stations: np.ndarray,
    threshold_mm: float = 1.0,
    save_path: Optional[Path] = None,
):
    """Distribution of consecutive dry-spell and wet-spell lengths.

    Helps lay people understand rainfall persistence patterns.
    """
    dry_lengths, wet_lengths = [], []
    for st in np.unique(stations):
        y = rain_mm[stations == st]
        is_wet = y >= threshold_mm
        # Run-length encoding
        if len(is_wet) < 2:
            continue
        changes = np.diff(is_wet.astype(int))
        starts = np.where(changes != 0)[0] + 1
        starts = np.concatenate([[0], starts, [len(is_wet)]])
        for i in range(len(starts) - 1):
            length = starts[i + 1] - starts[i]
            if is_wet[starts[i]]:
                wet_lengths.append(length)
            else:
                dry_lengths.append(length)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, title, colour in [
        (axes[0], dry_lengths, "Dry Spell Lengths", "#e67e22"),
        (axes[1], wet_lengths, "Wet Spell Lengths", "#2980b9"),
    ]:
        if data:
            bins = np.arange(0.5, min(max(data), 60) + 1.5, 1)
            ax.hist(data, bins=bins, color=colour, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(data), color="red", ls="--", lw=1.5,
                       label=f"Mean = {np.mean(data):.1f} days")
            ax.axvline(np.median(data), color="black", ls=":", lw=1.5,
                       label=f"Median = {np.median(data):.0f} days")
            ax.legend(fontsize=8)
        ax.set_xlabel("Consecutive Days")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\n(threshold = {threshold_mm} mm)")
        ax.grid(alpha=0.3)

    fig.suptitle("Dry & Wet Spell Distributions Across All Stations", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_rainfall_by_station_boxplot(
    rain_mm: np.ndarray,
    stations: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Box-and-whisker plot of rainfall per station — intuitive summary for lay people."""
    import pandas as pd
    df = pd.DataFrame({"station": [str(s) for s in stations], "rain_mm": rain_mm})
    # Order by median rainfall
    order = df.groupby("station")["rain_mm"].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 6))
    df_rainy = df[df["rain_mm"] > 0]
    bp = ax.boxplot(
        [df_rainy[df_rainy["station"] == st]["rain_mm"].values for st in order],
        labels=order, vert=True, patch_artist=True, showfliers=False,
        medianprops=dict(color="red", lw=2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xlabel("Station")
    ax.set_ylabel("Daily Rainfall (mm, rainy days only)")
    ax.set_title("Rainfall Distribution by Station\n(rainy days only, outliers hidden for clarity)")
    plt.xticks(rotation=45, ha="right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_annual_rainfall_trends(
    rain_mm: np.ndarray,
    years: np.ndarray,
    stations: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Annual total rainfall trends per station — shows long-term patterns."""
    import pandas as pd
    df = pd.DataFrame({
        "station": [str(s) for s in stations],
        "year": years.astype(int),
        "rain_mm": rain_mm,
    })
    annual = df.groupby(["station", "year"])["rain_mm"].sum().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: individual station trends
    ax = axes[0]
    for st in annual["station"].unique():
        sub = annual[annual["station"] == st].sort_values("year")
        ax.plot(sub["year"], sub["rain_mm"], alpha=0.4, lw=1)
    # Overall mean
    overall = annual.groupby("year")["rain_mm"].mean().sort_index()
    ax.plot(overall.index, overall.values, color="red", lw=2.5, label="Mean across stations")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Total Rainfall (mm)")
    ax.set_title("Annual Rainfall by Station")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: mean annual rainfall bar chart per station
    ax = axes[1]
    mean_annual = annual.groupby("station")["rain_mm"].mean().sort_values(ascending=False)
    ax.barh(mean_annual.index, mean_annual.values, color="steelblue", alpha=0.8)
    ax.set_xlabel("Mean Annual Rainfall (mm)")
    ax.set_title("Mean Annual Rainfall by Station")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


# ===================================================================
# DEM on geographic map
# ===================================================================

def plot_dem_on_map(
    local_dem: np.ndarray,
    regional_dem: np.ndarray,
    stations: np.ndarray,
    station_metadata: dict,
    n_samples: int = 6,
    local_km_per_cell: float = 1.0,
    regional_km_per_cell: float = 1.0,
    save_path: Optional[Path] = None,
):
    """Overlay DEM patches on a geographic scatter map of station locations.

    Shows local and regional DEM patches positioned at their station coordinates.
    """
    unique_st = np.unique(stations)
    pick_st = unique_st[np.linspace(0, len(unique_st) - 1, min(n_samples, len(unique_st)), dtype=int)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    panels = [
        (axes[0], local_dem, "Local DEM", float(local_km_per_cell)),
        (axes[1], regional_dem, "Regional DEM", float(regional_km_per_cell)),
    ]

    for ax, dem_arr, label, km_per_cell in panels:
        # Plot all stations as dots
        for sname, info in station_metadata.items():
            lat = info.get("latitude", info.get("lat", None))
            lon = info.get("longitude", info.get("lon", None))
            if lat is None or lon is None:
                continue
            ax.plot(lon, lat, 'k.', markersize=3, zorder=3)

        # Overlay DEM patches for selected stations
        for st in pick_st:
            idx = int(np.where(stations == st)[0][0])
            sname = str(st)
            info = station_metadata.get(sname, {})
            lat = info.get("latitude", info.get("lat", None))
            lon = info.get("longitude", info.get("lon", None))
            if lat is None or lon is None:
                continue
            patch_raw = dem_arr[idx]
            # Use elevation band only for map overlay (band 0 for multi-band, or direct for single-band)
            patch = patch_raw[0] if patch_raw.ndim == 3 else patch_raw
            h, w = patch.shape
            # Scale patch extent based on physical size.
            # Approximate degrees per km (sufficient for small extents around American Samoa).
            half_km_x = (w / 2.0) * km_per_cell
            half_km_y = (h / 2.0) * km_per_cell
            deg_per_km_lat = 1.0 / 110.574
            deg_per_km_lon = 1.0 / (111.320 * max(np.cos(np.deg2rad(lat)), 1e-6))
            half_deg_lon = half_km_x * deg_per_km_lon
            half_deg_lat = half_km_y * deg_per_km_lat
            extent = [lon - half_deg_lon, lon + half_deg_lon, lat - half_deg_lat, lat + half_deg_lat]
            ax.imshow(patch, extent=extent, cmap="terrain", alpha=0.7, zorder=2,
                      interpolation="bilinear")
            ax.annotate(sname, (lon, lat), fontsize=5, xytext=(3, 3),
                        textcoords="offset points", zorder=4)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{label} Patches on Map")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.suptitle("DEM Patches Overlaid on Station Locations", fontsize=12)
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


# ===================================================================
# Model architecture visualization
# ===================================================================

def plot_model_architecture(
    model: "torch.nn.Module",
    model_name: str = "Model",
    input_data: Optional[dict] = None,
    save_path: Optional[Path] = None,
):
    """Generate a dynamic architecture diagram from a live PyTorch model.

    Uses torchviz (graphviz-based) if available, otherwise falls back to a
    custom matplotlib block diagram built from the model's actual layers.
    """
    import torch

    fig = _plot_architecture_matplotlib(model, model_name)

    if save_path:
        _save_and_close(fig, save_path)

    # Also try torchviz for a computation-graph PDF/PNG
    if input_data is not None:
        try:
            from torchviz import make_dot
            import shutil
            if shutil.which("dot") is None:
                print("  Graphviz 'dot' not found on PATH — skipping torchviz computation graph")
                return fig
            model.eval()
            # torchviz needs grad_fn, so run with gradients enabled
            if isinstance(input_data, dict):
                out = model(input_data)
            else:
                out = model(input_data)
            dot = make_dot(out, params=dict(model.named_parameters()),
                           show_attrs=False, show_saved=False)
            graph_path = str(save_path).replace(".png", "_graph") if save_path else "model_graph"
            dot.render(graph_path, format="png", cleanup=True)
            print(f"  Saved torchviz graph to {graph_path}.png")
        except ImportError:
            print("  torchviz not installed — skipping computation graph (pip install torchviz)")
        except Exception as e:
            print(f"  WARNING: torchviz graph failed: {e}")

    return fig


def _plot_architecture_matplotlib(model: "torch.nn.Module", model_name: str):
    """Build a block diagram of the model architecture from its actual layers."""
    import torch.nn as nn

    # Collect layer info
    blocks = []
    for name, module in model.named_modules():
        if name == "":
            continue
        # Only show leaf modules (no containers)
        children = list(module.children())
        if len(children) > 0:
            continue
        n_params = sum(p.numel() for p in module.parameters())
        layer_type = type(module).__name__
        # Get shape info
        shape_str = ""
        if isinstance(module, nn.Linear):
            shape_str = f"{module.in_features}→{module.out_features}"
        elif isinstance(module, nn.Conv2d):
            shape_str = (f"{module.in_channels}→{module.out_channels} "
                         f"k={module.kernel_size} g={module.groups}")
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nf = module.num_features if hasattr(module, 'num_features') else module.normalized_shape
            shape_str = f"features={nf}"
        elif isinstance(module, nn.Dropout):
            shape_str = f"p={module.p}"

        blocks.append({
            "name": name,
            "type": layer_type,
            "shape": shape_str,
            "params": n_params,
        })

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Determine branch structure by name prefix
    branch_colours = {
        "climate": "#3498db",
        "clim": "#3498db",
        "ld_": "#27ae60",
        "rd_": "#e67e22",
        "mo_": "#9b59b6",
        "fc_": "#e74c3c",
        "bn_": "#e74c3c",
        "backbone": "#2c3e50",
        "out": "#c0392b",
        "dropout": "#95a5a6",
    }

    def _get_colour(name):
        for prefix, colour in branch_colours.items():
            if prefix in name:
                return colour
        return "#34495e"

    # Draw
    n = len(blocks)
    fig_height = max(6, n * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, n + 1)
    ax.axis("off")

    for i, b in enumerate(blocks):
        y = n - i - 0.5
        colour = _get_colour(b["name"])
        # Draw box
        rect = plt.Rectangle((1, y - 0.35), 8, 0.7, facecolor=colour, alpha=0.15,
                              edgecolor=colour, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        # Layer type + name
        ax.text(1.2, y + 0.1, f"{b['type']}", fontsize=8, fontweight="bold",
                va="center", color=colour, zorder=3)
        ax.text(1.2, y - 0.15, f"{b['name']}", fontsize=6, va="center",
                color="gray", zorder=3)
        # Shape info
        if b["shape"]:
            ax.text(5.5, y, b["shape"], fontsize=7, va="center", ha="center",
                    color="#2c3e50", zorder=3)
        # Param count
        if b["params"] > 0:
            ax.text(8.8, y, f"{b['params']:,}", fontsize=7, va="center", ha="right",
                    color="#7f8c8d", zorder=3)
        # Arrow to next
        if i < n - 1:
            ax.annotate("", xy=(5, y - 0.35), xytext=(5, y - 0.65),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # Header
    ax.text(5, n + 0.5, f"{model_name}  ({total_params:,} trainable parameters)",
            fontsize=12, fontweight="bold", ha="center", va="center")
    # Column headers
    ax.text(1.2, n + 0.0, "Layer", fontsize=8, fontweight="bold", color="gray")
    ax.text(5.5, n + 0.0, "Shape", fontsize=8, fontweight="bold", ha="center", color="gray")
    ax.text(8.8, n + 0.0, "Params", fontsize=8, fontweight="bold", ha="right", color="gray")

    plt.tight_layout()
    return fig


def plot_wetdry_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_mm: float = 1.0,
    title: str = "Wet/Dry Day Evaluation",
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """Four-panel wet/dry day evaluation figure.

    Panels:
        1. Confusion matrix (TP/FP/FN/TN counts and rates)
        2. Scatter plot on observed wet days (y_true ≥ threshold)
        3. KDE / histogram of wet-day amounts (observed vs predicted)
        4. Bar chart of classification skill scores (POD, FAR, CSI, ETS, HSS)

    Args:
        y_true: observed rainfall in mm.
        y_pred: predicted rainfall in mm.
        threshold_mm: wet-day threshold in mm (default 1.0).
        title: figure suptitle.
        save_path: if provided, save and close the figure.

    Returns:
        matplotlib Figure.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    obs_wet = yt >= threshold_mm
    pred_wet = yp >= threshold_mm

    tp = int(np.sum(obs_wet & pred_wet))
    fn = int(np.sum(obs_wet & ~pred_wet))
    fp = int(np.sum(~obs_wet & pred_wet))
    tn = int(np.sum(~obs_wet & ~pred_wet))
    n = len(yt)

    def _safe(num, denom):
        return float(num / denom) if denom > 0 else float("nan")

    pod = _safe(tp, tp + fn)
    far = _safe(fp, tp + fp)
    freq_bias = _safe(tp + fp, tp + fn)
    csi = _safe(tp, tp + fp + fn)
    tc = (tp + fp) * (tp + fn) / n if n > 0 else 0.0
    ets_denom = tp + fp + fn - tc
    ets = _safe(tp - tc, ets_denom)
    hss_num = 2.0 * (tp * tn - fp * fn)
    hss_denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = _safe(hss_num, hss_denom)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"{title}\n(wet-day threshold = {threshold_mm:.1f} mm)", fontsize=13, fontweight="bold")

    # --- Panel 1: Confusion matrix ---
    ax_cm = axes[0, 0]
    cm = np.array([[tp, fp], [fn, tn]], dtype=float)
    labels = np.array([
        [f"Hit\n{tp:,}", f"False\nAlarm\n{fp:,}"],
        [f"Miss\n{fn:,}", f"Correct\nNeg\n{tn:,}"],
    ])
    im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Obs Wet", "Obs Dry"], fontsize=10)
    ax_cm.set_yticklabels(["Pred Wet", "Pred Dry"], fontsize=10)
    ax_cm.set_title("Contingency Table", fontsize=11)
    for r in range(2):
        for c in range(2):
            ax_cm.text(c, r, labels[r, c], ha="center", va="center",
                       fontsize=9, color="white" if cm[r, c] > cm.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # --- Panel 2: Scatter on observed wet days ---
    ax_sc = axes[0, 1]
    wet_mask = obs_wet
    if wet_mask.sum() >= 2:
        yt_w, yp_w = yt[wet_mask], yp[wet_mask]
        hi = max(float(yt_w.max()), float(yp_w.max())) * 1.05
        ax_sc.scatter(yt_w, yp_w, s=4, alpha=0.25, color="steelblue", rasterized=True)
        ax_sc.plot([0, hi], [0, hi], "k--", lw=1, label="1:1")
        res = yp_w - yt_w
        r2 = float(1.0 - np.sum(res ** 2) / np.sum((yt_w - yt_w.mean()) ** 2)) if yt_w.std() > 0 else float("nan")
        ax_sc.set_title(
            f"Wet-day scatter (n={int(wet_mask.sum()):,})\n"
            f"MAE={float(np.mean(np.abs(res))):.2f} mm  R²={r2:.3f}",
            fontsize=10,
        )
    else:
        ax_sc.set_title("Wet-day scatter (insufficient data)")
    ax_sc.set_xlabel("Observed (mm)", fontsize=9)
    ax_sc.set_ylabel("Predicted (mm)", fontsize=9)
    ax_sc.set_xlim(left=0)
    ax_sc.set_ylim(bottom=0)
    ax_sc.grid(alpha=0.3)

    # --- Panel 3: KDE of wet-day amounts ---
    ax_kd = axes[1, 0]
    if wet_mask.sum() >= 5:
        yt_w = yt[wet_mask]
        # Predicted amounts on observed wet days
        yp_w_obs = yp[wet_mask]
        # All predicted wet-day amounts (for predicted wet days)
        yp_w_pred = yp[pred_wet] if pred_wet.sum() >= 2 else np.array([])

        q99 = float(np.nanpercentile(yt_w, 99))
        bins = np.linspace(threshold_mm, q99, 40)
        ax_kd.hist(yt_w, bins=bins, alpha=0.55, color="steelblue", density=True,
                   label=f"Observed wet days (n={int(wet_mask.sum()):,})")
        ax_kd.hist(yp_w_obs, bins=bins, alpha=0.55, color="coral", density=True,
                   label=f"Predicted | obs wet (n={int(wet_mask.sum()):,})")
        ax_kd.set_title(f"Wet-day amount distribution (≥{threshold_mm:.1f} mm)", fontsize=10)
        ax_kd.legend(fontsize=8)
    else:
        ax_kd.set_title("Wet-day distribution (insufficient data)")
    ax_kd.set_xlabel("Rainfall (mm)", fontsize=9)
    ax_kd.set_ylabel("Density", fontsize=9)
    ax_kd.grid(alpha=0.3)

    # --- Panel 4: Skill score bar chart ---
    ax_bar = axes[1, 1]
    skill_scores = {
        "POD": pod,
        "1-FAR": (1.0 - far) if np.isfinite(far) else float("nan"),
        "CSI": csi,
        "ETS": ets,
        "HSS": hss,
    }
    names = list(skill_scores.keys())
    vals = [v if np.isfinite(v) else 0.0 for v in skill_scores.values()]
    colours_bar = ["steelblue" if v >= 0 else "tomato" for v in vals]
    bars = ax_bar.bar(names, vals, color=colours_bar, alpha=0.8, edgecolor="white")
    ax_bar.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_bar.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax_bar.set_ylim(-0.1, 1.15)
    ax_bar.set_title(
        f"Classification skill scores\n"
        f"Freq Bias={freq_bias:.3f}  n_total={n:,}",
        fontsize=10,
    )
    ax_bar.set_ylabel("Score", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, skill_scores.values()):
        if np.isfinite(val):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig


def plot_per_station_comparison(
    station_metrics: Dict[str, Dict[str, Dict[str, float]]],
    metric_name: str = "rmse",
    save_path: Optional[Path] = None,
):
    """Grouped bar chart of a metric per station across models."""
    import pandas as pd
    rows = []
    for model_name, sm in station_metrics.items():
        for st, m in sm.items():
            rows.append({"model": model_name, "station": st, metric_name: m.get(metric_name, np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot = df.pivot_table(index="station", columns="model", values=metric_name)
    pivot.plot.bar(ax=ax)
    ax.set_title(f"{metric_name.upper()} by station")
    ax.set_ylabel(metric_name.upper())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        _save_and_close(fig, save_path)
    return fig
