"""Spatio-temporal cross-validation splits for American Samoa daily rainfall.

Split modes
-----------
1. **spatiotemporal** (LAND): hold out entire stations *and* use different
   year ranges.  Produces five named splits:

   - ``train``          – train stations × train years
   - ``val_spatial``    – held-out val stations × val years  (spatial + temporal)
   - ``test_spatial``   – held-out test stations × test years
   - ``val_temporal``   – train stations × val years  (temporal only)
   - ``test_temporal``  – train stations × test years

2. **station_proportional** (site-specific models): per-station chronological
   70/20/10 split.

Year boundaries can be computed from data so that ~70 % of samples fall in
train years, ~20 % in val years, and ~10 % in test years.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np

from Daily_Modeling import config


# ── helpers ────────────────────────────────────────────────────────────────

def compute_station_year_ranges(
    stations: np.ndarray,
    years: np.ndarray,
) -> Dict[str, Tuple[int, int]]:
    """Compute actual (min_year, max_year) for each station from the data arrays."""
    ranges: Dict[str, Tuple[int, int]] = {}
    for s in np.unique(stations):
        mask = stations == s
        yrs = years[mask].astype(int)
        if len(yrs) > 0:
            ranges[str(s)] = (int(yrs.min()), int(yrs.max()))
    return ranges


def compute_year_boundaries(
    years: np.ndarray,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Find chronological year cutoffs so that sample counts match target fractions.

    Returns (train_range, val_range, test_range) as ``(start_year, end_year)``
    tuples (inclusive).
    """
    yr = np.sort(years.astype(int))
    n = len(yr)
    train_end_idx = int(n * train_frac) - 1
    val_end_idx = int(n * (train_frac + val_frac)) - 1

    train_end_year = int(yr[train_end_idx])
    val_end_year = int(yr[val_end_idx])
    min_yr, max_yr = int(yr[0]), int(yr[-1])

    train_range = (min_yr, train_end_year)
    val_range = (train_end_year + 1, val_end_year)
    test_range = (val_end_year + 1, max_yr)

    # Print summary
    tr_n = int(np.sum((yr >= train_range[0]) & (yr <= train_range[1])))
    va_n = int(np.sum((yr >= val_range[0]) & (yr <= val_range[1])))
    te_n = int(np.sum((yr >= test_range[0]) & (yr <= test_range[1])))
    print(f"  Year boundaries (data-driven):")
    print(f"    train: {train_range[0]}–{train_range[1]}  ({tr_n:,d} samples, {100*tr_n/n:.1f}%)")
    print(f"    val:   {val_range[0]}–{val_range[1]}  ({va_n:,d} samples, {100*va_n/n:.1f}%)")
    print(f"    test:  {test_range[0]}–{test_range[1]}  ({te_n:,d} samples, {100*te_n/n:.1f}%)")
    return train_range, val_range, test_range


# ── station assignment ─────────────────────────────────────────────────────

def assign_station_groups(
    station_names: List[str],
    n_val: int = config.N_VAL_STATIONS,
    n_test: int = config.N_TEST_STATIONS,
    seed: int = config.RANDOM_SEED,
    station_year_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    val_years: Tuple[int, int] = config.VAL_YEAR_RANGE,
    test_years: Tuple[int, int] = config.TEST_YEAR_RANGE,
) -> Dict[str, str]:
    """Deterministically assign each station to 'train', 'val', or 'test'.

    If *station_year_ranges* is provided (``{name: (start_yr, end_yr)}``),
    only stations whose data overlaps the val / test year ranges are
    eligible for those roles.  This avoids empty splits.

    Returns ``{station_name: role}``.
    """
    if station_year_ranges is not None:
        def _overlaps(syr, target_range):
            return syr[0] <= target_range[1] and syr[1] >= target_range[0]
        eligible_test = sorted([
            s for s in station_names
            if s in station_year_ranges and _overlaps(station_year_ranges[s], test_years)
        ])
        eligible_val = sorted([
            s for s in station_names
            if s in station_year_ranges and _overlaps(station_year_ranges[s], val_years)
        ])
    else:
        eligible_test = sorted(station_names)
        eligible_val = sorted(station_names)

    rng = np.random.RandomState(seed)

    et = list(eligible_test); rng.shuffle(et)
    test_stations = set(et[:n_test])

    ev = [s for s in eligible_val if s not in test_stations]
    rng.shuffle(ev)
    val_stations = set(ev[:n_val])

    mapping: Dict[str, str] = {}
    for n in station_names:
        if n in test_stations:
            mapping[n] = "test"
        elif n in val_stations:
            mapping[n] = "val"
        else:
            mapping[n] = "train"
    return mapping


# ── spatiotemporal split (LAND) ────────────────────────────────────────────

def spatiotemporal_split(
    stations: np.ndarray,
    years: np.ndarray,
    station_groups: Dict[str, str],
    train_years: Tuple[int, int] = config.TRAIN_YEAR_RANGE,
    val_years: Tuple[int, int] = config.VAL_YEAR_RANGE,
    test_years: Tuple[int, int] = config.TEST_YEAR_RANGE,
) -> Dict[str, np.ndarray]:
    """Return index arrays for a spatio-temporal split.

    Five splits are returned:

    - **train**          – train stations in train years
    - **val_spatial**    – held-out *val* stations in val years
                           (tests spatial + temporal generalisation)
    - **test_spatial**   – held-out *test* stations in test years
    - **val_temporal**   – *train* stations in val years
                           (tests temporal-only generalisation)
    - **test_temporal**  – *train* stations in test years
    """
    n = len(stations)
    idx = np.arange(n)

    roles = np.array([station_groups.get(str(s), "train") for s in stations])
    yr = years.astype(int)

    train_mask = (roles == "train") & (yr >= train_years[0]) & (yr <= train_years[1])
    val_sp_mask = (roles == "val") & (yr >= val_years[0]) & (yr <= val_years[1])
    test_sp_mask = (roles == "test") & (yr >= test_years[0]) & (yr <= test_years[1])
    val_tm_mask = (roles == "train") & (yr >= val_years[0]) & (yr <= val_years[1])
    test_tm_mask = (roles == "train") & (yr >= test_years[0]) & (yr <= test_years[1])

    splits = {
        "train": idx[train_mask],
        "val_spatial": idx[val_sp_mask],
        "test_spatial": idx[test_sp_mask],
        "val_temporal": idx[val_tm_mask],
        "test_temporal": idx[test_tm_mask],
    }
    total = sum(len(v) for v in splits.values())
    for k, v in splits.items():
        pct = 100 * len(v) / total if total else 0
        print(f"  {k:16s}: {len(v):>7,d} samples  ({pct:5.1f}%)")
    return splits


# ── temporal-only split ────────────────────────────────────────────────────

def temporal_split(
    years: np.ndarray,
    train_years: Tuple[int, int] = config.TRAIN_YEAR_RANGE,
    val_years: Tuple[int, int] = config.VAL_YEAR_RANGE,
    test_years: Tuple[int, int] = config.TEST_YEAR_RANGE,
) -> Dict[str, np.ndarray]:
    """Simple year-based split (for site-specific models)."""
    yr = years.astype(int)
    idx = np.arange(len(yr))
    splits = {
        "train": idx[(yr >= train_years[0]) & (yr <= train_years[1])],
        "val": idx[(yr >= val_years[0]) & (yr <= val_years[1])],
        "test": idx[(yr >= test_years[0]) & (yr <= test_years[1])],
    }
    for k, v in splits.items():
        print(f"  {k:12s}: {len(v):>7,d} samples")
    return splits


# ── per-station chronological split (site-specific) ───────────────────────

def station_temporal_split(
    stations: np.ndarray,
    years: np.ndarray,
    target_station: str,
    train_years: Tuple[int, int] = config.TRAIN_YEAR_RANGE,
    val_years: Tuple[int, int] = config.VAL_YEAR_RANGE,
    test_years: Tuple[int, int] = config.TEST_YEAR_RANGE,
) -> Dict[str, np.ndarray]:
    """Return indices for a single station, split only by year."""
    mask = np.array([str(s) == target_station for s in stations])
    yr = years.astype(int)
    idx = np.arange(len(stations))
    splits = {
        "train": idx[mask & (yr >= train_years[0]) & (yr <= train_years[1])],
        "val": idx[mask & (yr >= val_years[0]) & (yr <= val_years[1])],
        "test": idx[mask & (yr >= test_years[0]) & (yr <= test_years[1])],
    }
    return splits


def station_proportional_split(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    target_station: str,
    train_frac: float = config.SITE_TRAIN_FRAC,
    val_frac: float = config.SITE_VAL_FRAC,
) -> Dict[str, np.ndarray]:
    """Split a single station's data chronologically by proportion.

    Sorts the station's samples by date, then assigns the first
    *train_frac* to train, the next *val_frac* to val, and the
    remainder to test.  This guarantees every station with enough
    data gets all three splits regardless of its year range.
    """
    mask = np.array([str(s) == target_station for s in stations])
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"train": np.array([], dtype=int),
                "val": np.array([], dtype=int),
                "test": np.array([], dtype=int)}

    yr = years[idx].astype(int)
    mo = months[idx].astype(int)
    dy = days[idx].astype(int)
    date_order = np.lexsort((dy, mo, yr))
    sorted_idx = idx[date_order]

    n = len(sorted_idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": sorted_idx[:n_train],
        "val": sorted_idx[n_train:n_train + n_val],
        "test": sorted_idx[n_train + n_val:],
    }


# ── visualisation ──────────────────────────────────────────────────────────

def plot_split_heatmap(
    stations: np.ndarray,
    years: np.ndarray,
    station_groups: Dict[str, str],
    train_years: Tuple[int, int],
    val_years: Tuple[int, int],
    test_years: Tuple[int, int],
    save_path=None,
    title: str = "Spatiotemporal Split",
):
    """Generate a station × year heatmap showing which cells belong to which split.

    Colours:
      0 = no data (white), 1 = train (blue), 2 = val_spatial (green),
      3 = test_spatial (red), 4 = val_temporal (light green),
      5 = test_temporal (orange), 6 = unused (grey)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    unique_stations = sorted(set(str(s) for s in stations))
    yr_int = years.astype(int)
    unique_years = sorted(set(yr_int))
    s2i = {s: i for i, s in enumerate(unique_stations)}
    y2j = {y: j for j, y in enumerate(unique_years)}

    grid = np.zeros((len(unique_stations), len(unique_years)), dtype=int)
    # Track whether any sample in a given (station, year) belongs to each split.
    has_train = np.zeros_like(grid, dtype=bool)
    has_val = np.zeros_like(grid, dtype=bool)
    has_test = np.zeros_like(grid, dtype=bool)

    for k in range(len(stations)):
        si = s2i[str(stations[k])]
        yj = y2j[int(yr_int[k])]
        role = station_groups.get(str(stations[k]), "train")
        yr_val = int(yr_int[k])

        in_train_yr = train_years[0] <= yr_val <= train_years[1]
        in_val_yr = val_years[0] <= yr_val <= val_years[1]
        in_test_yr = test_years[0] <= yr_val <= test_years[1]

        if role == "train" and in_train_yr:
            grid[si, yj] = 1  # train
        elif role == "val" and in_val_yr:
            grid[si, yj] = 2  # val_spatial
        elif role == "test" and in_test_yr:
            grid[si, yj] = 3  # test_spatial
        elif role == "train" and in_val_yr:
            grid[si, yj] = 4  # val_temporal
        elif role == "train" and in_test_yr:
            grid[si, yj] = 5  # test_temporal
        else:
            if grid[si, yj] == 0:
                grid[si, yj] = 6  # unused (val/test station in train years)

    colours = ["white", "#4c72b0", "#55a868", "#c44e52",
               "#b5cf6b", "#f4a460", "#cccccc"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(14, len(unique_years) * 0.22),
                                    max(5, len(unique_stations) * 0.35)))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # Axis labels
    ax.set_yticks(range(len(unique_stations)))
    ax.set_yticklabels(unique_stations, fontsize=7)
    # Show every 5th year
    step = max(1, len(unique_years) // 15)
    ax.set_xticks(range(0, len(unique_years), step))
    ax.set_xticklabels([unique_years[i] for i in range(0, len(unique_years), step)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Station")
    ax.set_title(title)

    # Legend
    labels = ["No data", "Train", "Val spatial", "Test spatial",
              "Val temporal", "Test temporal", "Unused"]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
             fontsize=7, frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Split heatmap saved to {save_path}")
    plt.close(fig)
    return fig


def plot_station_proportional_split_daily_raster(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    train_frac: float = config.SITE_TRAIN_FRAC,
    val_frac: float = config.SITE_VAL_FRAC,
    save_path=None,
    title: str = "Site Model Station-Proportional Split (per-day)",
):
    import datetime as _dt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    unique_stations = sorted(set(str(s) for s in stations))
    idx_all = np.arange(len(stations))

    xs = []
    ys = []
    cs = []

    c_train = "#4c72b0"
    c_val = "#55a868"
    c_test = "#c44e52"
    c_none = "white"

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
        tr_idx = sorted_idx[:n_train]
        va_idx = sorted_idx[n_train:n_train + n_val]
        te_idx = sorted_idx[n_train + n_val:]

        for k in tr_idx:
            xs.append(mdates.date2num(_dt.date(int(years[k]), int(months[k]), int(days[k]))))
            ys.append(si)
            cs.append(c_train)
        for k in va_idx:
            xs.append(mdates.date2num(_dt.date(int(years[k]), int(months[k]), int(days[k]))))
            ys.append(si)
            cs.append(c_val)
        for k in te_idx:
            xs.append(mdates.date2num(_dt.date(int(years[k]), int(months[k]), int(days[k]))))
            ys.append(si)
            cs.append(c_test)

    fig, ax = plt.subplots(figsize=(16, max(5, len(unique_stations) * 0.35)))
    if len(xs) > 0:
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
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        print(f"  Daily split raster saved to {save_path}")
    plt.close(fig)
    return fig


def plot_station_proportional_cv_folds_heatmap(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    cv_folds: int,
    train_frac: float = config.SITE_TRAIN_FRAC,
    val_frac: float = config.SITE_VAL_FRAC,
    save_path=None,
    title: str = "Site Model CV Folds (expanding-window)",
):
    """Visualize expanding-window folds per station based on actual timestamps.

    Produces a heatmap with rows = station×fold and columns = year.
    Colours: 0 no data, 1 train, 2 val.

    Notes:
    - This is computed from the station's *train+val* timeline (i.e., excluding test)
      so that folds are forward-chaining and data-driven.
    - Test is not shown here because folds are about how train/val are formed.
    """
    if cv_folds <= 1:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

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
        if len(sorted_idx) == 0:
            for f in range(1, cv_folds + 1):
                row_labels.append(f"{st}  [fold {f}]")
            continue

        n = len(sorted_idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        trainval_sorted = sorted_idx[: n_train + n_val]

        folds = _expanding_folds(trainval_sorted, cv_folds)
        for f in range(1, cv_folds + 1):
            row_labels.append(f"{st}  [fold {f}]")
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

    fig, ax = plt.subplots(figsize=(max(14, len(unique_years) * 0.22),
                                    max(7, n_rows * 0.18)))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6)
    step = max(1, len(unique_years) // 15)
    ax.set_xticks(range(0, len(unique_years), step))
    ax.set_xticklabels([unique_years[i] for i in range(0, len(unique_years), step)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Station × Fold")
    ax.set_title(title)

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  CV fold heatmap saved to {save_path}")
    plt.close(fig)
    return fig


def plot_station_proportional_split_heatmap(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    train_frac: float = config.SITE_TRAIN_FRAC,
    val_frac: float = config.SITE_VAL_FRAC,
    save_path=None,
    title: str = "Site Model Station-Proportional Split",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

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

    # Reduce presence flags into a single label per year.
    # Priority is chosen to avoid the misleading "no train" appearance when a year
    # contains both train+test samples (common when the split boundary falls within a year).
    grid[has_test] = 3
    grid[has_val] = 2
    grid[has_train] = 1

    colours = ["white", "#4c72b0", "#55a868", "#c44e52"]
    labels = ["No data", "Train", "Val", "Test"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(14, len(unique_years) * 0.22),
                                    max(5, len(unique_stations) * 0.35)))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_yticks(range(len(unique_stations)))
    ax.set_yticklabels(unique_stations, fontsize=7)
    step = max(1, len(unique_years) // 15)
    ax.set_xticks(range(0, len(unique_years), step))
    ax.set_xticklabels([unique_years[i] for i in range(0, len(unique_years), step)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Station")
    ax.set_title(title)

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Split heatmap saved to {save_path}")
    plt.close(fig)
    return fig
