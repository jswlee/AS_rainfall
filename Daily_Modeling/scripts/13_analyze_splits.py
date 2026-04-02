"""
Deep split analysis for Daily_Modeling spatio-temporal data splits.

Outputs a reproducibility-focused audit of station/year/sample separation to:
    Daily_Modeling/output/eda/split_analysis/

Run:
    python -m Daily_Modeling.scripts.13_analyze_splits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz
from Daily_Modeling.data_utils.load_raw import load_station_metadata
from Daily_Modeling.data_utils.splits import (
    assign_station_groups,
    compute_station_year_ranges,
    compute_year_boundaries,
    spatiotemporal_split,
)
from Daily_Modeling.utils.device import select_device
from Daily_Modeling.utils.io_utils import save_json
from Daily_Modeling.utils.visualization import (
    plot_split_heatmap,
    plot_split_year_counts,
    plot_station_role_map,
    plot_station_sample_counts,
)


OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "eda" / "split_analysis"


def _to_station_role_df(
    stations: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    rain_mm: np.ndarray,
    station_groups: Dict[str, str],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "station": [str(s) for s in stations],
            "year": years.astype(int),
            "month": months.astype(int),
            "day": days.astype(int),
            "rain_mm": rain_mm.astype(float),
        }
    )
    df["station_role"] = df["station"].map(lambda s: station_groups.get(str(s), "train"))
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    return df


def _assign_split_labels(
    df: pd.DataFrame,
    train_years: tuple[int, int],
    val_years: tuple[int, int],
    test_years: tuple[int, int],
) -> pd.Series:
    y = df["year"].to_numpy(dtype=int)
    role = df["station_role"].astype(str).to_numpy()

    labels = np.full(len(df), "unused", dtype=object)
    labels[(role == "train") & (y >= train_years[0]) & (y <= train_years[1])] = "train"
    labels[(role == "val") & (y >= val_years[0]) & (y <= val_years[1])] = "val_spatial"
    labels[(role == "test") & (y >= test_years[0]) & (y <= test_years[1])] = "test_spatial"
    labels[(role == "train") & (y >= val_years[0]) & (y <= val_years[1])] = "val_temporal"
    labels[(role == "train") & (y >= test_years[0]) & (y <= test_years[1])] = "test_temporal"
    return pd.Series(labels, index=df.index, name="split")


def _station_year_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st, g in df.groupby("station", sort=True):
        years = g["year"].to_numpy(dtype=int)
        dates = g["date"].dropna()
        rain = g["rain_mm"].to_numpy(dtype=float)
        rows.append(
            {
                "station": st,
                "role": str(g["station_role"].iloc[0]),
                "n_samples": int(len(g)),
                "n_years": int(pd.Series(years).nunique()),
                "first_year": int(years.min()),
                "last_year": int(years.max()),
                "first_date": str(dates.min().date()) if len(dates) else None,
                "last_date": str(dates.max().date()) if len(dates) else None,
                "pct_zero": float(100.0 * np.mean(rain <= 0.0)),
                "mean_mm": float(np.mean(rain)),
                "p95_mm": float(np.quantile(rain, 0.95)),
                "max_mm": float(np.max(rain)),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "station"]).reset_index(drop=True)


def _split_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_order = ["train", "val_spatial", "test_spatial", "val_temporal", "test_temporal", "unused"]
    total_n = len(df)
    for split_name in split_order:
        g = df[df["split"] == split_name]
        if g.empty:
            rows.append({"split": split_name, "n_samples": 0, "pct_samples": 0.0})
            continue
        rain = g["rain_mm"].to_numpy(dtype=float)
        rows.append(
            {
                "split": split_name,
                "n_samples": int(len(g)),
                "pct_samples": float(100.0 * len(g) / max(total_n, 1)),
                "n_stations": int(g["station"].nunique()),
                "n_years": int(g["year"].nunique()),
                "min_year": int(g["year"].min()),
                "max_year": int(g["year"].max()),
                "pct_zero": float(100.0 * np.mean(rain <= 0.0)),
                "mean_mm": float(np.mean(rain)),
                "std_mm": float(np.std(rain)),
                "p50_mm": float(np.quantile(rain, 0.50)),
                "p90_mm": float(np.quantile(rain, 0.90)),
                "p95_mm": float(np.quantile(rain, 0.95)),
                "p99_mm": float(np.quantile(rain, 0.99)),
                "max_mm": float(np.max(rain)),
            }
        )
    return pd.DataFrame(rows)


def _station_by_split_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["station", "split"], observed=False)
        .size()
        .rename("n_samples")
        .reset_index()
        .sort_values(["station", "split"])
    )
    return out


def _year_by_split_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["year", "split"], observed=False)
        .size()
        .rename("n_samples")
        .reset_index()
        .sort_values(["year", "split"])
    )
    return out


def _station_year_grid(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["station", "year", "split"], observed=False)
        .size()
        .rename("n_samples")
        .reset_index()
        .sort_values(["station", "year", "split"])
    )
    return g


def _compute_temporal_overlap_checks(
    df: pd.DataFrame,
    train_years: tuple[int, int],
    val_years: tuple[int, int],
    test_years: tuple[int, int],
) -> dict:
    train_stations = sorted(df.loc[df["station_role"] == "train", "station"].unique().tolist())
    val_stations = sorted(df.loc[df["station_role"] == "val", "station"].unique().tolist())
    test_stations = sorted(df.loc[df["station_role"] == "test", "station"].unique().tolist())

    test_temporal_stations = sorted(df.loc[df["split"] == "test_temporal", "station"].unique().tolist())
    val_temporal_stations = sorted(df.loc[df["split"] == "val_temporal", "station"].unique().tolist())
    val_spatial_stations = sorted(df.loc[df["split"] == "val_spatial", "station"].unique().tolist())
    test_spatial_stations = sorted(df.loc[df["split"] == "test_spatial", "station"].unique().tolist())

    checks = {
        "train_vs_val_station_overlap_count": int(len(set(train_stations) & set(val_stations))),
        "train_vs_test_station_overlap_count": int(len(set(train_stations) & set(test_stations))),
        "val_vs_test_station_overlap_count": int(len(set(val_stations) & set(test_stations))),
        "val_temporal_uses_only_train_stations": bool(set(val_temporal_stations).issubset(set(train_stations))),
        "test_temporal_uses_only_train_stations": bool(set(test_temporal_stations).issubset(set(train_stations))),
        "val_spatial_station_set_equals_role_val": sorted(val_spatial_stations) == sorted(val_stations),
        "test_spatial_station_set_equals_role_test": sorted(test_spatial_stations) == sorted(test_stations),
        "train_year_range": list(train_years),
        "val_year_range": list(val_years),
        "test_year_range": list(test_years),
        "train_val_year_overlap_count": int(max(0, min(train_years[1], val_years[1]) - max(train_years[0], val_years[0]) + 1)),
        "train_test_year_overlap_count": int(max(0, min(train_years[1], test_years[1]) - max(train_years[0], test_years[0]) + 1)),
        "val_test_year_overlap_count": int(max(0, min(val_years[1], test_years[1]) - max(val_years[0], test_years[0]) + 1)),
        "test_temporal_is_spatially_distinct_from_train": False,
        "test_temporal_is_temporal_only_holdout": True,
        "explanation": (
            "test_temporal uses train-role stations in held-out test years, so it is temporally distinct "
            "but not spatially distinct from train. It can still look different spatially in aggregate if the "
            "available train-role stations are unevenly sampled across years, but by design it is not a spatial holdout."
        ),
    }
    return checks


def _pairwise_station_distance_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def _station_distance_summary(station_df: pd.DataFrame, station_meta: Dict[str, dict]) -> pd.DataFrame:
    coords = []
    for _, row in station_df.iterrows():
        st = row["station"]
        meta = station_meta.get(st)
        if meta is None:
            continue
        coords.append(
            {
                "station": st,
                "role": row["role"],
                "latitude": float(meta["latitude"]),
                "longitude": float(meta["longitude"]),
            }
        )
    cdf = pd.DataFrame(coords)
    rows = []
    roles = sorted(cdf["role"].unique().tolist()) if not cdf.empty else []
    for src_role in roles:
        src = cdf[cdf["role"] == src_role]
        for dst_role in roles:
            dst = cdf[cdf["role"] == dst_role]
            dists = []
            for _, srow in src.iterrows():
                for _, drow in dst.iterrows():
                    if src_role == dst_role and srow["station"] == drow["station"]:
                        continue
                    dists.append(
                        _pairwise_station_distance_km(
                            srow["latitude"], srow["longitude"], drow["latitude"], drow["longitude"]
                        )
                    )
            if len(dists) == 0:
                rows.append({"src_role": src_role, "dst_role": dst_role, "n_pairs": 0})
            else:
                arr = np.asarray(dists, dtype=float)
                rows.append(
                    {
                        "src_role": src_role,
                        "dst_role": dst_role,
                        "n_pairs": int(len(arr)),
                        "min_km": float(arr.min()),
                        "p25_km": float(np.quantile(arr, 0.25)),
                        "median_km": float(np.quantile(arr, 0.50)),
                        "p75_km": float(np.quantile(arr, 0.75)),
                        "max_km": float(arr.max()),
                    }
                )
    return pd.DataFrame(rows)


def _nearest_cross_role_distances(station_df: pd.DataFrame, station_meta: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    role_map = station_df.set_index("station")["role"].to_dict()
    stations = station_df["station"].tolist()
    for st in stations:
        meta = station_meta.get(st)
        if meta is None:
            continue
        src_role = role_map[st]
        for dst_role in sorted(set(role_map.values())):
            if dst_role == src_role:
                continue
            candidates = [s for s in stations if role_map[s] == dst_role and s in station_meta]
            if not candidates:
                rows.append({"station": st, "src_role": src_role, "dst_role": dst_role, "nearest_station": None, "nearest_km": None})
                continue
            dists = []
            for cand in candidates:
                cm = station_meta[cand]
                d = _pairwise_station_distance_km(meta["latitude"], meta["longitude"], cm["latitude"], cm["longitude"])
                dists.append((cand, float(d)))
            nearest_station, nearest_km = min(dists, key=lambda x: x[1])
            rows.append(
                {
                    "station": st,
                    "src_role": src_role,
                    "dst_role": dst_role,
                    "nearest_station": nearest_station,
                    "nearest_km": float(nearest_km),
                }
            )
    return pd.DataFrame(rows).sort_values(["src_role", "station", "dst_role"]).reset_index(drop=True)


def _build_text_report(
    df: pd.DataFrame,
    station_df: pd.DataFrame,
    split_df: pd.DataFrame,
    overlap_checks: dict,
    nearest_df: pd.DataFrame,
) -> str:
    train_stations = station_df.loc[station_df["role"] == "train", "station"].tolist()
    val_stations = station_df.loc[station_df["role"] == "val", "station"].tolist()
    test_stations = station_df.loc[station_df["role"] == "test", "station"].tolist()

    lines = []
    lines.append("Daily_Modeling split analysis")
    lines.append("=" * 80)
    lines.append(f"Total samples: {len(df):,}")
    lines.append(f"Unique stations: {df['station'].nunique()}")
    lines.append(f"Global year range: {int(df['year'].min())}-{int(df['year'].max())}")
    lines.append("")
    lines.append("Station role assignment")
    lines.append("-" * 80)
    lines.append(f"Train stations ({len(train_stations)}): {', '.join(train_stations)}")
    lines.append(f"Val stations   ({len(val_stations)}): {', '.join(val_stations)}")
    lines.append(f"Test stations  ({len(test_stations)}): {', '.join(test_stations)}")
    lines.append("")
    lines.append("Split overlap checks")
    lines.append("-" * 80)
    for k, v in overlap_checks.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Key interpretation")
    lines.append("-" * 80)
    lines.append(
        "test_temporal is not spatially held out by design. It uses train-role stations in held-out test years. "
        "If it appears spatially distinct in visual summaries, that reflects coverage imbalance or missing data over time, not the split rule itself."
    )
    lines.append("")
    lines.append("Split summary table")
    lines.append("-" * 80)
    lines.append(split_df.to_string(index=False))
    lines.append("")
    lines.append("Nearest cross-role station distances (km): first 20 rows")
    lines.append("-" * 80)
    if nearest_df.empty:
        lines.append("No station metadata available for spatial distance checks.")
    else:
        lines.append(nearest_df.head(20).to_string(index=False))
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = select_device()
    tensors, meta = load_tensors_from_npz(device=device)
    station_meta = load_station_metadata()

    stations = meta["stations"]
    years = meta["years"].astype(int)
    months = meta["months"].astype(int)
    days = meta["days"].astype(int)
    rain_mm = tensors["targets"].detach().cpu().numpy().astype(float)

    train_years, val_years, test_years = compute_year_boundaries(years)
    station_year_ranges = compute_station_year_ranges(stations, years)
    station_groups = assign_station_groups(
        sorted(set(str(s) for s in stations)),
        station_year_ranges=station_year_ranges,
        val_years=val_years,
        test_years=test_years,
    )
    splits = spatiotemporal_split(
        stations,
        years,
        station_groups,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
    )

    df = _to_station_role_df(stations, years, months, days, rain_mm, station_groups)
    df["split"] = _assign_split_labels(df, train_years, val_years, test_years)

    station_df = _station_year_table(df)
    split_df = _split_summary_table(df)
    station_split_df = _station_by_split_counts(df)
    year_split_df = _year_by_split_counts(df)
    station_year_df = _station_year_grid(df)
    overlap_checks = _compute_temporal_overlap_checks(df, train_years, val_years, test_years)
    distance_df = _station_distance_summary(station_df, station_meta)
    nearest_df = _nearest_cross_role_distances(station_df, station_meta)

    station_df.to_csv(OUT_DIR / "station_role_summary.csv", index=False)
    split_df.to_csv(OUT_DIR / "split_summary.csv", index=False)
    station_split_df.to_csv(OUT_DIR / "station_by_split_counts.csv", index=False)
    year_split_df.to_csv(OUT_DIR / "year_by_split_counts.csv", index=False)
    station_year_df.to_csv(OUT_DIR / "station_year_split_grid.csv", index=False)
    distance_df.to_csv(OUT_DIR / "station_role_distance_summary.csv", index=False)
    nearest_df.to_csv(OUT_DIR / "nearest_cross_role_station_distances.csv", index=False)

    split_payload = {
        "train_years": list(train_years),
        "val_years": list(val_years),
        "test_years": list(test_years),
        "station_groups": station_groups,
        "station_year_ranges": {k: list(v) for k, v in station_year_ranges.items()},
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "overlap_checks": overlap_checks,
    }
    save_json(split_payload, OUT_DIR / "split_reproducibility_summary.json")

    report = _build_text_report(df, station_df, split_df, overlap_checks, nearest_df)
    (OUT_DIR / "split_analysis_report.txt").write_text(report, encoding="utf-8")

    plot_split_heatmap(
        stations,
        years,
        station_groups,
        train_years,
        val_years,
        test_years,
        save_path=OUT_DIR / "split_heatmap_spatiotemporal.png",
        title="LAND spatio-temporal split audit",
    )
    plot_station_sample_counts(
        stations,
        {k: v for k, v in splits.items() if len(v) > 0},
        save_path=OUT_DIR / "station_sample_counts_by_split.png",
    )
    plot_split_year_counts(df, OUT_DIR / "split_year_counts.png")
    plot_station_role_map(station_df, station_meta, OUT_DIR / "station_role_map.png")

    print(f"Saved split analysis to {OUT_DIR}")
    print(f"  - split_analysis_report.txt")
    print(f"  - split_reproducibility_summary.json")
    print(f"  - split_summary.csv")
    print(f"  - station_role_summary.csv")
    print(f"  - station_by_split_counts.csv")
    print(f"  - year_by_split_counts.csv")
    print(f"  - station_year_split_grid.csv")
    print(f"  - station_role_distance_summary.csv")
    print(f"  - nearest_cross_role_station_distances.csv")
    print(f"  - split_heatmap_spatiotemporal.png")
    print(f"  - station_sample_counts_by_split.png")
    print(f"  - year_counts_by_split.png")
    print(f"  - station_role_map.png")


if __name__ == "__main__":
    main()
