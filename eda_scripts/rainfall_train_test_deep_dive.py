"""Deep dive: why do train vs test weekly rainfall distributions differ?

Data lineage: raw_data/AS/daily_wide_4302025.csv -> final_rainfall_per_station/*.csv
-> load_daily_rainfall (inches*25.4, NaN dropped) -> 7-day ISO weeks (complete only).

Run from repo root:  venv/Scripts/python.exe eda_scripts/rainfall_train_test_deep_dive.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAIN_DIR = ROOT / "raw_data" / "AS" / "final_rainfall_per_station"
META = ROOT / "raw_data" / "AS" / "station_locations.csv"
TEST_STATIONS = ["aasu_UH", "afono_UH", "aunuu_UH", "poloa_UH", "vaipito_UH"]
TRAIN_YEAR_END = 2016

meta = pd.read_csv(META).rename(columns={"Station": "station"})
meta["elev_ft"] = pd.to_numeric(meta["elev_ft"].astype(str).str.replace(",", ""), errors="coerce")
meta = meta.set_index("station")

# ---------------------------------------------------------------- load daily
daily = {}
for f in sorted(RAIN_DIR.glob("*.csv")):
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y")
    if "precip_in" in df.columns:
        df["rain_mm"] = pd.to_numeric(df["precip_in"], errors="coerce") * 25.4
    else:
        df["rain_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
    daily[f.stem] = df[["date", "rain_mm"]].set_index("date").sort_index()

print("=" * 90)
print("1. PER-STATION DAILY RECORDS  (network, span, observed days, zero stats)")
print("=" * 90)
rows = []
for name, df in daily.items():
    obs = df["rain_mm"].dropna()
    in_meta = name in meta.index
    rows.append({
        "station": name,
        "org": meta.loc[name, "Organization"] if in_meta else "??",
        "elev_ft": meta.loc[name, "elev_ft"] if in_meta else np.nan,
        "span": f"{obs.index.min().date()}..{obs.index.max().date()}" if len(obs) else "-",
        "n_days": len(obs),
        "na_days": int(df["rain_mm"].isna().sum()),
        "zero_pct": 100 * (obs <= 0).mean() if len(obs) else np.nan,
        "mean_mm": obs.mean(),
        "max_mm": obs.max(),
        "min_gap_days": 0,
        "test": name in TEST_STATIONS,
    })
st = pd.DataFrame(rows).set_index("station")
print(st.round(2).to_string())

# value granularity: are daily values quantized to 0.01in?
print("\nDaily-value quantization check (fraction of nonzero obs that are multiples of 0.01in):")
for name in ["vaipito2000", "aunuu", "pioa_afono", "siufaga_WRCC", "aasu_UH", "vaipito_UH"]:
    obs = daily[name]["rain_mm"].dropna()
    nz = obs[obs > 0] / 25.4  # back to inches
    frac_001 = np.mean(np.isclose((nz * 100).round(0), nz * 100, atol=1e-6))
    print(f"  {name:16s} multiples-of-0.01in: {frac_001:.3f}   "
          f"min nonzero: {nz.min():.4f}in ({nz.min()*25.4:.3f}mm)")

# ------------------------------------------------- zero runs + accumulation
print("\n" + "=" * 90)
print("2. ZERO-RUN / ACCUMULATION SIGNATURES")
print("   Long runs of zeros ending in a large total suggest multi-day accumulation")
print("   written as zeros; zeros adjacent to NA gaps suggest missing-as-zero risk.")
print("=" * 90)

def run_lengths(series):
    """Lengths of consecutive zero runs in the observed (non-NA) series."""
    vals = series.values
    runs, cur = [], 0
    for v in vals:
        if np.isnan(v):
            if cur:
                runs.append(cur)
                cur = 0
            continue
        if v <= 0:
            cur += 1
        else:
            if cur:
                runs.append(cur)
                cur = 0
    if cur:
        runs.append(cur)
    return runs

print(f"{'station':16s} {'zero_runs':>9s} {'run>=7d':>8s} {'maxrun':>7s} "
      f"{'postrun>run':>11s} {'zeros_in_runs>=7':>16s}")
for name, df in sorted(daily.items()):
    s = df["rain_mm"]
    runs = run_lengths(s)
    if not runs:
        continue
    runs = np.asarray(runs)
    # accumulation check: total rain in week AFTER a >=5-day zero run vs typical weekly rain
    vals = s.values
    accum_flags = 0
    n_long = 0
    i = 0
    while i < len(vals):
        if not np.isnan(vals[i]) and vals[i] <= 0:
            j = i
            while j < len(vals) and not np.isnan(vals[j]) and vals[j] <= 0:
                j += 1
            if j - i >= 5 and j < len(vals) and not np.isnan(vals[j]):
                n_long += 1
                # is the day after the run unusually wet? (>3x station daily mean)
                if vals[j] > 3 * np.nanmean(vals[vals > 0]):
                    accum_flags += 1
            i = j
        else:
            i += 1
    print(f"{name:16s} {len(runs):9d} {(runs >= 7).sum():8d} {runs.max():7d} "
          f"{accum_flags:5d}/{n_long:<5d} {int(runs[runs >= 7].sum()):16d}")

# ------------------------------------------------------- weekly aggregation
print("\n" + "=" * 90)
print("3. WEEKLY TOTALS PER STATION (complete 7-day ISO weeks, as in pipeline)")
print("=" * 90)

weekly_rows = []
for name, df in daily.items():
    s = df["rain_mm"].dropna()
    if s.empty:
        continue
    wk = s.resample("W-SUN", label="left", closed="left")  # ISO week starting Monday
    wsum = wk.sum()
    wcnt = wk.count()
    wsum = wsum[wcnt == 7]
    for monday, val in wsum.items():
        weekly_rows.append({"station": name, "week": monday, "year": monday.year,
                            "rain_mm": val})
wk_df = pd.DataFrame(weekly_rows)
wk_df["role"] = np.where(wk_df["station"].isin(TEST_STATIONS), "test_station", "train_station")
wk_df["split"] = np.select(
    [(~wk_df["station"].isin(TEST_STATIONS)) & (wk_df["year"] <= TRAIN_YEAR_END),
     (wk_df["station"].isin(TEST_STATIONS)) & (wk_df["year"] > TRAIN_YEAR_END),
     (~wk_df["station"].isin(TEST_STATIONS)) & (wk_df["year"] > TRAIN_YEAR_END)],
    ["TRAIN", "TEST", "BRIDGE(post2016 train stn)"], default="unused")

summ = wk_df.groupby(["split", "station"]).agg(
    n=("rain_mm", "size"), yr0=("year", "min"), yr1=("year", "max"),
    zero_pct=("rain_mm", lambda x: 100 * (x <= 0).mean()),
    lt5_pct=("rain_mm", lambda x: 100 * (x < 5).mean()),
    mean=("rain_mm", "mean"), med=("rain_mm", "median"),
    q90=("rain_mm", lambda x: x.quantile(0.9)),
    maxx=("rain_mm", "max"),
).round(2).reset_index().sort_values(["split", "zero_pct"], ascending=[True, False])
print(summ.to_string(index=False))

print("\nPooled split comparison:")
pooled = wk_df[wk_df["split"] != "unused"].groupby("split")["rain_mm"].agg(
    ["size", "mean", "median", "std",
     lambda x: 100 * (x <= 0).mean(), lambda x: 100 * (x < 5).mean(),
     lambda x: x.quantile(0.9), lambda x: x.quantile(0.99)])
pooled.columns = ["n", "mean", "median", "std", "zero%", "<5mm%", "q90", "q99"]
print(pooled.round(2).to_string())

# ------------------------------------------------- co-located network pairs
print("\n" + "=" * 90)
print("4. CO-LOCATED PAIRS: same site, different network/era")
print("   Separates 'UH network measures differently' from '2017+ was wetter'.")
print("=" * 90)
pairs = [("aunuu", "aunuu_UH"), ("vaipito2000", "vaipito_UH"),
         ("vaipito_res", "vaipito_UH"), ("pioa_afono", "afono_UH"),
         ("aasufou80", "aasu_UH"), ("aoloafou", "aasu_UH"),
         ("maloata", "poloa_UH")]
for old, new in pairs:
    a = wk_df[(wk_df["station"] == old)]["rain_mm"]
    b = wk_df[(wk_df["station"] == new)]["rain_mm"]
    if len(a) == 0 or len(b) == 0:
        continue
    print(f"{old:14s} (n={len(a):4d}) zero%={100*(a<=0).mean():5.1f} mean={a.mean():6.1f} "
          f"med={a.median():6.1f} | {new:10s} (n={len(b):4d}) zero%={100*(b<=0).mean():5.1f} "
          f"mean={b.mean():6.1f} med={b.median():6.1f}")

# ------------------------------------------------------------- seasonality
print("\n" + "=" * 90)
print("5. MONTHLY CLIMATOLOGY BY GROUP (weekly mean, mm)")
print("=" * 90)
wk_df["month"] = wk_df["week"].dt.month
clim = wk_df[wk_df["split"].isin(["TRAIN", "TEST", "BRIDGE(post2016 train stn)"])].pivot_table(
    index="month", columns="split", values="rain_mm", aggfunc="mean")
print(clim.round(1).to_string())

zero_clim = wk_df[wk_df["split"].isin(["TRAIN", "TEST"])].assign(
    zero=lambda d: d["rain_mm"] <= 0).pivot_table(
    index="month", columns="split", values="zero", aggfunc="mean") * 100
print("\nZero-week % by month:")
print(zero_clim.round(2).to_string())

# ------------------------------------------------------------- yearly means
print("\n" + "=" * 90)
print("6. YEARLY MEAN WEEKLY RAINFALL: pooled train stations vs test stations")
print("=" * 90)
yr = wk_df[wk_df["split"] != "unused"].groupby(["year", "split"])["rain_mm"].mean().unstack()
yr["n_train_stations"] = wk_df[wk_df["split"] == "TRAIN"].groupby("year")["station"].nunique()
print(yr.round(1).to_string())

# ------------------------------------------------- daily zero-rate by era
print("\n" + "=" * 90)
print("7. DAILY zero rate by station-era (daily resolution, not weekly)")
print("=" * 90)
rows = []
for name, df in daily.items():
    obs = df["rain_mm"].dropna()
    for era, m in [("pre2017", obs.index.year <= 2016), ("post2016", obs.index.year > 2016)]:
        sub = obs[m]
        if len(sub) < 100:
            continue
        rows.append({"station": name, "era": era, "n": len(sub),
                     "zero%": 100 * (sub <= 0).mean(),
                     "mean": sub.mean(), "med": sub.median()})
dz = pd.DataFrame(rows)
print(dz.pivot_table(index="station", columns="era", values=["zero%", "mean", "n"]).round(2).to_string())

# ----------------------------------- examine suspicious stations in detail
print("\n" + "=" * 90)
print("8. SUSPECT STATIONS: aunuu (12% zero) vs aunuu_UH (0%); vaipito2000")
print("=" * 90)
for name in ["aunuu", "aunuu_UH", "vaipito2000", "pioa_afono"]:
    obs = daily[name]["rain_mm"].dropna()
    by_year = obs.groupby(obs.index.year).agg(
        n="size", zero_pct=lambda x: 100 * (x <= 0).mean(), mean="mean")
    print(f"\n--- {name} ---")
    print(by_year.round(2).to_string())

# accumulation forensic: for the big-zero stations, check whether the day after
# a >=3-day zero run tends to equal ~ sum of a plausible multi-day event
print("\n9. ACCUMULATION TEST: value on first wet day after >=3-day zero run,")
print("   expressed in units of station median wet-day rain (>2 = suspicious):")
for name in ["aunuu", "vaipito2000", "pioa_afono", "fagaitua", "vaipito_res", "aasu_UH"]:
    s = daily[name]["rain_mm"]
    vals = s.values
    med_wet = np.nanmedian(vals[vals > 0])
    ratios = []
    i = 0
    while i < len(vals):
        if not np.isnan(vals[i]) and vals[i] <= 0:
            j = i
            while j < len(vals) and not np.isnan(vals[j]) and vals[j] <= 0:
                j += 1
            if 3 <= j - i <= 20 and j < len(vals) and not np.isnan(vals[j]):
                ratios.append(vals[j] / med_wet)
            i = j
        else:
            i += 1
    if ratios:
        r = np.asarray(ratios)
        print(f"  {name:14s} n_runs={len(r):4d} median_ratio={np.median(r):.2f} "
              f"frac>2x={100*(r>2).mean():.0f}%  frac>4x={100*(r>4).mean():.0f}%")

# check if weekly zeros come in runs (multi-week dry spells) per station
print("\n10. Do weekly zeros cluster in runs? (run-length histogram, zero weeks only)")
for name in ["aunuu", "vaipito2000", "pioa_afono", "fagaitua", "vaipito_res", "TRAIN_ALL", "TEST_ALL"]:
    if name == "TRAIN_ALL":
        zz = wk_df[(wk_df["split"] == "TRAIN")]
    elif name == "TEST_ALL":
        zz = wk_df[(wk_df["split"] == "TEST")]
    else:
        zz = wk_df[wk_df["station"] == name]
    zz = zz.sort_values("week")
    isz = (zz["rain_mm"] <= 0).values
    runs, cur = [], 0
    for v in isz:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    runs = np.asarray(runs) if runs else np.array([0])
    print(f"  {name:14s} zero_weeks={isz.sum():4d} in {len(runs)} runs, "
          f"run_len_hist={dict(zip(*np.unique(runs, return_counts=True))) if len(runs) else {}}")
