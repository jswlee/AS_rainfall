"""Second pass: stats on the assembled weekly_dataset.npz (what the model sees)
plus artifact forensics on the raw daily files.

Run: venv/Scripts/python.exe eda_scripts/rainfall_npz_and_artifact_checks.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAIN_DIR = ROOT / "raw_data" / "AS" / "final_rainfall_per_station"
TEST_STATIONS = ["aasu_UH", "afono_UH", "aunuu_UH", "poloa_UH", "vaipito_UH"]

npz = np.load(ROOT / "LAND_AS" / "data" / "weekly_dataset.npz", allow_pickle=True)
stations = npz["stations"].astype(str)
years = npz["years"].astype(int)
months = npz["months"].astype(int)
days = npz["days"].astype(int)
rain = npz["rainfall_mm_raw"].astype(float)
week = pd.to_datetime({"year": years, "month": months, "day": days})
df = pd.DataFrame({"station": stations, "week": week, "year": years, "rain_mm": rain})
df["split"] = np.select(
    [(~df.station.isin(TEST_STATIONS)) & (df.year <= 2016),
     (df.station.isin(TEST_STATIONS)) & (df.year > 2016),
     (~df.station.isin(TEST_STATIONS)) & (df.year > 2016)],
    ["TRAIN", "TEST", "BRIDGE"], default="unused")

print("A. NPZ per-station stats (what the model actually trains/tests on)")
g = df[df.split != "unused"].groupby(["split", "station"]).agg(
    n=("rain_mm", "size"), yr0=("year", "min"), yr1=("year", "max"),
    zero_pct=("rain_mm", lambda x: 100 * (x <= 0).mean()),
    lt5=("rain_mm", lambda x: 100 * (x < 5).mean()),
    mean=("rain_mm", "mean"), med=("rain_mm", "median"),
    q90=("rain_mm", lambda x: x.quantile(.9)),
).round(2).sort_values(["split", "zero_pct"], ascending=[True, False])
print(g.to_string())

print("\nB. Pooled")
print(df[df.split != "unused"].groupby("split")["rain_mm"].agg(
    n="size", mean="mean", median="median", std="std",
    zero_pct=lambda x: 100 * (x <= 0).mean(),
    lt5_pct=lambda x: 100 * (x < 5).mean(),
    q90=lambda x: x.quantile(.9), q99=lambda x: x.quantile(.99),
    maxx="max").round(2).to_string())

print("\nC. Weekly-quantization signature: is a weekly total divisible into")
print("   daily steps? Test set should show smoother low end (finer daily")
print("   reporting). Fraction of weekly totals < 25.4mm that equal a small")
print("   multiple of 2.54mm (0.1in):")
for sp in ["TRAIN", "TEST", "BRIDGE"]:
    v = df.loc[df.split == sp, "rain_mm"].values
    low = v[v < 25.4]
    frac = np.mean(np.isclose(low / 2.54, np.round(low / 2.54), atol=1e-4)) if len(low) else np.nan
    print(f"   {sp}: n<25.4mm={len(low)}, frac ~ multiples of 2.54mm: {frac:.2f}")

# ---------------------------------------------------------------- raw daily
daily = {}
for f in sorted(RAIN_DIR.glob("*.csv")):
    d = pd.read_csv(f)
    d["date"] = pd.to_datetime(d["datetime"], format="%m/%d/%Y")
    col = "precip_in" if "precip_in" in d.columns else "precip_mm"
    scale = 25.4 if col == "precip_in" else 1.0
    d["rain_mm"] = pd.to_numeric(d[col], errors="coerce") * scale
    daily[f.stem] = d[["date", "rain_mm"]].set_index("date").sort_index()

print("\nD. Effective daily reporting resolution per station")
print("   (min nonzero daily value; ~2.54mm means 0.1in reporting step)")
rows = []
for name, d in daily.items():
    nz = d["rain_mm"].dropna()
    nz = nz[nz > 0]
    if len(nz) == 0:
        continue
    small = nz[nz <= 25.4]
    rows.append({"station": name, "min_nonzero_mm": nz.min(),
                 "pct_nonzero_multiple_of_2.54mm":
                     100 * np.mean(np.isclose(small / 2.54, np.round(small / 2.54), atol=0.02)) if len(small) else np.nan,
                 "daily_zero_pct": 100 * (d["rain_mm"].dropna() <= 0).mean()})
res = pd.DataFrame(rows).set_index("station").sort_values("min_nonzero_mm", ascending=False)
print(res.round(3).to_string())

print("\nE. afono_UH 53-day zero run: real drought or artifact?")
s = daily["afono_UH"]["rain_mm"]
vals = s.values; dates = s.index
i = 0
while i < len(vals):
    if not np.isnan(vals[i]) and vals[i] <= 0:
        j = i
        while j < len(vals) and not np.isnan(vals[j]) and vals[j] <= 0:
            j += 1
        if j - i >= 14:
            print(f"   afono_UH zero run {dates[i].date()}..{dates[j-1].date()} ({j-i}d); "
                  f"next wet day {vals[j]:.1f}mm" if j < len(vals) else f"   run to end")
        i = j
    else:
        i += 1
# neighbors over the same window
print("   neighbor station totals over afono_UH's longest run windows:")
for name in ["vaipito_UH", "aunuu_UH", "toa_ridge_WRCC", "siufaga_WRCC"]:
    d = daily[name]["rain_mm"]
    mask = (d.index >= "2022-08-01") & (d.index <= "2022-10-15")
    print(f"   {name:15s} 2022-08..10 mean daily={d[mask].mean():.1f}mm "
          f"zero%={100*(d[mask]<=0).mean():.0f}%  n={mask.sum()}")

print("\nF. Zero-run synchrony: do nearby stations share the same zero weeks?")
# pick overlapping-era pairs
def weekly_zero_set(name):
    s = daily[name]["rain_mm"].dropna()
    wsum = s.resample("W-SUN", label="left", closed="left").sum()
    wcnt = s.resample("W-SUN", label="left", closed="left").count()
    wsum = wsum[wcnt == 7]
    return set(wsum.index[wsum <= 0]), set(wsum.index)

pairs = [("aunuu", "fagaitua"), ("aunuu", "satala"), ("aunuu", "maloata"),
         ("vaipito2000", "vaipito_res"), ("vaipito2000", "malaeimi_1691"),
         ("pioa_afono", "aasufou80"), ("vaipito_res", "malaeimi_1691"),
         ("siufaga_WRCC", "toa_ridge_WRCC"),
         ("afono_UH", "vaipito_UH"), ("afono_UH", "aasu_UH"), ("afono_UH", "poloa_UH"),
         ("vaipito_UH", "poloa_UH"), ("aunuu_UH", "afono_UH")]
for a, b in pairs:
    za, wa = weekly_zero_set(a)
    zb, wb = weekly_zero_set(b)
    common = wa & wb
    if not common:
        print(f"   {a:14s} vs {b:14s}: no overlapping complete weeks")
        continue
    za_c = {w for w in common if w in za}
    zb_c = {w for w in common if w in zb}
    joint = za_c & zb_c
    # expected overlap if independent
    exp = len(za_c) * len(zb_c) / max(len(common), 1)
    print(f"   {a:14s} vs {b:14s}: overlap_wks={len(common):4d} "
          f"zeros {len(za_c):3d}/{len(zb_c):3d} joint={len(joint):3d} "
          f"(indep-expected {exp:.1f})  Jaccard={len(joint)/max(len(za_c|zb_c),1):.2f}")

print("\nG. vaipito2000 detail: decade-level collapse")
v = daily["vaipito2000"]["rain_mm"].dropna()
vv = pd.DataFrame({"mm": v})
vv["decade"] = (vv.index.year // 10) * 10
print(vv.groupby("decade")["mm"].agg(
    n="size", zero_pct=lambda x: 100 * (x <= 0).mean(), mean="mean",
    med="median", maxx="max").round(2).to_string())

print("\nH. Wet-week distributions: means conditioned on rain>0, by split")
for sp in ["TRAIN", "TEST", "BRIDGE"]:
    v = df.loc[(df.split == sp) & (df.rain_mm > 0), "rain_mm"]
    print(f"   {sp}: n={len(v)} mean={v.mean():.1f} med={v.median():.1f} "
          f"q90={v.quantile(.9):.1f} q99={v.quantile(.99):.1f}")

print("\nI. Seasonal amplitudes + post-2016 wet-season months in TEST")
tt = df[df.split == "TEST"]
print(tt.groupby(tt.week.dt.month)["rain_mm"].agg(["size", "mean",
      lambda x: 100 * (x <= 0).mean()]).round(1).rename(columns={"<lambda_0>": "zero%"}).to_string())
