"""
Deep Dive Diagnostics for LAND model data pipeline and training behavior.
Outputs to Daily_Modeling/output/Deep_Dive/
Run: python -m Daily_Modeling.scripts.deep_dive_diagnostics
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output" / "Deep_Dive"
OUT.mkdir(parents=True, exist_ok=True)

from Daily_Modeling import config
from Daily_Modeling.data_utils.dataset import load_tensors_from_npz, normalize_tensors
from Daily_Modeling.data_utils.splits import (
    assign_station_groups, spatiotemporal_split,
    compute_station_year_ranges, compute_year_boundaries,
)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("=" * 70); print("  1. LOADING DATA"); print("=" * 70)
device = torch.device("cpu")
tensors, meta = load_tensors_from_npz(device=device)
stations  = meta["stations"]
years     = meta["years"]
months    = meta["months"].astype(int)
variables = list(meta["variables"]) if len(meta["variables"]) else []

print(f"  Total samples  : {len(stations):,}")
print(f"  Unique stations: {len(np.unique(stations))}")
print(f"  Year range     : {int(years.min())}–{int(years.max())}")
print(f"  Climate shape  : {tuple(tensors['climate'].shape[1:])}")
print(f"  Local DEM      : {tuple(tensors['local_dem'].shape[1:])}")
print(f"  Regional DEM   : {tuple(tensors['regional_dem'].shape[1:])}")
print(f"  Variables ({len(variables)}): {variables}")

# ── 2. Splits ─────────────────────────────────────────────────────────────────
train_yr, val_yr, test_yr = compute_year_boundaries(years)
yr_ranges = compute_station_year_ranges(stations, years)
groups = assign_station_groups(
    sorted(set(str(s) for s in stations)),
    station_year_ranges=yr_ranges, val_years=val_yr, test_years=test_yr,
)
splits = spatiotemporal_split(stations, years, groups,
                              train_years=train_yr, val_years=val_yr, test_years=test_yr)
n_train = len(splits["train"])
train_stn = sum(1 for v in groups.values() if v == "train")
val_stn   = sum(1 for v in groups.values() if v == "val")
test_stn  = sum(1 for v in groups.values() if v == "test")

# ── 3. Target analysis (raw) ──────────────────────────────────────────────────
print("\n" + "=" * 70); print("  2. TARGET DISTRIBUTION (RAW mm)"); print("=" * 70)
raw = tensors["targets"].numpy()
tgt_tr = raw[splits["train"]]
tgt_va = raw[splits["val_spatial"]]
wet_tr = tgt_tr[tgt_tr > 0]

for name, arr in [("ALL", raw), ("TRAIN", tgt_tr), ("VAL_SP", tgt_va)]:
    pz = 100 * (arr == 0).mean()
    print(f"  {name:<8s}: n={len(arr):7,}  mean={arr.mean():7.2f}  "
          f"std={arr.std():7.2f}  max={arr.max():7.1f}  zero%={pz:.1f}%")

print(f"\n  Train WET: n={len(wet_tr):,}  mean={wet_tr.mean():.2f}  "
      f"median={np.median(wet_tr):.2f}  std={wet_tr.std():.2f}  "
      f"p95={np.percentile(wet_tr,95):.1f}  max={wet_tr.max():.1f}")

# ── 4. Normalisation ──────────────────────────────────────────────────────────
print("\n" + "=" * 70); print("  3. NORMALISATION AUDIT"); print("=" * 70)
clim_raw = tensors["climate"].numpy()
print(f"  Climate PRE-normalisation  (shape {clim_raw.shape}):")
print(f"  {'Channel':<35s} {'mean':>10s} {'std':>10s} {'NaN%':>7s}")
for i in range(clim_raw.shape[1]):
    vals = clim_raw[:, i].ravel()
    fin  = vals[np.isfinite(vals)]
    nan_pct = 100 * (1 - len(fin) / len(vals))
    vname = variables[i] if i < len(variables) else f"ch{i}"
    print(f"  {vname:<35s} {fin.mean():>10.3f} {fin.std():>10.3f} {nan_pct:>6.2f}%")

tensors, stats = normalize_tensors(tensors, splits["train"])
target_scale = stats["target_std_mm"]
norm_wet_tr = tgt_tr[tgt_tr > 0] / target_scale

print(f"\n  target_scale (train std) = {target_scale:.4f} mm")
print(f"  Normalised wet days: mean={norm_wet_tr.mean():.4f}  "
      f"std={norm_wet_tr.std():.4f}  max={norm_wet_tr.max():.4f}  "
      f"p95={np.percentile(norm_wet_tr, 95):.4f}")

clim_n = tensors["climate"].numpy()
clim_va = clim_n[splits["val_spatial"]]
clim_tr2 = clim_n[splits["train"]]
print(f"\n  Climate POST-norm train vs val shift:")
print(f"  {'Channel':<35s} {'tr_mean':>8s} {'va_mean':>8s} {'shift':>8s}")
for i in range(clim_n.shape[1]):
    tr_m = np.nanmean(clim_tr2[:, i])
    va_m = np.nanmean(clim_va[:, i]) if len(clim_va) else float("nan")
    shift = va_m - tr_m
    flag = " <<< LARGE" if abs(shift) > 0.5 else ""
    vname = variables[i] if i < len(variables) else f"ch{i}"
    print(f"  {vname:<35s} {tr_m:>8.3f} {va_m:>8.3f} {shift:>+8.3f}{flag}")

# ── 5. DEM audit ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70); print("  4. DEM PATCH AUDIT"); print("=" * 70)
for dk in ("local_dem", "regional_dem"):
    d = tensors[dk].numpy()
    tr_f = d[splits["train"]].ravel()
    tr_f = tr_f[np.isfinite(tr_f)]
    print(f"  {dk}  shape={d.shape}")
    print(f"    mean={tr_f.mean():.4f}  std={tr_f.std():.4f}  "
          f"min={tr_f.min():.3f}  max={tr_f.max():.3f}  "
          f"neg%={100*(tr_f<0).mean():.1f}%")

# ── 6. Architecture comparison ────────────────────────────────────────────────
print("\n" + "=" * 70); print("  5. ARCHITECTURE COMPARISON"); print("=" * 70)
C, H, W = tensors["climate"].shape[1:]
print(f"  Our climate input shape : {C}ch × {H}×{W} = {C*H*W} features")
print(f"  Paper climate input     : 16ch × 2×3 = 96 features (2×3 grid)")
print(f"  Our DEM inputs          : local {tuple(tensors['local_dem'].shape[1:])} "
      f"+ regional {tuple(tensors['regional_dem'].shape[1:])} (separate flat branches)")
print(f"  Paper DEM inputs        : 2 DEMs STACKED as 2-channel 10×10 image → Conv2d branch")
print(f"\n  ** KEY ARCH DIFFERENCE: Paper uses a Conv2d on a 2-channel DEM image.")
print(f"     We use separate linear branches on each DEM independently.")
print(f"     This means we lose any spatial structure WITHIN each DEM patch.")
print(f"\n  ** KEY LOSS DIFFERENCE:")
print(f"     Paper: trains final model with Gamma NLL on WET DAYS ONLY (no zero days)")
print(f"     Paper: skips Bernoulli – 'we skip the Bernoulli output to ensure spatially smooth output'")
print(f"     Ours : Bernoulli-Gamma (includes dry days in BCE term)")
print(f"\n  ** KEY TRAINING DIFFERENCE:")
print(f"     Paper: 30,000 gradient steps, no LR schedule, no early stopping, no val split")
bs_typical = 512
steps_per_ep = n_train // bs_typical
equiv_epochs = 30_000 / steps_per_ep
print(f"     Paper 30k steps @ bs=512 ≈ {equiv_epochs:.0f} epochs of our train set")
print(f"     Ours: max 100 epochs, early stop patience=20, cosine LR decay")

# ── 7. Station scale problem ──────────────────────────────────────────────────
print("\n" + "=" * 70); print("  6. STATION COUNT PROBLEM"); print("=" * 70)
print(f"  Total stations   : {len(groups)}")
print(f"  Train stations   : {train_stn}")
print(f"  Val stations     : {val_stn}   (held-out ENTIRELY)")
print(f"  Test stations    : {test_stn}   (held-out ENTIRELY)")
print(f"  Paper (Hawaii)   : ~1894 stations; 60 held out for val = 3.2%")
print(f"  Ours (AS)        : {len(groups)} stations; {val_stn+test_stn} held out = "
      f"{100*(val_stn+test_stn)/len(groups):.0f}%  ← MUCH LARGER FRACTION")
print(f"\n  With {val_stn} val stations held out, val_spatial tests BOTH temporal")
print(f"  AND spatial generalisation simultaneously. This is a very hard test")
print(f"  given only {train_stn} training stations for spatial coverage.")

# ── 8. Training dynamics analysis ────────────────────────────────────────────
print("\n" + "=" * 70); print("  7. TRAINING DYNAMICS ASSESSMENT"); print("=" * 70)
print(f"  Observed training pattern (typical trial):")
print(f"    Epoch 10: Train~0.81  Val~0.81  ValMAE~9.1mm")
print(f"    Epoch 20: Train~0.62  Val~0.97  ValMAE~9.7mm")
print(f"    Epoch 50: Train~0.16  Val~2.34  ValMAE~9.2mm")
print(f"\n  ANALYSIS:")
print(f"  - Train NLL drops consistently (model IS learning to fit the distribution)")
print(f"  - Val NLL diverges from epoch ~8-10 onward")
print(f"  - Val MAE (mm) is relatively FLAT (~9-10mm) across ALL epochs")
print(f"  - This means: Val MAE doesn't improve despite train NLL dropping 5x")
print(f"\n  CONCLUSION: The model is memorising station-specific patterns that")
print(f"  don't transfer to held-out spatial locations. The NLL divergence")
print(f"  is overconfidence (alpha/beta values tuned to training distribution).")
print(f"  The flat Val MAE suggests the mean prediction is not improving spatially.")
print(f"\n  ROOT CAUSES (ranked):")
print(f"  1. [CRITICAL] Too few training stations ({train_stn}) for spatial generalisation.")
print(f"     Paper had ~1834 train stations; we have {train_stn}.")
print(f"  2. [HIGH] val_spatial uses stations in BOTH held-out years AND held-out locations")
print(f"     — double shift. Consider using val_temporal for early stopping instead.")
print(f"  3. [HIGH] Model capacity ({list(range(256,2048,128))} hidden units) >> data size")
print(f"     for spatial generalisation with only {train_stn} locations.")
print(f"  4. [MEDIUM] Bernoulli component BCE is optimised on TRAINING station raininess")
print(f"     patterns; val stations may have different wet-day fractions.")
print(f"  5. [MEDIUM] CosineWarmup LR decays too fast — paper uses constant LR for 30k steps.")
print(f"  6. [LOW] DEM branches separate (not stacked) — loses local vs regional spatial context.")

# ── 9. Recommendations ────────────────────────────────────────────────────────
print("\n" + "=" * 70); print("  8. ACTIONABLE RECOMMENDATIONS"); print("=" * 70)
print("""
  IMMEDIATE (likely to help most):
  A. Use val_temporal for early stopping, NOT val_spatial.
     val_temporal = train stations in val years = temporal-only shift.
     This removes the spatial shift from the training signal and gives a
     fairer measure of whether the model is learning the climate->rain mapping.
     val_spatial can still be reported as final eval metric.

  B. Increase dropout dramatically (0.5+) and add weight decay (1e-3+).
     With {train_stn} stations, the model has far too much capacity.

  C. Try MSE first with val_temporal stopping — establish a baseline
     to confirm the pipeline works before adding BG complexity.

  MEDIUM-TERM:
  D. Use val_temporal for early stopping objective throughout tuning.
     Change val_key in 04_tune_land.py to always prefer val_temporal.

  E. Reduce model capacity search space: na in [64,256], nb in [32,128].
     The paper uses ra_dim=512, dem_dim=512, final=1024 but has 1834 train stations.
     With {train_stn} train stations, na>512 is almost certainly overfitting.

  F. Consider ensemble approach (paper ensembles 10 models) to reduce variance.

  ARCHITECTURAL:
  G. Stack local and regional DEM as a 2-channel image and use Conv2d,
     matching the paper more closely (captures spatial structure within patch).

  H. Use Gamma-only (no Bernoulli) as paper does for monthly, at least initially.
     Bernoulli adds complexity that may hurt spatial generalisation with few stations.
""".format(train_stn=train_stn))

# ── 10. Plots ─────────────────────────────────────────────────────────────────
# Target distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(tgt_tr, bins=100, log=True, color="#4c72b0", edgecolor="none")
axes[0].set_title("Train targets (raw mm, log scale)"); axes[0].set_xlabel("mm")
axes[1].hist(wet_tr, bins=100, log=True, color="#55a868", edgecolor="none")
axes[1].set_title("Train WET only (raw mm)"); axes[1].set_xlabel("mm")
axes[2].hist(norm_wet_tr, bins=100, log=True, color="#c44e52", edgecolor="none")
axes[2].axvline(norm_wet_tr.mean(), color="k", linestyle="--",
                label=f"mean={norm_wet_tr.mean():.2f}")
axes[2].set_title(f"Train WET normalised (÷{target_scale:.0f}mm)"); axes[2].set_xlabel("y/σ")
axes[2].legend()
plt.tight_layout()
fig.savefig(OUT / "01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Train vs val target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
bins = np.linspace(0, min(wet_tr.max(), 200), 60)
axes[0].hist(wet_tr, bins=bins, alpha=0.7, density=True, color="#4c72b0",
             label=f"train wet (n={len(wet_tr):,})", log=True)
axes[0].hist(tgt_va[tgt_va > 0], bins=bins, alpha=0.7, density=True, color="#c44e52",
             label=f"val_sp wet (n={len(tgt_va[tgt_va>0]):,})", log=True)
axes[0].set_xlabel("Rainfall (mm)"); axes[0].legend(fontsize=8)
axes[0].set_title("Train vs Val_spatial wet days (density)")

monthly_data = [tgt_tr[months[splits["train"]] == m] for m in range(1, 13)]
axes[1].boxplot(monthly_data, labels=[str(m) for m in range(1, 13)], showfliers=False)
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Rainfall (mm)")
axes[1].set_title("Monthly rainfall (train)")
plt.tight_layout()
fig.savefig(OUT / "02_target_split_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Climate distribution train vs val
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.ravel()
for i in range(min(C, 16)):
    tr_v = clim_tr2[:, i].ravel()
    va_v = clim_va[:, i].ravel() if len(clim_va) else np.array([])
    vname = variables[i] if i < len(variables) else f"ch{i}"
    all_v = np.concatenate([tr_v, va_v]) if len(va_v) else tr_v
    bins = np.linspace(np.nanpercentile(all_v, 1), np.nanpercentile(all_v, 99), 40)
    axes[i].hist(tr_v, bins=bins, alpha=0.6, density=True, color="#4c72b0", label="train")
    if len(va_v):
        axes[i].hist(va_v, bins=bins, alpha=0.6, density=True, color="#c44e52", label="val_sp")
    axes[i].set_title(vname[:25], fontsize=7)
    axes[i].tick_params(labelsize=6)
for j in range(C, 16):
    axes[j].set_visible(False)
axes[0].legend(fontsize=7)
plt.suptitle("Climate feature distributions: train vs val_spatial (post-norm)", y=1.01)
plt.tight_layout()
fig.savefig(OUT / "03_climate_train_val_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n  Saved 01_target_distribution.png")
print(f"  Saved 02_target_split_comparison.png")
print(f"  Saved 03_climate_train_val_distributions.png")
print(f"\n  See output/Deep_Dive/findings.md for full writeup.")
print("=" * 70)
print("  DIAGNOSTICS COMPLETE")
print("=" * 70)
