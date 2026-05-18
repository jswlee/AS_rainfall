# Inference Metrics Reference
### Daily Rainfall Downscaling — American Samoa LAND Model

This document explains every metric produced by the inference pipeline
(`07_infer_land_ensemble.py` / `run_ensemble_inference_from_dir`).  All metrics
are computed on held-out test data that was never seen during training or
hyperparameter tuning.

---

## Output Files

After running inference on a trained run directory the following files are
written to `<run_dir>/inference/`:

| File | Contents |
|---|---|
| `metrics_test_temporal.json` | Standard regression metrics — temporally held-out test set |
| `metrics_test_spatial.json` | Standard regression metrics — spatially held-out test set |
| `metrics_test_all.json` | Standard regression metrics — both test sets combined |
| `wetdry_metrics_test_temporal.json` | Wet/dry classification + intensity metrics — temporal |
| `wetdry_metrics_test_spatial.json` | Wet/dry classification + intensity metrics — spatial |
| `wetdry_metrics_test_all.json` | Wet/dry classification + intensity metrics — combined |
| `predictions_test_temporal.npz` | `y_true`, `y_pred_mean`, `y_pred_std`, `stations` arrays |
| `predictions_test_spatial.npz` | Same, spatial split |
| `predictions_test_all.npz` | Same, combined |
| `wetdry_eval_test_temporal.png` | 4-panel wet/dry evaluation figure — temporal |
| `wetdry_eval_test_spatial.png` | 4-panel wet/dry evaluation figure — spatial |
| `wetdry_eval_test_all.png` | 4-panel wet/dry evaluation figure — combined |
| `inference_manifest.json` | Run metadata: n_models, checkpoints used, threshold |

---

## Test Split Definitions

Understanding which split is reporting matters as much as the metric values
themselves.

### `test_temporal`
Stations that were seen during training, evaluated on **held-out years** (the
most recent ~10 % of the time series by default).  This tests whether the model
generalises across time at known locations.

- **Easier** than spatial generalisation because the model has learned the
  topographic character of these stations.
- A model can score well here simply by learning the seasonal/interannual
  climatology of a station.
- Small sample: the temporal test window is short (~2–3 years), so metrics
  are **higher variance** and should be interpreted with caution when `n < 500`.

### `test_spatial`
**Held-out stations** — locations the model has never seen, evaluated across
all years that overlap the training period.  This is the primary test of
spatial downscaling skill.

- **Harder** because the model must extrapolate its learned
  topographic–climate relationships to new locations.
- High performance here is the real goal: it demonstrates the model has
  learned transferable physical relationships, not just memorised station
  behaviour.
- Typically has a larger sample than `test_temporal`.

### `test_all`
Concatenation of both test sets.  Useful for a single headline number but can
be misleading if the two sets have very different characteristics or sizes.

---

## Section 1 — Standard Regression Metrics

These appear in `metrics_*.json`.  They are computed over **all days**
(wet and dry combined) in the respective test split.

---

### `rmse` — Root Mean Squared Error (mm)

```
RMSE = sqrt( mean( (y_pred - y_true)² ) )
```

RMSE penalises large errors more heavily than small ones due to the squaring.
For daily rainfall this means **extreme-event errors dominate the score**.

| Range | Interpretation |
|---|---|
| < 5 mm | Excellent — rare in practice for daily data at the station scale |
| 5–10 mm | Good — competitive with published statistical downscaling methods |
| 10–15 mm | Moderate — the model is useful but misses many heavy-rain days |
| 15–25 mm | Weak — heavy-rain errors are large; check for systematic wet bias |
| > 25 mm | Poor — model may be producing unrealistic extreme values |

**Context for American Samoa:** Daily rainfall is highly skewed (many dry days,
occasional multi-day events exceeding 100 mm).  An RMSE around 12–15 mm on a
dataset that includes typhoon/tropical-storm days is not unusual for a
regression model.  Compare against the **climatological baseline RMSE**
(`baseline_rmse` in `cv_val` outputs), which is the RMSE you would get by
always predicting the training-set mean.  Your model must beat this.

---

### `mae` — Mean Absolute Error (mm)

```
MAE = mean( |y_pred - y_true| )
```

MAE treats all errors equally regardless of magnitude.  It is more robust to
outliers than RMSE and easier to interpret as "on average, how many mm off is
the model?"

| Range | Interpretation |
|---|---|
| < 3 mm | Excellent |
| 3–6 mm | Good |
| 6–10 mm | Moderate |
| > 10 mm | Weak — revisit wet-day intensity calibration |

Because most days are dry (rainfall = 0 mm), MAE computed over all days is
partly driven by false alarms (predicting rain on dry days).  If MAE is
disproportionately large relative to wet-day MAE (`wet_mae`), the model has a
false-alarm problem.

---

### `mbe` — Mean Bias Error (mm)

```
MBE = mean( y_pred - y_true )
```

MBE is signed.  Positive = the model over-predicts on average (wet bias);
negative = under-predicts (dry bias).

| Range | Interpretation |
|---|---|
| −1 to +1 mm | Negligible bias — well-calibrated |
| ±1 to ±3 mm | Acceptable |
| ±3 to ±6 mm | Noticeable bias — check if it is structural (e.g. output head) |
| > ±6 mm | Severe bias — likely a calibration or loss function issue |

**Important:** A near-zero MBE does not mean the model is accurate — it can
cancel out large positive and negative errors.  Always read MBE alongside RMSE
and MAE.  For Gamma/Bernoulli-Gamma output heads a small positive MBE is
structurally expected because the output distribution has no hard lower bound.

---

### `r2` — Coefficient of Determination (R²)

```
R² = 1 - SS_res / SS_tot
     where SS_res = sum((y_pred - y_true)²)
           SS_tot = sum((y_true - mean(y_true))²)
```

R² measures how much of the **total variance** in observed rainfall the model
explains.  R² = 1 is perfect; R² = 0 means the model is no better than always
predicting the observed mean; R² < 0 means the model is actively worse than the
mean predictor.

| Range | Interpretation |
|---|---|
| > 0.7 | Strong — model explains most variability |
| 0.5–0.7 | Good — clearly useful predictive signal |
| 0.3–0.5 | Moderate — detectable skill, room to improve |
| 0.1–0.3 | Weak — model captures broad patterns only |
| < 0.1 | Very weak — close to a null model |
| < 0 | Harmful — worse than climatology |

**Why R² can be misleading for daily rainfall:** Because daily rainfall is
zero-inflated (many dry days), the variance is dominated by a few extreme days.
A model that gets the climatological frequency right but misses individual event
magnitudes can still produce R² ≈ 0.2–0.4.  This is common in the literature
for station-scale daily statistical downscaling and does not necessarily indicate
a broken model — but it does indicate the model cannot reliably reproduce
individual-event intensities.

---

### `spearman_r` — Spearman Rank Correlation

Spearman correlation measures the **monotonic rank relationship** between
observed and predicted.  Unlike Pearson correlation or R², it is insensitive to
the exact magnitude of extreme values and focuses on whether the model correctly
ranks days by rainfall intensity.

| Range | Interpretation |
|---|---|
| > 0.7 | Strong ranking skill |
| 0.5–0.7 | Moderate — model captures the ordering reasonably well |
| 0.3–0.5 | Weak — model struggles to rank individual days |
| < 0.3 | Near-random ordering of predictions |

`spearman_p` is the two-tailed p-value for the null hypothesis that
`spearman_r = 0`.  With large sample sizes (n > 500) this will always be
effectively zero even for modest correlations; focus on the value of `r` itself,
not the p-value.

---

## Section 2 — Extreme Event Metrics

These appear in `metrics_cv_val.json` (training diagnostics) and in the CV
summary.  They are designed to evaluate tail-event performance specifically.

---

### `pctl_rel_bias` — Percentile Relative Bias

```
pctl_rel_bias = (pctl_pred - pctl_true) / pctl_true
```

Computed at the 98th percentile by default (`--extreme-percentile`).  Measures
whether the model systematically under- or over-predicts the intensity of the
top-2% heaviest rain days.

| Range | Interpretation |
|---|---|
| −0.05 to +0.05 | Excellent — predicted and observed tail quantiles are very close |
| ±0.05 to ±0.15 | Acceptable |
| ±0.15 to ±0.30 | Moderate bias — the model mutes or inflates extreme events |
| > ±0.30 | Severe — model is not capturing the tail of the distribution |

Negative values mean the model under-predicts extremes (common with MSE loss,
which is penalised heavily by large errors and encourages conservatism).
Positive values mean the model over-predicts extremes.

---

### `csi` (heavy-rain CSI) — Critical Success Index at High Threshold

```
CSI = TP / (TP + FP + FN)    applied at a heavy-rain threshold (default: 50 mm)
```

This is the same formula as the wet/dry CSI (Section 3) but applied at a much
higher threshold to specifically evaluate heavy-rain event detection.  At 50 mm
this captures tropical-storm-scale events.

| Range | Interpretation |
|---|---|
| > 0.4 | Good — model reliably identifies major rain events |
| 0.2–0.4 | Moderate |
| 0.1–0.2 | Weak — many events are missed or falsely triggered |
| < 0.1 | Very poor skill at heavy-rain detection |

---

## Section 3 — Wet/Dry Day Classification Metrics

These appear in `wetdry_metrics_*.json`.  They are computed using a binary
wet/dry classification at `threshold_mm` (default **1.0 mm**, the WMO standard
for a "rain day").

All of these metrics are derived from a **2×2 contingency table**:

```
                    Observed
                   Wet    Dry
           Wet  [  TP  |  FP  ]   ← predicted wet
Predicted  Dry  [  FN  |  TN  ]   ← predicted dry
```

- **TP (Hit):** Model correctly predicts a wet day
- **FP (False Alarm):** Model predicts wet, but it was actually dry
- **FN (Miss):** Model predicts dry, but it was actually wet
- **TN (Correct Negative):** Model correctly predicts a dry day

---

### `pod` — Probability of Detection (Hit Rate)

```
POD = TP / (TP + FN)
```

Fraction of actual wet days that the model correctly predicts as wet.  Also
called **recall** or **sensitivity** for the wet class.

| Range | Interpretation |
|---|---|
| > 0.85 | Good detection — few wet days are missed |
| 0.70–0.85 | Moderate |
| 0.50–0.70 | Weak — model misses many rain events |
| < 0.50 | Poor — model is missing more rain events than it catches |

**Caveat:** POD can be trivially inflated by predicting "wet" for everything.
A model with POD = 1.0 but FAR = 0.9 is useless.  Always read POD and FAR
together, or use CSI/ETS/HSS which account for both.

---

### `far` — False Alarm Ratio

```
FAR = FP / (TP + FP)
```

Fraction of the model's wet-day predictions that are actually dry days.
**Lower is better.**

| Range | Interpretation |
|---|---|
| < 0.15 | Excellent |
| 0.15–0.30 | Good |
| 0.30–0.45 | Moderate — roughly 1 in 3 predicted wet days is actually dry |
| 0.45–0.60 | Weak — more than half of wet predictions are wrong |
| > 0.60 | Poor — model is generating far too many false alarms |

A persistently high FAR with high POD indicates the model is predicting "wet"
too liberally.  For Gamma output heads (which produce a continuous positive
value) this is a structural issue — the output never reaches exactly zero, so
even small predicted amounts are classified as "wet" at a 1 mm threshold.
Switching to the `bernoulli_gamma` loss introduces an explicit dry-day
probability component that directly addresses this.

---

### `freq_bias` — Frequency Bias

```
freq_bias = (TP + FP) / (TP + FN)
          = n_pred_wet / n_obs_wet
```

Ratio of predicted wet-day frequency to observed wet-day frequency.

| Range | Interpretation |
|---|---|
| 0.9–1.1 | Near-perfect frequency calibration |
| 0.7–0.9 or 1.1–1.3 | Slight under/over-prediction of wet frequency |
| 0.5–0.7 or 1.3–1.5 | Moderate bias |
| < 0.5 or > 1.5 | Severe — model wet-day frequency is very different from observed |

`freq_bias > 1` means the model predicts more wet days than observed (wet
frequency bias).  `freq_bias < 1` means fewer wet days (dry frequency bias).
Note that a perfect `freq_bias = 1.0` can still coexist with poor placement
(high FAR and high miss rate simultaneously).

---

### `csi` — Critical Success Index (at wet/dry threshold)

```
CSI = TP / (TP + FP + FN)
```

Fraction of all events that were either observed or predicted that were
correctly classified.  Penalises both false alarms and misses equally.
Also called the **Threat Score**.

| Range | Interpretation |
|---|---|
| > 0.6 | Good |
| 0.4–0.6 | Moderate |
| 0.2–0.4 | Weak |
| < 0.2 | Poor |

**Limitation:** CSI ignores correct negatives (TN), so it is sensitive to the
base rate of wet days.  For datasets where dry days greatly outnumber wet days
(which is the case here), CSI will be bounded above by the wet-day frequency.

---

### `ets` — Equitable Threat Score

```
ETS = (TP - Tc) / (TP + FP + FN - Tc)
Tc  = (TP + FP)(TP + FN) / N          ← expected hits by random chance
```

ETS corrects CSI for the number of hits that would occur by random chance given
the observed frequencies.  It is the **gold-standard skill score** for
precipitation occurrence verification (WMO standard).

| Range | Interpretation |
|---|---|
| > 0.3 | Good skill |
| 0.1–0.3 | Moderate skill |
| 0.05–0.1 | Low but detectable skill |
| 0–0.05 | Near-zero skill — close to a random predictor |
| < 0 | Model is worse than random |

ETS is harder to inflate than POD or CSI and is robust to class imbalance.
Values in the literature for statistical downscaling of daily precipitation at
individual stations typically range from 0.05–0.25, so ETS = 0.10–0.20 is
a reasonable target.  Values approaching 0.3 are considered good.

---

### `hss` — Heidke Skill Score

```
HSS = 2(TP·TN - FP·FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]
```

HSS measures the fractional improvement of the model over a random classifier
that uses the same observed base rates.  Ranges from −∞ to 1.

| Range | Interpretation |
|---|---|
| > 0.4 | Good skill |
| 0.2–0.4 | Moderate skill |
| 0.1–0.2 | Low but meaningful skill |
| 0–0.1 | Near-zero skill |
| < 0 | Worse than random |

HSS and ETS are closely related and usually tell the same story.  HSS is
slightly more sensitive to correct negatives (TN), making it more informative
when the dry-day class is dominant.

---

## Section 4 — Wet-Day Conditional Intensity Metrics

These are computed **only on observed wet days** (`y_true ≥ threshold_mm`).
They answer the question: *given that it actually rained, how accurately does
the model predict the amount?*

This is the most physically important question for hydrological applications.

---

### `wet_rmse` / `wet_mae` / `wet_mbe` (mm)

Same definitions as the all-day metrics (Section 1), but restricted to the
subset of days where observed rainfall exceeded the threshold.

- **`wet_rmse`** is dominated by errors on heavy-rain days.
- **`wet_mae`** gives the average absolute error in mm on rainy days.
- **`wet_mbe`** tells you whether the model systematically under- or
  over-predicts amounts on actual rain days (distinct from the all-day MBE
  which includes dry-day false alarms).

Typical good/moderate/weak thresholds are the same as for the overall RMSE/MAE
in Section 1, but values will be **higher** since dry days (which have small
errors of ~0) are excluded.

---

### `wet_r2` — R² on Wet Days Only

The fraction of variance in observed **wet-day** amounts explained by the model.
This is the most important single metric for assessing downscaling skill in the
hydrological literature.

| Range | Interpretation |
|---|---|
| > 0.5 | Good — model captures a meaningful portion of event-scale variability |
| 0.3–0.5 | Moderate — detectable spatial downscaling signal |
| 0.1–0.3 | Weak — model mostly tracks the climatological mean; event structure is poor |
| < 0.1 | Negligible — model has essentially no skill on wet-day amounts |
| < 0 | Harmful — predicting the wet-day mean would be better |

**Literature context:** Published statistical and machine-learning
downscaling studies for tropical island stations commonly report wet-day R²
in the range 0.15–0.45.  Values below 0.2 are common when the spatial density
of training stations is low (as in American Samoa), the reanalysis grid is
coarse relative to the island, or orographic effects dominate at sub-km scales
that neither the DEM nor the reanalysis can resolve.

---

### `wet_spearman_r` — Spearman Correlation on Wet Days

Same as overall Spearman (Section 1) but restricted to wet days.  Measures
whether the model correctly ranks rain events by intensity — important for
threshold-based applications (e.g. flood warning).

| Range | Interpretation |
|---|---|
| > 0.6 | Good ranking skill on wet days |
| 0.4–0.6 | Moderate |
| 0.2–0.4 | Weak |
| < 0.2 | Near-random ranking |

---

### `wet_mean_obs` / `wet_mean_pred` (mm)

Mean observed and predicted rainfall amounts on observed wet days.  Together
these reveal **intensity bias** on rain days independent of occurrence errors.

```
intensity_bias = wet_mean_pred / wet_mean_obs
```

- `intensity_bias > 1`: model over-predicts rain amounts on wet days
- `intensity_bias < 1`: model under-predicts rain amounts on wet days (common
  with MSE loss, which shrinks predictions toward the mean)
- `intensity_bias ≈ 1` with poor `wet_r2` means the model has the right mean
  but no event-scale resolution — it is effectively predicting the climatological
  wet-day mean for every rain day

---

## Section 5 — Reading the 4-Panel Wet/Dry Evaluation Figure

Each `wetdry_eval_<split>.png` contains four panels:

### Panel 1 (top-left): Contingency Table
A heat-map of the 2×2 table with raw counts.  The colour intensity is
proportional to count.  Ideally the diagonal (Hits + Correct Negatives) is
bright and the off-diagonal (Misses + False Alarms) is dim.

### Panel 2 (top-right): Wet-Day Scatter
Scatter of observed vs predicted amounts restricted to days where the gauge
recorded rainfall.  Points should cluster around the 1:1 line.  Systematic
under-prediction shows points below the line; systematic over-prediction shows
points above.  The subtitle shows `wet_mae` and `wet_r2` directly.

### Panel 3 (bottom-left): Wet-Day Amount Distribution
Overlaid histograms of observed and predicted amounts on observed wet days.
Ideally the two distributions overlap closely.  A rightward shift in the
predicted distribution indicates over-prediction; leftward indicates
under-prediction (amount muting).

### Panel 4 (bottom-right): Skill Score Bar Chart
Bar heights for POD, (1 − FAR), CSI, ETS, and HSS.  All are bounded [0, 1]
for a perfect model, with ETS and HSS potentially negative.  The dashed line at
1.0 is the perfect score; the horizontal line at 0.0 is the no-skill baseline.
`Freq Bias` is shown in the subtitle.

---

## Section 6 — Comparing Splits

When interpreting results, always compare `test_temporal` vs `test_spatial`:

| Scenario | Likely cause |
|---|---|
| Temporal >> Spatial | Model memorised station-specific patterns; poor spatial transfer |
| Spatial >> Temporal | Unusual — check if test_temporal sample size is very small |
| Both similar | Model has generalised well across both dimensions |
| Both poor | Fundamental signal is weak or training data has quality issues |

The **gap between temporal and spatial** ETS/HSS is the clearest indicator of
how well the learned topographic–climate relationships transfer to new locations.
A large gap (e.g. temporal ETS = 0.3, spatial ETS = 0.05) means the model has
overfit the spatial character of training stations and the DEM/climate embeddings
are not transferable.

---

## Quick-Reference Table

| Metric | Perfect | Good | Moderate | Weak |
|---|---|---|---|---|
| RMSE (mm) | 0 | < 8 | 8–15 | > 15 |
| MAE (mm) | 0 | < 5 | 5–8 | > 8 |
| MBE (mm) | 0 | ±1 | ±1–3 | ±3+ |
| R² | 1 | > 0.5 | 0.3–0.5 | < 0.3 |
| Spearman r | 1 | > 0.65 | 0.4–0.65 | < 0.4 |
| POD | 1 | > 0.85 | 0.7–0.85 | < 0.7 |
| FAR | 0 | < 0.20 | 0.20–0.40 | > 0.40 |
| Freq Bias | 1 | 0.9–1.1 | 0.75–1.25 | outside |
| CSI (1 mm) | 1 | > 0.6 | 0.4–0.6 | < 0.4 |
| ETS | 1 | > 0.20 | 0.05–0.20 | < 0.05 |
| HSS | 1 | > 0.35 | 0.10–0.35 | < 0.10 |
| wet_R² | 1 | > 0.4 | 0.15–0.4 | < 0.15 |
| wet_MAE (mm) | 0 | < 8 | 8–15 | > 15 |

---

*Generated automatically by the Daily_Modeling inference pipeline.
See `Daily_Modeling/utils/metrics.py` for exact formulas.*
