# Evaluation Metrics Guide

Metrics used in `09_evaluate.py` for daily rainfall downscaling.

---

## Continuous Metrics (all predictions)

### RMSE — Root Mean Squared Error
```
RMSE = sqrt(mean((y_pred - y_true)²))
```
Average prediction error in **mm**. Penalizes large errors heavily. Lower is better.

### MAE — Mean Absolute Error
```
MAE = mean(|y_pred - y_true|)
```
Average absolute error in **mm**. More robust to outliers than RMSE. Lower is better.

### R² — Coefficient of Determination
```
R² = 1 - SS_res / SS_tot
```
Fraction of rainfall variance explained by the model. Range: (−∞, 1].
- **1.0** = perfect
- **0.0** = no better than predicting the mean
- **< 0** = worse than predicting the mean

---

## Wet/Dry Classification Metrics

Computed after thresholding predictions and observations at **1.0 mm** (wet = ≥ 1 mm).

The 2×2 contingency table:

|  | Predicted Wet | Predicted Dry |
|--|--|--|
| **Observed Wet** | Hit (H) | Miss (M) |
| **Observed Dry** | False Alarm (FA) | Correct Negative (CN) |

### POD — Probability of Detection (Hit Rate)
```
POD = H / (H + M)
```
Fraction of observed wet days that were correctly predicted wet. Range: [0, 1]. Higher is better.
- **1.0** = all wet days caught (but says nothing about false alarms)

### FAR — False Alarm Ratio
```
FAR = FA / (H + FA)
```
Fraction of predicted wet days that were actually dry. Range: [0, 1]. Lower is better.
- High FAR with high POD = wet bias (model predicts rain too often)

### CSI — Critical Success Index (Threat Score)
```
CSI = H / (H + M + FA)
```
Accounts for both misses and false alarms. Range: [0, 1]. Higher is better.
Does **not** account for correct negatives (easy in rainy climates).

### ETS — Equitable Threat Score (Gilbert Skill Score)
```
H_random = (H + M)(H + FA) / N
ETS = (H - H_random) / (H + M + FA - H_random)
```
Like CSI but **corrected for random chance**. Range: [−1/3, 1].
- **0** = no skill beyond random
- Preferred over CSI for imbalanced wet/dry datasets

### HSS — Heidke Skill Score
```
HSS = 2(H·CN - FA·M) / [(H+M)(M+CN) + (H+FA)(FA+CN)]
```
Skill relative to random chance using the full contingency table. Range: [−1, 1].
- **0** = no skill, **1** = perfect
- Similar interpretation to ETS but computed differently

---

## Wet-Day Amount Metrics

Computed only on days where **both** observation and prediction are ≥ 1 mm.

### wet_RMSE
RMSE restricted to wet days. Measures amount prediction accuracy on rainy days. Units: mm.

### wet_R²
R² restricted to wet days. Measures how well the model explains variance in rainfall **amounts** (not just occurrence). Typically lower than overall R².

---

## Evaluation Sets

| Set | Description |
|-----|-------------|
| `test_temporal` | Held-out **years** at stations seen during training |
| `test_spatial` | Held-out **stations** not seen during training (any year) |
| `test_all` | Union of both holdout sets |

`test_spatial` is the hardest generalization task — it tests whether the model learned transferable physical relationships rather than station-specific patterns.
