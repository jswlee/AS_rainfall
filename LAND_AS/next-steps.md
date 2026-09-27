# Next experiments for `LAND_AS`

This document separates experiments that are statistically defensible with the
current 24-station dataset (26 stations minus the QC exclusions below) from
exploratory ideas that need more data or a new validation design.

The governing rule remains:

> Select models only on training-station validation predictions. The five
> post-2016 test stations are for final reporting, not for choosing losses,
> weights, monitors, or calibration parameters.

The original implementation is in `../LocationAgnosticNeuralDownscaling`. Its
most relevant ideas are:

- Gamma-distributed rainfall output (`land/train.py`,
  `land/model_utils_gamma.py`);
- independent atmosphere, DEM, and month encoders (`land/model.py`);
- repeated temporal folds and station-omission ensembles (`README.md`,
  `land/train.py`);
- GLM and Gaussian-process baselines (`baselines/`);
- climatology, QQ, rainfall-elevation, and spatial-map diagnostics
  (`results/*.ipynb`, `results/climatology.py`,
  `land/visualization_utils.py`).

`Daily_Modeling` is also relevant, though `LAND_AS` no longer imports it at
runtime: the data builders and QC config it used are vendored under
`LAND_AS/daily_modeling/` so the package is self-contained. The vendored
dataset already uses complete ISO weeks and doubles each reanalysis channel
into within-week means plus within-week standard deviations. `LAND_AS` now
also supports fold-local normalization, temporal year blocks, and
`mean`/`median` fold aggregation in `tune.py`. The most useful remaining
imports from the external `Daily_Modeling` project are optimization metrics
beyond MAE/MSE (CSI and percentile bias), Tweedie/Bernoulli-Gamma losses, and
more systematic architecture search. Do not compare its reported metrics
directly with `LAND_AS`: it uses a different split and does not include the
same rainfall-lag feature contract.

## Immediate finding: low-end target shift — QC applied

The data-prep notebook includes a dedicated low-end audit
(`notebooks/01_data_prep_eda.ipynb`), and `eda_scripts/rainfall_*.py` trace the
same question back to the raw daily files. The training/test difference is not
a temporal rainfall-regime difference — it is station/source composition:

- pre-QC: training 197/6,686 exact-zero weeks (2.95%), test 17/1,188 (1.43%);
- post-QC: training 143/6,078 (2.35%), test 3/1,170 (0.26%), bridge 2/734 (0.27%);
- wet-week conditional means were nearly identical all along (77.3 vs 76.8 mm);
  the entire shift was low-end mass plus a slightly heavier train tail.

Root causes found in the raw daily records:

- `aunuu` reports in 0.1-inch increments (min nonzero = 2.54 mm/day), so
  drizzle reads as zero — **excluded** via `QC_EXCLUDE_STATIONS`.
- `vaipito2000` degrades across decades (1970s-90s daily median = 0, weekly
  mean ~40 mm vs ~82 mm at co-located `vaipito_res`/`vaipito_UH`) —
  **excluded**.
- `afono_UH` has three flat-zero runs of 30-53 days that neighbouring stations
  contradict (gauge-offline stored as 0.000) — **masked to missing** via
  `QC_MASK_DATE_RANGES`, which drops the affected incomplete weeks.
- Remaining legacy zeros (`pioa_afono`, `fagaitua`, `vaipito_res`, `satala`,
  `aasufou80`, `malaeimi_1691`: ~90% of what is left) are largely real dry
  weeks — they co-occur across neighbouring stations at 3-15x the independence
  rate — but are inflated by 0.01-inch reporting floors.

This changes the experiment order: the QC is done; station-sensitivity
ablation on the remaining high-zero gauges is the next controlled experiment,
and another scalar loss sweep stays low priority. A hurdle/Bernoulli-Gamma
model remains possible, but a leakage-free occurrence classifier improves
existing OOF amount predictions by only about 0.2-0.3 mm MAE and still predicts
tens of millimeters on dry weeks — and after QC there are only 143 training
zeros to learn occurrence from anyway.

### Post-QC model outcome

The three leading configurations were retrained on the cleaned dataset
(`_qc` runs) and all baselines were refit on the cleaned training rows
(1,170-week test set):

| Model | RMSE | MAE | Bias | R2 |
|---|---:|---:|---:|---:|
| Ridge | **48.37** | 34.71 | -2.07 | **0.500** |
| GBM | 48.81 | **33.50** | -4.68 | 0.490 |
| `weekly_land_v6_huber_rw_t36_qc` | 49.32 | 35.46 | +2.54 | 0.480 |
| `v5_qc_gamma_huber_cv_mse` (blend) | 50.30 | 35.71 | +1.05 | 0.459 |
| `weekly_land_v5_qc` | 50.59 | 35.91 | +0.76 | 0.453 |
| `weekly_land_v5_huber_rw_msemon_qc` | 50.62 | 36.10 | +1.76 | 0.452 |

Two conclusions:

1. **The QC did not close the neural-vs-baseline gap; it widened it.** All
   models improved because ~14 fake-dry `afono_UH` weeks (large misses for
   everyone) were removed, but Ridge gained more than the best neural model
   (49.35 to 48.37 vs 49.58 to 49.32 RMSE). The dominant open question is no
   longer target cleanliness — it is whether the neural spatial features add
   anything over pooled tabular regression.
2. **The neural ordering did not change.** Weighted Huber still beats Gamma
   and plain Huber; temporal-CV tuning still beats spatial k-fold (t36 remains
   best). The `_qc` blend selected Huber weight 0.288 on OOF — mostly Gamma —
   and still sits mid-pack.

The active experiment is therefore a `--search-space broad` Optuna study
(`weekly_land_v7_huber_rw_temporal_broad`) that finally retunes model width,
dropout, batch size, and `dem_size` — the axes frozen since v5. Command and
per-knob rationale are in `README.md` section 7 ("v7: broad architecture
search on cleaned data").

## 1. Highest-priority controlled experiments

### 1.1 Source and site quality control — DONE (2026-09-26)

Audited in `eda_scripts/rainfall_train_test_deep_dive.py` and
`rainfall_npz_and_artifact_checks.py`. Answers to the planned questions:

- **Missing encoded as zero?** Yes, at `afono_UH`: three flat-zero runs
  (2022-08-14..10-05, 2022-10-18..11-16, 2024-07-12..08-15) during which
  `vaipito_UH`, `aunuu_UH`, and both WRCC gauges recorded 3-6 mm/day. Now
  masked in `LAND_AS/daily_modeling/config.py::QC_MASK_DATE_RANGES`.
- **Multi-day accumulation?** Moderate at `pioa_afono` (post-zero-run wet days
  exceed 4x the median wet day 18% of the time) — flagged, not yet removed.
- **Zero rate changes at a record discontinuity?** Yes, dramatically at
  `vaipito2000`: 0% zeros + ~30 mm/day in 1958-60, then 51-70% zeros and a
  median of 0 in the 1970s-90s — **excluded**.
- **Are dry weeks physically plausible?** Mostly yes for the retained legacy
  gauges: zero weeks co-occur across neighbours at 3-15x the independence rate.
- **Why does `aunuu` differ from `aunuu_UH`?** Reporting resolution: `aunuu`'s
  minimum nonzero daily value is exactly 0.1 inch — **excluded**.

Mechanism: `QC_EXCLUDE_STATIONS` / `QC_MASK_DATE_RANGES` in
`LAND_AS/daily_modeling/config.py` (the canonical copy for this package,
vendored from `Daily_Modeling/config.py`), applied in `load_daily_rainfall`.
Pre-QC dataset preserved as `LAND_AS/data/weekly_dataset_pre_qc.npz`.

### 1.2 Station-sensitivity ablation — `pioa_afono` result: RETAIN

The first exclusion set (`aunuu` + `vaipito2000`) is applied. The
`pioa_afono` ablation has been run and answered
(`weekly_land_v6_huber_rw_t36_qc_nopioa_v2`, same t36 config, 18 folds):

| | RMSE | MAE | Bias | R2 |
|---|---:|---:|---:|---:|
| `t36_qc` (19 folds, pioa kept) | **49.32** | **35.46** | +2.54 | **0.480** |
| `t36_qc_nopioa_v2` (18 folds) | 50.02 | 35.77 | +3.11 | 0.465 |

Exclusion worsened every headline metric far beyond the ~0.01 mm run-to-run
noise measured by an accidental identical-config rerun — pioa's 666 weeks
carry more real signal than artifact, so it stays in
`QC_EXCLUDE_STATIONS`-exempt status. Optional follow-ups on `fagaitua`,
`vaipito_res`, `satala` are lower expected value; run them only if a specific
new artifact is found, not as a batch.

Mechanism for any future ablation: add names to `QC_EXCLUDE_STATIONS` in
`LAND_AS/daily_modeling/config.py`, rerun `LAND_AS.prepare`, train with a new
run name (never reuse a run dir — fold indexing changes and stale checkpoints
would be incorrectly skipped/reused), then restore `config.py` and the npz.

Sequencing caution: `load_data()` reads `weekly_dataset.npz` once at process
start. Do not run an ablation npz swap while a tuning study (or any other
training run) is in flight — the two processes would silently train on
different station pools. Run the swap, `prepare`, and ablation train
sequentially, then restore.

Evaluation should include:

- ordinary LOSO OOF metrics;
- OOF metrics for retained versus removed stations;
- predicted probability below 1/5/10 mm;
- mean prediction on observed dry weeks;
- post-2016 metrics only after the ablation is selected.

This tests whether the legacy dry-week tail is helping the model learn real
occurrence behavior or teaching it an untransferable low-rain mass.

### 1.3 Huber delta sweep

The initial delta sweep is now complete (`delta=0.25`, `1.0`, `2.0` against the
retained `0.5` variants). It changed mostly the bias/extreme tradeoff and did
not produce a decisive OOF or test improvement. A further sweep is therefore
lower priority unless paired with the weighted loss and MSE monitor under the
same seeds.

The completed sweep showed that Huber's robustness parameter mostly moves the
MAE/bias/extreme tradeoff rather than solving the low-end distribution problem.
Do not run another delta grid until a stronger feature or data change is
available.

### 1.4 Rainfall-weight strength

The current weighted loss uses `log1p(target)` in normalized-target units. A
natural extension is an explicit weight exponent or cap:

```text
w_i = log1p(max(y_i, 0)) ** weight_power
w_i = min(w_i, weight_cap)
```

Candidate powers: `0.5`, `1.0`, `1.5`. This requires adding a
`--weight-power` argument and keeping the current run unchanged.

Why try it: ordinary Huber improved MAE but became negatively biased; the
present `log1p` weight corrected that bias. The correct next question is how
much wet-week emphasis is optimal, not merely whether weighting helps.

### 1.5 Increase ensemble size

The original LAND experiments used ten ensemble members per setting. The
current American Samoa runs use three seeds per LOSO fold. Five or seven seeds
would reduce seed noise and make validation comparisons less sensitive to a
lucky initialization.

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_rw_msemon_qc `
  --loss-type huber_weighted `
  --monitor mse `
  --seeds 5 --epochs 500 --patience 50 --workers 4
```

Existing checkpoints are skipped; seeds 45 and 46 are added. Also evaluate seed
dispersion, not just ensemble mean performance. If seed spread is large, model
selection is not yet stable enough for subtle architecture changes.

### 1.6 Station-balanced sampling

`--balanced-stations` already exists and gives each training station equal
expected representation. This is a defensible experiment because the target is
spatial transfer, but it may hurt stations with long reliable records by
down-weighting their information.

```powershell
.\venv\Scripts\python.exe -m LAND_AS.train_v5_huber `
  --run weekly_land_v5_huber_rw_msemon_balanced_qc `
  --loss-type huber_weighted `
  --monitor mse `
  --balanced-stations `
  --seeds 3 --epochs 500 --patience 50 --workers 4
```

Acceptance should be based on mean and worst-station OOF metrics, not only
pooled OOF metrics.

### 1.7 Compare checkpoint monitors under paired folds

`--monitor mae` and `--monitor mse` do not change the loss; they choose which
validation score retains the checkpoint. MSE monitoring has won every
comparison so far (the two best neural runs pre- and post-QC, `t36` and
`v5_huber_rw_msemon`, both use it), but monitor selection should still be
compared using the same seeds and folds across delta values.

A useful summary is a paired table:

```text
(delta, loss, monitor) × fold × seed → OOF MAE, MSE, bias, p98 ratio, CSI
```

Do not promote a monitor based only on its pooled mean; inspect variance across
stations and seeds.

## 2. Better prediction combination

### 2.1 Blend LAND with Ridge and GBM

Ridge is currently the strongest aggregate model and GBM has the lowest MAE.
The obvious next ensemble is therefore not Gamma+Huber only; it is a pooled
stack or convex blend involving Ridge, GBM, and the best neural model.

To avoid leakage, every member must generate LOSO out-of-fold predictions:

1. Refit Ridge and GBM inside each training fold.
2. Predict that fold's held-out station.
3. Stack those OOF predictions with the saved neural OOF predictions.
4. Select non-negative weights summing to one.
5. Refit all members on all training stations.
6. Apply the OOF-selected weights once to test predictions.

This is more defensible than selecting a neural-vs-baseline weight on test
metrics. It may also reveal whether LAND contributes unique information beyond
tabular regression.

### 2.2 OOF calibration, not test calibration

A global scalar correction or isotonic map could be fitted to OOF predictions.
This is only legitimate if fitted without test observations.

Candidate calibrations:

- additive correction by rainfall regime;
- isotonic regression on OOF predictions;
- quantile mapping for the ensemble mean;
- station-blind calibration, because station-specific calibration cannot be
  learned for unseen test stations.

A global correction is unlikely to help because OOF and test biases differ.
Regime-aware or quantile calibration is more plausible but must beat the
uncalibrated model on OOF first.

### 2.3 Optimize the blend objective explicitly

The existing blend script supports MSE selection. It could also report weights
for MAE, high-quantile error, or a composite objective. A composite could be:

```text
mean squared error + lambda * absolute bias
                       + gamma * upper-quantile absolute error
```

The composite should be fixed before looking at test metrics. Otherwise the
objective itself becomes another source of test-set tuning.

## 3. Alternative losses and output heads

### 3.1 Neural Tweedie likelihood

The existing `tweedie_glm` baseline applies a Tweedie model to flattened
features. A neural Tweedie head would let the feature extractor learn the same
kind of positive, right-skewed response while retaining nonlinear spatial
features.

Implementation notes:

- predict mean and optionally dispersion/power;
- fix the Tweedie power near `1 < p < 2` for a compound Poisson-Gamma model;
- `power=2` is Gamma-like and comparable to the retained baseline;
- optimize negative log-likelihood, preferably with a stable parameterization.

This is a stronger comparison than the GLM alone because it isolates the effect
of learned spatial features.

### 3.2 Bernoulli-Gamma / hurdle model

A two-head model can predict:

1. probability of a wet week;
2. positive rainfall amount conditional on a wet week.

`Daily_Modeling` implements this as `bernoulli_gamma`
(`Daily_Modeling/models/losses.py`). It is scientifically defensible, but the
weekly American Samoa target is only mildly zero-inflated:

```text
training weeks: 143 / 6078 exact zeros (2.35%)   # post-QC; was 197/6686 (2.95%)
test weeks:        3 / 1170 exact zeros (0.26%)  # post-QC; was 17/1188 (1.43%)
```

The existing Gamma loss drops dry weeks from the amount fit. A Bernoulli-Gamma
model would use those weeks to train an occurrence head, but the occurrence
signal is sparse at weekly resolution. The new leakage-free diagnostic in
`01_data_prep_eda.ipynb` finds LOSO wet/dry AUC around `0.88`, yet probability
scaling only reduces OOF MAE by about `0.2-0.3 mm` and does not make dry-week
predictions close to zero. `Daily_Modeling`'s own weekly tuning also favors
ordinary Gamma over Bernoulli-Gamma. None of this rules out a hurdle model, but
it means occurrence alone is unlikely to resolve the train/test low-end shift.

If tested, keep it as a controlled challenger and compare:

- overall MAE/RMSE/R2 and bias;
- wet/dry POD, FAR, CSI, ETS, and HSS;
- conditional wet-week amount error;
- upper-quantile bias;
- seed/station-fold variance.

Fit wet/dry probability thresholds and `lambda_bce` only on LOSO validation
predictions, never on the post-2016 test stations.

### 3.3 Quantile regression

Predict several quantiles, such as `0.1, 0.25, 0.5, 0.75, 0.9, 0.98`, using
pinball loss. Advantages:

- directly measures tail behavior;
- avoids imposing a Gamma family;
- can provide prediction intervals.

Risks:

- many outputs with only 19 spatial training folds;
- crossing quantiles unless constrained;
- the 98th percentile may still be poorly estimated because extreme test weeks
  are scarce.

Start with fewer quantiles (`0.1, 0.5, 0.9`) before attempting a direct 98th-
percentile model.

### 3.4 CRPS or distribution scoring

For the Gamma model, optimize or at least evaluate a probabilistic score such as
closed-form/approximate CRPS rather than only NLL and point metrics. NLL can
reward a distributional shape that does not translate into the best scalar mean.

Use this first as an additional OOF metric. Making it the training loss is a
larger change.

## 4. Input and architecture experiments

### 4.1 Revisit daily atmosphere with fewer simultaneous changes

The dataset still contains `reanalysis_daily` shaped like
`(samples, variables, 7, 3, 3)`, but the active loader ignores it. The prior
temporal branch changed architecture, output head, training objective, and
training protocol at once. A fair revisit should keep the v5 scalar head and
LOSO protocol while changing only the atmospheric encoder.

Candidate encoders:

- average daily embeddings;
- one-dimensional temporal convolution;
- small GRU;
- attention pooling over the seven days.

Keep the same `hidden_units`, dropout, output head, loss, seeds, and checkpoint
monitor. Otherwise the comparison will repeat the previous confounding problem.

### 4.2 Larger atmospheric context

The original LAND used larger atmospheric patches (`6x7`) than the retained
American Samoa `3x3` patch. Island rainfall may depend on synoptic-scale
moisture and winds outside the immediate station patch.

Possible tests:

- re-extract or cache a `5x5`/`7x7` reanalysis patch;
- retain the same model head and only change atmospheric context;
- use LOSO OOF to test whether larger context helps.

This requires preprocessing changes, so it is lower priority than loss and
ensemble experiments.

### 4.3 Antecedent and persistence features

Persistence is weak overall, but antecedent conditions may still matter in
interaction with atmosphere and terrain. Existing rainfall lags can be extended
with:

- lagged `dry_days` or `wet_days` within each prior week;
- two-week accumulated rainfall;
- four-week accumulated rainfall;
- previous-week anomaly from station climatology;
- count or duration of recent wet weeks;
- a missingness-aware summary rather than raw lags only.

The first item is the most directly motivated by the low-end audit: the current
model sees that the previous week totaled, for example, 10 mm, but not whether
that total occurred in one day or was spread across several days. Lagged
within-week occurrence counts could therefore improve dry-spell persistence
without exposing current-week target information.

These features are easy to evaluate in Ridge/GBM before adding them to LAND.

### 4.4 Topography alignment

The original repository's `topography_alignment.ipynb` examines whether station
rainfall gradients align with terrain. For American Samoa, test whether:

- local DEM patch orientation is consistent;
- windward/leeward predictors can be derived from aspect and prevailing wind;
- regional DEM radius is too small or too large;
- rainfall-elevation relationships differ by season.

This is more scientifically informative than blindly increasing DEM model size.

## 5. Validation and uncertainty

### 5.1 Add temporal CV within the training period

Partly implemented: `tune.py` already supports `--cv-mode temporal` and
`--cv-mode both` (LOSO + temporal), temporal CV produced the two best neural
models (`t36` pre- and post-QC), and the active v7 study uses temporal folds
for search. What remains is making joint station × year evaluation a standard
*finalist* protocol rather than only a tuning screen: trained runs still use
LOSO alone for OOF predictions and checkpoint selection, while the test
requires both spatial and temporal transfer.

The remaining stronger validation scheme would evaluate:

```text
LOSO station folds                  → spatial transfer
pre-2017 blocked year folds         → temporal transfer
nested station-year folds           → joint transfer
```

No new test evaluation is needed to extend this to finalists. It is a better
selection criterion for methods intended to work after 2016.

The unused post-2016 WRCC rows should remain a diagnostic bridge, not be folded
into training before evaluation. They are useful for checking whether a selected
method improves temporal extrapolation independently of the five UH test
stations. If source-aware features are ever added, remember that the UH observing
network does not occur in training, so network identity alone cannot explain the
final test transfer.

### 5.2 Station-cluster uncertainty intervals

Report confidence intervals by bootstrapping held-out stations, not individual
weeks. Weekly rows from one station are correlated, so sample-level bootstrap
intervals are too narrow.

For each candidate, report:

- mean OOF/test metric;
- station-cluster interval;
- worst-station metric;
- seed-to-seed interval.

This matters because Ridge, GBM, and the neural models differ by only a few
tenths of a millimeter on some metrics.

### 5.3 Paired prediction tests

Use paired prediction differences on identical samples. For example, compare
`ridge - neural` residuals station by station and test whether the mean absolute
difference is consistently positive or negative across stations. This is more
informative than declaring a winner from pooled MAE alone.

## 6. Hawaii transfer learning or pooled training

Hawaii data could help, but it should be treated as a separate experiment with
clear domain-shift risks.

Potential benefits:

- the original LAND was developed for Hawaiian station rainfall;
- Hawaii has far more stations and longer spatial diversity;
- its atmosphere/DEM/month architecture is conceptually compatible.

Risks:

- Hawaii stations sample a different orographic and climatic regime;
- raw variables, grids, units, or missing-value conventions may differ;
- weekly Hawaii rainfall distributions may not match American Samoa;
- pooled training may simply teach Hawaii-specific spatial relationships.

Defensible sequence:

1. Harmonize the feature schema exactly: same variables, channel order,
   normalization rules, patch sizes, DEM bands, and weekly aggregation.
2. Train Hawaii-only LAND using pre-2017 data and validate on held-out Hawaiian
   stations.
3. Evaluate zero-shot transfer to American Samoa training stations only.
4. Fine-tune on American Samoa training stations with a small learning rate.
5. Compare against AS-only training using AS LOSO OOF metrics.
6. Only then run the untouched AS test set once.

Do not select pretraining, freezing, or fine-tuning choices using the five AS
test stations.

## 7. Baselines worth adding

### 7.1 Gaussian-process spatial residual baseline

The original LAND comparison used GLM+GP for unseen sites. An American Samoa
analog could model residuals as a function of coordinates, elevation, and DEM
summary features. With only 24 stations this may be unstable, but it is a useful
spatial-interpolation reference.

### 7.2 Quantile or generalized additive baseline

A GAM/quantile baseline could test smooth nonlinear effects without deep
learning. Keep it pooled across stations and use the same flattened features.

### 7.3 Simple stacked baseline

A Ridge meta-model over OOF predictions from Ridge, GBM, Tweedie, Gamma LAND,
and Huber LAND is a natural final competitor. It must be trained only on OOF
predictions.

## 8. Recommended order

1. ~~Complete station/source QC for the high-zero legacy gauges~~ — done;
   `aunuu`/`vaipito2000` excluded and `afono_UH` offline runs masked (see 1.1).
2. ~~Retrain the retained finalists on the cleaned dataset and regenerate
   baselines~~ — done: `weekly_land_v5_qc`, `weekly_land_v5_huber_rw_msemon_qc`,
   `weekly_land_v6_huber_rw_t36_qc`, and the `v5_qc_gamma_huber_cv_mse` blend;
   pre-QC baseline metrics preserved in `output/baselines_pre_qc/`. Post-QC
   Ridge still beats every neural run, so the open question is architecture,
   not data cleanliness.
3. **Active:** run the `weekly_land_v7_huber_rw_temporal_broad` Optuna study
   (see README 7 — broad architecture space, temporal CV, `mse_ratio`,
   median fold aggregation). Validate its finalists under one common protocol
   (`--cv-mode both` or LOSO) before training winners; never compare raw
   objectives across `cv-mode` values.
4. ~~Station-sensitivity ablation on `pioa_afono`~~ — done: exclusion worsened
   RMSE 49.32→50.02, so it is retained (see 1.2). Further gauge exclusions are
   low priority without new artifact evidence.
5. Add lagged `dry_days`/`wet_days` features, first to Ridge/GBM and then to
   LAND only if the baselines improve on OOF low-end diagnostics.
6. Increase the leading formulation from three to five seeds.
7. Build OOF predictions for Ridge/GBM and test a leakage-free multi-model
   blend — the Ridge stack is now the most promising ensemble member given
   the post-QC leaderboard.
8. Add station-year and expanding-window temporal validation summaries.
9. Revisit daily atmospheric encoding under the v5-sized scalar setup.
10. Consider neural Tweedie, quantile regression, or a controlled hurdle head.
11. Consider Hawaii transfer only after the AS source/validation issues are
    stable enough that domain shift can be interpreted.

## 9. Promotion criteria

A new model should not replace the current candidates unless it:

- improves OOF MAE or MSE under paired station folds;
- improves or preserves OOF low-end diagnostics (`<1/5/10 mm` rates and mean
  prediction on dry weeks);
- does not materially worsen OOF bias or high-quantile bias;
- remains competitive under station-cluster bootstrap intervals;
- has acceptable worst-station and worst-source-group performance;
- retains a test prediction file aligned to the same observed/station arrays;
- includes code, environment, data-hash, and command provenance.

A test-set win by a fraction of a millimeter is not enough evidence by itself.
Given Ridge's current strength, every neural improvement should also answer the
question:

> What does the learned spatial representation explain that Ridge and GBM do
> not already explain?
