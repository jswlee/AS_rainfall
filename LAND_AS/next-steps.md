# Next experiments for `LAND_AS`

This document separates experiments that are statistically defensible with the
current 26-station dataset from exploratory ideas that need more data or a new
validation design.

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

`Daily_Modeling` is also relevant. `LAND_AS.prepare` already imports its weekly
aggregation, so the current dataset already uses complete ISO weeks and doubles
each reanalysis channel into within-week means plus within-week standard
deviations. `LAND_AS` now also supports fold-local normalization, temporal year
blocks, and `mean`/`median` fold aggregation in `tune.py`. The most useful
remaining imports are optimization metrics beyond MAE/MSE (CSI and percentile
bias), Tweedie/Bernoulli-Gamma losses, and more systematic architecture search.
Do not compare its reported metrics directly with `LAND_AS`: it uses a
different split and does not include the same rainfall-lag feature contract.

## Immediate finding: low-end target shift

The data-prep notebook now includes a dedicated low-end audit
(`notebooks/01_data_prep_eda.ipynb`). The key result is that the training/test
difference is not just a temporal rainfall-regime difference:

- training: 197/6,686 exact-zero weeks (2.95%);
- post-2016 test stations: 17/1,188 exact-zero weeks (1.43%);
- post-2016 rows for the two training WRCC stations: 2/734 (0.27%);
- those same WRCC stations had a 0.73% pre-2017 zero rate.

Most training zeros come from a small set of legacy gauges; `pioa_afono`,
`vaipito2000`, `aunuu`, `fagaitua`, and `vaipito_res` contribute about 66% of
them. Complete-week aggregation looks correct. The dominant issue is therefore
station/network heterogeneity plus a smaller temporal component.

This changes the experiment order: source/site quality control and
station-sensitivity ablations are now higher priority than another scalar loss
sweep. A hurdle/Bernoulli-Gamma model remains possible, but a leakage-free
occurrence classifier improves existing OOF amount predictions by only about
0.2-0.3 mm MAE and still predicts tens of millimeters on dry weeks.

## 1. Highest-priority controlled experiments

### 1.1 Source and site quality control

Before deleting stations or adding an occurrence head, inspect the high-zero
legacy records directly:

- `aunuu`
- `vaipito2000`
- `pioa_afono`
- `fagaitua`
- `vaipito_res`

Questions to answer:

- are missing observations encoded as zero?
- are multi-day accumulations represented by zeros followed by a large total?
- does the zero rate change at a gauge move, source change, or unit change?
- are dry weeks physically plausible given neighboring stations and atmosphere?
- why does `aunuu` differ so sharply from `aunuu_UH`?

The output should be a station-level QC table with evidence, not merely a list
of stations to drop.

### 1.2 Station-sensitivity ablation

After QC, train a controlled run excluding the stations whose dry behavior is
least trustworthy or least transferable. Start with the smallest defensible
removal set (for example `aunuu` and `vaipito2000`) rather than dropping all
high-zero stations.

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
  --run weekly_land_v5_huber_rw_msemon `
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
  --run weekly_land_v5_huber_rw_msemon_balanced `
  --loss-type huber_weighted `
  --monitor mse `
  --balanced-stations `
  --seeds 3 --epochs 500 --patience 50 --workers 4
```

Acceptance should be based on mean and worst-station OOF metrics, not only
pooled OOF metrics.

### 1.7 Compare checkpoint monitors under paired folds

`--monitor mae` and `--monitor mse` do not change the loss; they choose which
validation score retains the checkpoint. The weighted-Huber/MSE-monitor run is
currently the best neural standalone RMSE candidate, but monitor selection
should be compared using the same seeds and folds across delta values.

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
training weeks: 197 / 6686 exact zeros (2.95%)
test weeks:      17 / 1188 exact zeros (1.43%)
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

- many outputs with only 21 spatial training folds;
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

The original repository used year folds as well as station omission
(`LocationAgnosticNeuralDownscaling/README.md`). `LAND_AS` currently uses only
LOSO validation within pre-2017 years, while the test requires both spatial and
temporal transfer.

A stronger validation scheme would evaluate:

```text
LOSO station folds                  → spatial transfer
pre-2017 blocked year folds         → temporal transfer
nested station-year folds           → joint transfer
```

No new test evaluation is needed to implement this. It is a better selection
criterion for methods intended to work after 2016.

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
summary features. With only 26 stations this may be unstable, but it is a useful
spatial-interpolation reference.

### 7.2 Quantile or generalized additive baseline

A GAM/quantile baseline could test smooth nonlinear effects without deep
learning. Keep it pooled across stations and use the same flattened features.

### 7.3 Simple stacked baseline

A Ridge meta-model over OOF predictions from Ridge, GBM, Tweedie, Gamma LAND,
and Huber LAND is a natural final competitor. It must be trained only on OOF
predictions.

## 8. Recommended order

1. Complete station/source QC for the high-zero legacy gauges and document
   whether their zeros are physical observations or reporting artifacts.
2. Run the smallest defensible station-sensitivity ablation; keep the removed
   stations in validation reports rather than silently dropping them.
3. Evaluate the completed v6 spatial and temporal candidates under one common
   LOSO finalist protocol. Do not compare raw spatial and temporal tuning MSE.
4. Add lagged `dry_days`/`wet_days` features, first to Ridge/GBM and then to
   LAND only if the baselines improve on OOF low-end diagnostics.
5. Increase the leading formulation from three to five seeds.
6. Build OOF predictions for Ridge/GBM and test a leakage-free multi-model
   blend.
7. Add station-year and expanding-window temporal validation summaries.
8. Revisit daily atmospheric encoding under the v5-sized scalar setup.
9. Consider neural Tweedie, quantile regression, or a controlled hurdle head.
10. Consider Hawaii transfer only after the AS source/validation issues are
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
