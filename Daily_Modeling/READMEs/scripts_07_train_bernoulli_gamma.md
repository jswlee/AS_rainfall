# `scripts/07_train_bernoulli_gamma.py` — Train Bernoulli-Gamma GLM

## Purpose
Seventh pipeline step. Fits a two-part Bernoulli-Gamma Generalized Linear Model per station: (1) logistic regression for rain/no-rain classification, (2) Gamma GLM with log link for rain amount given rain > 0. This is the classical statistical baseline from the paper.

## Relation to the Deep Downscaling Paper
Hatanaka et al. (2025) Section 3c describes the Bernoulli-Gamma GLM as the baseline model. It is site-specific (one model per station), uses the same flattened feature vector as the MLP, and serves as a reference point for evaluating the neural network models. The two-part structure is standard for zero-inflated continuous targets — the Bernoulli component models the probability of rain, and the Gamma component models the amount conditional on rain occurring. This is more principled than regressing directly on rainfall, which includes many zeros.

## Line-by-Line Walkthrough

### Imports (lines 1–33)
```python
from Daily_Modeling.models.bernoulli_gamma import BernoulliGammaGLM, flatten_features_numpy
```
- `BernoulliGammaGLM` wraps scikit-learn's `LogisticRegression` + a Gamma GLM (via `statsmodels` or manual fitting).
- `flatten_features_numpy` concatenates climate (15×3×3=135), local DEM, regional DEM, and month one-hot into a single feature vector — the numpy equivalent of `_FlatDataset`.
- This script runs on **CPU only** — GLMs don't benefit from GPU.

### `main()` (lines 35–151)

**Lines 40–57: Load data + splits**
Same pattern as other training scripts: load tensors (CPU), compute data-driven year boundaries, assign station groups, spatio-temporal split, normalise.

**Lines 59–60: Normalisation report**
Prints the full normalisation report for transparency.

**Lines 67–72: Convert to numpy**
```python
climate_np = tensors["climate"].numpy()
local_dem_np = tensors["local_dem"].numpy()
regional_dem_np = tensors["regional_dem"].numpy()
month_np = tensors["temporal"].numpy()
rain_np = tensors["targets"].numpy()  # raw mm
```
Converts PyTorch tensors to numpy for sklearn/statsmodels. Note: `rain_np` is in raw mm (not normalised), because the GLM directly predicts mm.

**Lines 83–122: Per-station training loop**
```python
for station_name in unique:
    sp = station_proportional_split(stations, years, meta["months"], meta["days"], station_name)
    if len(sp["train"]) < 50:
        continue
    X_train = flatten_features_numpy(climate_np[sp["train"]], ...)
    y_train = rain_np[sp["train"]]
    glm = BernoulliGammaGLM()
    glm.fit(X_train, y_train)
```
For each station:
1. Computes a per-station chronological 70/20/10 split (same as site MLP).
2. Skips stations with fewer than 50 training samples.
3. Flattens features into a 2-D array `(N, D)`.
4. Fits the Bernoulli-Gamma model.
5. Evaluates on val and test splits, computing RMSE/MAE/MBE/R²/Spearman for each.

**Lines 124–126: Save models**
```python
with open(out_dir / "glm_models.pkl", "wb") as f:
    pickle.dump(models, f)
```
Pickles all fitted GLMs for potential later use (inference on new data).

**Lines 128–147: Aggregate metrics**
Concatenates all per-station predictions and computes aggregate metrics. Also computes baseline (mean predictor) metrics for comparison. Saves a scatter plot and prediction NPZ.

## Data Manipulation
- **Per-station proportional split**: Each station's data is sorted by date, then split 70/20/10. This guarantees every station with sufficient data gets val/test samples, regardless of its year range.
- **No DEM cropping**: The Bernoulli-Gamma model uses the full (max-size) DEM patches, flattened. This means its feature vector is larger than the MLP's when DEM cropping is used. This is a known inconsistency — the GLM doesn't benefit from the multi-resolution DEM tuning.
- **Raw targets**: Unlike the neural models which normalise targets by dividing by `target_std_mm`, the GLM works directly in mm. The Gamma component's log link naturally handles the skewed distribution.

## Architecture Decisions
- **Site-specific models**: Unlike LAND (one model for all stations), the GLM fits independently per station. This means it cannot generalise to unseen stations — it's a temporal-only baseline.
- **CPU-only**: GLMs are fast to fit and don't benefit from GPU. The entire script runs in ~1 minute.
- **Two-part model**: Separating rain occurrence from rain amount is more principled than a single regression, because the rainfall distribution has a point mass at zero that Gaussian/MSE regression handles poorly.
- **Pickle serialisation**: Using pickle for model saving is simple but fragile (tied to sklearn version). JSON-based coefficient storage would be more robust.

## Areas of Improvement
- **Add DEM cropping**: Apply the same DEM crop config as the neural models for a fair comparison. Currently the GLM sees 11×11+25×25=746 DEM features while the MLP might see only 3×3+3×3=18.
- **Regularisation tuning**: The logistic regression and Gamma GLM regularisation strengths are not tuned. A quick grid search over the `C` parameter could improve performance.
- **Feature selection**: With 135 climate + 746 DEM + 12 month = 893 features, the GLM may suffer from multicollinearity. PCA or L1 regularisation could help.
- **Zero-inflated alternatives**: A Tweedie GLM (compound Poisson-Gamma) could replace the two-part model with a single regression that naturally handles the zero-inflation.
- **Spatial evaluation**: Currently only evaluates temporally (same station, future years). Adding a spatial evaluation mode would allow comparison with LAND on held-out stations.
