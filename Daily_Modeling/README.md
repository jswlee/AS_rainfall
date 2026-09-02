# Daily Rainfall Downscaling for American Samoa

Standalone project for daily rainfall prediction using three model families, following the methodology of Hatanaka et al. (2025) *"Super Resolution Statistical Downscaling from Sparse Observations with Deep Learning"*, adapted for **daily** data and **American Samoa** (26 stations).

## Models

1. **LAND** — Location-Agnostic Neural Downscaler (multi-branch NN; predicts at arbitrary locations)
2. **Bernoulli-Gamma GLM** — Site-specific generalized linear model with Bernoulli (rain/no-rain) + Gamma (amount) components
3. **Site-specific MLP** — Feed-forward MLP on flattened features (one model per station)
4. **SiteGLU (optional)** — GLU-gated variant of the site model (one model per station)

## Directory Layout

```
Daily_Modeling/
├── config.py                 # Paths, constants, variable configs
├── data/                     # Data loading, feature building, assembly, splits
│   ├── load_raw.py           # Load raw rainfall CSVs + station metadata
│   ├── build_features.py     # Build reanalysis & DEM patches (calls existing pipeline)
│   ├── assemble_dataset.py   # Combine into single NPZ
│   ├── splits.py             # Spatio-temporal cross-validation
│   └── dataset.py            # PyTorch Dataset / DataLoader utilities
├── models/                   # Model definitions
│   ├── land.py               # LAND architecture
│   ├── bernoulli_gamma.py    # Bernoulli-Gamma GLM (statsmodels)
│   ├── site_mlp.py           # Site-specific models (SiteMLP + SiteLSTM)
│   └── losses.py             # MSE, Gamma NLL, Bernoulli-Gamma NLL
├── utils/                    # Shared helpers
│   ├── training.py           # Training loop, early stopping, LR schedule
│   ├── metrics.py            # RMSE, MAE, MBE, Spearman, R², Wasserstein
│   ├── visualization.py      # EDA plots, training curves, scatter plots, maps
│   └── io_utils.py           # Save/load helpers
├── scripts/                  # Numbered entry-point scripts
│   ├── 01_build_features.py
│   ├── 02_assemble_dataset.py
│   ├── 03_eda.py
│   ├── 04_tune_land.py
│   ├── 05_tune_site_mlp.py
│   ├── 06_train_land.py
│   ├── 07_train_bernoulli_gamma.py
│   ├── 08_train_site_mlp.py
│   └── 09_evaluate_compare.py
└── output/                   # Created at runtime
```

## Workflow

```bash
# 1. Build intermediate features (reanalysis patches, DEM patches)
python -m Daily_Modeling.scripts.01_build_features

# 2. Assemble into a single NPZ
python -m Daily_Modeling.scripts.02_assemble_dataset

# 3. Exploratory data analysis
python -m Daily_Modeling.scripts.03_eda

# 4-5. Hyperparameter tuning (LAND, site MLP)
python -m Daily_Modeling.scripts.04_tune_land
python -m Daily_Modeling.scripts.05_tune_site_mlp

# 6-8. Train final models
python -m Daily_Modeling.scripts.06_train_land
python -m Daily_Modeling.scripts.07_train_bernoulli_gamma
python -m Daily_Modeling.scripts.08_train_site_mlp

# 9. Evaluate and compare all models
python -m Daily_Modeling.scripts.09_evaluate_compare
```

## Weekly aggregation

The pipeline can model ISO calendar weeks (Monday–Sunday) instead of days.
Step 01 is unchanged — weekly samples are built from the same daily features.

```bash
# Assemble a weekly dataset (rainfall summed; each reanalysis channel reduced
# to its within-week mean and std, so the channel count doubles).
python -m Daily_Modeling.scripts.02_assemble_dataset --freq weekly

# Every downstream script reads the matching NPZ via this env var
$env:AS_RAINFALL_FREQ = "weekly"          # PowerShell
export AS_RAINFALL_FREQ=weekly            # bash

python -m Daily_Modeling.scripts.03b_inspect_dataset
python -m Daily_Modeling.scripts.04_tune_land --n-trials 100
python -m Daily_Modeling.scripts.06_train_land \
    --hp-dir Daily_Modeling/output/weekly/tuning/land_weekly_gamma_... \
    --run-name land_weekly_gamma
```

There is no `--freq` flag on steps 03-11: they all follow `AS_RAINFALL_FREQ`.
Only step 02 takes `--freq`, because it is the step that builds the dataset.

**Loss type defaults to `gamma` for weekly** (vs `bernoulli_gamma` for daily).
Weekly rainfall is almost never zero (~96-100% of weeks are wet at 1mm), so the
Bernoulli wet/dry gate has no signal and the Gamma NLL alone is the standard
loss for strictly-positive, right-skewed aggregates.  The default is applied
automatically via `config.DEFAULT_LOSS_TYPE`; pass `--loss-type` to override.

Notes:

- `--min-days-per-week` (default 7) sets how many daily records a week needs to
  be kept; incomplete weeks are dropped, and the kept count is stored in the
  NPZ as `n_days`.  `AS_RAINFALL_MIN_DAYS_PER_WEEK` sets the default.
- Weeks are stamped with the date of their Monday, so `years`/`months`/`days`
  and the month one-hot keep the same meaning for splits and plots.
- `variables` are suffixed `_mean` / `_std`.  Because the LAND climate branch
  uses a grouped conv, **`climate_units` must be divisible by the channel
  count** — 30 for weekly vs 15 for daily.  `04_tune_land` derives its search
  space from the data (`num_climate_vars`) and handles this automatically;
  only the hardcoded `LAND_DEFAULT_HP` (`climate_units=64`) does not, so pass
  `--hp-dir` when training weekly rather than relying on the defaults.
- Outputs are fully separated by resolution, so daily and weekly runs never
  overwrite each other:

  ```
  output/
    features/                 # shared - step 01, frequency-independent
    daily/{assembled,eda,tuning,results,evaluation}/
    weekly/{assembled,eda,tuning,results,evaluation}/
  ```

## SiteGLU: gated feature mixing without recurrent overhead

This repository includes an optional site model called **`SiteGLU`** (defined in
`Daily_Modeling/models/site_mlp.py`) that replaces the earlier `SiteLSTM` experiment.

### Why GLU instead of LSTM?

The original `SiteLSTM` used an LSTM cell on a **single time-step** (seq_len=1).
That means the forget gate, cell state transfer, and hidden state carryover were
all wasted — they always operate on zeros.  You were paying for a full recurrent
cell just to get multiplicative gating.

A **Gated Linear Unit (GLU)** (Dauphin et al., 2017) provides the exact same
multiplicative gating with roughly half the parameters and no recurrent overhead:

    GLU(x) = (W₁x + b₁) ⊙ σ(W₂x + b₂)

- One linear projection produces a "value" and a "gate" (via sigmoid).
- Element-wise multiplication lets the gate suppress or amplify each feature.
- No forget gate, no cell state, no hidden state — just gating.

### Architecture: SiteMLP vs SiteGLU

Both models take the **same flattened input vector** (climate + DEM + month).

#### SiteMLP

- `[Linear → LayerNorm → ReLU → Dropout] × L`
- `Linear → softplus`

Every layer is a generic dense transform with additive (ReLU) nonlinearity.

#### SiteGLU

- `GLUBlock(input_size → hidden_sizes[0])` — **multiplicative** gating
- `[Linear → LayerNorm → ReLU → Dropout] × (L-1)` — MLP head from `hidden_sizes[1:]`
- `Linear → softplus`

The first layer is the GLU; remaining layers are standard MLP. For
`hidden_sizes = [256, 128, 128]`:

- GLU: `Linear(input, 512)` → chunk → `value * sigmoid(gate)` → `(batch, 256)`
- MLP head: `256 → 128 → 128`
- Output: `Linear(128, 1) → softplus`

### Why multiplicative gating helps

Your input vector mixes several conceptually distinct feature groups:

- **Climate** (13×3×3 = 117 features)
- **Local DEM** (11×11 = 121 features)
- **Regional DEM** (25×25 = 625 features)
- **Month** (12 features)

A plain `Linear → ReLU` layer can only *additively* combine these blocks.
A GLU can *multiplicatively suppress* entire feature groups when they are
irrelevant for a particular input pattern — e.g., zeroing out regional DEM
features when local topography dominates.  This acts as a learned, soft
feature selector.

### Tweedie p tuning — objective metric fix

When `tweedie_p` is tuned inside Optuna, different `p` values change the
**scale** of the Tweedie deviance loss (the exponent `p` appears in every
term of the deviance formula).  This means raw deviance values at `p=1.05`
are not comparable to `p=1.5` — Optuna can "cheat" by picking the `p` that
mathematically shrinks the loss number.

**Fix:** the Optuna objective now always returns **validation RMSE in mm**
(a scale-invariant metric), regardless of the training loss type.  The model
still trains with Tweedie (good gradients for zero-inflated data), but trials
are compared by RMSE so that `p` cannot game the objective.

### How to enable/tune SiteGLU

1. **Tuning-time:** include `arch_type` as a hyperparameter

   ```bash
   python -m Daily_Modeling.scripts.05_tune_site_mlp \
     --per-station-tuning \
     --loss-types tweedie \
     --tune-arch-type
   ```

   This adds `arch_type ∈ {"mlp", "glu"}` to the Optuna search space.

2. **Training-time:** `08_train_site_mlp.py` reads `arch_type` from
   `best_hyperparameters.json` (defaults to `"mlp"` if absent) and
   constructs the model via `build_model(arch_type, ...)`.

## Cross-Validation Strategy

Because we have only 26 stations, we use **spatio-temporal** CV:
- **Train stations** (~18): data from 1980–2015
- **Val stations** (~4, held-out locations): data from 2016–2020
- **Test stations** (~4, held-out locations): data from 2021–2024

Station assignments are deterministic (sorted by name, then assigned round-robin or by geographic clustering). This tests generalization to **both** unseen locations and unseen time periods.

For **site-specific** models (MLP, GLM), a temporal-only split is used within each station.

## Key Differences from the Paper

| Paper (Hawai'i, monthly) | This project (American Samoa, daily) |
|---|---|
| ~1,900 stations | 26 stations |
| Monthly rainfall totals | Daily rainfall totals |
| Gamma output distribution | Bernoulli-Gamma for daily zeros |
| 16 reanalysis channels | 13 reanalysis channels |
| One-hot or positional month embedding | One-hot month encoding only |
| 6-fold temporal CV | Spatio-temporal CV (location + time) |
