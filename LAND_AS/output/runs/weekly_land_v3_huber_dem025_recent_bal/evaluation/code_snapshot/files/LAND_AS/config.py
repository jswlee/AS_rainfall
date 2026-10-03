"""Shared project configuration: output paths, train/test split, lag depth.

Defines where the weekly dataset and all run/tuning outputs live, plus the
strict spatiotemporal split (``TEST_STATIONS`` x years after
``TRAIN_YEAR_END``) used by every stage. Imported by s1-s5; sibling module
``LAND_AS.s1_prepare.config`` holds the raw-data paths used at prep time.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
FEATURE_DIR = DATA_DIR / "features"
DATASET_PATH = DATA_DIR / "weekly_dataset.npz"
DAILY_DATASET_PATH = DATA_DIR / "daily_dataset.npz"
TUNING_DIR = OUTPUT_DIR / "tuning"
RUNS_DIR = OUTPUT_DIR / "runs"

SEED = 42

# Strict spatiotemporal train/test split. Train and test are disjoint in both
# station ID and year. Training uses the stations below during years up to and
# including TRAIN_YEAR_END; testing uses TEST_STATIONS during years after
# TRAIN_YEAR_END.
TRAIN_YEAR_END = 2016
TEST_STATIONS = ["aasu_UH", "aunuu_UH", "poloa_UH", "vaipito_UH"]

# Kept for backwards compatibility with old notebooks / code that reference it.
N_TEST_STATIONS = len(TEST_STATIONS)

# Number of prior periods provided to the model as temporal context. Each
# sample's inputs include the reanalysis patch and observed rainfall from up
# to N earlier periods of the same station; missing lag periods (record
# gaps or starts) are zero-filled and flagged with a mask feature.
# LAG_WEEKS applies to weekly_dataset.npz, LAG_DAYS to daily_dataset.npz.
LAG_WEEKS = 3
LAG_DAYS = 7

# loso_recent CV mode: LOSO folds whose validation is restricted to the held-out
# station's years >= this value, so early stopping / model selection sees the
# same unseen-station + late-era difficulty as the test set (post-2016).
# Training folds are unchanged (all other stations, all train years).
LOSO_RECENT_YEAR_START = 2010


def dataset_path_for(freq):
    """Dataset NPZ for a sampling frequency ("weekly" or "daily")."""
    if freq == "daily":
        return DAILY_DATASET_PATH
    if freq == "weekly":
        return DATASET_PATH
    raise ValueError(f"Unknown freq {freq!r}; use 'weekly' or 'daily'")

# Frozen hyperparameter values (v5 tuning showed these have no effect on
# val MSE -- see tuning param importance). Only learning_rate, weight_decay,
# rain_lag_weeks, and the DEM extent cfgs remain tunable.
DEFAULTS = {
    "climate_units": 240,
    "dem_units": 64,
    "month_units": 16,
    "hidden_units": 320,
    "dropout": 0.3,
    "dem_size": 10,
    "climate_lag_weeks": 0,
    "rain_lag_weeks": 3,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
}
