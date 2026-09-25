from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
FEATURE_DIR = DATA_DIR / "features"
DATASET_PATH = DATA_DIR / "weekly_dataset.npz"
TUNING_DIR = OUTPUT_DIR / "tuning"
RUNS_DIR = OUTPUT_DIR / "runs"

SEED = 42

# Strict spatiotemporal train/test split. Train and test are disjoint in both
# station ID and year. Training uses the stations below during years up to and
# including TRAIN_YEAR_END; testing uses TEST_STATIONS during years after
# TRAIN_YEAR_END.
TRAIN_YEAR_END = 2016
TEST_STATIONS = ["aasu_UH", "afono_UH", "aunuu_UH", "poloa_UH", "vaipito_UH"]

# Kept for backwards compatibility with old notebooks / code that reference it.
N_TEST_STATIONS = len(TEST_STATIONS)

DEFAULTS = {
    "climate_units": 120,
    "dem_units": 64,
    "month_units": 32,
    "hidden_units": 256,
    "dropout": 0.3,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
}
