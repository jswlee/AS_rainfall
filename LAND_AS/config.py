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
N_VAL_STATIONS = 5
N_TEST_STATIONS = 3
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.20

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
