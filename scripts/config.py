# scripts/config.py
from pathlib import Path

# Global settings
# ------------------------
SEED = 42

# Data paths
# ------------------------
RAW_DATA = Path("data/raw/kc_house_data.csv")
PROCESSED_DATA = Path("data/processed/train_test.npz")

# Model registry
# ------------------------
REGISTRY = Path("models/registry")
REGISTRY.mkdir(parents=True, exist_ok=True)

LATEST_JSON = REGISTRY / "latest.json"

# Baseline
# ------------------------
BASELINE_DIR = Path("models/baseline")
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_METRICS = BASELINE_DIR / "metrics.json"

# Reports
# ------------------------
EVAL_DIR = Path("reports/evaluations")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_FILE = Path("reports/comparison.json")
COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

# Packaging
# ------------------------
PACKAGE_DIR = Path("models/packaged")
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
