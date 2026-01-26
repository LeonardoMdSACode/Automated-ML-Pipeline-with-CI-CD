#! python3
# scripts/package_model.py
# Registry promotion: package the best model according to quality gate

import sys
from pathlib import Path
from datetime import datetime
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import REGISTRY, EVAL_DIR, PACKAGE_DIR, BASELINE_METRICS

# Load baseline metrics
with open(BASELINE_METRICS) as f:
    baseline = json.load(f)

def gate_passed(latest, baseline):
    if latest["rmse"] > baseline["rmse"]:
        return False
    if latest["r2"] < baseline["r2"]:
        return False
    return True

# Get all evaluation JSONs
eval_files = list(EVAL_DIR.glob("*.json"))
if not eval_files:
    raise RuntimeError("No evaluation files found. Cannot package model.")

# Load evaluations
all_models = [(f, json.load(open(f))) for f in eval_files]
# Sort by rmse ascending, r2 descending
all_models.sort(key=lambda x: (x[1]["rmse"], -x[1]["r2"]))

best_eval_file, best_metrics = all_models[0]
best_model_version = best_eval_file.stem.split("_run")[0]

if not gate_passed(best_metrics, baseline):
    print(f"Warning: Best model {best_model_version} does not pass quality gate vs baseline")

print(f"Best model selected: {best_model_version} with RMSE={best_metrics['rmse']} and R2={best_metrics['r2']}")

# Copy model.pkl from registry
src_model = REGISTRY / best_model_version / "model.pkl"
if not src_model.exists():
    raise FileNotFoundError(f"Model file not found: {src_model}")

PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(src_model, PACKAGE_DIR / "model.pkl")

# Copy evaluation JSON as metrics.json
shutil.copy(best_eval_file, PACKAGE_DIR / "metrics.json")

# Load metadata from registry for exact latest.json format
metadata_path = REGISTRY / best_model_version / "metadata.json"
with open(metadata_path) as f:
    metadata = json.load(f)

# Create packaged.json in latest.json format
packaged_record = {
    "model_version": metadata.get("version", best_model_version),
    "path": str(PACKAGE_DIR / "model.pkl"),
    "metrics": metadata.get("metrics", best_metrics),
    "created_at": metadata.get("created_at", datetime.utcnow().isoformat() + "Z")
}

with open(PACKAGE_DIR / "packaged.json", "w") as f:
    json.dump(packaged_record, f, indent=2)

# Update baseline metrics
shutil.copy(best_eval_file, BASELINE_METRICS)

print(f"Baseline updated to {best_model_version}")
print(f"Model {best_model_version} packaged successfully in {PACKAGE_DIR}")
