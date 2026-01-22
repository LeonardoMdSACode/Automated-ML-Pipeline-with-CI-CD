#! python3
# scripts/package_model.py
# Registry promotion: package the best model according to quality gate

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import shutil

from scripts.config import (
    REGISTRY,
    EVAL_DIR,
    PACKAGE_DIR,
    BASELINE_METRICS,
)

with open(BASELINE_METRICS) as f:
    baseline = json.load(f)

def gate_passed(latest, baseline):
    if latest["rmse"] > baseline["rmse"]:
        return False
    if latest["r2"] < baseline["r2"]:
        return False
    return True

eval_files = list(EVAL_DIR.glob("*.json"))
if not eval_files:
    raise RuntimeError("No evaluation files found. Cannot package model.")

print([f.name for f in eval_files])

all_models = [(f, json.load(open(f))) for f in eval_files]
all_models.sort(key=lambda x: (x[1]["rmse"], -x[1]["r2"]))

best_model_file, best_metrics = all_models[0]
best_model_version = best_model_file.stem

if not gate_passed(best_metrics, baseline):
    print(f"Warning: Best model {best_model_version} does not pass quality gate vs baseline")

print(f"Best model selected: {best_model_version} with RMSE={best_metrics['rmse']} and R2={best_metrics['r2']}")

model_folder = best_model_file.stem.split("_run")[0]
src_model = REGISTRY / model_folder / "model.pkl"

if not src_model.exists():
    raise FileNotFoundError(f"Model file not found: {src_model}")

shutil.copy(src_model, PACKAGE_DIR / "model.pkl")
shutil.copy(best_model_file, PACKAGE_DIR / "metrics.json")

record = {
    "version": best_model_version,
    "rmse": best_metrics["rmse"],
    "r2": best_metrics["r2"]
}
with open(PACKAGE_DIR / "packaged.json", "w") as f:
    json.dump(record, f, indent=2)

shutil.copy(best_model_file, BASELINE_METRICS)
print(f"Baseline updated to {best_model_version}")
print(f"Model {best_model_version} packaged successfully in {PACKAGE_DIR}")
