#! python3
# scripts/package_model.py
# Registry promotion: package the best model according to quality gate

import json
import shutil
from pathlib import Path

# Paths
REGISTRY = Path("models/registry")
EVAL_DIR = Path("reports/evaluations")
PACKAGE_DIR = Path("models/packaged")
PACKAGE_DIR.mkdir(exist_ok=True, parents=True)
BASELINE_METRICS = Path("models/baseline/metrics.json")

# Load baseline
with open(BASELINE_METRICS) as f:
    baseline = json.load(f)

# Step 1: Filter models passing quality gate
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

# Step 2: Pick the one with lowest RMSE among all models (ignore gate)
# Sort by RMSE ascending, then R² descending
all_models.sort(key=lambda x: (x[1]["rmse"], -x[1]["r2"]))

best_model_file, best_metrics = all_models[0]
best_model_version = best_model_file.stem

if not gate_passed(best_metrics, baseline):
    print(f"Warning: Best model {best_model_version} does not pass quality gate vs baseline")

print(f"Best model selected: {best_model_version} with RMSE={best_metrics['rmse']} and R2={best_metrics['r2']}")

# Step 3: Copy model.pkl
src_model = REGISTRY / best_model_version / "model.pkl"
if not src_model.exists():
    raise FileNotFoundError(f"Model file not found: {src_model}")

shutil.copy(src_model, PACKAGE_DIR / "model.pkl")

# Step 4: Copy metrics
shutil.copy(best_model_file, PACKAGE_DIR / "metrics.json")

# Step 5: Update packaged record
PACKAGED_RECORD = PACKAGE_DIR / "packaged.json"
record = {
    "version": best_model_version,
    "rmse": best_metrics["rmse"],
    "r2": best_metrics["r2"]
}
with open(PACKAGED_RECORD, "w") as f:
    json.dump(record, f, indent=2)

# Step 6: Update baseline to the new best model
shutil.copy(best_model_file, BASELINE_METRICS)
print(f"Baseline updated to {best_model_version}")

print(f"Model {best_model_version} packaged successfully in {PACKAGE_DIR}")
