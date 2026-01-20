#! python3
# scripts/package_model.py
# Registry promotion

import json
import shutil
from pathlib import Path

# Paths
REGISTRY = Path("models/registry")
EVAL_DIR = Path("reports/evaluations")
PACKAGE_DIR = Path("models/packaged")
PACKAGE_DIR.mkdir(exist_ok=True, parents=True)

# Step 1: Find best model by RMSE
eval_files = list(EVAL_DIR.glob("*.json"))
if not eval_files:
    raise RuntimeError("No evaluation files found. Cannot package model.")

best_rmse = float("inf")
best_model_file = None

for f in eval_files:
    with open(f) as ef:
        data = json.load(ef)
        if data["rmse"] < best_rmse:
            best_rmse = data["rmse"]
            best_model_file = f

best_model_version = best_model_file.stem  # e.g., model_v004
print(f"Best model selected: {best_model_version} with RMSE={best_rmse}")

# Step 2: Copy model.pkl
src_model = REGISTRY / best_model_version / "model.pkl"
if not src_model.exists():
    raise FileNotFoundError(f"Model file not found: {src_model}")

shutil.copy(src_model, PACKAGE_DIR / "model.pkl")

# Step 3: Copy metrics
src_metrics = best_model_file
shutil.copy(src_metrics, PACKAGE_DIR / "metrics.json")

# Step 4: Update packaged record
PACKAGED_RECORD = PACKAGE_DIR / "packaged.json"
record = {
    "version": best_model_version,
    "rmse": best_rmse
}
with open(PACKAGED_RECORD, "w") as f:
    json.dump(record, f, indent=2)

print(f"Model {best_model_version} packaged successfully in {PACKAGE_DIR}")
