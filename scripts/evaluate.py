#! python3
# scripts/evaluate.py
# Metrics computation with baseline initialization

import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path
import time

# Paths
PROCESSED_DATA = Path("data/processed/train_test.npz")
REGISTRY = Path("models/registry")         # contains model version folders and latest.json
LATEST_JSON = REGISTRY / "latest.json"
BASELINE = Path("models/baseline")
BASELINE.mkdir(exist_ok=True, parents=True)
BASELINE_METRICS = BASELINE / "metrics.json"

EVAL_DIR = Path("reports/evaluations")
EVAL_DIR.mkdir(exist_ok=True, parents=True)

COMPARISON_FILE = Path("reports/comparison.json")
COMPARISON_FILE.parent.mkdir(exist_ok=True, parents=True)

# Load latest model version
with open(LATEST_JSON) as f:
    latest_version = json.load(f)["latest_version"]

# Correct path: each version is a folder
MODEL_PATH = REGISTRY / latest_version / "model.pkl"
model = joblib.load(MODEL_PATH)

# Load data
data = np.load(PROCESSED_DATA)
X_test = data["X_test"]
y_test = data["y_test"]

# Predict
y_pred = model.predict(X_test)

# Compute metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Evaluation JSON ---
timestamp = int(time.time() * 1_000_000)
eval_file = EVAL_DIR / f"{latest_version}_run{timestamp}.json"

evaluation = {
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "model_version": latest_version
}

# Save evaluation report
with open(eval_file, "w") as f:
    json.dump(evaluation, f, indent=2)

# Save baseline metrics
with open(BASELINE_METRICS, "w") as f:
    json.dump(evaluation, f, indent=2)

print(f"Evaluation complete: {evaluation}")
print(f"Baseline metrics updated at {BASELINE_METRICS}")
