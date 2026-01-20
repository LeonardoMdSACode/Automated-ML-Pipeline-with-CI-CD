#! python3
# scripts/evaluate.py
# Metrics computation with versioned evaluations + baseline initialization

import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path

# Paths
PROCESSED_DATA = Path("data/processed/train_test.npz")
REGISTRY = Path("models/registry")
LATEST_JSON = REGISTRY / "latest.json"
BASELINE_METRICS = Path("models/baseline/metrics.json")
EVAL_DIR = Path("reports/evaluations")
EVAL_DIR.mkdir(exist_ok=True, parents=True)

# Load latest model version
with open(LATEST_JSON) as f:
    latest_version = json.load(f)["latest_version"]

MODEL_PATH = REGISTRY / latest_version / "model.pkl"
model = joblib.load(MODEL_PATH)

# Load data
data = np.load(PROCESSED_DATA)
X_test = data["X_test"]
y_test = data["y_test"]

# Predict
y_pred = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Evaluation dict
evaluation = {
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "model_version": latest_version
}

# Save versioned evaluation report
EVAL_PATH = EVAL_DIR / f"{latest_version}.json"
with open(EVAL_PATH, "w") as f:
    json.dump(evaluation, f, indent=4)

print(f"Evaluation complete: {evaluation}")
print(f"Saved versioned evaluation report: {EVAL_PATH}")

# Initialize baseline if not exists
if not BASELINE_METRICS.exists() or BASELINE_METRICS.stat().st_size == 0:
    with open(BASELINE_METRICS, "w") as f:
        json.dump(evaluation, f, indent=4)
    print("Baseline metrics initialized.")
