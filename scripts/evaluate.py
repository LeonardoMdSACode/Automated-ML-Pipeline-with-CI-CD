#! python3
# scripts/evaluate.py
# Metrics computation + baseline initialization

import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- Paths ----------------
PROCESSED_DATA = Path("data/processed/train_test.npz")

REGISTRY = Path("models/registry")
LATEST = REGISTRY / "latest.json"

REPORTS_DIR = Path("reports")
EVAL_PATH = REPORTS_DIR / "evaluation.json"

BASELINE_DIR = Path("models/baseline")
BASELINE_METRICS = BASELINE_DIR / "metrics.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load latest model ----------------
with open(LATEST, "r") as f:
    latest_version = json.load(f)["latest_version"]

MODEL_PATH = REGISTRY / latest_version / "model.pkl"
model = joblib.load(MODEL_PATH)

# ---------------- Load test data ----------------
data = np.load(PROCESSED_DATA)
X_test = data["X_test"]
y_test = data["y_test"]

# ---------------- Predict ----------------
y_pred = model.predict(X_test)

# ---------------- Metrics ----------------
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "model_version": latest_version
}

# ---------------- Write evaluation report ----------------
with open(EVAL_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------- Initialize baseline if missing ----------------
if not BASELINE_METRICS.exists() or BASELINE_METRICS.stat().st_size == 0:
    with open(BASELINE_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Baseline metrics initialized.")

print("Evaluation complete:", metrics)
