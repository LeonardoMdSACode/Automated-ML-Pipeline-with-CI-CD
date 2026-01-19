#! python3
# scripts\evaluate.py
# Metrics computation

import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from pathlib import Path

# Paths
PROCESSED_DATA = Path("data/processed/train_test.npz")
REGISTRY = Path("models/registry")
LATEST = REGISTRY / "latest.json"
EVAL_PATH = Path("reports/evaluation.json")
EVAL_PATH.parent.mkdir(exist_ok=True, parents=True)

# Load latest model version
with open(LATEST) as f:
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

# Save evaluation
evaluation = {
    "rmse": float(rmse),
    "mae": float(mae),
    "model_version": latest_version
}
with open(EVAL_PATH, "w") as f:
    json.dump(evaluation, f)

print("Evaluation complete:", evaluation)
