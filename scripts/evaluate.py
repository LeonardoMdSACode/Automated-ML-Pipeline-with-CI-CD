#! python3
# scripts/evaluate.py
# Metrics computation for candidate model

import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import PROCESSED_DATA, REGISTRY, LATEST_JSON, EVAL_DIR

# Load latest model version
with open(LATEST_JSON) as f:
    latest_version = json.load(f)["latest_version"]

MODEL_PATH = REGISTRY / latest_version / "model.pkl"
model = joblib.load(MODEL_PATH)

# Load test data
data = np.load(PROCESSED_DATA)
X_test = data["X_test"]
y_test = data["y_test"]

# Predict
y_pred = model.predict(X_test)

# Compute metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save evaluation only
timestamp = int(time.time() * 1_000_000)
eval_file = EVAL_DIR / f"{latest_version}_run{timestamp}.json"
evaluation = {
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "model_version": latest_version
}

EVAL_DIR.mkdir(parents=True, exist_ok=True)
with open(eval_file, "w") as f:
    json.dump(evaluation, f, indent=2)

print(f"Candidate evaluation complete: {evaluation}")
print(f"Saved at: {eval_file}")
