#! python3
# scripts/train.py
# Deterministic training

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import json
import random

from scripts.config import (
    SEED,
    RAW_DATA,
    PROCESSED_DATA,
    REGISTRY,
    LATEST_JSON,
)

random.seed(SEED)
np.random.seed(SEED)

# Determine new model version
existing = sorted([d.name for d in REGISTRY.iterdir() if d.name.startswith("model_v")])
if existing:
    last_version = max(int(d.split("_v")[-1]) for d in existing)
    new_version = f"model_v{last_version+1:03d}"
else:
    new_version = "model_v001"

MODEL_DIR = REGISTRY / new_version
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Load dataset
df = pd.read_csv(RAW_DATA)
X = df.drop(columns=["price", "id", "date"])
y = df["price"]

# Split data deterministically
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    random_state=SEED
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)

# Save train/test data
PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
np.savez(PROCESSED_DATA, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Save metadata
metadata = {
    "n_samples": len(df),
    "n_features": X.shape[1],
    "model_type": "RandomForestRegressor",
    "version": new_version
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

# Update latest pointer
with open(LATEST_JSON, "w") as f:
    json.dump({"latest_version": new_version}, f, indent=2)

print(f"Training complete. Model saved to {MODEL_DIR}")
