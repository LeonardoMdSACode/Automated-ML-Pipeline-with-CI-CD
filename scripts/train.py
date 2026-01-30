#! python3
# scripts/train.py
# Deterministic training with immediate evaluation and latest.json update

import sys
from pathlib import Path
from datetime import datetime
import random
import json
import os

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import SEED, RAW_DATA, PROCESSED_DATA, REGISTRY, LATEST_JSON

# Seed everything for deterministic behavior
random.seed(SEED)
np.random.seed(SEED)

# Determine new model version
existing = sorted([d.name for d in REGISTRY.iterdir() if d.name.startswith("model_v")])
if existing:
    last_version = max(int(d.split("_v")[-1]) for d in existing)
    new_version = f"model_v{last_version + 1:03d}"
else:
    new_version = "model_v001"

# Create model directory
MODEL_DIR = REGISTRY / new_version
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Load dataset
df = pd.read_csv(RAW_DATA)

TARGET = "price"
OPTIONAL_DROP = ["id", "date"]

if TARGET not in df.columns:
    raise ValueError("Target column 'price' not found in dataset")

cols_to_drop = [TARGET] + [c for c in OPTIONAL_DROP if c in df.columns]

X = df.drop(columns=cols_to_drop)
y = df[TARGET]

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

# Save processed train/test data
PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
np.savez(PROCESSED_DATA, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Compute metrics on test set
y_pred = model.predict(X_test)
metrics = {
    "r2": r2_score(y_test, y_pred),
    "mae": mean_absolute_error(y_test, y_pred)
}

# Save metadata
metadata = {
    "version": new_version,
    "n_samples": len(df),
    "n_features": X.shape[1],
    "model_type": "RandomForestRegressor",
    "metrics": metrics,
    "created_at": datetime.utcnow().isoformat() + "Z"
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

# Update latest.json pointer
latest_info = {
    "latest_version": new_version,
    "path": os.path.relpath(MODEL_PATH, start=ROOT),
    "metrics": metrics,
    "created_at": metadata["created_at"]
}
with open(LATEST_JSON, "w") as f:
    json.dump(latest_info, f, indent=2)

print(f"Training complete. Model saved to {MODEL_DIR}")
