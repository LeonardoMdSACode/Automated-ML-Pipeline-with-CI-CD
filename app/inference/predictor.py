# app/inference/predictor.py
from pathlib import Path
import json
import joblib
import pandas as pd
from app.core.config import LATEST_JSON

# Define the exact feature order used during training
FEATURE_ORDER = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat",
    "long", "sqft_living15", "sqft_lot15"
]

class Predictor:
    def __init__(self):
        self.latest_model_info = self._load_latest()
        self.model_version = self.latest_model_info["latest_version"]
        self.model = self._load_model()

    def _load_latest(self):
        with open(LATEST_JSON) as f:
            return json.load(f)

    def _load_model(self):
        model_path = (
            Path("models")
            / "registry"
            / self.model_version
            / "model.pkl"
        )
        return joblib.load(model_path)

    def predict(self, features: dict):
        # Fill missing features with 0
        X_df = pd.DataFrame([{f: features.get(f, 0) for f in FEATURE_ORDER}])
        return float(self.model.predict(X_df)[0])

