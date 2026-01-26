# app/inference/predictor.py
from pathlib import Path
import json
import joblib
import pandas as pd
from app.core.config import LATEST_JSON, PACKAGED_JSON

# Define the exact feature order used during training
FEATURE_ORDER = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat",
    "long", "sqft_living15", "sqft_lot15"
]

class Predictor:
    def __init__(self):
        # Load packaged model info
        self.packaged_info = self._load_packaged()
        self.model_version = self.packaged_info["model_version"]
        self.model = self._load_model()

    def _load_packaged(self):
        with open(PACKAGED_JSON) as f:
            return json.load(f)

    def _load_model(self):
        model_path = Path("models") / "packaged" / "model.pkl"
        return joblib.load(model_path)

    def predict(self, features: dict):
        # Ensure features are in correct order
        X_df = pd.DataFrame([features])[FEATURE_ORDER]
        return float(self.model.predict(X_df)[0])
    