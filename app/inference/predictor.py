# app/inference/predictor.py
from pathlib import Path
import json
import joblib
import pandas as pd
from app.core.config import PACKAGED_JSON

# Define the exact feature order used during training
FEATURE_ORDER = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat",
    "long", "sqft_living15", "sqft_lot15"
]


class Predictor:
    def __init__(self):
        self.model = None
        self.model_version = None

    def load(self):
        model_path = Path("models/packaged/model.pkl")

        if not model_path.exists():
            raise RuntimeError(
                "Packaged model not found at models/packaged/model.pkl. "
                "Ensure CI runs `package_model.py` before integration tests."
            )

        if not PACKAGED_JSON.exists():
            raise RuntimeError(
                f"Packaged metadata file missing at {PACKAGED_JSON}. "
                "Model packaging is incomplete or corrupted."
            )

        with open(PACKAGED_JSON) as f:
            info = json.load(f)

        if "model_version" not in info:
            raise RuntimeError(
                "Invalid packaged metadata: missing `model_version` field."
            )

        self.model_version = info["model_version"]
        self.model = joblib.load(model_path)

    def predict(self, features: dict):
        if self.model is None:
            self.load()

        # Enforce feature order strictly
        X_df = pd.DataFrame([features])[FEATURE_ORDER]
        return float(self.model.predict(X_df)[0])
