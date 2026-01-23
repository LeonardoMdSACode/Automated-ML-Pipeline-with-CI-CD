# app/inference/predictor.py
from pathlib import Path
import json
import joblib
import numpy as np

from app.core.config import LATEST_JSON

import pandas as pd


class Predictor:
    def __init__(self):
        self.latest_model_info = self._load_latest()
        self.model_version = self.latest_model_info["latest_version"]
        self.model = self._load_model()

    def _load_latest(self):
        with open(LATEST_JSON, "r") as f:
            return json.load(f)

    def _load_model(self):
        model_path = (
            Path("models")
            / "registry"
            / self.model_version
            / "model.pkl"
        )
        return joblib.load(model_path)

    def predict(self, features: dict) -> float:
        X = pd.DataFrame([features])
        return float(self.model.predict(X)[0])
