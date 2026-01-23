# app/inference/predictor.py
from pathlib import Path
import json
import joblib
import numpy as np

from app.core.config import LATEST_JSON


class Predictor:
    def __init__(self):
        self.latest_model_info = self._load_latest()
        self.model = self._load_model()
        self.model_version = self.latest_model_info["model_version"]

    def _load_latest(self):
        with open(LATEST_JSON, "r") as f:
            return json.load(f)

    def _load_model(self):
        model_path = Path(self.latest_model_info["path"])
        return joblib.load(model_path)

    def predict(self, features: list[float]) -> float:
        X = np.array(features).reshape(1, -1)
        return float(self.model.predict(X)[0])
