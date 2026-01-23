# app\schemas\request_response.py
from pydantic import BaseModel
from typing import Dict

class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
