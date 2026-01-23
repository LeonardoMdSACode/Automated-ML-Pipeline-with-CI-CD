# app\api\schemas.py
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
