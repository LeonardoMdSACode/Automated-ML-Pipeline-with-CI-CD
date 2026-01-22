# app/schemas/request_response.py
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    features: List[List[float]]  # 2D array: [ [f1, f2, ...], [f1, f2, ...], ... ]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
