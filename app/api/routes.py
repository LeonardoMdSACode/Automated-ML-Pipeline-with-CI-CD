# app/api/routes.py
from fastapi import APIRouter
from app.schemas.request_response import PredictionRequest, PredictionResponse
from app.inference.predictor import Predictor
import pandas as pd

router = APIRouter()
predictor = Predictor()

@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    X_df = pd.DataFrame(request.features)
    preds = predictor.predict(X_df)
    return PredictionResponse(predictions=preds, model_version=predictor.latest_model_info["model_version"])
