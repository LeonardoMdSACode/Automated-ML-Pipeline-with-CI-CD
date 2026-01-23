# app/api/routes.py
from fastapi import APIRouter

from app.api.schemas import PredictionRequest, PredictionResponse
from app.inference.predictor import Predictor

router = APIRouter()
predictor = Predictor()


@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    y = predictor.predict(request.features)
    return PredictionResponse(
        prediction=y,
        model_version=predictor.model_version
    )
