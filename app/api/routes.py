# app/api/routes.py
from fastapi import APIRouter
from app.schemas.request_response import PredictionRequest, PredictionResponse
from app.inference.predictor import Predictor

router = APIRouter()
predictor = Predictor()

@router.post("/predict")
def predict(payload: dict):
    return {"prediction": predictor.predict(payload)}

@router.get("/health")
def health():
    model_exists = (Path("models/packaged/model.pkl")).exists()
    return {
        "model_loaded": model_exists
    }
