#! python3
# app/main.py
from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(title="Automated ML Pipeline Inference API", version="1.0")

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Automated ML Pipeline Inference API is running."}

