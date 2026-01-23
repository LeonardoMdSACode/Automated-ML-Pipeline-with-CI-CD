# app/main.py
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config import APP_NAME, APP_VERSION
from app.core.logging import setup_logging

logger = setup_logging()

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.include_router(api_router, prefix="/api")

@app.on_event("startup")
def startup():
    logger.info("API startup complete")

@app.get("/")
def root():
    return {"status": "ok"}
