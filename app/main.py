# app/main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
from app.core.config import APP_NAME, APP_VERSION
from app.core.logging import setup_logging
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import sys

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    packaged_model = Path("models/packaged/model.pkl")
    if not packaged_model.exists():
        logger.warning(
            "No packaged model found. Please run scripts/bootstrap.py "
            "API will start but prediction endpoints will be unavailable."
        )

    yield

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(api_router, prefix="/api")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
