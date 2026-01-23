# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.core.config import APP_NAME, APP_VERSION
from app.core.logging import setup_logging

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api")

templates = Jinja2Templates(directory="app/templates")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
