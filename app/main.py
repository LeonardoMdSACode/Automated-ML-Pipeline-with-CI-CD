# app/main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
from app.core.config import APP_NAME, APP_VERSION
from app.core.logging import setup_logging
from contextlib import asynccontextmanager

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
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
