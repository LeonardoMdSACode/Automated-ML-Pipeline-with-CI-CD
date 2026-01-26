# app/core/config.py
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

ENV = os.getenv("APP_ENV", "local")

# Registry / model pointers
MODELS_DIR = ROOT / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
LATEST_JSON = REGISTRY_DIR / "latest.json"
PACKAGED_JSON = ROOT / "models" / "packaged" / "packaged.json"

# API settings
APP_NAME = "Automated ML Pipeline Inference API"
APP_VERSION = "1.0.0"

# Runtime flags
DEBUG = ENV != "production"
