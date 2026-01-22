#! python3
# scripts/versioning.py

import json
import shutil
from pathlib import Path

from scripts.config import REGISTRY, LATEST_JSON

def get_next_model_version(registry: Path = REGISTRY) -> str:
    if not LATEST_JSON.exists():
        return "model_v001"

    with open(LATEST_JSON) as f:
        latest_version = json.load(f).get("latest_version", "model_v000")

    num = int(latest_version.split("_v")[-1]) + 1
    return f"model_v{num:03d}"

def promote_model(model_path: Path, registry: Path = REGISTRY) -> str:
    new_version = get_next_model_version(registry)
    target_dir = registry / new_version
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_path, target_dir / "model.pkl")

    with open(LATEST_JSON, "w") as f:
        json.dump({"latest_version": new_version}, f, indent=2)

    return new_version
