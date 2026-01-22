#! python3
# scripts/versioning.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import shutil

def get_next_model_version(registry: Path) -> str:
    latest_json = registry / "latest.json"

    if not latest_json.exists():
        return "model_v001"

    with open(latest_json) as f:
        latest_version = json.load(f).get("latest_version", "model_v000")

    num = int(latest_version.split("_v")[-1]) + 1
    return f"model_v{num:03d}"

def promote_model(model_path: Path, registry: Path) -> str:
    registry.mkdir(parents=True, exist_ok=True)

    new_version = get_next_model_version(registry)
    target_dir = registry / new_version
    target_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_path, target_dir / "model.pkl")

    with open(registry / "latest.json", "w") as f:
        json.dump({"latest_version": new_version}, f, indent=2)

    return new_version
