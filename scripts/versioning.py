#! python3
# scripts/versioning.py
# Minimal model versioning for pipeline with optional registry path

import json
from pathlib import Path
import shutil

def get_next_model_version(registry: Path = Path("models/registry")) -> str:
    """
    Computes the next model version based on latest.json
    """
    latest_json = registry / "latest.json"
    if not latest_json.exists():
        return "model_v001"
    
    with open(latest_json) as f:
        latest_data = json.load(f)
    
    latest_version = latest_data.get("latest_version", "model_v000")
    num = int(latest_version.split("_v")[-1]) + 1
    return f"model_v{num:03d}"


def promote_model(model_path: Path, registry: Path = Path("models/registry")) -> str:
    """
    Promotes a model to the registry, updates latest.json
    
    Args:
        model_path: path to trained model (model.pkl)
        registry: registry directory (default: models/registry)
    
    Returns:
        new_version: str
    """
    latest_json = registry / "latest.json"
    new_version = get_next_model_version(registry)
    target_dir = registry / new_version
    target_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(model_path, target_dir / "model.pkl")
    
    # update latest.json
    with open(latest_json, "w") as f:
        json.dump({"latest_version": new_version}, f, indent=2)
    
    return new_version
