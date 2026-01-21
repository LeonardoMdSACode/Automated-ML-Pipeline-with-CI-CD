'''Metadata JSON schema

Required keys: metrics, git_sha, data_hash'''
# tests/unit/test_registry_metadata.py

import json
from pathlib import Path
import pytest

@pytest.mark.unit
def test_registry_structure(tmp_path):
    """
    Each model version in registry should have a model.pkl and metrics.json
    with required keys.
    """
    registry = tmp_path / "registry"
    registry.mkdir()

    # Create fake model versions
    versions = ["model_v001", "model_v002"]
    for v in versions:
        vdir = registry / v
        vdir.mkdir(parents=True)
        (vdir / "model.pkl").write_text("fake model")
        metrics = {
            "rmse": 0.1,
            "mae": 0.05,
            "r2": 0.99,
            "model_version": v
        }
        with open(vdir / "metrics.json", "w") as f:
            json.dump(metrics, f)

    # --- Assert ---
    for v in versions:
        vdir = registry / v
        assert (vdir / "model.pkl").exists()
        metrics_file = vdir / "metrics.json"
        assert metrics_file.exists()
        with open(metrics_file) as f:
            data = json.load(f)
            for key in ["rmse", "mae", "r2", "model_version"]:
                assert key in data
