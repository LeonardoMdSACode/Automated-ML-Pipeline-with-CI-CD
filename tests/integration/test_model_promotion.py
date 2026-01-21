'''Approved model copied to registry

latest pointer updated correctly

Metadata matches evaluation output'''
# tests\integration\test_model_promotion.py

import pytest
import shutil
import json
from pathlib import Path

from scripts.versioning import get_next_model_version, promote_model

@pytest.mark.integration
def test_model_promotion_creates_new_version(tmp_path):
    """
    Test that model promotion creates a new version folder and updates latest.json.
    """
    registry = tmp_path / "registry"
    registry.mkdir()

    model_path = tmp_path / "dummy_model.pkl"
    model_path.write_text("fake_model_content")

    # Promote first model
    promoted_version = promote_model(model_path, registry)
    assert promoted_version == "model_v001"
    assert (registry / promoted_version / "model.pkl").exists()

    # latest.json updated
    latest = registry / "latest.json"
    with open(latest) as f:
        latest_data = json.load(f)
    assert latest_data["latest_version"] == promoted_version

    # Promote second model
    model_path2 = tmp_path / "dummy_model2.pkl"
    model_path2.write_text("fake_model_content2")
    promoted_version2 = promote_model(model_path2, registry)
    assert promoted_version2 == "model_v002"
