'''Model object type is correct

Model exposes .predict'''
# tests/unit/test_train_outputs.py

import pytest
from pathlib import Path

@pytest.mark.unit
def test_train_output_file_exists():
    model_path = Path("models/registry/latest.json")
    assert model_path.exists(), "Latest model JSON must exist"
