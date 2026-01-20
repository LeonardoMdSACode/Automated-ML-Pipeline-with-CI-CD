'''train

evaluate

gate

package

Verifies final registry state'''

# tests/integration/test_pipeline_flow.py
import pytest
import subprocess
from pathlib import Path
import json
import sys

@pytest.mark.integration
def test_full_pipeline_run():
    """
    Minimal integration test: runs train -> evaluate -> compare
    and checks that the latest model exists and passes quality gate.
    """
    # Use current venv python
    python_exe = sys.executable

    # Run training
    train_result = subprocess.run([python_exe, "scripts/train.py"], capture_output=True, text=True)
    assert train_result.returncode == 0, f"Training failed: {train_result.stderr}"

    # Run evaluation
    eval_result = subprocess.run([python_exe, "scripts/evaluate.py"], capture_output=True, text=True)
    assert eval_result.returncode == 0, f"Evaluation failed: {eval_result.stderr}"

    # Run quality gate
    compare_result = subprocess.run([python_exe, "scripts/compare.py"], capture_output=True, text=True)
    assert compare_result.returncode == 0, f"Quality gate failed: {compare_result.stderr}"

    # Check latest model exists
    latest_json = Path("models/registry/latest.json")
    assert latest_json.exists(), "Latest model JSON not found"

    with open(latest_json) as f:
        latest_version = json.load(f)["latest_version"]

    model_path = Path(f"models/registry/{latest_version}/model.pkl")
    assert model_path.exists(), f"Trained model file not found: {model_path}"
