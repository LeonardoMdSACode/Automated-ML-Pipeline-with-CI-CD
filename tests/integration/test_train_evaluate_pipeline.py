'''
Runs train.py → evaluate.py

Produces evaluation.json

Metrics are non-null and numeric'''
# tests\integration\test_train_evaluate_pipeline.py

import pytest
import subprocess
import sys
import shutil
from pathlib import Path
import json
import os

@pytest.mark.integration
def test_train_evaluate_pipeline(tmp_path):
    """
    Full pipeline: train -> evaluate -> gate -> promotion.
    """

    python_exe = sys.executable

    # --- Arrange workspace ---
    workdir = tmp_path / "run"
    shutil.copytree(".", workdir, dirs_exist_ok=True)

    env = {**dict(**os.environ), "PYTHONHASHSEED": "0"}

    # --- Act: train ---
    train = subprocess.run(
        [python_exe, "scripts/train.py"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True
    )
    assert train.returncode == 0, train.stderr

    # --- Act: evaluate ---
    evaluate = subprocess.run(
        [python_exe, "scripts/evaluate.py"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True
    )
    assert evaluate.returncode == 0, evaluate.stderr

    # --- Assert: evaluation metrics exist ---
    eval_dir = workdir / "reports" / "evaluations"
    eval_files = list(eval_dir.glob("*.json"))
    assert len(eval_files) > 0
    with open(eval_files[-1]) as f:
        metrics = json.load(f)
    assert "rmse" in metrics and "mae" in metrics and "r2" in metrics

    # --- Act: promote model (simulate gate passing) ---
    from scripts.versioning import promote_model
    latest_model_file = workdir / "models" / "registry" / metrics["model_version"] / "model.pkl"
    registry_dir = workdir / "models" / "registry"
    promoted_version = promote_model(latest_model_file, registry_dir)
    assert (registry_dir / promoted_version / "model.pkl").exists()

    # latest.json updated
    latest_json = registry_dir / "latest.json"
    with open(latest_json) as f:
        latest = json.load(f)
    assert latest["latest_version"] == promoted_version
