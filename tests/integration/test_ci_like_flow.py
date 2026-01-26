# tests/integration/test_ci_like_flow.py

import pytest
import subprocess
from pathlib import Path
import json
import sys
import os
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def stage_data(tmp_path):
    """Ensure minimal required raw data exists in temp test folder."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    src_file = PROJECT_ROOT / "data" / "raw" / "kc_house_data.csv"
    dst_file = data_dir / "kc_house_data.csv"
    if not src_file.exists():
        raise FileNotFoundError(f"Test cannot run, missing {src_file}")
    shutil.copy(src_file, dst_file)

@pytest.mark.integration
def test_full_pipeline_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Stage raw data for training
    stage_data(tmp_path)

    python_exe = sys.executable

    train = [python_exe, str(PROJECT_ROOT / "scripts" / "train.py")]
    evaluate = [python_exe, str(PROJECT_ROOT / "scripts" / "evaluate.py")]
    compare = [python_exe, str(PROJECT_ROOT / "scripts" / "compare.py")]
    package = [python_exe, str(PROJECT_ROOT / "scripts" / "package_model.py")]

    # -------------------------------
    # First run: bootstrap baseline
    # -------------------------------
    assert subprocess.run(train, capture_output=True, text=True).returncode == 0
    assert subprocess.run(evaluate, capture_output=True, text=True).returncode == 0

    gate1 = subprocess.run(
        compare,
        capture_output=True,
        text=True,
        env={**os.environ, "CI": "true"},
    )
    assert gate1.returncode == 0, gate1.stderr

    # comparison.json MUST exist after first gate
    comparison = tmp_path / "reports" / "comparison.json"
    assert comparison.exists(), "comparison.json not created during bootstrap"

    # -------------------------------
    # Second run: real comparison
    # -------------------------------
    assert subprocess.run(train, capture_output=True, text=True).returncode == 0
    assert subprocess.run(evaluate, capture_output=True, text=True).returncode == 0

    gate2 = subprocess.run(
        compare,
        capture_output=True,
        text=True,
        env={**os.environ, "CI": "true"},
    )
    assert gate2.returncode == 0, gate2.stderr

    # -------------------------------
    # Package & verify registry
    # -------------------------------
    assert subprocess.run(package, capture_output=True, text=True).returncode == 0

    latest = tmp_path / "models" / "registry" / "latest.json"
    assert latest.exists()

    with open(latest) as f:
        version = json.load(f)["latest_version"]

    model_file = tmp_path / "models" / "registry" / version / "model.pkl"
    assert model_file.exists()
