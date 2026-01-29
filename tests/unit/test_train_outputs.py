# tests/unit/test_train_outputs.py

import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.unit
def test_train_output_file_exists(tmp_path, monkeypatch):
    """
    Training must create models/registry/latest.json
    in a clean environment.
    """

    # Isolate filesystem
    monkeypatch.chdir(tmp_path)

    python_exe = sys.executable
    train_script = PROJECT_ROOT / "scripts" / "train.py"

    result = subprocess.run(
        [python_exe, str(train_script)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    latest = tmp_path / "models" / "registry" / "latest.json"
    assert latest.exists(), "latest.json must be created by train.py"
