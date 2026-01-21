'''Fixed seed → identical model hash / coefficients'''
# tests\unit\test_train_deterministic.py

import subprocess
import sys
import hashlib
from pathlib import Path
import shutil
import json
import os
import pytest


def file_hash(path: Path) -> str:
    """Return SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.unit
def test_training_is_deterministic(tmp_path):
    """
    Running training twice with the same seed and data
    must produce identical model artifacts.
    """

    python_exe = sys.executable

    # --- Arrange isolated workspace ---
    workdir = tmp_path / "run"
    shutil.copytree(".", workdir, dirs_exist_ok=True)

    model_dir = workdir / "models" / "registry"
    model_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **dict(**os.environ),
        "PYTHONHASHSEED": "0"
    }

    # --- Act: run training twice ---
    r1 = subprocess.run(
        [python_exe, "scripts/train.py"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True
    )
    assert r1.returncode == 0, r1.stderr

    r2 = subprocess.run(
        [python_exe, "scripts/train.py"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True
    )
    assert r2.returncode == 0, r2.stderr

    # --- Assert: compare last two model artifacts ---
    model_versions = sorted(model_dir.iterdir())
    assert len(model_versions) >= 2, "Expected at least two trained models"

    m1 = model_versions[-2] / "model.pkl"
    m2 = model_versions[-1] / "model.pkl"

    assert m1.exists()
    assert m2.exists()

    assert file_hash(m1) == file_hash(m2), "Model artifacts differ — training is not deterministic"
