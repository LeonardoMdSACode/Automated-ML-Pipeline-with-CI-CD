# tests\unit\test_evaluate_deterministic.py

import subprocess
import sys
import json
import shutil
import hashlib
from pathlib import Path
import os
import pytest
import math


def dict_hash(d: dict) -> str:
    """Stable hash for a dict (order-independent)."""
    payload = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.unit
def test_evaluation_is_deterministic(tmp_path):
    """
    Running evaluation twice on the same model and data
    must produce identical metrics.
    """

    python_exe = sys.executable

    # --- Arrange isolated workspace ---
    workdir = tmp_path / "run"
    shutil.copytree(".", workdir, dirs_exist_ok=True)

    env = {
        **os.environ,
        "PYTHONHASHSEED": "0"
    }

    # Ensure one trained model exists
    train = subprocess.run(
        [python_exe, "scripts/train.py"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True
    )
    assert train.returncode == 0, train.stderr

    # --- Act: run evaluation twice ---
    e1 = subprocess.run(
        [python_exe, "scripts/evaluate.py"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True
    )
    assert e1.returncode == 0, e1.stderr

    e2 = subprocess.run(
        [python_exe, "scripts/evaluate.py"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True
    )
    assert e2.returncode == 0, e2.stderr

    eval_dir = workdir / "reports" / "evaluations"
    eval_files = sorted(eval_dir.glob("*.json"))
    assert len(eval_files) >= 2, "Expected at least two evaluation files"

    with open(eval_files[-2]) as f:
        m1 = json.load(f)

    with open(eval_files[-1]) as f:
        m2 = json.load(f)



    # --- Assert: metrics content only with tolerance ---
    keys_numeric = ["rmse", "mae", "r2"]
    for k in keys_numeric:
        assert math.isclose(m1[k], m2[k], rel_tol=1e-12, abs_tol=0), f"Metric {k} differs: {m1[k]} != {m2[k]}"
    
    # Exact match for version
    assert m1["model_version"] == m2["model_version"]
    