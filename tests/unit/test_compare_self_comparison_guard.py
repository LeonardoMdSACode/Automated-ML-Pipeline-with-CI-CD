# tests/unit/test_compare_self_comparison_guard.py

import json
import subprocess
import sys
from pathlib import Path
import pytest
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EVAL_DIR = PROJECT_ROOT / "reports" / "evaluations"
BASELINE_DIR = PROJECT_ROOT / "models" / "baseline"


@pytest.mark.unit
def test_compare_fails_when_candidate_equals_baseline(tmp_path):
    """
    If baseline exists and only one evaluation exists with same model_version,
    compare.py must fail hard (RuntimeError).
    """

    # --- Setup isolated dirs ---
    eval_dir = tmp_path / "reports" / "evaluations"
    baseline_dir = tmp_path / "models" / "baseline"
    eval_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    # --- Fake model metrics ---
    metrics = {
        "model_version": "model_v007",
        "rmse": 100.0,
        "r2": 0.9,
        "mae": 50.0,
    }

    # Write evaluation
    with open(eval_dir / "model_v007.json", "w") as f:
        json.dump(metrics, f)

    # Write baseline (same model)
    with open(baseline_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # --- Run compare.py as subprocess ---
    env = {
        **dict(os.environ),
        "PYTHONPATH": str(PROJECT_ROOT),
        "CI": "false",  # force non-CI mode
    }

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "compare.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    # --- Assertion ---
    assert result.returncode != 0
    assert "INVALID STATE" in result.stderr or "INVALID STATE" in result.stdout
