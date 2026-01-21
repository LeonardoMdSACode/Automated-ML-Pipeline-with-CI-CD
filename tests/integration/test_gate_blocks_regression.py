'''Inject worse metrics

compare.py exits non-zero'''
# tests\integration\test_gate_blocks_regression.py

import pytest
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scripts.metric_gate import check_gate

@pytest.mark.integration
def test_gate_blocks_low_quality(tmp_path):
    """
    Low-quality model metrics should fail the gate.
    """
    rmse, mae, r2 = 10.0, 5.0, 0.1  # Poor metrics
    thresholds = {"rmse": 5.0, "mae": 3.0, "r2": 0.5}

    result = check_gate({"rmse": rmse, "mae": mae, "r2": r2}, thresholds)
    assert result is False

@pytest.mark.integration
def test_gate_passes_high_quality(tmp_path):
    """
    High-quality model metrics should pass the gate.
    """
    rmse, mae, r2 = 1.0, 0.5, 0.95  # Good metrics
    thresholds = {"rmse": 5.0, "mae": 3.0, "r2": 0.5}

    result = check_gate({"rmse": rmse, "mae": mae, "r2": r2}, thresholds)
    assert result is True
