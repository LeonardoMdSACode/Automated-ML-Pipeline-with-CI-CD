'''Passes when metrics ≥ baseline

Fails when metrics regress

Threshold logic (absolute / relative delta)'''
# tests/unit/test_metric_gate.py

import pytest

@pytest.mark.unit
def test_quality_gate_pass():
    """
    Gate passes if current metrics better than baseline.
    """
    baseline = {"rmse": 1.0, "r2": 0.8}
    current = {"rmse": 0.5, "r2": 0.9}

    # Gate logic
    passed = current["rmse"] <= baseline["rmse"] and current["r2"] >= baseline["r2"]
    assert passed

@pytest.mark.unit
def test_quality_gate_fail():
    """
    Gate fails if current metrics worse than baseline.
    """
    baseline = {"rmse": 0.5, "r2": 0.9}
    current = {"rmse": 1.0, "r2": 0.85}

    passed = current["rmse"] <= baseline["rmse"] and current["r2"] >= baseline["r2"]
    assert not passed

@pytest.mark.unit
def test_quality_gate_edge_equal():
    """
    Gate passes if metrics exactly match baseline.
    """
    baseline = {"rmse": 1.0, "r2": 0.8}
    current = {"rmse": 1.0, "r2": 0.8}

    passed = current["rmse"] <= baseline["rmse"] and current["r2"] >= baseline["r2"]
    assert passed
