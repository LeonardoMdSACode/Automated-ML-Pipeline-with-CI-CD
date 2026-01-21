# tests/unit/test_compare_gate_logic.py
import pytest


def gate_passed(latest, baseline):
    if latest["rmse"] > baseline["rmse"]:
        return False
    if latest["r2"] < baseline["r2"]:
        return False
    return True


@pytest.mark.unit
def test_gate_passes_when_model_improves_both_metrics():
    baseline = {"rmse": 200.0, "r2": 0.70}
    latest = {"rmse": 180.0, "r2": 0.75}

    assert gate_passed(latest, baseline) is True


@pytest.mark.unit
def test_gate_fails_when_rmse_is_worse():
    baseline = {"rmse": 150.0, "r2": 0.80}
    latest = {"rmse": 160.0, "r2": 0.85}

    assert gate_passed(latest, baseline) is False


@pytest.mark.unit
def test_gate_fails_when_r2_is_worse():
    baseline = {"rmse": 150.0, "r2": 0.80}
    latest = {"rmse": 140.0, "r2": 0.78}

    assert gate_passed(latest, baseline) is False


@pytest.mark.unit
def test_gate_passes_when_equal_metrics():
    baseline = {"rmse": 150.0, "r2": 0.80}
    latest = {"rmse": 150.0, "r2": 0.80}

    assert gate_passed(latest, baseline) is True


@pytest.mark.unit
def test_gate_fails_when_both_metrics_worse():
    baseline = {"rmse": 150.0, "r2": 0.80}
    latest = {"rmse": 170.0, "r2": 0.60}

    assert gate_passed(latest, baseline) is False
