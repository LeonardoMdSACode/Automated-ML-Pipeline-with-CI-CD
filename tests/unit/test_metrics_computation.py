'''Metric values are within valid range

Handles edge cases (single class, empty preds)'''
# tests/unit/test_metrics_computation.py

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@pytest.mark.unit
def test_metric_computation_known_values():
    """
    Given known predictions and targets, computed metrics should match expected.
    """
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])  # perfect prediction

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    assert rmse == 0.0
    assert mae == 0.0
    assert r2 == 1.0

@pytest.mark.unit
def test_metric_computation_non_perfect():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 1])

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Precomputed expected values
    assert round(rmse, 5) == 0.81650
    assert round(mae, 5) == 0.66667
    assert round(r2, 5) == 0.0
