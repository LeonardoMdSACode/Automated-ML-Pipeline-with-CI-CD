#! python3
# scripts/metric_gate.py
# Minimal metric gate logic for regression metrics

from typing import Dict

def check_gate(metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    """
    Checks if all metrics pass the defined thresholds.
    
    Args:
        metrics: dict with metric values, e.g. {"rmse": 150000, "r2": 0.83}
        thresholds: dict with threshold values, e.g. {"rmse": 160000, "r2": 0.8}
    
    Returns:
        True if all thresholds are satisfied, False otherwise
    """
    for key, threshold in thresholds.items():
        if key not in metrics:
            raise KeyError(f"Metric {key} missing from evaluation metrics")
        
        # Gate rule: RMSE must be <= threshold, R2 must be >= threshold
        if key.lower() == "r2":
            if metrics[key] < threshold:
                return False
        else:
            if metrics[key] > threshold:
                return False
    return True
