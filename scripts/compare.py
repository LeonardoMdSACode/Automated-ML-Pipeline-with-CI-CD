#! python3
# scripts/compare.py
# Compare latest evaluation to baseline

import json
from pathlib import Path

BASELINE_METRICS = Path("models/baseline/metrics.json")
EVAL_DIR = Path("reports/evaluations")

# Load baseline metrics
with open(BASELINE_METRICS) as f:
    baseline = json.load(f)

# Load latest evaluation (by version)
latest_version = max(EVAL_DIR.glob("*.json"), key=lambda x: x.stem)
with open(latest_version) as f:
    latest = json.load(f)

print(f"Comparing latest ({latest['model_version']}) vs baseline ({baseline['model_version']})")

# Define quality gate rules
def gate_passed(latest_metrics, baseline_metrics):
    # Fail if RMSE increased or R² decreased
    if latest_metrics["rmse"] > baseline_metrics["rmse"]:
        return False
    if latest_metrics["r2"] < baseline_metrics["r2"]:
        return False
    return True

if gate_passed(latest, baseline):
    print("QUALITY GATE PASSED")
else:
    print("QUALITY GATE FAILED")
    exit(1)
