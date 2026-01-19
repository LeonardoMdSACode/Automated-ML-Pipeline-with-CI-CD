#! python3
# scripts\compare.py
# Quality gate (FAILS CI)

import json
from pathlib import Path
import sys

# Paths
EVAL_PATH = Path("reports/evaluation.json")
BASELINE_PATH = Path("models/baseline/metrics.json")
GATE_PATH = Path("reports/comparison.json")
GATE_PATH.parent.mkdir(exist_ok=True, parents=True)

# Load metrics
with open(EVAL_PATH) as f:
    current = json.load(f)

with open(BASELINE_PATH) as f:
    baseline = json.load(f)

# Quality gate: fail if RMSE increases
rmse_diff = current["rmse"] - baseline["rmse"]
passed = current["rmse"] <= baseline["rmse"]

# Save comparison
comparison = {
    "current_rmse": current["rmse"],
    "baseline_rmse": baseline["rmse"],
    "rmse_diff": rmse_diff,
    "passed_gate": passed,
    "model_version": current["model_version"]
}

with open(GATE_PATH, "w") as f:
    json.dump(comparison, f)

if not passed:
    print("QUALITY GATE FAILED: RMSE increased")
    sys.exit(1)

print("QUALITY GATE PASSED")
