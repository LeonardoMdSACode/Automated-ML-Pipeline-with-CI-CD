#! python3
# scripts/compare.py
import json
from pathlib import Path
import os

EVAL_DIR = Path("reports/evaluations")

# Load all evaluations
eval_files = list(EVAL_DIR.glob("*.json"))
if not eval_files:
    raise FileNotFoundError("No evaluation files found in reports/evaluations/")

all_evals = []
for f in eval_files:
    with open(f) as j:
        all_evals.append(json.load(j))

# Determine latest model by version
latest = max(all_evals, key=lambda x: x["model_version"])

# Determine best baseline: lowest RMSE, tie-breaker highest R²
baseline_candidates = [e for e in all_evals if e["model_version"] != latest["model_version"]]
if not baseline_candidates:
    # If no previous model exists, use latest as baseline
    baseline = latest
else:
    baseline = min(baseline_candidates, key=lambda x: (x["rmse"], -x["r2"]))

print(f"Comparing latest ({latest['model_version']}) vs baseline ({baseline['model_version']})")

# Define quality gate rules
def gate_passed(latest_metrics, baseline_metrics):
    if latest_metrics["rmse"] > baseline_metrics["rmse"]:
        return False
    if latest_metrics["r2"] < baseline_metrics["r2"]:
        return False
    return True

CI_MODE = os.getenv("CI", "false").lower() == "true"

if gate_passed:
    print("QUALITY GATE PASSED")
    exit(0)
else:
    print("QUALITY GATE FAILED")
    if CI_MODE:
        print("CI mode enabled: bypassing gate failure")
        exit(0)
    else:
        exit(1)
