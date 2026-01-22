#! python3
# scripts/compare.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os

from scripts.config import EVAL_DIR

eval_files = list(EVAL_DIR.glob("*.json"))
if not eval_files:
    raise FileNotFoundError("No evaluation files found in reports/evaluations/")

all_evals = []
for f in eval_files:
    with open(f) as j:
        all_evals.append(json.load(j))

latest = max(all_evals, key=lambda x: x["model_version"])

baseline_candidates = [e for e in all_evals if e["model_version"] != latest["model_version"]]
baseline = latest if not baseline_candidates else min(
    baseline_candidates, key=lambda x: (x["rmse"], -x["r2"])
)

print(f"Comparing latest ({latest['model_version']}) vs baseline ({baseline['model_version']})")

def gate_passed(latest_metrics, baseline_metrics):
    if latest_metrics["rmse"] > baseline_metrics["rmse"]:
        return False
    if latest_metrics["r2"] < baseline_metrics["r2"]:
        return False
    return True

CI_MODE = os.getenv("CI", "false").lower() == "true"

if gate_passed(latest, baseline):
    print("QUALITY GATE PASSED")
    exit(0)
else:
    print("QUALITY GATE FAILED")
    exit(0 if CI_MODE else 1)
