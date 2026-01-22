#! python3
# scripts/compare.py

import sys
import json
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import EVAL_DIR, BASELINE_METRICS, COMPARISON_FILE

# --- Load evaluations ---
eval_files = list(EVAL_DIR.glob("*.json"))
if not eval_files:
    raise FileNotFoundError("No evaluation files found in reports/evaluations/")

all_evals = []
for f in eval_files:
    with open(f) as j:
        all_evals.append(json.load(j))

# Latest by version
latest = max(all_evals, key=lambda x: x["model_version"])

# Baseline logic
if BASELINE_METRICS.exists():
    with open(BASELINE_METRICS) as f:
        baseline = json.load(f)
else:
    baseline_candidates = [
        e for e in all_evals if e["model_version"] != latest["model_version"]
    ]
    baseline = latest if not baseline_candidates else min(
        baseline_candidates, key=lambda x: (x["rmse"], -x["r2"])
    )

print(f"Comparing latest ({latest['model_version']}) vs baseline ({baseline['model_version']})")

# --- Quality gate ---
def gate_passed(latest_metrics, baseline_metrics):
    if latest_metrics["rmse"] > baseline_metrics["rmse"]:
        return False
    if latest_metrics["r2"] < baseline_metrics["r2"]:
        return False
    return True

passed = gate_passed(latest, baseline)

# --- Write comparison artifact (ALWAYS) ---
COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

comparison = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "candidate_model": latest["model_version"],
    "baseline_model": baseline["model_version"],
    "metrics": {
        "rmse": {
            "candidate": latest["rmse"],
            "baseline": baseline["rmse"],
            "delta": latest["rmse"] - baseline["rmse"],
        },
        "r2": {
            "candidate": latest["r2"],
            "baseline": baseline["r2"],
            "delta": latest["r2"] - baseline["r2"],
        },
    },
    "passed_gate": passed,
    "decision": "promote" if passed else "reject",
}

with open(COMPARISON_FILE, "w") as f:
    json.dump(comparison, f, indent=2)

# --- Exit semantics ---
CI_MODE = os.getenv("CI", "false").lower() == "true"

if passed:
    print("QUALITY GATE PASSED")
    sys.exit(0)
else:
    print("QUALITY GATE FAILED")
    sys.exit(0 if CI_MODE else 1)
