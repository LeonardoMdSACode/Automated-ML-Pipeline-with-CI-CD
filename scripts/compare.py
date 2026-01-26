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

# -------------------------------
# Load all candidate evaluations
# -------------------------------
eval_files = list(EVAL_DIR.glob("*.json"))
if not eval_files:
    raise FileNotFoundError("No evaluation files found in reports/evaluations/")

evaluations = []
for f in eval_files:
    with open(f) as j:
        evaluations.append(json.load(j))

# Candidate = latest model version
candidate = max(evaluations, key=lambda x: x["model_version"])

CI_MODE = os.getenv("CI", "false").lower() == "true"

# -------------------------------
# Load baseline if exists
# -------------------------------
if BASELINE_METRICS.exists():
    with open(BASELINE_METRICS) as f:
        baseline = json.load(f)
else:
    baseline = None

# -------------------------------
# FIRST RUN: bootstrap baseline
# -------------------------------
if baseline is None:
    comparison = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "candidate_model": candidate["model_version"],
        "baseline_model": None,
        "metrics": {},
        "passed_gate": True,
        "decision": "bootstrap",
    }

    BASELINE_METRICS.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Baseline is only set here during bootstrap
    with open(BASELINE_METRICS, "w") as f:
        json.dump(candidate, f, indent=2)

    with open(COMPARISON_FILE, "w") as f:
        json.dump(comparison, f, indent=2)

    print("NO BASELINE FOUND — bootstrap gate passed")
    sys.exit(0)

# -------------------------------
# Prevent self-comparison if not CI bootstrap
# -------------------------------
if candidate["model_version"] == baseline["model_version"] and not CI_MODE:
    raise RuntimeError(
        f"INVALID STATE: candidate model ({candidate['model_version']}) "
        f"is identical to pre-existing baseline model"
    )

print(f"Comparing candidate ({candidate['model_version']}) vs baseline ({baseline['model_version']})")

# -------------------------------
# Quality gate
# -------------------------------
def gate_passed(c, b):
    return c["rmse"] <= b["rmse"] and c["r2"] >= b["r2"]

passed = gate_passed(candidate, baseline)

# -------------------------------
# Write comparison artifact
# -------------------------------
COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)

comparison = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "candidate_model": candidate["model_version"],
    "baseline_model": baseline["model_version"],
    "metrics": {
        "rmse": {
            "candidate": candidate["rmse"],
            "baseline": baseline["rmse"],
            "delta": candidate["rmse"] - baseline["rmse"],
        },
        "r2": {
            "candidate": candidate["r2"],
            "baseline": baseline["r2"],
            "delta": candidate["r2"] - baseline["r2"],
        },
    },
    "passed_gate": passed,
    "decision": "promote" if passed else "reject",
}

with open(COMPARISON_FILE, "w") as f:
    json.dump(comparison, f, indent=2)

# -------------------------------
# Update baseline only if candidate passes gate
# -------------------------------
if passed:
    with open(BASELINE_METRICS, "w") as f:
        json.dump(candidate, f, indent=2)
    print("QUALITY GATE PASSED — baseline updated")
else:
    print("QUALITY GATE FAILED — baseline unchanged")

# -------------------------------
# Exit semantics
# -------------------------------
sys.exit(0 if passed or CI_MODE else 1)
