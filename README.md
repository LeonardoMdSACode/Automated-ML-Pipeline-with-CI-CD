---
title: Automated ML Pipeline with CI/CD
emoji: 🤖
colorFrom: gray
colorTo: red
sdk: docker
app_file: Dockerfile
pinned: false
license: mit
---

# Under Construction

# Notes

# Repo Structure (Present)

```text
Automated ML Pipeline with CI-CD/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml          # CI: train → evaluate → gate → package
│
├── app/
│   ├── main.py                      # FastAPI entrypoint (HF Spaces)
│   ├── api/
│   │   └── routes.py                # /predict, /health
│   ├── inference/
│   │   └── predictor.py             # Loads latest approved model
│   ├── schemas/
│   │   └── request_response.py
│   └── core/
│       ├── config.py                # Paths, env flags
│       └── logging.py
│
├── data/
│   ├── raw/                          # DVC-tracked
│   ├── processed/                   # DVC-tracked
│   └── reference/                   # Baseline dataset (for regression tests)
│
├── models/
│   ├── registry/
│   │   ├── model_v001/
│   │   │   ├── model.pkl
│   │   │   └── metadata.json        # metrics, git_sha, data_hash
│   │   ├── model_v002/
│   │   └── latest -> model_v002     # symlink or pointer file
│   └── baseline/
│       └── metrics.json             # Last approved metrics
│
├── reports/
│   ├── evaluation.json               # CI output
│   └── comparison.json               # baseline vs candidate
│
├── scripts/
│   ├── train.py                      # Deterministic training
│   ├── evaluate.py                   # Metrics computation
│   ├── compare.py                    # Quality gate (FAILS CI)
│   └── package_model.py              # Registry promotion
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_training_reproducible.py
│   └── test_metrics_thresholds.py
│
├── dvc.yaml                          # Pipeline stages (train/eval)
├── .dvc/
├── .dvcignore
│
├── Makefile
│   ├── train
│   ├── evaluate
│   ├── gate
│   ├── package
│   └── serve
│
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
│
├── Dockerfile                        # HF Spaces compatible
├── README.md
└── LICENSE
```
