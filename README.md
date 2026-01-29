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

Raw dataset: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

py -3.10 -m venv .venv

.\\.venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements-dev.txt

python scripts/bootstrap.py

uvicorn app.main:app --reload

# Repo Structure (Initial)

```text
Automated-ML-Pipeline-with-CI-CD/
├── Dockerfile
├── Makefile
├── README.md
├── LICENSE
├── dvc.yaml
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── inference/
│   │   └── predictor.py
│   └── schemas/
│       └── request_response.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
├── models/
│   ├── baseline/
│   │   └── metrics.json
│   └── registry/
│       ├── model_v001/
│       └── model_v002/
├── reports/
│   ├── evaluation.json
│   └── comparison.json
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── compare.py
│   └── package_model.py
└── tests/
    ├── unit/
    │   ├── test_data_schema.py
    │   ├── test_feature_extraction.py
    │   ├── test_metric_gate.py
    │   ├── test_metrics_computation.py
    │   ├── test_registry_metadata.py
    │   ├── test_train_deterministic.py
    │   ├── test_train_outputs.py
    │   └── test_version_increment.py
    └── integration/
        ├── test_ci_like_flow.py
        ├── test_gate_blocks_regression.py
        ├── test_model_promotion.py
        └── test_train_evaluate_pipeline.py
```
