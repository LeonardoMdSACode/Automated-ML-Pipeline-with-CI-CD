# tests/integration/test_api_predict.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.mark.integration
def test_predict_endpoint_happy_path():
    client = TestClient(app)

    payload = {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "yr_built": 1995,
        "yr_renovated": 0,
        "zipcode": 98178,
        "lat": 47.51,
        "long": -122.25,
        "sqft_living15": 1800,
        "sqft_lot15": 5000,
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "prediction" in data
    assert "model_version" in data
