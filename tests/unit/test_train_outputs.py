# tests/unit/test_train_outputs.py

import subprocess
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.unit
def test_train_output_file_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # --- Create minimal raw dataset ---
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    df = pd.DataFrame({
        "bedrooms": [3, 2, 4],
        "bathrooms": [2, 1, 3],
        "sqft_living": [1800, 1200, 2500],
        "sqft_lot": [5000, 4000, 6000],
        "floors": [1, 1, 2],
        "waterfront": [0, 0, 0],
        "view": [0, 0, 0],
        "condition": [3, 3, 4],
        "grade": [7, 6, 8],
        "sqft_above": [1800, 1200, 2500],
        "sqft_basement": [0, 0, 0],
        "yr_built": [1995, 1980, 2005],
        "yr_renovated": [0, 0, 0],
        "zipcode": [98178, 98178, 98178],
        "lat": [47.51, 47.51, 47.51],
        "long": [-122.25, -122.25, -122.25],
        "sqft_living15": [1800, 1200, 2500],
        "sqft_lot15": [5000, 4000, 6000],
        "price": [300000, 250000, 400000],
    })

    csv_path = raw_dir / "kc_house_data.csv"
    df.to_csv(csv_path, index=False)

    # --- Run training ---
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    # --- Assert artifact ---
    latest = tmp_path / "models" / "registry" / "latest.json"
    assert latest.exists()
