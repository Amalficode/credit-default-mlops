"""Автоматические проверки API."""

import json

from fastapi.testclient import TestClient
import joblib
import pandas as pd
import pytest

from src.api.app import create_app
from src.models.pipeline import create_pipeline


@pytest.fixture
def payload():
    data = {
        "LIMIT_BAL": 100000.0,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
    }
    for month in range(1, 7):
        data[f"BILL_AMT{month}"] = 1000.0
        data[f"PAY_AMT{month}"] = 100.0
    return data


@pytest.fixture
def model_bundle(tmp_path, payload):
    second = payload.copy()
    second.update({"AGE": 42, "PAY_0": 2})
    data = pd.DataFrame([payload, second])

    model = create_pipeline(C=0.1)
    model.fit(data, [0, 1])
    joblib.dump(model, tmp_path / "model.joblib")

    config = {
        "C": 0.1,
        "decision_threshold": 0.3,
        "input_columns": data.columns.tolist(),
    }
    (tmp_path / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, model


@pytest.fixture
def api_client(model_bundle):
    directory, _ = model_bundle
    with TestClient(create_app(model_dir=directory)) as client:
        yield client


def test_health(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_prediction_matches_saved_model(api_client, payload, model_bundle):
    _, model = model_bundle
    expected = float(model.predict_proba(pd.DataFrame([payload]))[0, 1])

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["default_probability"] == pytest.approx(expected)
    assert body["prediction"] == int(expected >= 0.3)
    assert body["threshold"] == 0.3


@pytest.mark.parametrize(
    "column,value",
    [
        ("PAY_AMT1", -100.0),
        ("PAY_0", 99),
        ("ID", 123),
    ],
)
def test_invalid_request_is_rejected(api_client, payload, column, value):
    payload[column] = value

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    assert any(error["loc"][-1] == column for error in response.json()["detail"])


def test_missing_age_is_rejected(api_client, payload):
    del payload["AGE"]

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    assert any(error["loc"][-1] == "AGE" for error in response.json()["detail"])
