"""Проверки PSI и согласованности API с эталонной моделью."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from threading import Thread

import numpy as np
import pandas as pd
import pytest

from src.monitoring.monitor import (
    calculate_psi,
    run_monitoring,
    validate_api_prediction,
)


def test_identical_distributions_have_zero_psi():
    reference = np.linspace(0, 1, 101)
    result = calculate_psi(reference, reference[::-1])

    assert result["value"] == pytest.approx(0, abs=1e-14)
    assert result["level"] == "low"


def test_psi_matches_hand_calculation():
    reference = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    current = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    result = calculate_psi(reference, current, n_bins=2)

    # Доли с псевдочастотой 0.5: p=(1/2,1/2), q=(5/18,13/18).
    expected = (2 / 9) * math.log(13 / 5)
    assert result["value"] == pytest.approx(expected)
    assert [row["current_count"] for row in result["bins"]] == [2, 6]


def test_empty_current_bins_produce_finite_psi():
    reference = np.linspace(0.001, 0.999, 1000)
    result = calculate_psi(reference, np.full(200, 0.95))

    assert math.isfinite(result["value"])
    assert result["value"] > 0.25
    assert sum(row["current_share"] for row in result["bins"]) == pytest.approx(1)
    assert sum(row["reference_count"] for row in result["bins"]) == 1000
    assert sum(row["current_count"] for row in result["bins"]) == 200


def test_current_data_does_not_change_bin_boundaries():
    reference = np.linspace(0.1, 0.9, 1000)
    low = calculate_psi(reference, np.full(100, 0.05))
    high = calculate_psi(reference, np.full(100, 0.95))

    assert [(x["left"], x["right"]) for x in low["bins"]] == [
        (x["left"], x["right"]) for x in high["bins"]
    ]


def test_constant_reference_uses_fixed_bins_and_detects_shift():
    reference = np.full(100, 0.25)
    result = calculate_psi(reference, np.full(100, 0.75))

    assert result["binning"] == "equal_width_for_degenerate_reference"
    assert math.isfinite(result["value"])
    assert result["value"] > 0
    assert calculate_psi(reference, reference)["value"] == pytest.approx(0)


@pytest.mark.parametrize("bad", [[], [float("nan")], [-0.1], [1.1], [[0.5]]])
@pytest.mark.parametrize("side", ["reference", "current"])
def test_invalid_probabilities_are_rejected(bad, side):
    data = {"reference": [0.1, 0.9], "current": [0.2, 0.8]}
    data[side] = bad

    with pytest.raises(ValueError):
        calculate_psi(**data)


@pytest.mark.parametrize(
    "response",
    [
        {"default_probability": 0.7, "prediction": 1, "threshold": 0.5},
        {"default_probability": 0.7, "prediction": 0, "threshold": 0.3},
        {"default_probability": 0.8, "prediction": 1, "threshold": 0.3},
        {},
    ],
)
def test_incompatible_api_response_is_rejected(response):
    with pytest.raises(RuntimeError):
        validate_api_prediction(response, expected_probability=0.7, threshold=0.3)


def test_monitoring_sends_sample_over_http_and_saves_reports(tmp_path, monkeypatch):
    class ReferenceModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            probabilities = frame["AGE"].to_numpy() / 100
            return np.column_stack((1 - probabilities, probabilities))

    received = []

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == "/health"
            self.send_json({"status": "ok", "model_loaded": True})

        def do_POST(self):
            assert self.path == "/predict"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            probability = payload["AGE"] / 100
            self.send_json(
                {
                    "default_probability": probability,
                    "prediction": int(probability >= 0.3),
                    "threshold": 0.3,
                }
            )

        def log_message(self, *args):
            pass

    train_path = tmp_path / "train.csv"
    current_path = tmp_path / "test.csv"
    pd.DataFrame({"ID": [1, 2, 3, 4], "AGE": [20, 30, 40, 50]}).to_csv(
        train_path, index=False
    )
    pd.DataFrame({"ID": [5, 6, 7, 8, 9], "AGE": [35, 45, 55, 65, 75]}).to_csv(
        current_path, index=False
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.joblib").write_bytes(b"test model placeholder")
    (model_dir / "model_config.json").write_text(
        json.dumps({"input_columns": ["AGE"], "decision_threshold": 0.3}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.monitoring.monitor.joblib.load", lambda _: ReferenceModel()
    )
    report_path = tmp_path / "reports/drift_report.json"
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args = argparse.Namespace(
                train=train_path,
                current=current_path,
                model_dir=model_dir,
                api_url=f"http://127.0.0.1:{server.server_port}",
                sample_size=3,
                seed=42,
                bins=2,
                timeout=5,
                report=report_path,
            )
            run_monitoring(args)
        finally:
            server.shutdown()
            thread.join(timeout=5)

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(report_path.with_name("drift_report_predictions.csv"))
    assert received == [{"AGE": 45}, {"AGE": 75}, {"AGE": 55}]
    assert predictions["ID"].tolist() == [6, 9, 7]
    assert predictions["default_probability"].tolist() == pytest.approx(
        [0.45, 0.75, 0.55]
    )
    assert saved["scenario"] == "held_out_test_sample_simulation"
    assert saved["psi"]["reference_rows"] == 4
    assert saved["psi"]["current_rows"] == 3
    assert saved["all_api_predictions_match_saved_model"] is True
    assert saved["max_api_probability_difference"] == 0
