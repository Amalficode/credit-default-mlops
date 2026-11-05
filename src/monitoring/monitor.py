"""Имитация новых запросов к API и PSI распределения вероятностей дефолта."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd


def _probabilities(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name}: ожидается непустой одномерный массив")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: вероятности должны быть конечными")
    if ((result < 0) | (result > 1)).any():
        raise ValueError(f"{name}: вероятности должны находиться между 0 и 1")
    return result


def calculate_psi(reference, current, n_bins=10, pseudocount=0.5):
    """PSI по общим интервалам; их границы определяются только по reference."""
    reference = _probabilities(reference, "reference")
    current = _probabilities(current, "current")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("Число интервалов должно быть целым и не меньше 2")
    if not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("Псевдочастота должна быть конечной и положительной")

    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        edges = np.linspace(0, 1, n_bins + 1)
        strategy = "equal_width_for_degenerate_reference"
    else:
        inner = quantiles[1:-1]
        inner = inner[(inner > 0) & (inner < 1)]
        edges = np.unique(np.concatenate(([0.0], inner, [1.0])))
        strategy = "training_quantiles"

    reference_counts = np.histogram(reference, bins=edges)[0]
    current_counts = np.histogram(current, bins=edges)[0]
    count = len(edges) - 1
    reference_share = (reference_counts + pseudocount) / (
        reference.size + pseudocount * count
    )
    current_share = (current_counts + pseudocount) / (
        current.size + pseudocount * count
    )
    contributions = (current_share - reference_share) * np.log(
        current_share / reference_share
    )
    psi = float(contributions.sum())
    if psi < 0.1:
        level = "low"
    elif psi < 0.25:
        level = "moderate"
    else:
        level = "high"

    return {
        "value": psi,
        "level": level,
        "binning": strategy,
        "requested_bins": n_bins,
        "actual_bins": count,
        "pseudocount": pseudocount,
        "reference_rows": int(reference.size),
        "current_rows": int(current.size),
        "thresholds": {"moderate": 0.1, "high": 0.25},
        "bins": [
            {
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "reference_count": int(reference_counts[i]),
                "current_count": int(current_counts[i]),
                "reference_share": float(reference_share[i]),
                "current_share": float(current_share[i]),
                "psi_contribution": float(contributions[i]),
            }
            for i in range(count)
        ],
    }


def validate_api_prediction(response, expected_probability, threshold):
    """Не смешиваем результаты другой модели или другого порога с эталоном."""
    required = {"default_probability", "prediction", "threshold"}
    if not isinstance(response, dict) or not required.issubset(response):
        raise RuntimeError("Ответ API не соответствует ожидаемой схеме")
    probability = response["default_probability"]
    api_threshold = response["threshold"]
    if type(probability) not in (int, float):
        raise RuntimeError("API вернул некорректную вероятность")
    _probabilities([probability], "API")
    if type(api_threshold) not in (int, float) or not math.isclose(
        api_threshold, threshold, rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError("Порог API отличается от настроек эталонной модели")
    if type(response["prediction"]) is not int or response["prediction"] != int(
        probability >= threshold
    ):
        raise RuntimeError("Класс в ответе API не соответствует вероятности и порогу")
    if not math.isclose(
        probability, expected_probability, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise RuntimeError("Прогноз API отличается от сохранённой эталонной модели")
    return float(probability)


def _request_json(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"API вернул HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"API недоступен по адресу {url}: {error}") from error


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_monitoring(args):
    """Отправляем фиксированную случайную часть теста в работающий API."""
    if args.sample_size < 1 or args.timeout <= 0 or not math.isfinite(args.timeout):
        raise ValueError("Размер выборки и timeout должны быть положительными")
    if args.bins < 2:
        raise ValueError("Число интервалов должно быть не меньше 2")
    base_url = args.api_url.rstrip("/")
    health = _request_json(f"{base_url}/health", timeout=args.timeout)
    if (
        not isinstance(health, dict)
        or health.get("status") != "ok"
        or not health.get("model_loaded")
    ):
        raise RuntimeError("API не подтвердил готовность модели")

    config_path = args.model_dir / "model_config.json"
    model_path = args.model_dir / "model.joblib"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    columns = config["input_columns"]
    threshold = float(config["decision_threshold"])
    if not columns or len(set(columns)) != len(columns) or not 0 < threshold < 1:
        raise ValueError("Некорректные входные столбцы или порог в конфигурации")

    train = pd.read_csv(args.train)
    incoming = pd.read_csv(args.current)
    if train.empty or incoming.empty:
        raise ValueError("Обучающая и текущая таблицы должны быть непустыми")
    for name, frame in [("train", train), ("current", incoming)]:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{name}: отсутствуют признаки {missing}")
    if "ID" in train and "ID" in incoming:
        if set(train["ID"]) & set(incoming["ID"]):
            raise ValueError("Эталонная и текущая выборки пересекаются по ID")

    sample = incoming.sample(
        n=min(args.sample_size, len(incoming)), random_state=args.seed
    ).reset_index(drop=True)
    model = joblib.load(model_path)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Ожидались классы модели 0 и 1")
    reference = _probabilities(model.predict_proba(train[columns])[:, 1], "train")
    expected = _probabilities(model.predict_proba(sample[columns])[:, 1], "sample")

    print(f"Эталон: {len(train)} обучающих строк", flush=True)
    print(
        f"Имитация потока: {len(sample)} из {len(incoming)} тестовых строк", flush=True
    )
    records = sample[columns].to_dict(orient="records")
    probabilities = []
    for index, record in enumerate(records):
        response = _request_json(
            f"{base_url}/predict", payload=record, timeout=args.timeout
        )
        probabilities.append(
            validate_api_prediction(response, float(expected[index]), threshold)
        )
        if (index + 1) % 50 == 0 or index + 1 == len(records):
            print(f"Ответы API: {index + 1}/{len(records)}", flush=True)

    current = np.asarray(probabilities)
    psi = calculate_psi(reference, current, n_bins=args.bins)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "held_out_test_sample_simulation",
        "api_url": base_url,
        "sample_seed": args.seed,
        "requested_sample_size": args.sample_size,
        "current_source_rows": len(incoming),
        "model_run_id": config.get("run_id"),
        "decision_threshold": threshold,
        "all_api_predictions_match_saved_model": True,
        "max_api_probability_difference": float(np.abs(current - expected).max()),
        "reference_mean_probability": float(reference.mean()),
        "current_mean_probability": float(current.mean()),
        "psi": psi,
        "input_sha256": {
            "train": _sha256(args.train),
            "current": _sha256(args.current),
            "model": _sha256(model_path),
            "config": _sha256(config_path),
        },
        "notes": [
            "Использована часть отложенного теста как учебная имитация новых данных.",
            "Границы интервалов построены только по обучающим вероятностям.",
            "Псевдочастота 0.5 обеспечивает конечный PSI при пустых интервалах.",
            "Пороги 0.1 и 0.25 условны; PSI зависит от размера выборки и интервалов.",
            "PSI отражает сдвиг распределения оценок, а не качество классификации.",
            "Модель и порог не переобучаются и не изменяются этим скриптом.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.report.with_name(args.report.stem + "_predictions.csv")
    predictions = pd.DataFrame(
        {
            "default_probability": current,
            "prediction": (current >= threshold).astype(int),
            "threshold": threshold,
        }
    )
    if "ID" in sample:
        predictions.insert(0, "ID", sample["ID"].to_numpy())
    predictions.to_csv(predictions_path, index=False)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    levels = {"low": "низкий", "moderate": "умеренный", "high": "выраженный"}
    print(f"PSI вероятностей дефолта: {psi['value']:.6f}")
    print(f"Условный уровень сдвига: {levels[psi['level']]}")
    print("Прогнозы API совпадают с сохранённой моделью.")
    print(f"Отчёт: {args.report}")
    print(f"Предсказания: {predictions_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--current", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/reproduced"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/drift_report.json")
    )
    run_monitoring(parser.parse_args())


if __name__ == "__main__":
    main()
