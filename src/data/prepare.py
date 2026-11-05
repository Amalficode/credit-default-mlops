"""Проверка исходных данных и воспроизводимое разделение выборки."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.validation.validate_data import TARGET, validate_credit_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/UCI_Credit_Card.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.json"),
    )
    args = parser.parse_args()

    params = json.loads(args.params.read_text(encoding="utf-8"))
    split_params = params["prepare"]

    data = pd.read_csv(args.input)
    validation = validate_credit_data(data)

    train, test = train_test_split(
        data,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=data[TARGET],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    summary = {
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "validation_success": bool(validation.success),
        "validation_statistics": validation.statistics,
        "split_parameters": split_params,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_default_rate": float(train[TARGET].mean()),
        "test_default_rate": float(test[TARGET].mean()),
    }
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Проверка данных пройдена.")
    print(f"Обучение: {len(train)} строк")
    print(f"Тест: {len(test)} строк")
    print(f"Файлы сохранены: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
