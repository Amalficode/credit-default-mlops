"""Воспроизводимое обучение модели с фиксированными параметрами."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.pipeline import create_pipeline
from src.validation.validate_data import TARGET, validate_credit_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/reproduced"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/reproduced_metrics.json"),
    )
    parser.add_argument("--params", type=Path, default=Path("params.json"))
    args = parser.parse_args()

    params = json.loads(args.params.read_text(encoding="utf-8"))
    model_params = params["train"]
    threshold = float(model_params["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Порог должен находиться между 0 и 1")

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    validate_credit_data(train)
    validate_credit_data(test)

    if set(train["ID"]) & set(test["ID"]):
        raise ValueError("Обучающая и тестовая выборки пересекаются по ID")

    X_train = train.drop(columns=["ID", TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=["ID", TARGET])
    y_test = test[TARGET]

    # Отдельное хранилище для запусков воспроизведения
    tracking_dir = Path("experiments/reproduction").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")

    experiment_name = "PD_Credit_Scoring_Reproduction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(tracking_dir / "artifacts").as_uri(),
        )
    mlflow.set_experiment(experiment_name)

    model = create_pipeline(C=float(model_params["C"]))

    with mlflow.start_run(run_name="reproduce_fixed_model") as run:
        mlflow.log_params(model.named_steps["classifier"].get_params())
        mlflow.log_params(
            {
                "threshold": threshold,
                "training_rows": len(train),
                "test_rows": len(test),
            }
        )

        print("Началось обучение...", flush=True)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        metrics = {
            "test_roc_auc": roc_auc_score(y_test, probabilities),
            "test_precision": precision_score(y_test, predictions, zero_division=0),
            "test_recall": recall_score(y_test, predictions, zero_division=0),
            "test_f1": f1_score(y_test, predictions, zero_division=0),
        }

        args.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.model_dir / "model.joblib"
        joblib.dump(model, model_path, compress=3)

        config = {
            "run_id": run.info.run_id,
            "C": float(model_params["C"]),
            "decision_threshold": threshold,
            "input_columns": X_train.columns.tolist(),
            "target": TARGET,
            "model_file": str(model_path),
            "python_version": platform.python_version(),
        }
        config_path = args.model_dir / "model_config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(args.metrics))
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(args.params))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="pd_model",
            serialization_format="cloudpickle",
            input_example=X_train.head(5),
            code_paths=["src"],
        )

        print("Обучение завершено.")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("Run ID:", run.info.run_id)
        print("Модель:", model_path.resolve())


if __name__ == "__main__":
    main()
