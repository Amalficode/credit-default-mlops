"""API прогнозирования дефолта."""

from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import FastAPI, Request
import joblib
import pandas as pd

from src.api.schemas import CreditApplication, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(model_dir=None):
    directory = (
        Path(model_dir) if model_dir is not None else PROJECT_ROOT / "models/reproduced"
    )

    @asynccontextmanager
    async def lifespan(application):
        config = json.loads(
            (directory / "model_config.json").read_text(encoding="utf-8")
        )
        model = joblib.load(directory / "model.joblib")
        threshold = float(config["decision_threshold"])
        columns = config["input_columns"]

        if not 0 < threshold < 1:
            raise ValueError("Некорректный порог модели")

        if set(columns) != set(CreditApplication.model_fields) or len(columns) != len(
            CreditApplication.model_fields
        ):
            raise ValueError("Схема API не соответствует признакам модели")

        if list(model.classes_) != [0, 1]:
            raise ValueError("Ожидались классы модели 0 и 1")

        if float(config["C"]) != model.named_steps["classifier"].C:
            raise ValueError("Настройки не соответствуют модели")

        application.state.model = model
        application.state.threshold = threshold
        application.state.input_columns = columns

        try:
            yield
        finally:
            application.state.model = None

    application = FastAPI(
        title="Credit Default Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/")
    def read_root():
        return {"message": "Credit Default Prediction API is alive!"}

    @application.get("/health")
    def health(request: Request):
        return {
            "status": "ok",
            "model_loaded": request.app.state.model is not None,
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: CreditApplication, request: Request):
        state = request.app.state

        data = pd.DataFrame(
            [payload.model_dump()],
            columns=state.input_columns,
        )
        probability = float(state.model.predict_proba(data)[0, 1])

        return PredictionResponse(
            default_probability=probability,
            prediction=int(probability >= state.threshold),
            threshold=state.threshold,
        )

    return application


app = create_app()
