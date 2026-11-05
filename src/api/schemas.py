"""Формат входных данных и результата прогнозирования."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

PaymentStatus = Annotated[int, Field(ge=-2, le=9)]
PaymentAmount = Annotated[float, Field(ge=0)]


class CreditApplication(BaseModel):
    """Данные одного клиента: 23 исходных признака."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)

    PAY_0: PaymentStatus
    PAY_2: PaymentStatus
    PAY_3: PaymentStatus
    PAY_4: PaymentStatus
    PAY_5: PaymentStatus
    PAY_6: PaymentStatus

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: PaymentAmount
    PAY_AMT2: PaymentAmount
    PAY_AMT3: PaymentAmount
    PAY_AMT4: PaymentAmount
    PAY_AMT5: PaymentAmount
    PAY_AMT6: PaymentAmount


class PredictionResponse(BaseModel):
    """Вероятность дефолта и класс при сохранённом пороге."""

    model_config = ConfigDict(allow_inf_nan=False)

    default_probability: float = Field(ge=0, le=1)
    prediction: Literal[0, 1]
    threshold: float = Field(gt=0, lt=1)
