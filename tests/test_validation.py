"""Проверяем, какие данные валидатор принимает и отклоняет."""

import pandas as pd
import pytest

from src.validation.validate_data import validate_credit_data


@pytest.fixture
def sample():
    """Каждый тест получает собственную корректную таблицу."""
    first = {
        "ID": 1,
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
        "default.payment.next.month": 0,
    }
    for month in range(1, 7):
        first[f"BILL_AMT{month}"] = 1000.0
        first[f"PAY_AMT{month}"] = 100.0

    second = first.copy()
    second.update(
        {
            "ID": 2,
            "AGE": 42,
            "PAY_0": 2,
            "default.payment.next.month": 1,
        }
    )

    return pd.DataFrame([first, second])


def test_valid_data(sample):
    result = validate_credit_data(sample)
    assert result.success


def test_negative_payment_is_rejected(sample):
    sample.loc[0, "PAY_AMT1"] = -100.0

    with pytest.raises(ValueError, match="PAY_AMT1"):
        validate_credit_data(sample)


def test_unknown_status_is_rejected(sample):
    sample.loc[0, "PAY_0"] = 99

    with pytest.raises(ValueError, match="PAY_0"):
        validate_credit_data(sample)


def test_negative_bill_is_allowed(sample):
    sample.loc[0, "BILL_AMT1"] = -100.0

    result = validate_credit_data(sample)
    assert result.success
