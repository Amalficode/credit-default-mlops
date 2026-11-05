"""Очистка категорий и создание признаков кредитного скоринга."""

import pandas as pd

STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]


def clean_categories(data):
    """Объединяем недокументированные коды образования в категорию 0."""
    result = data.copy()
    result["EDUCATION"] = result["EDUCATION"].replace({5: 0, 6: 0})
    return result


def build_features(data):
    """Добавляем шесть признаков, сохраняя исходные столбцы."""
    result = data.copy()

    # Число месяцев, в которых была просрочка
    result["late_months_count"] = (result[STATUS_COLUMNS] > 0).sum(axis=1)

    # Максимальная длительность просрочки
    result["max_delay"] = result[STATUS_COLUMNS].clip(lower=0).max(axis=1)

    # Средняя сумма счёта и средняя сумма платежа за шесть месяцев
    result["bill_mean"] = result[BILL_COLUMNS].mean(axis=1)
    result["payment_mean"] = result[PAYMENT_COLUMNS].mean(axis=1)

    # Последний счёт относительно кредитного лимита
    result["latest_bill_to_limit"] = (
        result["BILL_AMT1"].clip(lower=0) / result["LIMIT_BAL"]
    )

    # Возрастная группа
    result["AGE_GROUP"] = pd.cut(
        result["AGE"],
        bins=[18, 30, 45, 60, float("inf")],
        labels=["18–29", "30–44", "45–59", "60+"],
        right=False,
    )

    return result


def prepare_features(data):
    """Последовательно очищаем категории и создаём признаки."""
    return build_features(clean_categories(data))
