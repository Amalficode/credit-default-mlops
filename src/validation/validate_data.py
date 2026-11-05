"""Проверка исходных данных кредитного скоринга."""

from functools import lru_cache

import great_expectations as gx
import pandas as pd

from src.features.build_features import (
    BILL_COLUMNS,
    PAYMENT_COLUMNS,
    STATUS_COLUMNS,
)

TARGET = "default.payment.next.month"
EXPECTED_COLUMNS = (
    ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + STATUS_COLUMNS
    + BILL_COLUMNS
    + PAYMENT_COLUMNS
    + [TARGET]
)


@lru_cache(maxsize=1)
def _get_validation_definition():
    """Создаём набор правил один раз на процесс Python."""
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name="project_credit_data")
    asset = source.add_dataframe_asset(name="credit_clients")
    batch = asset.add_batch_definition_whole_dataframe(name="whole_table")

    suite = gx.ExpectationSuite(name="credit_raw_checks")

    # Для каждого столбца: отсутствие пропусков и допустимый тип
    amount_columns = ["LIMIT_BAL"] + BILL_COLUMNS + PAYMENT_COLUMNS
    for column in EXPECTED_COLUMNS:
        types = ["int64", "float64"] if column in amount_columns else ["int64"]
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInTypeList(
                column=column, type_list=types
            )
        )

    # Структура таблицы и идентификаторы
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_COLUMNS, exact_match=True
        )
    )
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="ID"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="ID", min_value=0, strict_min=True
        )
    )

    # Возраст и кредитный лимит
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="AGE", min_value=18, max_value=100
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="LIMIT_BAL", min_value=0, strict_min=True
        )
    )

    # Допустимые категории исходных данных
    category_sets = {
        TARGET: [0, 1],
        "SEX": [1, 2],
        "EDUCATION": list(range(7)),
        "MARRIAGE": [0, 1, 2, 3],
    }
    for column, values in category_sets.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=column, value_set=values)
        )

    for column in STATUS_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column, value_set=list(range(-2, 10))
            )
        )

    # Платежи неотрицательные; отрицательные счета допускаются
    for column in PAYMENT_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(column=column, min_value=0)
        )

    suite = context.suites.add(suite)
    definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="validate_raw_credit_data",
            data=batch,
            suite=suite,
        )
    )
    return context, definition


def validate_credit_data(data):
    """Возвращаем результат GX или отклоняем некорректные данные."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Ожидалась таблица pandas DataFrame")

    # Проверяем структуру до расчёта метрик отдельных столбцов
    missing = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    extra = sorted(set(data.columns) - set(EXPECTED_COLUMNS))
    if missing or extra or not data.columns.is_unique:
        raise ValueError(
            f"Некорректные столбцы. Отсутствуют: {missing}; "
            f"лишние: {extra}; "
            f"имена уникальны: {data.columns.is_unique}"
        )
    if data.empty:
        raise ValueError("Таблица не должна быть пустой")

    _, definition = _get_validation_definition()
    result = definition.run(batch_parameters={"dataframe": data})

    if not result.success:
        failed = []
        for check in result.results:
            if not check.success:
                config = check.expectation_config
                column = config.kwargs.get("column", "вся таблица")
                failed.append(f"{column}: {config.type}")
        raise ValueError("Данные не прошли валидацию:\n" + "\n".join(failed))

    return result
