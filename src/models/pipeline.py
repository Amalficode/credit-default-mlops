"""Создание Pipeline кредитного скоринга."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from src.features.build_features import (
    BILL_COLUMNS,
    PAYMENT_COLUMNS,
    STATUS_COLUMNS,
    prepare_features,
)

NUMERIC_FEATURES = (
    ["LIMIT_BAL", "AGE"]
    + BILL_COLUMNS
    + PAYMENT_COLUMNS
    + [
        "late_months_count",
        "max_delay",
        "bill_mean",
        "payment_mean",
        "latest_bill_to_limit",
    ]
)

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"] + STATUS_COLUMNS + ["AGE_GROUP"]


def create_pipeline(C=0.1):
    """Возвращаем новый, ещё не обученный Pipeline."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        [
            (
                "features",
                FunctionTransformer(prepare_features, validate=False),
            ),
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(C=C, max_iter=2000)),
        ]
    )
