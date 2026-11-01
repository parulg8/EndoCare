"""Loading, validation, splitting, and preprocessing without test-set leakage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


class DatasetSchemaError(ValueError):
    """Raised when the on-disk dataset does not match the expected schema."""


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise DatasetSchemaError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    extra = [c for c in df.columns if c not in FEATURE_COLUMNS + [TARGET]]
    if extra:
        df = df[FEATURE_COLUMNS + [TARGET]].copy()
    else:
        df = df.copy()
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLUMNS + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def drop_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    out = df.dropna(subset=FEATURE_COLUMNS + [TARGET]).copy()
    dropped_na = before - len(out)
    before_dup = len(out)
    out = out.drop_duplicates()
    dropped_dup = before_dup - len(out)
    out[TARGET] = out[TARGET].astype(int)
    invalid_target = ~out[TARGET].isin([0, 1])
    dropped_target = int(invalid_target.sum())
    out = out.loc[~invalid_target].copy()
    return out, {
        "dropped_missing": int(dropped_na),
        "dropped_duplicates": int(dropped_dup),
        "dropped_invalid_target": dropped_target,
        "rows_after_cleaning": int(len(out)),
    }


def stratified_split(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = df[FEATURE_COLUMNS]
    y = df[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return x_train, x_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Fit-only-on-train transformer. Scaling is required for LR and SVM.

    Tree models are insensitive to monotone scaling; a shared preprocessor
    keeps the inference contract identical for every saved pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_feature_names() -> list[str]:
    return NUMERIC_FEATURES + BINARY_FEATURES


def to_model_frame(payload: dict[str, Any]) -> pd.DataFrame:
    row = {col: payload[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def clip_warning(payload: dict[str, Any], train_stats: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    ranges = train_stats.get("feature_ranges", {})
    for col in FEATURE_COLUMNS:
        if col not in ranges:
            continue
        value = float(payload[col])
        lo, hi = ranges[col]["min"], ranges[col]["max"]
        if value < lo or value > hi:
            warnings.append(
                f"{col}={value} is outside the training range [{lo:.3f}, {hi:.3f}]. "
                "The estimate is an extrapolation and is even less reliable."
            )
    return warnings


def summarise_train_stats(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    ranges = {}
    for col in FEATURE_COLUMNS:
        ranges[col] = {
            "min": float(np.min(x_train[col])),
            "max": float(np.max(x_train[col])),
            "mean": float(np.mean(x_train[col])),
            "std": float(np.std(x_train[col])),
        }
    return {
        "n_train": int(len(x_train)),
        "positive_rate_train": float(y_train.mean()),
        "feature_ranges": ranges,
    }
