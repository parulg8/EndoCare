"""Dataset audit: quality, distributions, leakage checks, and provenance caveats."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import BINARY_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from src.preprocessing import coerce_types, load_dataset


def _iqr_outlier_counts(series: pd.Series) -> dict[str, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(lower),
        "upper_fence": float(upper),
        "n_outliers": int(mask.sum()),
        "pct_outliers": float(mask.mean() * 100),
    }


def audit_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    raw = coerce_types(df)
    n = len(raw)
    missing = raw.isna().sum().astype(int).to_dict()
    dtypes = {c: str(raw[c].dtype) for c in raw.columns}
    duplicates = int(raw.duplicated().sum())
    class_counts = raw[TARGET].value_counts(dropna=False).to_dict()
    class_counts = {str(k): int(v) for k, v in class_counts.items()}
    pos_rate = float(raw[TARGET].mean()) if raw[TARGET].notna().any() else None

    numeric_summary = raw[NUMERIC_FEATURES + BINARY_FEATURES + [TARGET]].describe().to_dict()
    cleaned_summary = {
        col: {k: float(v) for k, v in stats.items()}
        for col, stats in numeric_summary.items()
    }

    outliers = {col: _iqr_outlier_counts(raw[col].dropna()) for col in NUMERIC_FEATURES}

    correlations = (
        raw[FEATURE_COLUMNS + [TARGET]]
        .corr(numeric_only=True)
        .round(6)
        .to_dict()
    )

    target_corr = (
        raw[FEATURE_COLUMNS + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    leakage_flags = []
    for feature, corr in target_corr.items():
        if abs(corr) >= 0.95:
            leakage_flags.append(
                f"{feature} is almost perfectly correlated with Diagnosis (r={corr:.3f}). "
                "This would suggest leakage or a constructed label."
            )
    if not leakage_flags:
        leakage_flags.append(
            "No feature is nearly perfectly correlated with Diagnosis. "
            "That does not prove the labels are independent of the features by construction, "
            "only that a trivial one-to-one mapping is not visible."
        )

    group_rates = {}
    for col in BINARY_FEATURES:
        rates = raw.groupby(col)[TARGET].mean().to_dict()
        group_rates[col] = {str(k): float(v) for k, v in rates.items()}

    bmi_floor = int((raw["BMI"] == 15.0).sum()) if "BMI" in raw else 0
    pain_floor = int((raw["Chronic_Pain_Level"] == 0.0).sum())
    pain_ceil = int((raw["Chronic_Pain_Level"] == 10.0).sum())

    uniqueness = raw.nunique(dropna=False).astype(int).to_dict()

    # Point-biserial / t-tests for numeric features vs label (descriptive only)
    association_tests = {}
    for col in NUMERIC_FEATURES:
        a = raw.loc[raw[TARGET] == 0, col].dropna()
        b = raw.loc[raw[TARGET] == 1, col].dropna()
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        association_tests[col] = {
            "mean_class_0": float(a.mean()),
            "mean_class_1": float(b.mean()),
            "welch_t": float(t_stat),
            "p_value": float(p_val),
        }

    chi_square = {}
    for col in BINARY_FEATURES:
        table = pd.crosstab(raw[col], raw[TARGET])
        chi2, p_val, dof, _ = stats.chi2_contingency(table)
        chi_square[col] = {"chi2": float(chi2), "p_value": float(p_val), "dof": int(dof)}

    observations = [
        f"The file contains {n} rows and {raw.shape[1]} columns.",
        f"Missing values: {int(sum(missing.values()))} across all cells.",
        f"Exact duplicate rows: {duplicates}.",
        f"Diagnosis class counts: {class_counts} (positive rate {pos_rate:.4f}).",
        "The imbalance is moderate, not extreme; stratified splits and class-weight options are still appropriate.",
        "Chronic_Pain_Level and BMI have thousands of unique values, which is consistent with continuous (possibly generated) measurements rather than a short ordinal scale.",
        f"BMI equals 15.0 in {bmi_floor} rows, suggesting a lower clip. Chronic_Pain_Level hits 0.0 in {pain_floor} rows and 10.0 in {pain_ceil} rows.",
        "Age is only weakly associated with Diagnosis in this file; hormone abnormality and chronic pain show the largest (still modest) associations.",
        "Feature definitions are not accompanied by a data dictionary in the repository. Coding of binary fields should be treated as file-specific, not clinical standard.",
        "Provenance is uncertain. This project treats the table as a structured Kaggle-style research file, not as a verified clinical registry.",
    ]

    return {
        "n_rows": int(n),
        "n_columns": int(raw.shape[1]),
        "columns": list(raw.columns),
        "dtypes": dtypes,
        "missing_by_column": missing,
        "duplicate_rows": duplicates,
        "class_counts": class_counts,
        "positive_rate": pos_rate,
        "nunique": uniqueness,
        "describe": cleaned_summary,
        "iqr_outliers": outliers,
        "correlations": correlations,
        "target_correlations": {k: float(v) for k, v in target_corr.items()},
        "binary_positive_rates": group_rates,
        "association_tests_numeric": association_tests,
        "chi_square_binary": chi_square,
        "bmi_equals_15": bmi_floor,
        "pain_equals_0": pain_floor,
        "pain_equals_10": pain_ceil,
        "leakage_notes": leakage_flags,
        "observations": observations,
        "provenance": {
            "filename": "structured_endometriosis_data.csv",
            "source_claim": "Supplied locally as a structured endometriosis table intended for academic ML work.",
            "unknowns": [
                "Collection site, years, inclusion/exclusion criteria, and ethics approval are not documented in-file.",
                "It is not known whether Diagnosis is surgically confirmed, clinically coded, self-reported, or synthetically assigned.",
                "Binary feature definitions (irregularity, hormone abnormality, infertility) have no codebook.",
                "The combination of clipped BMI, dense unique continuous values, and modest linear associations is compatible with simulated or heavily processed data.",
                "Licence and redistribution terms were not bundled with the CSV; the packaged ZIP therefore omits the dataset file.",
            ],
        },
    }


def audit_file(path) -> dict[str, Any]:
    df = load_dataset(path)
    return audit_dataframe(df)
