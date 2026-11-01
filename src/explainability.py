"""Global and local explanations: impurity/coef importance, permutation, SHAP."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import PERMUTATION_REPEATS, RANDOM_STATE, SHAP_SAMPLE_SIZE
from src.preprocessing import transformed_feature_names


def _estimator_from_pipeline(pipeline):
    return pipeline.named_steps["model"]


def native_feature_importance(pipeline) -> pd.DataFrame | None:
    names = transformed_feature_names()
    model = _estimator_from_pipeline(pipeline)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity_or_gain"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float).ravel()
        values = np.abs(coef)
        kind = "absolute_coefficient"
    else:
        inner = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
        if inner is not None and hasattr(inner, "coef_"):
            values = np.abs(np.asarray(inner.coef_, dtype=float).ravel())
            kind = "absolute_coefficient"
        elif hasattr(model, "calibrated_classifiers_"):
            coefs = []
            for cc in model.calibrated_classifiers_:
                est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                if est is not None and hasattr(est, "coef_"):
                    coefs.append(np.asarray(est.coef_, dtype=float).ravel())
            if coefs:
                values = np.abs(np.mean(np.vstack(coefs), axis=0))
                kind = "absolute_coefficient"
            else:
                return None
        else:
            return None
    if values.shape[0] != len(names):
        return None
    df = pd.DataFrame({"feature": names, "importance": values, "kind": kind})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_table(pipeline, x, y) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        x,
        y,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": list(x.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def shap_explanations(pipeline, x: pd.DataFrame) -> dict[str, Any] | None:
    try:
        import shap
    except ImportError:
        return None

    sample = x
    if len(x) > SHAP_SAMPLE_SIZE:
        sample = x.sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    x_trans = preprocessor.transform(sample)
    feature_names = transformed_feature_names()

    try:
        explainer = shap.Explainer(model, x_trans, feature_names=feature_names)
        explanation = explainer(x_trans)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 1]
        base_values = np.asarray(explanation.base_values)
        if base_values.ndim > 1:
            base_values = base_values[:, 1] if base_values.shape[-1] == 2 else base_values.mean(axis=-1)
        mean_abs = np.mean(np.abs(values), axis=0)
        global_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        return {
            "available": True,
            "n_samples": int(len(sample)),
            "feature_names": feature_names,
            "mean_abs_shap": global_df.to_dict(orient="records"),
            "sample_index": sample.index.astype(int).tolist(),
            "shap_values": values.tolist(),
            "base_values": np.asarray(base_values).reshape(-1).tolist(),
            "data": np.asarray(x_trans).tolist(),
            "raw_sample": sample.reset_index(drop=True).to_dict(orient="list"),
            "method": type(explainer).__name__,
        }
    except Exception as exc:  # noqa: BLE001 - SHAP backends vary by estimator
        try:
            explainer = shap.PermutationExplainer(pipeline.predict_proba, sample, seed=RANDOM_STATE)
            explanation = explainer(sample)
            values = np.asarray(explanation.values)
            if values.ndim == 3:
                values = values[:, :, 1]
            mean_abs = np.mean(np.abs(values), axis=0)
            global_df = pd.DataFrame(
                {"feature": list(sample.columns), "mean_abs_shap": mean_abs}
            ).sort_values("mean_abs_shap", ascending=False)
            return {
                "available": True,
                "n_samples": int(len(sample)),
                "feature_names": list(sample.columns),
                "mean_abs_shap": global_df.to_dict(orient="records"),
                "shap_values": values.tolist(),
                "method": "PermutationExplainer",
                "note": f"Primary SHAP explainer failed ({exc}); permutation SHAP was used instead.",
            }
        except Exception as exc2:  # noqa: BLE001
            return {
                "available": False,
                "error": f"{type(exc).__name__}: {exc} | fallback {type(exc2).__name__}: {exc2}",
            }


def local_prediction_contributions(pipeline, x_row: pd.DataFrame) -> dict[str, Any]:
    """Per-row contributions using SHAP when valid, otherwise a model-anchored sensitivity fallback."""
    shap_result = shap_explanations(pipeline, x_row)
    if shap_result and shap_result.get("available") and shap_result.get("shap_values"):
        values = np.asarray(shap_result["shap_values"], dtype=float)
        if values.ndim == 3:
            values = values[:, :, 1]
        flat = np.asarray(values).reshape(-1)
        if flat.size and not np.allclose(flat, 0.0):
            names = shap_result["feature_names"]
            values_row = np.asarray(values[0], dtype=float) if values.ndim > 1 else np.asarray(values, dtype=float)
            pairs = sorted(zip(names, values_row), key=lambda t: abs(t[1]), reverse=True)
            return {
                "method": shap_result.get("method", "shap"),
                "contributions": [{"feature": n, "contribution": float(v)} for n, v in pairs],
                "base_value": float(shap_result.get("base_values", [0])[0]) if shap_result.get("base_values") else None,
            }

    names = transformed_feature_names()
    model = _estimator_from_pipeline(pipeline)
    base_score = float(pipeline.predict_proba(x_row)[0, 1])
    contributions = []
    for name in names:
        perturbed = x_row.copy()
        current = float(x_row[name].iloc[0])
        if name in ["Menstrual_Irregularity", "Hormone_Level_Abnormality", "Infertility"]:
            perturbed[name] = 1.0 - current
        else:
            perturbed[name] = current + 1.0
        perturbed_score = float(pipeline.predict_proba(perturbed)[0, 1])
        contributions.append({"feature": name, "contribution": float(perturbed_score - base_score)})
    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return {
        "method": "feature_sensitivity",
        "contributions": contributions,
        "base_value": base_score,
        "note": "SHAP produced a degenerate zero-valued output for this estimator, so a local feature sensitivity check was used instead. This is a model-based influence measure, not a causal claim.",
    }

    names = transformed_feature_names()
    model = _estimator_from_pipeline(pipeline)
    x_trans = pipeline.named_steps["preprocessor"].transform(x_row)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float).ravel()
        contrib = coef * np.asarray(x_trans).ravel()
        intercept = float(model.intercept_.ravel()[0]) if hasattr(model, "intercept_") else 0.0
        pairs = sorted(zip(names, contrib), key=lambda t: abs(t[1]), reverse=True)
        return {
            "method": "linear_log_odds_contribution",
            "contributions": [{"feature": n, "contribution": float(v)} for n, v in pairs],
            "base_value": intercept,
        }

    native = native_feature_importance(pipeline)
    proba = float(pipeline.predict_proba(x_row)[0, 1])
    if native is None:
        return {
            "method": "score_only",
            "contributions": [],
            "model_score": proba,
            "note": "This estimator does not expose coefficients or feature importances for a local breakdown.",
        }
    row_vals = np.asarray(x_trans).ravel()
    signed = []
    importance_map = dict(zip(native["feature"], native["importance"]))
    for name, z in zip(names, row_vals):
        signed.append((name, float(importance_map.get(name, 0.0) * z)))
    signed.sort(key=lambda t: abs(t[1]), reverse=True)
    return {
        "method": "scaled_native_importance_times_value",
        "contributions": [{"feature": n, "contribution": float(v)} for n, v in signed],
        "model_score": proba,
        "note": "Approximate local view for tree models when SHAP is unavailable. Not a causal effect.",
    }
