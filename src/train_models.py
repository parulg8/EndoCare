"""Train, tune, compare, and persist endometriosis risk models."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    CV_FOLDS,
    DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SEARCH_CV_FOLDS,
    SEARCH_ITER,
    TARGET,
)
from src.data_audit import audit_file
from src.evaluate_models import (
    calibration_payload,
    classification_metrics,
    composite_selection_score,
    curve_payloads,
    select_threshold_from_oof,
    threshold_table,
)
from src.explainability import native_feature_importance, permutation_table, shap_explanations
from src.preprocessing import (
    build_preprocessor,
    coerce_types,
    drop_invalid_rows,
    load_dataset,
    stratified_split,
    summarise_train_stats,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _maybe_xgboost():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier
    except Exception:
        return None


def _search_space(n_pos: int, n_neg: int) -> dict:
    scale = max(n_neg / max(n_pos, 1), 1.0)
    spaces = {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__C": [0.1, 0.5, 1.0, 3.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [4, 8, 12, None],
                "model__min_samples_leaf": [1, 5, 15, 30],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Support Vector Machine": {
            "estimator": CalibratedClassifierCV(
                LinearSVC(
                    random_state=RANDOM_STATE,
                    max_iter=8000,
                    dual="auto",
                ),
                method="sigmoid",
                cv=3,
            ),
            "params": {
                "model__estimator__C": [0.1, 0.5, 1.0, 3.0, 10.0],
                "model__estimator__class_weight": [None, "balanced"],
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "model__n_estimators": [80, 150, 250],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [10, 20, 40],
            },
        },
    }
    xgb_cls = _maybe_xgboost()
    if xgb_cls is not None:
        spaces["XGBoost"] = {
            "estimator": xgb_cls(
                random_state=RANDOM_STATE,
                n_jobs=1,
                eval_metric="logloss",
                verbosity=0,
            ),
            "params": {
                "model__n_estimators": [80, 160, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__scale_pos_weight": [1.0, float(scale)],
                "model__min_child_weight": [1, 5, 10],
            },
        }
    return spaces


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _cv_report(pipeline, x, y) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    raw = cross_validate(pipeline, x, y, cv=cv, scoring=scoring, n_jobs=1)
    out = {}
    for key in scoring:
        values = raw[f"test_{key}"]
        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": [float(v) for v in values],
        }
    return out


def _oof_scores(pipeline, x, y) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(pipeline, x, y, cv=cv, method="predict_proba")[:, 1]


def _plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No diagnosis (0)", "Diagnosis (1)"],
        cmap="YlGnBu",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_curves(all_curves: dict, path_roc: Path, path_pr: Path, prevalence: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name, payload in all_curves.items():
        ax.plot(payload["fpr"], payload["tpr"], label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa3ad", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (held-out test set)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path_roc, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name, payload in all_curves.items():
        ax.plot(payload["recall"], payload["precision"], label=name, linewidth=2)
    ax.axhline(prevalence, linestyle="--", color="#9aa3ad", linewidth=1, label="Prevalence baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves (held-out test set)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path_pr, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_calibration(cal: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa3ad", label="Perfect calibration")
    ax.plot(cal["mean_predicted_value"], cal["fraction_of_positives"], marker="o", linewidth=2, label="Selected model")
    ax.set_xlabel("Mean model score in bin")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Reliability diagram (test set)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_importance(df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plot_df = df.sort_values(value_col)
    ax.barh(plot_df["feature"], plot_df[value_col], color="#3A8A8C")
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " "))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_training() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place structured_endometriosis_data.csv in the data/ folder."
        )

    audit = audit_file(DATA_PATH)
    df = drop_invalid_rows(coerce_types(load_dataset(DATA_PATH)))[0]
    x_train, x_test, y_train, y_test = stratified_split(df)
    train_stats = summarise_train_stats(x_train, y_train)

    spaces = _search_space(int((y_train == 1).sum()), int((y_train == 0).sum()))
    inner_cv = StratifiedKFold(n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    comparison_rows = []
    cv_rows = []
    eval_store = {"y_test": y_test.astype(int).tolist(), "models": {}}
    all_curves = {}
    fitted = {}

    for name, spec in spaces.items():
        print(f"Tuning {name}...")
        pipe = build_pipeline(spec["estimator"])
        search = RandomizedSearchCV(
            pipe,
            spec["params"],
            n_iter=min(SEARCH_ITER, _grid_size(spec["params"])),
            scoring="roc_auc",
            cv=inner_cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train)
        best_pipe = search.best_estimator_
        fitted[name] = best_pipe

        cv_stats = _cv_report(best_pipe, x_train, y_train)
        oof = _oof_scores(best_pipe, x_train, y_train)
        oof_thr = select_threshold_from_oof(y_train, oof)

        y_score = best_pipe.predict_proba(x_test)[:, 1]
        y_pred_05 = (y_score >= 0.5).astype(int)
        y_pred_thr = (y_score >= oof_thr["threshold"]).astype(int)
        metrics_05 = classification_metrics(y_test, y_pred_05, y_score)
        metrics_thr = classification_metrics(y_test, y_pred_thr, y_score)
        curves = curve_payloads(y_test, y_score)
        all_curves[name] = curves

        _plot_confusion(
            y_test,
            y_pred_05,
            f"{name} — test set (threshold 0.50)",
            FIGURES_DIR / f"confusion_{_slug(name)}_t050.png",
        )

        row = {
            "model": name,
            "inner_cv_roc_auc": float(search.best_score_),
            "best_params": json.dumps(search.best_params_),
            "selected_threshold_oof_f1": oof_thr["threshold"],
            **{f"test_{k}": v for k, v in metrics_05.items() if k not in {"tp", "tn", "fp", "fn"}},
            "test_f1_at_oof_threshold": metrics_thr["f1"],
            "test_recall_at_oof_threshold": metrics_thr["recall"],
            "composite": composite_selection_score(metrics_05),
        }
        comparison_rows.append(row)

        for metric, payload in cv_stats.items():
            cv_rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "mean": payload["mean"],
                    "std": payload["std"],
                    **{f"fold_{i+1}": v for i, v in enumerate(payload["folds"])},
                }
            )

        eval_store["models"][name] = {
            "y_score": y_score.tolist(),
            "y_pred_0_5": y_pred_05.tolist(),
            "metrics_threshold_0_5": metrics_05,
            "metrics_oof_threshold": metrics_thr,
            "curves": curves,
            "thresholds": threshold_table(y_test, y_score),
            "oof_threshold": oof_thr,
            "cv": cv_stats,
            "best_params": _jsonify(search.best_params_),
            "inner_cv_roc_auc": float(search.best_score_),
        }
        joblib.dump(best_pipe, MODELS_DIR / f"{_slug(name)}.joblib")
        print(f"  inner ROC-AUC={search.best_score_:.4f}  test ROC-AUC={metrics_05['roc_auc']:.4f}")

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["inner_cv_roc_auc", "composite"],
        ascending=False,
    )
    best_name = str(comparison.iloc[0]["model"])
    best_pipe = fitted[best_name]
    best_eval = eval_store["models"][best_name]
    best_scores = np.asarray(best_eval["y_score"])
    cal = calibration_payload(y_test, best_scores)

    perm = permutation_table(best_pipe, x_test, y_test)
    native = native_feature_importance(best_pipe)
    shap_payload = shap_explanations(best_pipe, x_test)

    _plot_curves(
        all_curves,
        FIGURES_DIR / "roc_curves.png",
        FIGURES_DIR / "pr_curves.png",
        float(np.mean(y_test)),
    )
    _plot_calibration(cal, FIGURES_DIR / "calibration.png")
    _plot_importance(perm, "importance_mean", "Permutation importance (test ROC-AUC drop)", FIGURES_DIR / "permutation_importance.png")
    if native is not None:
        _plot_importance(native, "importance", "Native model importance", FIGURES_DIR / "native_importance.png")
        native.to_csv(REPORTS_DIR / "native_importance.csv", index=False)
    perm.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(REPORTS_DIR / "cv_results.csv", index=False)

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako", ax=ax, square=True)
    ax.set_title("Feature correlation (full cleaned table)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "best_model_name": best_name,
        "random_state": RANDOM_STATE,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "features": list(x_train.columns),
        "threshold_default": 0.5,
        "threshold_oof_f1": best_eval["oof_threshold"]["threshold"],
        "calibration": cal,
        "selection_rule": (
            "Hyperparameters and the deployed algorithm are chosen from training-set inner "
            "stratified ROC-AUC only. A composite of ROC-AUC, PR-AUC, F1 and recall is stored "
            "for reporting; it is not used to pick the saved model, so the test set stays isolated "
            "from model selection. Accuracy is reported but is not a selection criterion."
        ),
        "train_stats": train_stats,
        "xgboost_available": _maybe_xgboost() is not None,
        "disclaimer": (
            "ENDOCARE is an academic prototype. Outputs are experimental model scores, not diagnoses. "
            "A low score does not rule out endometriosis."
        ),
    }

    joblib.dump(best_pipe, MODELS_DIR / "best_pipeline.joblib")
    joblib.dump(
        {
            "pipeline": best_pipe,
            "metadata": metadata,
        },
        MODELS_DIR / "endocare_bundle.joblib",
    )

    shap_for_json = None
    if shap_payload:
        shap_for_json = {k: v for k, v in shap_payload.items() if k not in {"shap_values", "data", "raw_sample"}}
        joblib.dump(shap_payload, REPORTS_DIR / "shap_payload.joblib")

    reports = {
        "audit": audit,
        "metadata": metadata,
        "comparison": comparison.to_dict(orient="records"),
        "evaluation": eval_store,
        "calibration": cal,
        "shap_summary": shap_for_json,
        "best_model": best_name,
        "limitations": [
            "The dataset provenance, label generation process and feature codebooks are not verified.",
            "Performance estimates describe this file under a random stratified split, not prospective clinical use.",
            "Six tabular features cannot capture the clinical heterogeneity of endometriosis.",
            "Calibration was assessed after model selection; treat scores conservatively.",
            "SHAP and permutation importance describe model associations, not biological causation.",
        ],
    }
    _write_json(REPORTS_DIR / "training_report.json", reports)
    _write_json(REPORTS_DIR / "data_audit.json", audit)
    _write_json(MODELS_DIR / "metadata.json", metadata)

    print(f"Selected model: {best_name}")
    print(f"Test ROC-AUC: {eval_store['models'][best_name]['metrics_threshold_0_5']['roc_auc']:.4f}")
    print(f"Calibration ECE: {cal['ece']:.4f} -> {cal['output_label']}")
    return reports


def _grid_size(params: dict) -> int:
    size = 1
    for values in params.values():
        size *= max(len(list(values)), 1)
    return size


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None:
        return None
    return obj


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_jsonify), encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run_training()
