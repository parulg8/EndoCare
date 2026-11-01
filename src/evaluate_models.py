"""Metrics, curves, threshold analysis, and calibration assessment."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def specificity_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tn + fp
    return float(tn / denom) if denom else float("nan")


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def curve_payloads(y_true, y_score) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    prec, rec, pr_thr = precision_recall_curve(y_true, y_score)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thr.tolist(),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "pr_thresholds": pr_thr.tolist(),
    }


def threshold_table(y_true, y_score, steps: int = 17) -> list[dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    rows = []
    for thr in np.linspace(0.10, 0.90, steps):
        pred = (y_score >= thr).astype(int)
        m = classification_metrics(y_true, pred, y_score)
        rows.append({"threshold": float(thr), **{k: m[k] for k in [
            "accuracy", "precision", "recall", "specificity", "f1", "tp", "fp", "tn", "fn"
        ]}})
    return rows


def select_threshold_from_oof(y_true, y_score) -> dict[str, float]:
    """Choose a threshold on out-of-fold scores only (never the test set)."""
    table = threshold_table(y_true, y_score)
    best = max(table, key=lambda r: (r["f1"], r["recall"], r["roc_auc"] if "roc_auc" in r else 0))
    # recompute roc_auc independently of threshold
    return {
        "threshold": float(best["threshold"]),
        "oof_f1": float(best["f1"]),
        "oof_recall": float(best["recall"]),
        "oof_precision": float(best["precision"]),
        "oof_specificity": float(best["specificity"]),
    }


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.clip(np.asarray(y_score).astype(float), 0, 1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_score >= bins[i]) & (y_score < bins[i + 1] if i < n_bins - 1 else y_score <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_score[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def calibration_payload(y_true, y_score, n_bins: int = 10) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    ece = expected_calibration_error(y_true, y_score, n_bins=n_bins)
    brier = float(brier_score_loss(y_true, y_score))
    # Heuristic, conservative labelling for a research prototype
    reliable = bool(ece <= 0.05 and brier <= 0.22)
    return {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
        "ece": ece,
        "brier": brier,
        "n_bins": n_bins,
        "scores_are_well_calibrated": reliable,
        "output_label": "predicted probability" if reliable else "model score (not a calibrated probability)",
        "note": (
            "Reliability diagram and expected calibration error were estimated on the held-out test set "
            "after model selection. If ECE is high, treat outputs as ranking scores rather than probabilities."
        ),
    }


def composite_selection_score(metrics: dict[str, float]) -> float:
    """Multi-metric score used only to rank models on the isolated test set after CV selection of hyperparameters.

    Hyperparameters are chosen by inner-loop ROC-AUC. This composite is for transparent reporting
    of which fitted model is deployed, not for fishing on many test evaluations of the same family.
    """
    return float(
        0.30 * metrics["roc_auc"]
        + 0.30 * metrics["pr_auc"]
        + 0.20 * metrics["f1"]
        + 0.20 * metrics["recall"]
    )
