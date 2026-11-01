"""Inference using the saved preprocessing pipeline and selected model."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np

from src.config import FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR
from src.explainability import local_prediction_contributions
from src.preprocessing import clip_warning, to_model_frame


class ModelNotFoundError(FileNotFoundError):
    pass


def load_bundle() -> dict[str, Any]:
    bundle_path = MODELS_DIR / "endocare_bundle.joblib"
    pipeline_path = MODELS_DIR / "best_pipeline.joblib"
    meta_path = MODELS_DIR / "metadata.json"
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
    elif bundle_path.exists():
        pipeline = joblib.load(bundle_path)["pipeline"]
    else:
        raise ModelNotFoundError(
            "No trained pipeline found. Run `python -m src.train_models` from the project root."
        )
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"pipeline": pipeline, "metadata": metadata}


def predict_record(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = load_bundle()
    pipeline = bundle["pipeline"]
    metadata = bundle["metadata"]
    missing = [c for c in FEATURE_COLUMNS if c not in payload]
    if missing:
        raise ValueError(f"Missing features required by the model: {missing}")

    frame = to_model_frame(payload)
    score = float(pipeline.predict_proba(frame)[0, 1])
    threshold = float(metadata.get("threshold_default", 0.5))
    label = int(score >= threshold)
    calibration = metadata.get("calibration", {})
    score_name = calibration.get("output_label", "model score (not a calibrated probability)")
    well_calibrated = bool(calibration.get("scores_are_well_calibrated", False))

    contributions = local_prediction_contributions(pipeline, frame)
    ood = clip_warning(payload, metadata.get("train_stats", {}))

    band = _score_band(score)
    return {
        "model_score": score,
        "score_name": score_name,
        "well_calibrated": well_calibrated,
        "threshold_used": threshold,
        "predicted_class_at_threshold": label,
        "class_name": "higher model score class (label 1 in the training file)" if label else "lower model score class (label 0 in the training file)",
        "band": band,
        "contributions": contributions,
        "ood_warnings": ood,
        "model_name": metadata.get("best_model_name", "unknown"),
        "ece": calibration.get("ece"),
        "disclaimer": (
            "This is an experimental estimate from a research prototype trained on a structured table. "
            "It is not a diagnosis, screening decision, or clinical probability. "
            "A lower score does not rule out endometriosis. Please consult a qualified clinician for symptoms or concerns."
        ),
    }


def _score_band(score: float) -> dict[str, str]:
    if score < 0.35:
        key, blurb = "lower", "The model assigned a lower score relative to this training file."
    elif score < 0.55:
        key, blurb = "mid", "The model assigned a mid-range score. Overlap between classes is expected."
    else:
        key, blurb = "higher", "The model assigned a higher score relative to this training file."
    return {
        "key": key,
        "title": f"{key.title()}-range model score",
        "blurb": blurb
        + " Class overlap is substantial, so individuals with and without the training label can receive similar scores.",
    }


def load_training_report() -> dict[str, Any]:
    path = REPORTS_DIR / "training_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
