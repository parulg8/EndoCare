from pathlib import Path

import pytest

from src.config import FEATURE_COLUMNS, MODELS_DIR
from src.prediction import ModelNotFoundError, predict_record


def test_saved_pipeline_predicts_expected_schema():
    if not (MODELS_DIR / "best_pipeline.joblib").exists():
        pytest.skip("Trained model artefacts are not present yet.")
    result = predict_record(
        {
            "Age": 28,
            "Menstrual_Irregularity": 1,
            "Chronic_Pain_Level": 7.4,
            "Hormone_Level_Abnormality": 1,
            "Infertility": 0,
            "BMI": 24.1,
        }
    )
    assert 0.0 <= result["model_score"] <= 1.0
    assert "disclaimer" in result
    assert "not a diagnosis" in result["disclaimer"].lower() or "not a diagnosis" in result["disclaimer"]
    assert "does not rule out" in result["disclaimer"].lower() or "not rule" in result["disclaimer"].lower()
    assert result["model_name"]
    for col in FEATURE_COLUMNS:
        assert col in {
            "Age",
            "Menstrual_Irregularity",
            "Chronic_Pain_Level",
            "Hormone_Level_Abnormality",
            "Infertility",
            "BMI",
        }


def test_missing_model_message(tmp_path, monkeypatch):
    monkeypatch.setattr("src.prediction.MODELS_DIR", Path(tmp_path))
    with pytest.raises(ModelNotFoundError):
        predict_record(
            {
                "Age": 28,
                "Menstrual_Irregularity": 1,
                "Chronic_Pain_Level": 7.4,
                "Hormone_Level_Abnormality": 1,
                "Infertility": 0,
                "BMI": 24.1,
            }
        )
