from src.config import FEATURE_COLUMNS, TARGET
from src.preprocessing import build_preprocessor, coerce_types, drop_invalid_rows, to_model_frame
import pandas as pd


def test_to_model_frame_column_order():
    payload = {
        "BMI": 22.0,
        "Age": 30,
        "Infertility": 0,
        "Menstrual_Irregularity": 1,
        "Chronic_Pain_Level": 4.2,
        "Hormone_Level_Abnormality": 1,
    }
    frame = to_model_frame(payload)
    assert list(frame.columns) == FEATURE_COLUMNS


def test_drop_invalid_and_duplicates():
    df = pd.DataFrame(
        {
            "Age": [20, 20, None, 22],
            "Menstrual_Irregularity": [0, 0, 1, 1],
            "Chronic_Pain_Level": [1.0, 1.0, 2.0, 3.0],
            "Hormone_Level_Abnormality": [0, 0, 0, 1],
            "Infertility": [0, 0, 0, 0],
            "BMI": [21.0, 21.0, 22.0, 23.0],
            "Diagnosis": [0, 0, 1, 2],
        }
    )
    cleaned, stats = drop_invalid_rows(coerce_types(df))
    assert TARGET in cleaned.columns
    assert stats["dropped_duplicates"] >= 1
    assert stats["dropped_missing"] >= 1
    assert stats["dropped_invalid_target"] >= 1
    assert set(cleaned[TARGET].unique()).issubset({0, 1})


def test_preprocessor_output_width():
    prep = build_preprocessor()
    x = pd.DataFrame(
        [
            {
                "Age": 30,
                "Menstrual_Irregularity": 1,
                "Chronic_Pain_Level": 5.0,
                "Hormone_Level_Abnormality": 0,
                "Infertility": 0,
                "BMI": 22.0,
            }
        ]
    )
    xt = prep.fit_transform(x)
    assert xt.shape == (1, 6)
