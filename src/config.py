"""Project paths, feature schema, and reproducible experiment settings."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_CV_FOLDS = 3
SEARCH_ITER = 12
SHAP_SAMPLE_SIZE = 400
PERMUTATION_REPEATS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "structured_endometriosis_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = PROJECT_ROOT / "assets"

TARGET = "Diagnosis"
FEATURE_COLUMNS = [
    "Age",
    "Menstrual_Irregularity",
    "Chronic_Pain_Level",
    "Hormone_Level_Abnormality",
    "Infertility",
    "BMI",
]
NUMERIC_FEATURES = ["Age", "Chronic_Pain_Level", "BMI"]
BINARY_FEATURES = [
    "Menstrual_Irregularity",
    "Hormone_Level_Abnormality",
    "Infertility",
]

FEATURE_LABELS = {
    "Age": "Age (years)",
    "Menstrual_Irregularity": "Menstrual irregularity",
    "Chronic_Pain_Level": "Chronic pain level",
    "Hormone_Level_Abnormality": "Hormone-level abnormality",
    "Infertility": "Infertility",
    "BMI": "Body mass index (BMI)",
}

FEATURE_DESCRIPTIONS = {
    "Age": "Recorded age in years. In this file the observed range is 18–49.",
    "Menstrual_Irregularity": "Binary indicator (0/1). The file does not define coding rules, cycle-length cut-offs, or how irregularity was assessed.",
    "Chronic_Pain_Level": "Continuous value observed between 0 and 10. The instrument, recall window, and whether this is pelvic pain specifically are not documented.",
    "Hormone_Level_Abnormality": "Binary indicator (0/1). The hormone, assay, reference range, and abnormality rule are not documented.",
    "Infertility": "Binary indicator (0/1). Duration of trying to conceive, clinical definition, and self-report versus diagnosis are not documented.",
    "BMI": "Body mass index. Units are not stated in the file; values are consistent with kg/m² and appear clipped at 15.0.",
    "Diagnosis": "Binary label (0/1). The file does not state whether this is laparoscopic confirmation, clinical coding, self-report, or a generated label.",
}

POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0
