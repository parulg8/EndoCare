# ENDOCARE

**Your Pain Deserves to Be Understood.**

Academic research prototype: **Machine Learning-Based Endometriosis Risk Assessment and Symptom Analysis.**

ENDOCARE explores whether conventional machine learning can recover patterns associated with a binary endometriosis-related label in a structured table. It is **not a medical device, screening programme, or diagnostic tool**. A lower model score does **not** rule out endometriosis.

## What is in this folder

| Path | Role |
| --- | --- |
| `app.py` | Streamlit website |
| `run_training.py` | Train, evaluate, explain, and save artefacts |
| `src/` | Preprocessing, models, metrics, explainability, inference, UI pages |
| `data/` | Place `structured_endometriosis_data.csv` here |
| `models/` | Saved pipelines and metadata (created by training) |
| `reports/` | Metrics, audits, figures |
| `assets/` | Branding CSS and logo |
| `tests/` | Unit tests |
| `notebooks/` | Optional EDA notes |

## Dataset

Expected file:

```text
data/structured_endometriosis_data.csv
```

Expected columns: `Age`, `Menstrual_Irregularity`, `Chronic_Pain_Level`, `Hormone_Level_Abnormality`, `Infertility`, `BMI`, `Diagnosis`.

During development this file had 10,000 rows, no missing cells, no exact duplicate rows, and a Diagnosis=1 rate of about 0.41. Those facts are **file-specific**. Collection protocol, label generation, hormone definitions, and licence were **not documented in the CSV**. Treat provenance as uncertain. The packaged ZIP omits the CSV because redistribution rights are unclear; copy the file into `data/` yourself.

## Environment

Python 3.10+ recommended (developed with 3.12). From the project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train models

```bash
python run_training.py
```

This writes:

- `models/best_pipeline.joblib` — preprocessing + selected estimator
- `models/metadata.json` — split sizes, calibration, selection rule
- `reports/training_report.json` — comparison, curves, audit
- `reports/figures/` — ROC, PR, confusion, calibration, importance plots

The website reads these files. It does not retrain on page load.

## Launch the website

```bash
streamlit run app.py
```

Open the URL printed in the terminal (typically `http://localhost:8501`).

Pages:
1. Home
2. Understand Your Symptoms
3. Risk Assessment
4. About the Technology
5. About ENDOCARE

The **Symptom Awareness** questionnaire is educational only and is **not** passed into the model.

## Tests

```bash
python -m pytest -q
```

## Design of the ML pipeline

1. Schema check and numeric coercion.
2. Drop missing targets/features and duplicate rows.
3. Stratified 80/20 train–test split (`random_state=42`).
4. Inner stratified randomised search (ROC-AUC) on **training data only**.
5. Five-fold stratified CV on training data for reporting.
6. Single test-set evaluation per algorithm.
7. Operating threshold 0.50 for the primary table; an F1 threshold is chosen from out-of-fold **training** scores.
8. Calibration (ECE, Brier, reliability diagram) on the test scores of the selected model. If calibration is poor, the UI labels the output as a **model score**, not a probability.
9. Permutation importance and SHAP (when compatible).

Algorithms: Logistic Regression, Random Forest, linear SVM with `CalibratedClassifierCV`, Gradient Boosting, XGBoost if installed.

## Medical boundary

- Do not use ENDOCARE for diagnosis, triage, or self-treatment.
- Do not interpret a low score as absence of disease.
- Feature importance is association inside a model, not causation.

## Recreate the ZIP

```bash
python package_project.py
```

Creates `ENDOCARE.zip` in the parent directory of this folder, excluding virtual environments, caches, and the dataset CSV.
