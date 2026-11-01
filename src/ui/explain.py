"""Global and local explainability."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import REPORTS_DIR
from src.prediction import ModelNotFoundError, predict_record
from src.ui.common import apply_layout, disclaimer_box, load_report, models_ready, section_header


def render() -> None:
    section_header(
        "Interpretability",
        "Explainability",
        "Importance measures say which inputs the model relied on statistically. They do not establish medical causation.",
    )
    disclaimer_box()

    if not models_ready():
        st.error("Train the models first (`python run_training.py`) to populate explanation artefacts.")
        return

    report = load_report()
    best = report.get("best_model")
    st.write(f"Explanations below refer to the saved model: **{best}**.")

    perm_path = REPORTS_DIR / "permutation_importance.csv"
    native_path = REPORTS_DIR / "native_importance.csv"
    c1, c2 = st.columns(2)
    if perm_path.exists():
        perm = pd.read_csv(perm_path)
        fig = px.bar(perm, x="importance_mean", y="feature", error_x="importance_std", orientation="h")
        c1.plotly_chart(apply_layout(fig, "Permutation importance (test ROC-AUC drop)"), width="stretch")
    if native_path.exists():
        native = pd.read_csv(native_path)
        fig = px.bar(native, x="importance", y="feature", orientation="h")
        c2.plotly_chart(apply_layout(fig, f"Native importance ({native['kind'].iloc[0] if len(native) else ''})"), width="stretch")

    shap_summary = report.get("shap_summary") or {}
    st.markdown("### SHAP global view")
    if shap_summary.get("available"):
        sdf = pd.DataFrame(shap_summary.get("mean_abs_shap", []))
        if not sdf.empty:
            fig = px.bar(sdf, x="mean_abs_shap", y="feature", orientation="h")
            st.plotly_chart(apply_layout(fig, f"Mean |SHAP| ({shap_summary.get('method')}, n={shap_summary.get('n_samples')})"), width="stretch")
        if shap_summary.get("note"):
            st.caption(shap_summary["note"])
    else:
        st.write(
            "SHAP was unavailable or failed for the selected estimator. Permutation importance remains the primary model-agnostic view. "
            f"{shap_summary.get('error', '')}"
        )

    st.markdown("### Individual prediction explanation")
    st.write("Enter one hypothetical record using the same six training features.")
    cols = st.columns(3)
    age = cols[0].number_input("Age", 18, 55, 30)
    pain = cols[1].number_input("Chronic pain level", 0.0, 10.0, 6.5, step=0.1)
    bmi = cols[2].number_input("BMI", 14.0, 42.0, 24.0, step=0.1)
    cols = st.columns(3)
    irr = cols[0].selectbox("Menstrual irregularity", [0, 1], index=1, key="ex_irr")
    horm = cols[1].selectbox("Hormone abnormality", [0, 1], index=1, key="ex_h")
    inf = cols[2].selectbox("Infertility", [0, 1], index=0, key="ex_i")
    if st.button("Explain this record"):
        try:
            result = predict_record(
                {
                    "Age": age,
                    "Menstrual_Irregularity": irr,
                    "Chronic_Pain_Level": pain,
                    "Hormone_Level_Abnormality": horm,
                    "Infertility": inf,
                    "BMI": bmi,
                }
            )
        except ModelNotFoundError as exc:
            st.error(str(exc))
            return
        st.write(f"Model score: **{result['model_score']:.3f}** ({result['score_name']})")
        cdf = pd.DataFrame(result["contributions"].get("contributions", []))
        if not cdf.empty:
            fig = px.bar(cdf, x="contribution", y="feature", orientation="h")
            st.plotly_chart(apply_layout(fig, "Local contributions"), width="stretch")
        st.warning(result["disclaimer"])

    st.markdown("### How to read these plots")
    st.write(
        """
        - **Permutation importance** estimates how much test ROC-AUC falls when a column is shuffled.
        - **Native importance** is impurity/gain for trees or absolute coefficients for linear models.
        - **SHAP** attributes a score change to each feature for sampled rows.
        - None of these methods prove that changing a symptom in real life would change disease status.
        - Overlapping classes mean even “important” features still leave substantial uncertainty.
        """
    )
