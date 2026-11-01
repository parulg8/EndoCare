"""Experimental risk assessment form — model features only."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.prediction import ModelNotFoundError, predict_record
from src.ui.common import apply_layout, disclaimer_box, load_report, section_header


def render() -> None:
    section_header(
        "Inference",
        "Risk Assessment",
        "This form uses the same six fields the saved project pipeline was trained on. It does not include the symptom questionnaire items from the previous page.",
    )
    disclaimer_box(
        "The output is an experimental model score from a research prototype. It is not a diagnosis, not a medical test result, and it does not rule out endometriosis."
    )

    with st.form("risk_form"):
        st.caption("Enter the information the model expects. These fields are based on the saved training schema.")
        c1, c2 = st.columns(2)
        age = c1.slider("Age (years)", min_value=18, max_value=55, value=32)
        bmi = c2.slider("BMI", min_value=14.0, max_value=42.0, value=23.0, step=0.1)
        pain = st.slider("Chronic pain level (0–10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        r1, r2, r3 = st.columns(3)
        irregular = r1.selectbox("Menstrual irregularity", options=[0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")
        hormone = r2.selectbox("Hormone-level abnormality", options=[0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")
        infert = r3.selectbox("Infertility", options=[0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")
        submitted = st.form_submit_button("Analyse my responses")

    if not submitted:
        st.markdown(
            """
            <div class="soft-card">
              <h3>Before you run the assessment</h3>
              <p>Binary input fields follow the 0/1 coding used in the project dataset. The model was trained on this structure and cannot use the symptom questionnaire on the previous page.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    payload = {
        "Age": int(age),
        "Menstrual_Irregularity": int(irregular),
        "Chronic_Pain_Level": float(pain),
        "Hormone_Level_Abnormality": int(hormone),
        "Infertility": int(infert),
        "BMI": float(bmi),
    }
    try:
        result = predict_record(payload)
    except ModelNotFoundError as exc:
        st.error(str(exc))
        return

    score = result["model_score"]
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    a, b = st.columns([0.9, 1.1])
    with a:
        st.markdown(f'<p class="result-score">{score:.3f}</p>', unsafe_allow_html=True)
        st.caption(result["score_name"])
        st.write(f"**Band:** {result['band']['title']}")
        st.write(result["band"]["blurb"])
        st.write(f"Model: {result['model_name']} · Decision threshold: {result['threshold_used']:.2f}")
        st.write(f"Class at that threshold: {result['class_name']}")
        if not result["well_calibrated"]:
            st.warning(
                "This model output should be interpreted as a research score, not a calibrated probability."
            )
        else:
            st.info("This was labelled as a usable score for demonstrating model behaviour, but it remains non-clinical.")
    with b:
        st.success(result["disclaimer"])
        for warn in result["ood_warnings"]:
            st.error(warn)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### What influenced this estimate?")
    contrib = result.get("contributions", {})
    rows = contrib.get("contributions", [])
    if rows:
        cdf = pd.DataFrame(rows)
        fig = px.bar(cdf, x="contribution", y="feature", orientation="h")
        st.plotly_chart(apply_layout(fig, f"Local explanation ({contrib.get('method')})"), width="stretch")
        st.caption(
            "Positive bars pushed the model toward a higher score; negative bars pushed it lower. These are associations in the project data, not proof of causation."
        )
        if contrib.get("note"):
            st.write(contrib["note"])
    else:
        st.write("A local breakdown is not available for this estimator.")

    report = load_report()
    cal = report.get("metadata", {}).get("calibration", {})
    if cal:
        st.caption(f"Reference values: test ECE={cal.get('ece', float('nan')):.3f}, Brier={cal.get('brier', float('nan')):.3f}.")
