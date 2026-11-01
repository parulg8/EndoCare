"""Technical overview page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import REPORTS_DIR
from src.ui.common import apply_layout, disclaimer_box, load_report, models_ready, section_header


METRIC_LABELS = [
    ("test_roc_auc", "ROC-AUC"),
    ("test_pr_auc", "PR-AUC"),
    ("test_f1", "F1"),
    ("test_recall", "Recall / sensitivity"),
    ("test_specificity", "Specificity"),
    ("test_precision", "Precision"),
    ("test_accuracy", "Accuracy"),
]


def render() -> None:
    section_header(
        "Technology",
        "About the Technology",
        "A compact overview of the project’s pipeline, model comparison and explainability output.",
    )
    disclaimer_box("The figures below reflect the saved project reports and training artefacts. They are educational outputs, not clinical evidence.")

    if not models_ready():
        st.error("Training reports were not found. Run `python run_training.py` from the project root.")
        return

    report = load_report()
    comparison = pd.DataFrame(report.get("comparison", []))
    evaluation = report.get("evaluation", {}).get("models", {})
    best = report.get("best_model")
    meta = report.get("metadata", {})

    st.markdown(
        """
        <div class="soft-card">
          <h3>What is included</h3>
          <p>The project compares several conventional classifiers, stores the selected model and preprocessing pipeline, and reports how those outputs behaved on a held-out test set.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Overview", "Performance", "Explainability"])

    with tabs[0]:
        st.write("**Algorithms used:** Logistic Regression, Random Forest, Support Vector Machine, Gradient Boosting and XGBoost when available.")
        st.write("**Approach:** the model uses the six structured features in the training file, and the saved pipeline is reused for live inference on the project website.")
        if not meta.get("xgboost_available", True):
            st.warning("XGBoost was unavailable in this environment, so it was skipped during training.")

        if not comparison.empty:
            show_cols = ["model", "inner_cv_roc_auc", "composite"] + [c for c, _ in METRIC_LABELS if c in comparison.columns]
            pretty = comparison[show_cols].copy()
            st.dataframe(pretty, hide_index=True, width="stretch")

    with tabs[1]:
        if not comparison.empty:
            melt = comparison.melt(
                id_vars=["model"],
                value_vars=[c for c, _ in METRIC_LABELS if c in comparison.columns],
                var_name="metric",
                value_name="value",
            )
            fig = go.Figure()
            for metric, label in METRIC_LABELS:
                sub = melt[melt["metric"] == metric]
                fig.add_bar(x=sub["model"], y=sub["value"], name=label)
            fig.update_layout(barmode="group", yaxis_title="Score")
            st.plotly_chart(apply_layout(fig, "Test-set metrics by model"), width="stretch")

            roc = go.Figure()
            pr = go.Figure()
            prevalence = None
            y_test = report.get("evaluation", {}).get("y_test", [])
            if y_test:
                prevalence = float(np.mean(y_test))
            for name, payload in evaluation.items():
                curves = payload.get("curves", {})
                roc.add_trace(go.Scatter(x=curves.get("fpr"), y=curves.get("tpr"), mode="lines", name=name))
                pr.add_trace(go.Scatter(x=curves.get("recall"), y=curves.get("precision"), mode="lines", name=name))
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#65756C")))
            if prevalence is not None:
                pr.add_hline(y=prevalence, line_dash="dash", annotation_text="Prevalence")
            st.plotly_chart(apply_layout(roc, "ROC curves"), width="stretch")
            st.plotly_chart(apply_layout(pr, "Precision–recall curves"), width="stretch")

    with tabs[2]:
        perm_path = REPORTS_DIR / "permutation_importance.csv"
        native_path = REPORTS_DIR / "native_importance.csv"
        if perm_path.exists():
            perm = pd.read_csv(perm_path)
            fig = go.Figure(go.Bar(x=perm["importance_mean"], y=perm["feature"], orientation="h", error_x=dict(type="data", array=perm["importance_std"])))
            st.plotly_chart(apply_layout(fig, "Permutation importance"), width="stretch")
        if native_path.exists():
            native = pd.read_csv(native_path)
            fig = go.Figure(go.Bar(x=native["importance"], y=native["feature"], orientation="h"))
            st.plotly_chart(apply_layout(fig, "Native importance"), width="stretch")

        shap_summary = report.get("shap_summary") or {}
        if shap_summary.get("available"):
            sdf = pd.DataFrame(shap_summary.get("mean_abs_shap", []))
            if not sdf.empty:
                fig = go.Figure(go.Bar(x=sdf["mean_abs_shap"], y=sdf["feature"], orientation="h"))
                st.plotly_chart(apply_layout(fig, "Mean absolute SHAP value"), width="stretch")
        st.caption("Feature importance is interpreted as model association, not proof of causation or diagnosis.")

    st.markdown("### Model selection summary")
    if best:
        st.write(f"The saved model is **{best}**. The saved pipeline is reused in the risk assessment page.")
    cal = meta.get("calibration", {})
    if cal:
        st.write(f"Calibration summary: ECE={cal.get('ece', float('nan')):.3f}, Brier={cal.get('brier', float('nan')):.3f}. The score output is labelled as **{cal.get('output_label', 'experimental model score')}**.")
    st.write("This is a research interface, not a medical decision aid.")
