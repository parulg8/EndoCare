"""Dataset explorer."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import BINARY_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from src.ui.common import apply_layout, disclaimer_box, feature_help, load_audit, load_cleaned_frame, section_header


def render() -> None:
    section_header("Data", "Dataset Explorer", "Inspection of the file actually present on disk. Nothing here is invented.")
    disclaimer_box("Feature names are taken from the CSV header, not from a clinical codebook.")

    df = load_cleaned_frame()
    audit = load_audit()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{audit.get('n_rows', len(df)):,}")
    m2.metric("Columns", audit.get("n_columns", df.shape[1]))
    m3.metric("Missing cells", int(sum(audit.get("missing_by_column", {}).values())))
    m4.metric("Duplicate rows", audit.get("duplicate_rows", 0))
    pos = audit.get("positive_rate")
    m5.metric("Label 1 rate", f"{pos:.1%}" if pos is not None else "—")

    st.markdown("### Provenance and uncertainty")
    prov = audit.get("provenance", {})
    st.info(prov.get("source_claim", "Local structured table."))
    for item in prov.get("unknowns", []):
        st.write(f"- {item}")

    st.markdown("### Feature descriptions")
    st.markdown(feature_help())

    st.markdown("### Missing values and duplicates")
    miss = pd.DataFrame(
        {
            "column": list(audit.get("missing_by_column", {}).keys()),
            "missing": list(audit.get("missing_by_column", {}).values()),
        }
    )
    st.dataframe(miss, hide_index=True, width="stretch")
    st.caption(
        f"Exact duplicate rows: {audit.get('duplicate_rows', 0)}. "
        f"BMI exactly 15.0: {audit.get('bmi_equals_15', 'n/a')} rows (possible lower clip). "
        f"Pain 0.0 / 10.0: {audit.get('pain_equals_0')} / {audit.get('pain_equals_10')}."
    )

    st.markdown("### Class distribution")
    counts = audit.get("class_counts", {})
    if counts:
        cdf = pd.DataFrame({"Diagnosis": list(counts.keys()), "count": list(counts.values())})
        fig = px.bar(cdf, x="Diagnosis", y="count", text="count")
        st.plotly_chart(apply_layout(fig, "Diagnosis label counts"), width="stretch")
    st.write(
        "The positive class is smaller but not rare. Stratified splitting is used so both "
        "labels keep their proportions in train and test."
    )

    st.markdown("### Interactive feature distributions")
    feature = st.selectbox("Feature", FEATURE_COLUMNS, index=2)
    colored = st.checkbox("Split by Diagnosis label", value=True)
    if feature in NUMERIC_FEATURES:
        fig = px.histogram(
            df,
            x=feature,
            color=TARGET if colored else None,
            nbins=40,
            barmode="overlay",
            opacity=0.72,
        )
        st.plotly_chart(apply_layout(fig, f"Distribution of {feature}"), width="stretch")
        fig2 = px.box(df, x=TARGET if colored else None, y=feature, points="outliers")
        st.plotly_chart(apply_layout(fig2, f"{feature} by label" if colored else feature), width="stretch")
    else:
        tmp = df.groupby([feature, TARGET]).size().reset_index(name="count")
        fig = px.bar(tmp, x=feature, y="count", color=TARGET, barmode="group")
        st.plotly_chart(apply_layout(fig, f"{feature} counts by Diagnosis"), width="stretch")

    st.markdown("### Correlation heatmap")
    corr = df[FEATURE_COLUMNS + [TARGET]].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Tealgrn", aspect="auto")
    st.plotly_chart(apply_layout(fig, "Pearson correlation"), width="stretch")

    st.markdown("### Label rates for binary features")
    rate_rows = []
    for col in BINARY_FEATURES:
        for val, rate in audit.get("binary_positive_rates", {}).get(col, {}).items():
            rate_rows.append({"feature": col, "value": val, "diagnosis_rate": rate})
    if rate_rows:
        rdf = pd.DataFrame(rate_rows)
        fig = px.bar(rdf, x="feature", y="diagnosis_rate", color="value", barmode="group")
        st.plotly_chart(apply_layout(fig, "Empirical P(Diagnosis=1 | feature value)"), width="stretch")

    st.markdown("### Observations from this file")
    for line in audit.get("observations", []):
        st.write(f"- {line}")
    st.markdown("### Leakage check")
    for line in audit.get("leakage_notes", []):
        st.write(f"- {line}")

    with st.expander("Preview first 25 rows"):
        st.dataframe(df.head(25), width="stretch")
    with st.expander("Descriptive statistics"):
        st.dataframe(df.describe(), width="stretch")
