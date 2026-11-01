"""Shared UI helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ASSETS_DIR, DATA_PATH, FEATURE_DESCRIPTIONS, REPORTS_DIR
from src.preprocessing import coerce_types, drop_invalid_rows, load_dataset

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAF5",
    font=dict(family="Source Sans 3, Segoe UI, sans-serif", color="#263A32"),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#245C49", "#173F35", "#A8C3AE", "#D98989", "#65756C", "#E7F1EA"],
    legend=dict(bgcolor="rgba(255,255,255,0.72)", borderwidth=0, font=dict(color="#263A32"), title_font=dict(color="#263A32")),
    title_font=dict(color="#173F35"),
    hoverlabel=dict(bgcolor="#FFFFFF", font=dict(color="#263A32")),
)


def inject_css() -> None:
    css = (ASSETS_DIR / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def disclaimer_box(extra: str = "") -> None:
    st.markdown(
        f"""
        <div class="health-note">
          <strong>Research prototype — not medical advice.</strong>
          ENDOCARE does not diagnose endometriosis, does not replace a clinician,
          and a lower model score never rules the condition out.
          {extra}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, text: str = "") -> None:
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><h2 class="section-title">{title}</h2>',
        unsafe_allow_html=True,
    )
    if text:
        st.caption(text)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_cleaned_frame() -> pd.DataFrame:
    df = drop_invalid_rows(coerce_types(load_dataset(DATA_PATH)))[0]
    return df


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    return load_json(REPORTS_DIR / "training_report.json", {})


@st.cache_data(show_spinner=False)
def load_audit() -> dict:
    path = REPORTS_DIR / "data_audit.json"
    data = load_json(path, {})
    if data:
        return data
    from src.data_audit import audit_file

    return audit_file(DATA_PATH)


def models_ready() -> bool:
    return (REPORTS_DIR / "training_report.json").exists()


def apply_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=title, **PLOTLY_LAYOUT)
    fig.update_xaxes(
        title_font=dict(color="#263A32"),
        tickfont=dict(color="#263A32"),
        gridcolor="#D9E3DB",
        zeroline=False,
        linecolor="#D9E3DB",
    )
    fig.update_yaxes(
        title_font=dict(color="#263A32"),
        tickfont=dict(color="#263A32"),
        gridcolor="#D9E3DB",
        zeroline=False,
        linecolor="#D9E3DB",
    )
    fig.update_annotations(font=dict(color="#263A32"))
    fig.update_traces(
        marker_line_width=0,
        textfont=dict(color="#263A32"),
        selector=dict(type="bar"),
    )
    fig.update_layout(
        hoverlabel=dict(bgcolor="#FFFFFF", font=dict(color="#263A32")),
        legend=dict(font=dict(color="#263A32"), title_font=dict(color="#263A32")),
    )
    return fig


def feature_help() -> str:
    lines = [f"**{k}.** {v}" for k, v in FEATURE_DESCRIPTIONS.items()]
    return "\n\n".join(lines)
