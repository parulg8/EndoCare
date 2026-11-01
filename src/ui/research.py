"""About ENDOCARE page."""

from __future__ import annotations

import streamlit as st

from src.ui.common import disclaimer_box, section_header


def render() -> None:
    section_header(
        "About ENDOCARE",
        "About ENDOCARE",
        "A short overview of the project’s purpose, its research context and the boundaries of what it can and cannot do.",
    )
    disclaimer_box()

    st.markdown(
        """
        <div class="soft-card">
          <h3>Purpose</h3>
          <p>ENDOCARE was built to help people understand endometriosis symptoms in a simple, respectful way and to explore an experimental model-based risk assessment using the project’s saved training artefacts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What this project is")
    st.write(
        "- A student research project exploring whether a small, structured dataset can support a model-based assessment."
    )
    st.write(
        "- A patient-facing educational experience designed to improve clarity and reduce uncertainty around symptoms."
    )
    st.write(
        "- A demonstration of how a trained pipeline and explanation outputs can be presented in a user-friendly interface."
    )

    st.markdown("### What this project is not")
    st.write(
        "- It is not a diagnosis or a clinical screening tool."
    )
    st.write(
        "- It is not a replacement for a gynaecologist, primary care clinician or specialist assessment."
    )
    st.write(
        "- A lower model score does not rule out endometriosis or other causes of pain."
    )

    st.markdown("### Dataset-based approach")
    st.write(
        "The project uses the existing structured file and saved model pipeline from this repository. This means the interface remains grounded in the project’s actual data and training outputs, while clearly separating educational questions from model input."
    )

    st.markdown("### Limits and medical disclaimer")
    st.write(
        "This website is designed for research and awareness. It cannot assess a person’s individual health status, and it cannot replace professional evaluation for symptoms like chronic pelvic pain, painful periods, infertility concerns or changes in bowel or bladder habits."
    )

    st.warning(
        "If symptoms are persistent, worsening or affecting daily life, speak with a qualified healthcare professional. Seek urgent care for severe abdominal pain, fainting, heavy bleeding or sudden worsening of symptoms."
    )
