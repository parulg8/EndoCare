"""Home page."""

import streamlit as st

from src.ui.common import disclaimer_box


def render() -> None:
    if st.session_state.get("home_anchor") == "endometriosis":
        st.components.v1.html(
            """
            <script>
            const target = document.getElementById('endometriosis-section');
            if (target) {
              target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }
            </script>
            """,
            height=0,
            scrolling=False,
        )
        st.session_state.home_anchor = None

    st.markdown(
        """
        <div class="hero">
          <div class="brand-kicker">Women's health platform</div>
          <h1>ENDOCARE</h1>
          <div class="tagline">Your Pain Deserves to Be Understood.</div>
          <p>
            Endometriosis can cause pelvic pain, painful periods and other symptoms that are
            often dismissed or misunderstood. ENDOCARE helps people explore symptom patterns,
            understand the condition, and look at an experimental model-based assessment in a
            supportive, non-diagnostic way.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    disclaimer_box()

    st.markdown("### A clearer starting point")
    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "What is endometriosis?", "It is a condition where tissue similar to the lining of the uterus grows outside it, often causing pelvic pain, painful periods and other symptoms."),
        (c2, "What does ENDOCARE do?", "It offers a symptom-awareness guide and a model-based risk assessment built from the project’s saved training artefacts."),
        (c3, "What it is not", "This is not a diagnosis, not a clinical screening tool, and not a replacement for professional medical care."),
    ]
    for col, title, text in cards:
        col.markdown(
            f'<div class="soft-card"><h3>{title}</h3><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div id='endometriosis-section'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown("### Why this matters")
        st.write(
            "Endometriosis affects many people but is often under-recognised. Symptoms can vary from person to person, and the experience of pain can be very different from one cycle to the next."
        )
        st.write(
            "The goal of ENDOCARE is simple: help people understand symptoms, make the educational pathway clearer, and present the project’s experimental model in a safer, more informed way."
        )
    with right:
        st.markdown(
            """
            <div class="soft-card">
              <h3>How to use this site</h3>
              <ol>
                <li>Explore symptoms and learn what they can mean.</li>
                <li>Try the model-based risk assessment.</li>
                <li>Read the technology and limitations.</li>
                <li>Speak with a qualified clinician if symptoms are ongoing.</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Important boundaries")
    st.markdown(
        """
        - The symptom questionnaire is educational and separate from the model.
        - The risk assessment uses the saved project features only.
        - A lower model score does not rule out endometriosis.
        - This project is for learning, research and awareness, not diagnosis.
        """
    )
