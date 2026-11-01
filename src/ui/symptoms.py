"""Educational symptom awareness questionnaire — not used by the ML model."""

from __future__ import annotations

import streamlit as st

from src.ui.common import disclaimer_box, section_header

QUESTIONS = [
    ("severe_dysmenorrhea", "Severe menstrual pain that limits your usual activity"),
    ("chronic_pelvic", "Chronic pelvic pain outside your period"),
    ("dyspareunia", "Pain during intercourse"),
    ("dyschezia", "Painful bowel movements, especially around menstruation"),
    ("dysuria", "Painful urination, especially around menstruation"),
    ("bloating", "Persistent bloating or abdominal fullness"),
    ("fatigue", "Fatigue that is difficult to explain by sleep alone"),
    ("family_history", "Family history of endometriosis"),
    ("life_impact", "Symptoms affecting everyday life, work or relationships"),
]


def render() -> None:
    section_header(
        "Education",
        "Understand Your Symptoms",
        "This questionnaire is educational and intentionally separate from the model. These symptoms are not part of the saved training dataset.",
    )
    disclaimer_box(
        "No medical risk score is calculated here. This is a symptom-awareness tool only, not a diagnostic index."
    )

    st.info(
        "The model-based assessment on the next page uses the project’s saved six features only. This symptom page is kept separate by design."
    )

    answers = {}
    with st.form("awareness"):
        st.write("In recent months, have you experienced the following?")
        for key, prompt in QUESTIONS:
            answers[key] = st.radio(prompt, ["No", "Unsure", "Yes"], horizontal=True, index=1, key=key)
        submitted = st.form_submit_button("Review my responses")

    if not submitted:
        st.markdown(
            """
            <div class="soft-card">
              <h3>Why keep this separate?</h3>
              <p>These lived-experience questions are meaningful, but they are not part of the project’s training file. ENDOCARE keeps symptom awareness and model assessment in different steps so the model is never asked to guess for information it did not see.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    yes = sum(v == "Yes" for v in answers.values())
    unsure = sum(v == "Unsure" for v in answers.values())

    st.markdown("### Your symptom summary")
    st.write(f"You marked **{yes}** item(s) as yes and **{unsure}** as unsure out of {len(QUESTIONS)}.")
    st.write(
        "Symptoms like painful periods, pelvic pain, pain during sex, bowel or bladder pain, bloating, fatigue, family history and symptoms affecting everyday life are commonly discussed in endometriosis care."
    )

    if yes >= 3 or (yes >= 2 and answers["life_impact"] == "Yes"):
        st.markdown(
            """
            **What this may suggest:** A pattern of recurring pain or impact on daily life can be worth discussing with a qualified healthcare professional. A clinician can help assess whether this fits endometriosis or another diagnosis with overlapping symptoms.
            """
        )
    elif yes == 0 and unsure == 0:
        st.markdown(
            """
            **What this may suggest:** You are not reporting these symptoms in this check-in. That does not rule out endometriosis or other conditions, and symptoms can develop or change over time.
            """
        )
    else:
        st.markdown(
            """
            **What this may suggest:** Your answers are mixed or uncertain. Pain and fatigue can have many causes, so a clinician is the right person to help sort through symptoms and other possibilities.
            """
        )

    st.warning(
        "This guidance is educational and general. It is not a diagnosis, a screening tool, or a medical recommendation. Seek urgent medical attention for severe abdominal pain, fainting, heavy bleeding, or sudden worsening of symptoms."
    )

    st.markdown(
        """
        Helpful questions to ask at a clinic:
        - When did the symptoms start, and do they follow a menstrual pattern?
        - What treatments have you already tried?
        - How do symptoms affect your work, sleep, intimacy or everyday life?
        """
    )
