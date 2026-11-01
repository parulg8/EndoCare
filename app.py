"""ENDOCARE Streamlit application."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import ASSETS_DIR, DATA_PATH
from src.ui import dataset, explain, home, laboratory, research, risk, symptoms
from src.ui.common import inject_css, models_ready

st.set_page_config(
    page_title="ENDOCARE — Your Pain Deserves to Be Understood",
    page_icon=str(ASSETS_DIR / "logo.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

PAGES = {
    "Home": home,
    "Understand Your Symptoms": symptoms,
    "Risk Assessment": risk,
    "About the Technology": laboratory,
    "About ENDOCARE": research,
}

page_names = list(PAGES.keys())
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"
if "last_nav_page" not in st.session_state:
    st.session_state.last_nav_page = st.session_state.nav_page

with st.sidebar:
    logo = ASSETS_DIR / "logo.svg"
    if logo.exists():
        st.image(str(logo), width=72)
    st.markdown("## ENDOCARE")
    st.caption("Your Pain Deserves to Be Understood.")
    current_page = st.session_state.nav_page
    selected_page = st.radio(
        "Navigate",
        page_names,
        index=page_names.index(current_page) if current_page in page_names else 0,
        label_visibility="collapsed",
        key="page_selector",
    )
    st.session_state.nav_page = selected_page
    st.markdown("---")
    if DATA_PATH.exists():
        st.caption("Dataset file detected in `data/`.")
    else:
        st.error("Place `structured_endometriosis_data.csv` in the `data/` folder.")
    if models_ready():
        st.caption("Saved model artefacts ready.")
    else:
        st.caption("Models not trained yet. Run `python run_training.py`.")
    st.markdown("---")
    st.caption("Research prototype. Not a diagnostic device.")

if st.session_state.nav_page != st.session_state.last_nav_page:
    st.session_state.last_nav_page = st.session_state.nav_page
    st.components.v1.html(
        """
        <script>
        try {
          const main = document.querySelector('main');
          if (main) {
            main.scrollTop = 0;
            main.scrollTo({ top: 0, left: 0, behavior: 'auto' });
          }
          const appView = document.querySelector('[data-testid="stAppViewContainer"]');
          if (appView) {
            appView.scrollTop = 0;
            appView.scrollTo({ top: 0, left: 0, behavior: 'auto' });
          }
          window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
        } catch (error) {
          console.log('scroll reset unavailable', error);
        }
        </script>
        """,
        height=0,
        scrolling=False,
    )

PAGES[st.session_state.nav_page].render()
