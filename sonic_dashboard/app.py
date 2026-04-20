"""Streamlit entrypoint for the Sonic Dashboard multipage app."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Sonic Log Dashboard",
    layout="wide",
    page_icon="🔊",
)

st.title("🔊 Sonic Log Dashboard")
st.write(
    "Use the sidebar to navigate between the workflow pages in "
    "`sonic_dashboard/pages/`."
)

st.info(
    "Run the app with `streamlit run sonic_dashboard/app.py` so Streamlit "
    "auto-detects the sibling `pages/` directory."
)
