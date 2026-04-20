"""Cycle skipping placeholder page."""

import streamlit as st

from sonic_dashboard.utils.session_state import initialize_session_state


initialize_session_state()

st.title("Cycle Skipping Detection")
st.write("This module will detect cycle skipping in waveform and semblance outputs.")
st.info("Coming soon")
