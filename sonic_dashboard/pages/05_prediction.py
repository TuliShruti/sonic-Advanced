"""Prediction placeholder page."""

import streamlit as st

from sonic_dashboard.utils.session_state import initialize_session_state


initialize_session_state()

st.title("Sonic Log Prediction")
st.write("This module will predict sonic logs from other logs using ML.")
st.info("Coming soon")
