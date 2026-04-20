"""Centralized session state initialization for the sonic dashboard."""

import streamlit as st


SESSION_DEFAULTS = {
    "raw_data": None,
    "selected_file": None,
    "selected_frame": None,
    "selected_waveform": None,
    "selected_depth": None,
    "semblance_data": None,
    "prediction_results": None,
}


def initialize_session_state():
    """Initialize shared session state keys used across dashboard pages."""
    for key, default_value in SESSION_DEFAULTS.items():
        if st.session_state.get(key) is None and key not in st.session_state:
            st.session_state[key] = default_value
