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
    "cross_dipole_ran": False,
    "monopole_dipole_ran": False,
    "cycle_skipping_ran": False,
    "prediction_ran": False,
}


def initialize_pipeline_state():
    defaults = {
        "waveform": None,
        "cleaned_dtco": None,
        "cleaned_dtsm": None,
        "qc_flags": None,
        "stage_qc_done": False,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def initialize_session_state():
    """Initialize shared session state keys used across dashboard pages."""
    for key, default_value in SESSION_DEFAULTS.items():
        if st.session_state.get(key) is None and key not in st.session_state:
            st.session_state[key] = default_value
    initialize_pipeline_state()
