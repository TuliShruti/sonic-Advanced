"""Sonic log prediction page."""

import streamlit as st

from sonic_dashboard.utils.session_state import initialize_session_state


initialize_session_state()

st.title("Sonic Log Prediction")

st.info("This page predicts sonic logs from other logs. Independent of QC pipeline.")

df = st.session_state.waveform
if df is None:
    st.warning("Load data in the Monopole/Dipole page first.")
else:
    st.dataframe(df, use_container_width=True)
