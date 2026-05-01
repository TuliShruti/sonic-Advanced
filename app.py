"""Root Streamlit entrypoint for the Sonic Dashboard app."""

from __future__ import annotations

from pathlib import Path
import runpy

import streamlit as st

from sonic_dashboard.utils.session_state import initialize_pipeline_state


ROOT = Path(__file__).resolve().parent
PAGES_DIR = ROOT / "sonic_dashboard" / "pages"

PAGE_SPECS = [
    ("01_monopole_dipole.py", "Monopole / Dipole"),
    ("02_cross_dipole.py", "Cross Dipole"),
    ("03_cycle_skipping_and_outlier_detection.py", "Cycle Skipping and Outlier Detection"),
    ("05_prediction.py", "Prediction"),
]


st.set_page_config(
    page_title="Sonic Log Dashboard",
    layout="wide",
    page_icon="🔊",
)

initialize_pipeline_state()

with st.sidebar:
    st.write("QC Pipeline Status")
    st.write("QC Done:", st.session_state.stage_qc_done)


def _run_page(page_path: Path) -> None:
    """Execute a page script in the current Streamlit session."""
    runpy.run_path(str(page_path), run_name="__main__")


if hasattr(st, "navigation") and hasattr(st, "Page"):
    navigation = st.navigation(
        [
            st.Page(str(PAGES_DIR / filename), title=title)
            for filename, title in PAGE_SPECS
        ]
    )
    navigation.run()
else:
    st.sidebar.title("Navigation")
    selected_title = st.sidebar.radio(
        "Choose a workflow",
        [title for _, title in PAGE_SPECS],
    )
    selected_filename = next(
        filename for filename, title in PAGE_SPECS if title == selected_title
    )
    _run_page(PAGES_DIR / selected_filename)
