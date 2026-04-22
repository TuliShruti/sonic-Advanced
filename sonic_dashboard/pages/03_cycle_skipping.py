"""Cycle skipping diagnostics page."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from processing.cycle_skipping import detect_bad_zones
from sonic_dashboard.loaders.dlis_loader import load_dlis
from sonic_dashboard.utils.session_state import initialize_session_state


LOG_NAMES = [
    "DTCO",
    "DTSM",
    "RHOB",
    "NPHI",
    "GR",
    "CALI",
    "RT",
    "TENS",
    "SNR2",
    "DCI2",
    "DCI4",
    "VPVS",
]


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _get_optional_channel(frame_data: dict, *candidates: str) -> np.ndarray | None:
    channels = frame_data.get("channels", {})
    normalized_candidates = {_normalize_key(name) for name in candidates}

    for name, value in channels.items():
        if _normalize_key(name) in normalized_candidates:
            return value

    return None


def _as_1d_log(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None

    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim != 1:
        return None

    return array


def _get_frame_log(frame_data: dict, name: str) -> np.ndarray | None:
    logs = frame_data.get("logs", {})
    if name in logs:
        return logs[name]

    return _get_optional_channel(frame_data, name)


def _build_qc_dataframe(frame_data: dict, depth: np.ndarray) -> pd.DataFrame:
    depth_m = _as_1d_log(depth)
    arrays = {"DEPTH_M": depth_m * 0.00254 if depth_m is not None else None}

    for name in LOG_NAMES:
        arrays[name] = _as_1d_log(_get_frame_log(frame_data, name))

    available_lengths = [len(array) for array in arrays.values() if array is not None]
    if not available_lengths:
        return pd.DataFrame(columns=["DEPTH_M", *LOG_NAMES])

    n_rows = min(available_lengths)
    qc_data = {"DEPTH_M": arrays["DEPTH_M"][:n_rows] if arrays["DEPTH_M"] is not None else np.arange(n_rows)}
    for name in LOG_NAMES:
        array = arrays.get(name)
        qc_data[name] = array[:n_rows] if array is not None else np.full(n_rows, np.nan)

    return pd.DataFrame(qc_data)


def _hydrate_df_raw_from_session() -> None:
    if "df_raw" in st.session_state and st.session_state.df_raw is not None:
        return

    waveform = st.session_state.get("waveform")
    if isinstance(waveform, pd.DataFrame) and not waveform.empty:
        st.session_state["df_raw"] = waveform.copy()
        return

    data = st.session_state.get("raw_data") or st.session_state.get("data")
    selected_frame = st.session_state.get("selected_frame")
    if not data or not selected_frame:
        return

    frame_data = data.get("frames", {}).get(selected_frame)
    if frame_data is None or frame_data.get("depth") is None:
        return

    df = _build_qc_dataframe(frame_data, frame_data["depth"])
    if not df.empty:
        st.session_state["df_raw"] = df


def _zone_table(zone_intervals: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for idx, (top, base) in enumerate(zone_intervals, start=1):
        rows.append(
            {
                "Zone": f"Z{idx}",
                "Top": float(top),
                "Base": float(base),
                "Thickness": abs(float(base) - float(top)),
            }
        )
    return pd.DataFrame(rows, columns=["Zone", "Top", "Base", "Thickness"])


def _shade_zones(ax, zone_intervals: list[tuple[float, float]]) -> None:
    for top, base in zone_intervals:
        ax.axhspan(top, base, color="#f2c94c", alpha=0.28, linewidth=0)


def _plot_cycle_skipping(
    df: pd.DataFrame,
    dt_ch: str,
    cali_ch: str,
    bit_size_inch: float,
    washout_factor: float,
    spike_threshold: float,
    zone_intervals: list[tuple[float, float]],
) -> plt.Figure:
    depth = df["DEPTH_M"].to_numpy(dtype=float)
    washout_threshold = bit_size_inch * washout_factor

    fig, axs = plt.subplots(1, 4, figsize=(15, 9), dpi=120, sharey=True)

    if cali_ch in df.columns:
        axs[0].plot(df[cali_ch], depth, color="#1f77b4", linewidth=1.0, label=cali_ch)
    axs[0].axvline(bit_size_inch, color="#222222", linestyle="--", linewidth=1.0, label="Bit size")
    axs[0].axvline(washout_threshold, color="#d62728", linestyle="-.", linewidth=1.0, label="Washout")
    axs[0].set_title("Caliper")
    axs[0].set_xlabel("in")

    axs[1].plot(df[dt_ch], depth, color="#222222", linewidth=1.0, label="Raw")
    axs[1].plot(df["DT_ROLLING"], depth, color="#d62728", linewidth=1.0, label="Rolling median")
    axs[1].set_title("DT")
    axs[1].set_xlabel(dt_ch)

    axs[2].plot(df["DT_DEVIATION"], depth, color="#9467bd", linewidth=1.0, label="Deviation")
    axs[2].axvline(spike_threshold, color="#d62728", linestyle="--", linewidth=1.0, label="Threshold")
    axs[2].set_title("DT Deviation")
    axs[2].set_xlabel("abs deviation")

    axs[3].step(df["FLAG_BAD"].astype(int), depth, where="mid", color="#111111", linewidth=1.1)
    axs[3].set_xlim(-0.1, 1.1)
    axs[3].set_xticks([0, 1])
    axs[3].set_title("Bad Flag")
    axs[3].set_xlabel("FLAG_BAD")

    for idx, (top, base) in enumerate(zone_intervals, start=1):
        mid_depth = (float(top) + float(base)) / 2.0
        axs[3].text(1.04, mid_depth, f"Z{idx}", va="center", ha="left", fontsize=8)

    for ax in axs:
        _shade_zones(ax, zone_intervals)
        ax.grid(True, linewidth=0.4, alpha=0.55)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="best")

    axs[0].set_ylabel("Depth (m)")
    axs[0].invert_yaxis()
    fig.tight_layout()

    return fig


initialize_session_state()
_hydrate_df_raw_from_session()

st.title("Cycle Skipping")

if "df_raw" not in st.session_state or st.session_state.df_raw is None:
    with st.container():
        st.subheader("DLIS Upload")
        uploaded_file = st.file_uploader(
            "Upload DLIS file",
            type=["dlis"],
            help="Shown only when no df_raw data is available in session state.",
        )

        if uploaded_file is not None:

            @st.cache_data(show_spinner="Loading DLIS file...")
            def load(file_bytes: bytes) -> dict:
                return load_dlis(io.BytesIO(file_bytes))

            loaded_data = load(uploaded_file.read())
            st.session_state["raw_data"] = loaded_data
            st.session_state["data"] = loaded_data
            st.session_state["selected_file"] = uploaded_file.name

data = st.session_state.get("raw_data") or st.session_state.get("data")
if ("df_raw" not in st.session_state or st.session_state.df_raw is None) and data:
    frame_names = data.get("frame_names", [])
    if frame_names:
        selected_frame = st.selectbox("Select Frame", frame_names)
        st.session_state["selected_frame"] = selected_frame
        frame_data = data["frames"][selected_frame]
        if frame_data.get("depth") is None:
            st.error("Selected frame is missing depth data.")
        else:
            df_from_dlis = _build_qc_dataframe(frame_data, frame_data["depth"])
            if df_from_dlis.empty:
                st.error("No usable log channels were found in the selected frame.")
            else:
                st.session_state["df_raw"] = df_from_dlis

df_raw = st.session_state.get("df_raw")
if df_raw is None or df_raw.empty:
    st.info("Upload a DLIS file or run the Monopole/Dipole page to create df_raw data.")
    st.stop()

if "DEPTH_M" not in df_raw.columns:
    st.error("Cycle skipping requires a DEPTH_M column.")
    st.stop()

columns = list(df_raw.columns)
numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df_raw[col])]
if not numeric_columns:
    st.error("Cycle skipping requires numeric log columns.")
    st.stop()

with st.sidebar:
    st.subheader("Cycle Skipping Inputs")
    dt_default = numeric_columns.index("DTCO") if "DTCO" in numeric_columns else 0
    cali_default = numeric_columns.index("CALI") if "CALI" in numeric_columns else 0

    dt_ch = st.selectbox("DT channel", numeric_columns, index=dt_default)
    cali_ch = st.selectbox("Caliper channel", numeric_columns, index=cali_default)
    bit_size_inch = st.number_input("Bit size (in)", min_value=0.0, value=8.5, step=0.25)
    washout_factor = st.slider("Washout factor", min_value=1.0, max_value=2.0, value=1.15, step=0.01)
    spike_threshold = st.number_input("Spike threshold", min_value=0.0, value=12.0, step=1.0)
    rolling_window = st.slider("Rolling window", min_value=3, max_value=101, value=11, step=2)
    min_bad_run = st.slider("Minimum bad run", min_value=1, max_value=50, value=3, step=1)
    logic = st.selectbox("Combine logic", ["OR", "AND"])

st.caption(f"Rows: {len(df_raw):,} | Columns: {len(df_raw.columns):,}")

if st.button("Run Cycle Skipping Detection", type="primary"):
    with st.spinner("Detecting bad zones..."):
        df, zone_intervals, n_zones = detect_bad_zones(
            df_raw,
            dt_ch,
            cali_ch,
            bit_size_inch,
            washout_factor,
            spike_threshold,
            rolling_window,
            min_bad_run,
            logic=logic,
        )

    st.session_state.df_qc = df
    st.session_state.bad_mask = df["FLAG_BAD"]
    st.session_state.zone_intervals = zone_intervals
    st.session_state.cycle_skipping_summary = {
        "dt_ch": dt_ch,
        "cali_ch": cali_ch,
        "bit_size_inch": bit_size_inch,
        "washout_factor": washout_factor,
        "spike_threshold": spike_threshold,
        "rolling_window": rolling_window,
        "min_bad_run": min_bad_run,
        "logic": logic,
        "n_zones": n_zones,
    }

df_qc = st.session_state.get("df_qc")
zone_intervals = st.session_state.get("zone_intervals", [])
summary = st.session_state.get("cycle_skipping_summary")

if df_qc is None or "FLAG_BAD" not in df_qc.columns or summary is None:
    st.info("Run detection to generate FLAG_BAD and zone_intervals for Outlier Detection.")
    st.stop()

total_bad = int(df_qc["FLAG_BAD"].sum())
percent_bad = 100.0 * total_bad / len(df_qc) if len(df_qc) else 0.0
n_zones = int(summary["n_zones"])

st.subheader("Summary")
metric_cols = st.columns(3)
metric_cols[0].metric("Total bad points", f"{total_bad:,}")
metric_cols[1].metric("% bad", f"{percent_bad:.2f}%")
metric_cols[2].metric("Number of zones", f"{n_zones:,}")

st.subheader("Visualization")
fig = _plot_cycle_skipping(
    df_qc,
    summary["dt_ch"],
    summary["cali_ch"],
    summary["bit_size_inch"],
    summary["washout_factor"],
    summary["spike_threshold"],
    zone_intervals,
)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Zone Table")
zones_df = _zone_table(zone_intervals)
if zones_df.empty:
    st.info("No bad zones detected.")
else:
    st.dataframe(zones_df, use_container_width=True, hide_index=True)
