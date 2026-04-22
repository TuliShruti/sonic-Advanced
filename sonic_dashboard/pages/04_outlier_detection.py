"""ML-based outlier correction page."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from processing.outlier_detection import train_and_fill
from sonic_dashboard.utils.session_state import initialize_session_state


def _shade_zones(ax, zone_intervals: list[tuple[float, float]]) -> None:
    for top, base in zone_intervals:
        ax.axhspan(top, base, color="#f2c94c", alpha=0.28, linewidth=0)


def _resolve_cycle_inputs() -> tuple[pd.DataFrame, list[tuple[float, float]]]:
    if "df_qc" not in st.session_state or st.session_state.df_qc is None:
        st.warning("Run Cycle Skipping first. Outlier Detection requires df_qc with FLAG_BAD.")
        st.stop()

    df = st.session_state.df_qc
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("Cycle Skipping output is empty. Run Cycle Skipping again.")
        st.stop()

    if "FLAG_BAD" not in df.columns:
        st.warning("Cycle Skipping must create FLAG_BAD before Outlier Detection can run.")
        st.stop()

    if "DEPTH_M" not in df.columns:
        st.error("Outlier Detection requires DEPTH_M in df_qc.")
        st.stop()

    zone_intervals = st.session_state.get("zone_intervals", [])
    return df, zone_intervals


def _plot_outlier_correction(
    df: pd.DataFrame,
    dt_ch: str,
    cali_ch: str | None,
    washout_threshold: float | None,
    zone_intervals: list[tuple[float, float]],
) -> plt.Figure:
    depth = df["DEPTH_M"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 9), dpi=120, sharey=True)

    if cali_ch and cali_ch in df.columns:
        axs[0].plot(df[cali_ch], depth, color="#1f77b4", linewidth=1.0, label=cali_ch)
    if washout_threshold is not None:
        axs[0].axvline(washout_threshold, color="#d62728", linestyle="--", linewidth=1.0, label="Washout")
    axs[0].set_title("Caliper")
    axs[0].set_xlabel("in")

    axs[1].plot(df[dt_ch], depth, color="#222222", linewidth=1.0, label=f"Original {dt_ch}")
    axs[1].set_title("Original DT")
    axs[1].set_xlabel(dt_ch)

    original = df[df["DT_SOURCE"] == "original"]
    predicted = df[df["DT_SOURCE"] == "ml_predicted"]
    axs[2].plot(original["DT_FILLED"], original["DEPTH_M"], color="#1f77b4", linewidth=1.0, label="Original")
    if not predicted.empty:
        axs[2].scatter(
            predicted["DT_FILLED"],
            predicted["DEPTH_M"],
            color="#ff7f0e",
            s=14,
            label="ML predicted",
            zorder=3,
        )
    axs[2].set_title("DT_FILLED")
    axs[2].set_xlabel("DT_FILLED")

    for ax in axs:
        _shade_zones(ax, zone_intervals)
        ax.grid(True, linewidth=0.4, alpha=0.55)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="best")

    axs[0].set_ylabel("Depth (m)")
    axs[0].invert_yaxis()
    fig.tight_layout()

    return fig


def _output_table(df: pd.DataFrame, dt_ch: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Depth": df["DEPTH_M"],
            "DT": df[dt_ch],
            "DT_FILLED": df["DT_FILLED"],
            "DT_SOURCE": df["DT_SOURCE"],
        }
    )


initialize_session_state()

st.title("Outlier Detection")

df, zone_intervals = _resolve_cycle_inputs()
bad_count = int(df["FLAG_BAD"].sum())
if bad_count == 0:
    st.warning("Cycle Skipping did not flag any bad samples. There are no bad zones to fill.")
    st.stop()

numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
cycle_summary = st.session_state.get("cycle_skipping_summary", {})

with st.sidebar:
    st.subheader("ML Correction Inputs")

    dt_default_name = cycle_summary.get("dt_ch", "DTCO")
    dt_default = numeric_columns.index(dt_default_name) if dt_default_name in numeric_columns else 0
    dt_ch = st.selectbox("DT channel", numeric_columns, index=dt_default)

    excluded_predictors = {
        dt_ch,
        "DEPTH_M",
        "FLAG_BAD",
        "FLAG_BAD_POINT",
        "FLAG_SPIKE",
        "FLAG_WASHOUT",
        "DT_ROLLING",
        "DT_DIFF",
        "DT_DEVIATION",
        "DT_FILLED",
    }
    predictor_options = [col for col in numeric_columns if col not in excluded_predictors]
    default_predictors = [
        col for col in ["CALI", "RHOB", "NPHI", "GR", "RT", "TENS", "SNR2", "DCI2", "DCI4", "VPVS"]
        if col in predictor_options
    ]

    predictor_channels = st.multiselect(
        "Predictor channels",
        predictor_options,
        default=default_predictors,
    )
    n_estimators = st.slider("n_estimators", min_value=10, max_value=500, value=100, step=10)
    random_state = st.number_input("random_state", min_value=0, value=42, step=1)
    use_saved_model = st.checkbox("Use saved model if available", value=True)

st.caption(f"Bad samples from Cycle Skipping: {bad_count:,} | Zones: {len(zone_intervals):,}")

if st.button("Run ML Correction", type="primary"):
    if not predictor_channels:
        st.error("Select at least one predictor channel.")
    else:
        try:
            with st.spinner("Training ML model and filling bad zones..."):
                df_merged, metrics = train_and_fill(
                    df,
                    dt_ch,
                    predictor_channels,
                    n_estimators=n_estimators,
                    random_state=int(random_state),
                    use_saved_model=use_saved_model,
                )
        except Exception as exc:
            st.error(f"ML correction failed: {exc}")
        else:
            st.session_state.df_corrected = df_merged
            st.session_state.outlier_metrics = metrics
            st.session_state.outlier_params = {
                "dt_ch": dt_ch,
                "predictor_channels": predictor_channels,
                "n_estimators": n_estimators,
                "random_state": int(random_state),
            }

df_corrected = st.session_state.get("df_corrected")
metrics = st.session_state.get("outlier_metrics")
params = st.session_state.get("outlier_params")

if df_corrected is None or metrics is None or params is None:
    st.info("Run ML Correction to create DT_FILLED for velocity picking.")
    st.stop()

dt_ch = params["dt_ch"]

st.subheader("Metrics")
metric_cols = st.columns(2)
train_mae = metrics["train_mae"]
val_mae = metrics["val_mae"]
metric_cols[0].metric("Train MAE", "N/A" if np.isnan(train_mae) else f"{train_mae:.3f}")
metric_cols[1].metric("Validation MAE", "N/A" if np.isnan(val_mae) else f"{val_mae:.3f}")

cali_ch = cycle_summary.get("cali_ch")
if not cali_ch and "CALI" in df_corrected.columns:
    cali_ch = "CALI"

washout_threshold = None
if "bit_size_inch" in cycle_summary and "washout_factor" in cycle_summary:
    washout_threshold = cycle_summary["bit_size_inch"] * cycle_summary["washout_factor"]

st.subheader("Visualization")
fig = _plot_outlier_correction(
    df_corrected,
    dt_ch,
    cali_ch,
    washout_threshold,
    zone_intervals,
)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Output Table")
st.dataframe(_output_table(df_corrected, dt_ch), use_container_width=True, hide_index=True)
