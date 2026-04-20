"""Monopole / Dipole sonic workflow page."""

from __future__ import annotations

import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sonic_dashboard.loaders.dlis_loader import load_dlis
from sonic_dashboard.processing.semblance_processing import build_slowness_axis
from sonic_dashboard.utils.session_state import initialize_session_state
from sonic_dashboard.visualization.waveform_plot import plot_waveform


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _get_optional_channel(frame_data: dict, *candidates: str) -> np.ndarray | None:
    channels = frame_data.get("channels", {})
    normalized_candidates = {_normalize_key(name) for name in candidates}

    for name, value in channels.items():
        if _normalize_key(name) in normalized_candidates:
            return value

    return None


def _resolve_window_data(data: dict, selected_frame: str | None) -> dict[str, np.ndarray] | None:
    if all(key in data for key in ("depth", "pwf2", "t1r2", "t2r2")):
        return {
            "depth": data["depth"],
            "pwf2": data["pwf2"],
            "t1r2": data["t1r2"],
            "t2r2": data["t2r2"],
        }

    if not selected_frame:
        return None

    frame_data = data.get("frames", {}).get(selected_frame)
    if frame_data is None:
        return None

    return {
        "depth": frame_data.get("depth"),
        "pwf2": frame_data.get("waveforms", {}).get("PWF2"),
        "t1r2": _get_optional_channel(frame_data, "T1R2"),
        "t2r2": _get_optional_channel(frame_data, "T2R2"),
    }


@st.cache_data
def build_windows(data: dict[str, np.ndarray], dt: float = 1e-6) -> dict[str, tuple[int, int]]:
    """Build global P- and S-wave windows from median picks."""
    pwf2 = np.asarray(data["pwf2"])
    t1r2 = np.asarray(data["t1r2"])
    t2r2 = np.asarray(data["t2r2"])

    if pwf2.ndim != 3:
        raise ValueError("Expected PWF2 shape (depth, receivers, samples).")

    n_samples = pwf2.shape[-1]

    p_pick = float(np.nanmedian(t1r2))
    s_pick = float(np.nanmedian(t2r2))

    p_center = int(np.round(p_pick / dt))
    s_center = int(np.round(s_pick / dt))

    p_start = max(0, p_center - 20)
    p_end = min(n_samples - 1, p_center + 60)
    s_start = max(0, s_center - 20)
    s_end = min(n_samples - 1, s_center + 80)

    return {
        "p_window": (p_start, p_end),
        "s_window": (s_start, s_end),
    }


@st.cache_data
def extract_velocity_from_spr4(depth: np.ndarray, spr4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N, Ns = spr4.shape
    slowness = np.linspace(40, 240, Ns)

    vp = np.zeros(N)
    vs = np.zeros(N)

    for d in range(N):
        row = spr4[d]

        mask_p = (slowness >= 40) & (slowness <= 90)
        idx_p = np.argmax(row[mask_p])
        p_slow = slowness[mask_p][idx_p]

        mask_s = (slowness >= 90) & (slowness <= 220)
        idx_s = np.argmax(row[mask_s])
        s_slow = slowness[mask_s][idx_s]

        vp[d] = 1e6 / p_slow
        vs[d] = 1e6 / s_slow

    return vp, vs


@st.cache_data
def extract_velocity_improved(
    spr4: np.ndarray,
    dtco: np.ndarray,
    dtsm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth_placeholder = np.arange(spr4.shape[0])
    return extract_velocity_from_spr4(depth_placeholder, spr4)


def plot_window_at_depth(
    wf_depth: np.ndarray,
    depth_m: float,
    windows: dict[str, tuple[int, int]],
) -> plt.Figure:
    """Plot waveform heatmap with P and S windows overlaid."""

    wf = wf_depth.copy().astype(float)

    # ── 1. Clip to the display range before any normalisation ──────────────
    n_samples = wf.shape[-1]
    display_end = min(n_samples, 250)
    wf = wf[:, :display_end]

    # ── 2. Global percentile normalisation (avoids blown-out contrast) ─────
    #   Per-receiver max-norm maps every trace to [-1,1] which saturates the
    #   colourmap.  A global 2/98-percentile scale preserves relative amplitude.
    p_low, p_high = np.percentile(wf, [2, 98])
    scale = max(abs(p_low), abs(p_high), 1e-8)
    wf = np.clip(wf / scale, -1.0, 1.0)

    # ── 3. Figure geometry ─────────────────────────────────────────────────
    n_receivers = wf.shape[0]
    fig, ax = plt.subplots(figsize=(13, max(3.5, n_receivers * 0.55)), dpi=130)

    ax.imshow(
        wf,
        cmap="seismic",
        aspect="auto",
        interpolation="bilinear",          # smooth, no brick artefacts
        vmin=-1.0,
        vmax=1.0,
        extent=[-0.5, display_end - 0.5,  # x: time samples
                 n_receivers - 0.5, -0.5], # y: receiver index (0 at top)
    )

    # ── 4. Window overlays (guaranteed visible regardless of window values) ─
    p_start, p_end   = windows["p_window"]
    s_start, s_end   = windows["s_window"]

    kw_p = dict(color="lime",   linewidth=1.8, alpha=0.9)
    kw_s = dict(color="yellow", linewidth=1.8, alpha=0.9)

    ax.axvline(p_start, linestyle="--", label="P start", **kw_p)
    ax.axvline(p_end,   linestyle="-",  label="P end",   **kw_p)
    ax.axvline(s_start, linestyle="--", label="S start", **kw_s)
    ax.axvline(s_end,   linestyle="-",  label="S end",   **kw_s)

    ax.set_xlim(-0.5, display_end - 0.5)
    ax.set_ylim(n_receivers - 0.5, -0.5)

    ax.set_title(f"Wave Windows  —  Depth {depth_m:.2f} m", fontsize=10, pad=6)
    ax.set_xlabel("Time samples", fontsize=9)
    ax.set_ylabel("Receiver",     fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(
        loc="upper right", fontsize=8,
        framealpha=0.55, edgecolor="none",
    )

    fig.tight_layout(pad=1.0)
    return fig


def _render_semblance_panel(
    depth: np.ndarray,
    spr2: np.ndarray | None,
    spr4: np.ndarray | None,
    slowness: np.ndarray | None,
) -> None:
    st.divider()
    st.subheader("Semblance")

    if spr4 is None and spr2 is None:
        st.warning("No SPR4 or SPR2 semblance data available.")
        return

    semblance_type = st.selectbox(
        "Select Wave Mode",
        ["Compressional (SPR4)", "Shear (SPR2)"],
    )

    if "SPR4" in semblance_type:
        selected_spr = spr4
        wave_label = "P-wave semblance"
    else:
        selected_spr = spr2
        wave_label = "S-wave semblance"

    if selected_spr is None:
        st.warning(f"{wave_label} not available")
        return

    if selected_spr.shape[0] != depth.shape[0]:
        st.error("Shape mismatch between selected semblance and depth data.")
        return

    slowness_axis = build_slowness_axis(selected_spr, slowness)

    fig, ax = plt.subplots(figsize=(6, 10))

    ax.imshow(
        selected_spr,
        aspect="auto",
        cmap="jet",
        extent=[
            slowness_axis[0],
            slowness_axis[-1],
            depth[-1] * 0.00254,
            depth[0] * 0.00254,
        ],
    )

    ax.set_xlabel("Slowness (µs/ft)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("STC Semblance")

    st.pyplot(fig)


def _render_waveform_panel(wf: np.ndarray | None, idx: int, selected_depth: float) -> None:
    st.divider()
    st.subheader("Waveform")

    if wf is None:
        st.warning("PWF2 missing")
        return

    wf_depth = wf[idx]
    fig = plot_waveform(wf_depth, title=f"PWF2 at Depth = {selected_depth:.2f}")
    st.plotly_chart(fig, use_container_width=True)


def _render_velocity_picking_panel(depth: np.ndarray, spr4: np.ndarray | None) -> None:
    st.divider()
    st.subheader("Velocity Picking (SPR4)")

    if spr4 is None:
        st.warning("SPR4 not available for velocity picking")
        return

    depth_m = depth * 0.00254
    vp, vs = extract_velocity_from_spr4(depth, spr4)

    fig, ax = plt.subplots(figsize=(4, 6), dpi=120)

    ax.plot(vp, depth_m, color="blue", label="Vp (SPR4)")
    ax.plot(vs, depth_m, color="red", label="Vs (SPR4)")

    ax.invert_yaxis()

    ax.set_xlabel("Velocity (ft/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Velocity from SPR4")

    ax.legend()
    ax.grid(True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)


def _render_velocity_comparison_panel(
    depth: np.ndarray,
    spr4: np.ndarray | None,
    dtco: np.ndarray | None,
    dtsm: np.ndarray | None,
) -> None:
    st.divider()
    st.subheader("Velocity Comparison (SPR4 vs Tool Logs)")

    if any(x is None for x in [spr4, dtco, dtsm]):
        st.warning("DTCO/DTSM or SPR4 not available for comparison")
        return

    depth_m = depth * 0.00254

    vp_tool = 1e6 / dtco
    vs_tool = 1e6 / dtsm

    vp_spr, vs_spr = extract_velocity_improved(spr4, dtco, dtsm)

    fig, axs = plt.subplots(1, 2, figsize=(8, 6), dpi=120, sharey=True)

    # ---- P-wave ----
    axs[0].plot(vp_tool, depth_m, "k-", linewidth=1.2, label="DTCO")
    axs[0].plot(vp_spr, depth_m, "r--", linewidth=1.2, label="SPR4")
    axs[0].set_xlabel("Velocity (ft/s)", fontsize=9)
    axs[0].set_title("P-wave", fontsize=10)
    axs[0].invert_yaxis()
    axs[0].grid(True, linewidth=0.5)
    axs[0].legend(fontsize=8)

    # ---- S-wave ----
    axs[1].plot(vs_tool, depth_m, "k-", linewidth=1.2, label="DTSM")
    axs[1].plot(vs_spr, depth_m, "b--", linewidth=1.2, label="SPR4")
    axs[1].set_xlabel("Velocity (ft/s)", fontsize=9)
    axs[1].set_title("S-wave", fontsize=10)
    axs[1].grid(True, linewidth=0.5)
    axs[1].legend(fontsize=8)

    axs[0].set_ylabel("Depth (m)", fontsize=9)

    for ax in axs:
        ax.tick_params(labelsize=8)

    fig.tight_layout()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)


initialize_session_state()

st.title("Monopole / Dipole Sonic (Isotropic)")
st.write(
    "Isotropic sonic analysis pipeline: data upload -> frame selection -> "
    "semblance analysis -> waveform inspection."
)

with st.container():
    st.subheader("Data Upload")

    uploaded_file = st.file_uploader(
        "Upload a sonic data file",
        type=["dlis", "las", "bin"],
        help="Supported formats: .dlis, .las, .bin",
    )

    if uploaded_file is not None:

        @st.cache_data(show_spinner="Loading DLIS file...")
        def load(file_bytes: bytes, file_name: str) -> dict:
            return load_dlis(io.BytesIO(file_bytes))

        loaded_data = load(uploaded_file.read(), uploaded_file.name)
        st.session_state["raw_data"] = loaded_data
        st.session_state["data"] = loaded_data
        st.session_state["selected_file"] = uploaded_file.name

    selected_file = st.session_state.get("selected_file")
    st.caption(f"Selected file: {selected_file or 'None'}")

data = st.session_state.get("raw_data")
if data is None:
    data = st.session_state.get("data")
    if data is not None:
        st.session_state["raw_data"] = data

if data is None:
    st.info("Upload a DLIS file above to begin.")
else:
    st.divider()
    st.subheader("Frame Selection")

    frame_names = data.get("frame_names", [])
    if not frame_names:
        st.error("No frames found in this file.")
    else:
        default_idx = 0
        selected_frame = st.session_state.get("selected_frame")
        if selected_frame in frame_names:
            default_idx = frame_names.index(selected_frame)

        selected_frame = st.selectbox(
            "Select Frame",
            options=frame_names,
            index=default_idx,
        )
        st.session_state["selected_frame"] = selected_frame

        frame_data = data["frames"][selected_frame]

        depth = frame_data.get("depth")
        wf = frame_data.get("waveforms", {}).get("PWF2")
        spr2 = frame_data.get("semblance", {}).get("SPR2")
        spr4 = frame_data.get("semblance", {}).get("SPR4")
        slowness = frame_data.get("slowness")
        dtco = frame_data.get("channels", {}).get("DTCO")
        dtsm = frame_data.get("channels", {}).get("DTSM")

        if depth is None:
            st.error("Missing depth data.")
        else:
            selected_depth = st.slider(
                "Select Depth",
                float(depth.min()),
                float(depth.max()),
                float(depth[len(depth) // 2]),
            )

            idx = int(np.argmin(np.abs(depth - selected_depth)))

            _render_semblance_panel(depth, spr2, spr4, slowness)
            _render_waveform_panel(wf, idx, selected_depth)

            st.divider()
            st.subheader("Wave Windowing")

            show_wave_windows = st.checkbox("Show wave windows", value=False)

            if show_wave_windows:
                window_data = _resolve_window_data(st.session_state.get("data", {}), selected_frame)

                if window_data is None:
                    st.warning("Windowing data is not available.")
                elif any(window_data.get(key) is None for key in ("depth", "pwf2", "t1r2", "t2r2")):
                    st.warning("Windowing data is incomplete.")
                else:
                    window_depth = np.asarray(window_data["depth"])
                    pwf2 = np.asarray(window_data["pwf2"])

                    window_idx = int(np.argmin(np.abs(window_depth - selected_depth)))
                    wf_window_depth = pwf2[window_idx]

                    try:
                        windows = build_windows(window_data)
                    except ValueError as exc:
                        st.warning(str(exc))
                    else:
                        fig = plot_window_at_depth(
                            wf_window_depth,
                            float(window_depth[window_idx] * 0.00254),
                            windows,
                        )

                        st.pyplot(fig)

            _render_velocity_picking_panel(depth, spr4)
            _render_velocity_comparison_panel(depth, spr4, dtco, dtsm)
