"""Monopole / Dipole sonic workflow page."""

from __future__ import annotations

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from sonic_dashboard.loaders.dlis_loader import load_dlis
from sonic_dashboard.processing.semblance_processing import build_slowness_axis
from sonic_dashboard.utils.session_state import initialize_session_state
from sonic_dashboard.visualization.waveform_plot import plot_waveform
from processing.outlier_detection import run_outlier_detection


def despike(x, thresh=3.0):
    x = x.copy()
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    z = 0.6745 * (x - med) / mad
    x[np.abs(z) > thresh] = np.nan
    return x


def fill_nan(x):
    n = len(x)
    idx = np.arange(n)
    mask = ~np.isnan(x)
    return np.interp(idx, idx[mask], x[mask])


def compute_elastic_from_df(df):

    depth = df["DEPTH_M"].values

    dtco = df["DTCO"].values
    dtsm = df["DTSM"].values
    rhob = df["RHOB"].values

    dtco_si = dtco * 1e-6 / 0.3048
    dtsm_si = dtsm * 1e-6 / 0.3048

    vp = 1.0 / dtco_si
    vs = 1.0 / dtsm_si
    rho = rhob * 1000

    mask = (vs > 500) & (vp > vs)

    vp[~mask] = np.nan
    vs[~mask] = np.nan
    rho[~mask] = np.nan

    vp = despike(vp, 3)
    vs = despike(vs, 3)

    vp = fill_nan(vp)
    vs = fill_nan(vs)

    vp = medfilt(vp, 9)
    vs = medfilt(vs, 11)

    vp = gaussian_filter1d(vp, sigma=2)
    vs = gaussian_filter1d(vs, sigma=2)

    nu = (vp**2 - 2*vs**2) / (2*(vp**2 - vs**2 + 1e-12))
    nu = np.clip(nu, 0.0, 0.5)

    nu = medfilt(nu, 11)
    nu = gaussian_filter1d(nu, sigma=2)

    mu = rho * vs**2
    lam = rho * (vp**2 - 2*vs**2)

    E = mu * (3*lam + 2*mu) / (lam + mu + 1e-12)
    E = gaussian_filter1d(E, sigma=2)

    return depth, vp, vs, nu, E


def plot_elastic(depth, vp, vs, nu, E):

    fig, axs = plt.subplots(1, 4, figsize=(10, 15), sharey=True)

    axs[0].plot(vp, depth)
    axs[0].set_title("Vp (m/s)")
    axs[0].invert_yaxis()
    axs[0].grid()

    axs[1].plot(vs, depth)
    axs[1].set_title("Vs (m/s)")
    axs[1].grid()

    axs[2].plot(nu, depth)
    axs[2].set_title("Poisson's Ratio")
    axs[2].grid()

    axs[3].plot(E / 1e9, depth)
    axs[3].set_title("Young's Modulus (GPa)")
    axs[3].grid()

    axs[0].set_ylabel("Depth (m)")

    plt.suptitle("Elastic Properties (QC Corrected)")
    plt.tight_layout()

    return fig


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

    log_names = ["DTCO", "DTSM", "RHOB", "NPHI", "GR", "CALI", "RT", "TENS", "SNR2", "DCI2", "DCI4", "VPVS", "BS"]
    arrays = {"DEPTH_M": depth_m * 0.00254 if depth_m is not None else None}

    for name in log_names:
        arrays[name] = _as_1d_log(_get_frame_log(frame_data, name))

    available_lengths = [len(array) for array in arrays.values() if array is not None]
    if not available_lengths:
        return pd.DataFrame(columns=[*log_names, "DEPTH_M"])

    n_rows = min(available_lengths)
    qc_data = {}
    for name in log_names:
        array = arrays.get(name)
        qc_data[name] = array[:n_rows] if array is not None else np.full(n_rows, np.nan)
    qc_data["DEPTH_M"] = arrays["DEPTH_M"][:n_rows] if arrays["DEPTH_M"] is not None else np.arange(n_rows)

    return pd.DataFrame(qc_data)


def _plot_key_log_overview(df: pd.DataFrame) -> plt.Figure:
    tracks = [
        ("GR", "API", "lime", (0, 220), "GR (API)"),
        ("CALI", "in", "brown", (6, 20), "CALI (in)"),
        ("RHOB", "g/cc", "red", (1.8, 3.1), "RHOB (g/cc)"),
        ("NPHI", "v/v", "blue", (-0.05, 0.6), "NPHI (v/v)"),
        ("RT", "ohm.m", "black", None, "RT (ohm.m)"),
        ("DTCO", "us/ft", "darkblue", (40, 140), "DTCO (us/ft)"),
        ("DTSM", "us/ft", "darkcyan", (60, 250), "DTSM (us/ft)"),
        ("VPVS", "ratio", "purple", (1.3, 3.6), "Vp/Vs"),
    ]

    depth = df["DEPTH_M"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, len(tracks), figsize=(22, 14), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (col, unit, color, xlim, label) in zip(axes, tracks):
        ax.set_facecolor("white")
        vals = df[col].to_numpy(dtype=float).copy()
        ax.spines["bottom"].set_color("gray")
        ax.spines["top"].set_color("gray")
        ax.spines["left"].set_color("gray")
        ax.spines["right"].set_color("gray")
        ax.tick_params(colors="black", labelsize=7)
        ax.set_title(label, color="black", fontsize=8, fontweight="bold", pad=4)
        if col == "RT":
            valid = vals > 0
            ax.semilogx(np.where(valid, vals, np.nan), depth, color=color, lw=0.6)
            ax.set_xlim(0.1, 1000)
        else:
            ax.plot(vals, depth, color=color, lw=0.6)
            if xlim:
                ax.set_xlim(*xlim)
        if col == "CALI" and "BS" in df.columns and not df["BS"].dropna().empty:
            bs_value = float(df["BS"].dropna().iloc[0])
            ax.axvline(bs_value, color="yellow", lw=1.2, ls="--", label=f"BS={bs_value:.2f}")
            ax.legend(fontsize=6, labelcolor="black", framealpha=0.3)
        ax.invert_yaxis()
        ax.set_xlabel(unit, color="black", fontsize=7)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5)

    axes[0].set_ylabel("Depth (m)", color="black", fontsize=9)
    axes[0].yaxis.set_tick_params(labelcolor="black")
    fig.suptitle("Key Log Quality Control Overview", color="black", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig


def _plot_sonic_crossplots(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    washout = df["CALI"] - df["BS"] if "BS" in df.columns else df["CALI"]
    scatter_kw = dict(s=2, alpha=0.5, cmap="plasma")

    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("gray")

    sc = axes[0].scatter(df["DTCO"], df["DTSM"], c=df["GR"], **scatter_kw)
    axes[0].set_xlabel("DTCO (us/ft)", color="black")
    axes[0].set_ylabel("DTSM (us/ft)", color="black")
    axes[0].set_title("DTSM vs DTCO\n(colored by GR)", color="black", fontsize=10)
    plt.colorbar(sc, ax=axes[0]).ax.yaxis.set_tick_params(labelcolor="black")
    dtco_line = np.linspace(40, 140, 100)
    dtsm_castle = 1.16 * dtco_line + 36.6
    axes[0].plot(dtco_line, dtsm_castle, "r--", lw=1.5, label="Castagna Mudrock")
    axes[0].legend(fontsize=8, labelcolor="black", framealpha=0.3)

    sc2 = axes[1].scatter(df["DTCO"], df["VPVS"], c=washout, **scatter_kw)
    axes[1].set_xlabel("DTCO (us/ft)", color="black")
    axes[1].set_ylabel("Vp/Vs", color="black")
    axes[1].set_title("Vp/Vs vs DTCO\n(colored by Washout)", color="black", fontsize=10)
    axes[1].set_ylim(1.3, 3.6)
    plt.colorbar(sc2, ax=axes[1]).ax.yaxis.set_tick_params(labelcolor="black")
    axes[1].axhline(1.5, color="yellow", lw=1, ls="--", label="VpVs=1.5")
    axes[1].legend(fontsize=8, labelcolor="black", framealpha=0.3)

    sc3 = axes[2].scatter(df["RHOB"], df["DTCO"], c=df["GR"], **scatter_kw)
    axes[2].set_xlabel("RHOB (g/cc)", color="black")
    axes[2].set_ylabel("DTCO (us/ft)", color="black")
    axes[2].set_title("DTCO vs RHOB\n(colored by GR)", color="black", fontsize=10)
    plt.colorbar(sc3, ax=axes[2]).ax.yaxis.set_tick_params(labelcolor="black")

    fig.suptitle("Sonic Cross-Plot QC", color="black", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _plot_qc_flags(df: pd.DataFrame, flags: pd.DataFrame) -> plt.Figure:
    flag_specs = [
        ("washout", "peru"),
        ("poor_tension", "gray"),
        ("low_snr", "salmon"),
        ("cycle_skip_co", "royalblue"),
        ("cycle_skip_sm", "cyan"),
        ("vpvs_bad", "purple"),
        ("dci_qc", "olive"),
        ("bad_dtco", "red"),
        ("bad_dtsm", "teal"),
    ]
    available_specs = [(name, color) for name, color in flag_specs if name in flags.columns]
    if not available_specs:
        available_specs = [(name, color) for name, color in [("washout", "peru"), ("cycle_skip_co", "royalblue"), ("cycle_skip_sm", "cyan"), ("iso_anomaly", "purple"), ("bad_dtco", "red"), ("bad_dtsm", "teal"), ("bad_zone", "olive")] if name in flags.columns]

    depth = df["DEPTH_M"].to_numpy(dtype=float)
    n_samples = len(df)
    fig, axes = plt.subplots(1, len(available_specs) + 2, figsize=(26, 14), sharey=True)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("gray")

    axes[0].plot(df["DTCO"], depth, color="royalblue", lw=0.7)
    axes[0].set_xlim(40, 130)
    axes[0].set_title("DTCO", color="black", fontsize=8)
    axes[0].set_xlabel("us/ft", color="black", fontsize=7)
    axes[0].xaxis.set_label_position("top")
    axes[0].xaxis.tick_top()
    axes[0].set_ylabel("Depth (m)", color="black")
    axes[0].yaxis.set_tick_params(labelcolor="black")

    axes[1].plot(df["DTSM"], depth, color="cyan", lw=0.7)
    axes[1].set_xlim(60, 250)
    axes[1].set_title("DTSM", color="black", fontsize=8)
    axes[1].set_xlabel("us/ft", color="black", fontsize=7)
    axes[1].xaxis.set_label_position("top")
    axes[1].xaxis.tick_top()

    for ax, (flag_name, flag_color) in zip(axes[2:], available_specs):
        bad_idx = flags[flag_name].to_numpy(dtype=bool)
        for i in range(1, n_samples):
            if bad_idx[i]:
                ax.barh(
                    depth[i],
                    1,
                    height=abs(depth[i] - depth[i - 1]) + 0.15,
                    left=0,
                    color=flag_color,
                    alpha=0.7,
                    linewidth=0,
                )
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title(flag_name.replace("_", "\n"), color="black", fontsize=7, pad=3)

    for ax in axes:
        ax.invert_yaxis()
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5)

    patches = [Patch(facecolor=color, label=name) for name, color in available_specs]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=min(5, len(available_specs)),
        fontsize=8,
        labelcolor="black",
        framealpha=0.3,
        facecolor="white",
    )
    fig.suptitle("QC Flag Analysis", color="black", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    return fig


def _render_intermediate_qc_panel(frame_data: dict, depth: np.ndarray, selected_file: str | None, selected_frame: str) -> None:
    st.divider()
    st.header("Intermediate QC Diagnostics")

    qc_df = _build_qc_dataframe(frame_data, depth)
    if qc_df.empty or qc_df[["DTCO", "DTSM"]].isna().all().any():
        st.warning("DTCO/DTSM logs are required for QC diagnostics.")
        return

    result = _get_active_qc(selected_file, selected_frame)
    if result is None or "flags" not in result:
        st.caption("Run QC + Cleaning Pipeline in QC & Correction to display the diagnostic plots here.")
        return

    flags = result["flags"]

    st.subheader("Key Log Overview")
    fig1 = _plot_key_log_overview(qc_df)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    st.subheader("Sonic Crossplots")
    fig2 = _plot_sonic_crossplots(qc_df)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.subheader("QC Flags")
    fig3 = _plot_qc_flags(qc_df, flags)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)


def _build_data_summary(data: dict) -> pd.DataFrame:
    summary_rows = []

    for frame_name, frame_data in data.get("frames", {}).items():
        channels = (
            len(frame_data.get("channels", {}))
            + len(frame_data.get("waveforms", {}))
            + len(frame_data.get("semblance", {}))
        )

        depth = frame_data.get("depth")
        pwf2 = frame_data.get("waveforms", {}).get("PWF2")
        dtco = _get_frame_log(frame_data, "DTCO")

        summary_rows.append(
            {
                "Logical File": 0,
                "Frame": frame_name,
                "Channels": channels,
                "PWF2 Shape": str(np.asarray(pwf2).shape) if pwf2 is not None else "-",
                "TDEP Shape": str(np.asarray(depth).shape) if depth is not None else "-",
                "DTCO Shape": str(np.asarray(dtco).shape) if dtco is not None else "-",
            }
        )

    return pd.DataFrame(summary_rows)


def _flag_spans(depth_m: np.ndarray, flag_mask: np.ndarray) -> list[tuple[float, float]]:
    if depth_m.size == 0 or flag_mask.size == 0:
        return []

    spans = []
    start = None

    for idx, flagged in enumerate(flag_mask):
        if flagged and start is None:
            start = idx
        elif not flagged and start is not None:
            spans.append((float(depth_m[start]), float(depth_m[idx - 1])))
            start = None

    if start is not None:
        spans.append((float(depth_m[start]), float(depth_m[-1])))

    return spans


def _get_active_qc(selected_file: str | None, selected_frame: str) -> dict | None:
    qc = st.session_state.get("qc_result")
    if qc is None or st.session_state.get("qc_key") != (selected_file, selected_frame):
        return None

    return qc


def _get_qc_bad_mask(qc: dict | None, target_length: int) -> np.ndarray | None:
    if qc is None or "flags" not in qc:
        return None

    flags = qc["flags"]
    if "final_bad_dtco" in flags:
        bad_mask = flags["final_bad_dtco"].to_numpy(dtype=bool)
    elif "bad_dtco" in flags:
        bad_mask = flags["bad_dtco"].to_numpy(dtype=bool)
    elif "bad_zone" in flags:
        bad_mask = flags["bad_zone"].to_numpy(dtype=bool)
    else:
        return None

    if len(bad_mask) != target_length:
        resized = np.zeros(target_length, dtype=bool)
        resized[: min(target_length, len(bad_mask))] = bad_mask[:target_length]
        return resized

    return bad_mask


def _mask_spr4_bad_zones(spr4: np.ndarray | None, qc: dict | None) -> np.ndarray | None:
    if spr4 is None:
        return None

    bad_mask = _get_qc_bad_mask(qc, spr4.shape[0])
    if bad_mask is None:
        return spr4

    spr4_masked = np.asarray(spr4, dtype=float).copy()
    spr4_masked[bad_mask] = np.nan
    return spr4_masked


def _render_qc_panel(frame_data: dict, depth: np.ndarray, selected_file: str | None, selected_frame: str) -> None:
    st.divider()
    st.subheader("Sonic Log QC & Correction")

    qc_df = _build_qc_dataframe(frame_data, depth)
    if qc_df.empty or qc_df[["DTCO", "DTSM"]].isna().all().any():
        st.warning("DTCO/DTSM logs are required for QC correction.")
        return

    qc_key = (selected_file, selected_frame)
    st.session_state.waveform = qc_df

    if st.button("Run QC + Cleaning Pipeline", type="primary"):
        with st.spinner("Running sonic log QC and correction..."):
            qc_result = run_outlier_detection(st.session_state.waveform)
        st.session_state["qc_result"] = qc_result
        st.session_state["qc_key"] = qc_key
        st.session_state["stage_qc_done"] = True

    if not st.session_state.stage_qc_done or "qc_result" not in st.session_state or st.session_state.get("qc_key") != qc_key:
        st.caption("Run QC + Cleaning Pipeline to populate the intermediate diagnostics, correction plot, and corrected elastic properties.")
        return

    result = st.session_state.qc_result
    flags = result["flags"]
    dtco_corr = result["dtco_corr"]
    dtsm_corr = result["dtsm_corr"]
    st.session_state.cleaned_dtco = dtco_corr
    st.session_state.cleaned_dtsm = dtsm_corr
    st.session_state.qc_flags = flags

    df = st.session_state.waveform.copy()
    if st.session_state.stage_qc_done:
        df["DTCO"] = st.session_state.qc_result["dtco_corr"]
        df["DTSM"] = st.session_state.qc_result["dtsm_corr"]

    df_elastic = df.copy()

    st.subheader("Correction Plot")
    depth_m = qc_df["DEPTH_M"].to_numpy(dtype=float)
    bad_mask = _get_qc_bad_mask(result, len(qc_df))
    if bad_mask is None:
        bad_mask = np.zeros(len(qc_df), dtype=bool)

    fig, axs = plt.subplots(1, 2, figsize=(8, 7), dpi=120, sharey=True)
    plot_specs = [
        ("DTCO", dtco_corr, "Compressional slowness"),
        ("DTSM", dtsm_corr, "Shear slowness"),
    ]

    for ax, (name, corrected, title) in zip(axs, plot_specs):
        raw = qc_df[name].to_numpy(dtype=float)
        ax.plot(raw, depth_m, color="0.25", linewidth=1.0, label=f"Raw {name}")
        ax.plot(corrected, depth_m, color="#d62728", linewidth=1.1, label=f"Corrected {name}")

        for top, bottom in _flag_spans(depth_m, bad_mask):
            ax.axhspan(top, bottom, color="#f2c94c", alpha=0.25, linewidth=0)

        ax.set_xlabel(f"{name} (us/ft)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    axs[0].set_ylabel("Depth (m)", fontsize=9)
    axs[0].invert_yaxis()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.subheader("Elastic Properties (QC Corrected)")

    depth_elastic, vp, vs, nu, E = compute_elastic_from_df(df_elastic)

    fig = plot_elastic(depth_elastic, vp, vs, nu, E)

    st.pyplot(fig)
    plt.close(fig)


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
    st.subheader("Wave Mode")

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

    st.subheader("Semblance Plot")

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

        previous_file = st.session_state.get("selected_file")
        loaded_data = load(uploaded_file.read(), uploaded_file.name)
        st.session_state["raw_data"] = loaded_data
        st.session_state["data"] = loaded_data
        st.session_state["selected_file"] = uploaded_file.name
        if previous_file != uploaded_file.name:
            st.session_state["stage_qc_done"] = False
            st.session_state["qc_result"] = None
            st.session_state["cleaned_dtco"] = None
            st.session_state["cleaned_dtsm"] = None
            st.session_state["qc_flags"] = None

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
    st.subheader("Data Summary")
    st.dataframe(_build_data_summary(data), use_container_width=True)

    st.divider()
    st.subheader("Frame Selection")

    frame_names = data.get("frame_names", [])
    if not frame_names:
        st.error("No frames found in this file.")
    else:
        default_idx = 0
        selected_frame = st.session_state.get("selected_frame")
        previous_frame = selected_frame
        if selected_frame in frame_names:
            default_idx = frame_names.index(selected_frame)

        selected_frame = st.selectbox(
            "Select Frame",
            options=frame_names,
            index=default_idx,
        )
        st.session_state["selected_frame"] = selected_frame
        if previous_frame is not None and previous_frame != selected_frame:
            st.session_state["stage_qc_done"] = False
            st.session_state["qc_result"] = None
            st.session_state["cleaned_dtco"] = None
            st.session_state["cleaned_dtsm"] = None
            st.session_state["qc_flags"] = None

        frame_data = data["frames"][selected_frame]

        depth = frame_data.get("depth")
        wf = frame_data.get("waveforms", {}).get("PWF2")
        spr2 = frame_data.get("semblance", {}).get("SPR2")
        spr4 = frame_data.get("semblance", {}).get("SPR4")
        slowness = frame_data.get("slowness")
        dtco = _get_frame_log(frame_data, "DTCO")
        dtsm = _get_frame_log(frame_data, "DTSM")

        if depth is None:
            st.error("Missing depth data.")
        else:
            depth_raw = np.asarray(depth, dtype=np.float64)
            depth_m = depth_raw * 0.00254
            st.write(
                f"Depth range: {depth_m.min():.2f} m -> {depth_m.max():.2f} m"
            )

            selected_depth = st.slider(
                "Select Depth (m)",
                float(depth_m.min()),
                float(depth_m.max()),
                float(depth_m[len(depth_m) // 2]),
            )

            idx = int(np.argmin(np.abs(depth_m - selected_depth)))
            selected_depth = depth_m[idx]

            st.session_state.waveform = _build_qc_dataframe(frame_data, depth)

            dtco_used = _as_1d_log(dtco)
            dtsm_used = _as_1d_log(dtsm)

            _render_waveform_panel(wf, idx, selected_depth)

            st.divider()
            st.subheader("Window Selection")

            show_wave_windows = st.checkbox("Show wave windows", value=False)

            if show_wave_windows:
                window_data = _resolve_window_data(st.session_state.get("data", {}), selected_frame)

                if window_data is None:
                    st.warning("Windowing data is not available.")
                elif any(window_data.get(key) is None for key in ("depth", "pwf2", "t1r2", "t2r2")):
                    st.warning("Windowing data is incomplete.")
                else:
                    window_depth = np.asarray(window_data["depth"])
                    window_depth_m = window_depth * 0.00254
                    pwf2 = np.asarray(window_data["pwf2"])

                    window_idx = int(np.argmin(np.abs(window_depth_m - selected_depth)))
                    wf_window_depth = pwf2[window_idx]

                    try:
                        windows = build_windows(window_data)
                    except ValueError as exc:
                        st.warning(str(exc))
                    else:
                        fig = plot_window_at_depth(
                            wf_window_depth,
                            float(window_depth_m[window_idx]),
                            windows,
                        )

                        st.pyplot(fig)

            _render_semblance_panel(depth, spr2, spr4, slowness)
            _render_velocity_picking_panel(depth, spr4)
            _render_velocity_comparison_panel(depth, spr4, dtco_used, dtsm_used)
            _render_intermediate_qc_panel(frame_data, depth, selected_file, selected_frame)

            st.markdown("---")
            st.header("QC & Correction")
            _render_qc_panel(frame_data, depth, selected_file, selected_frame)
