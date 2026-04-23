"""Cross dipole sonic processing page."""

from pathlib import Path
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tempfile

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - fallback when numba is unavailable
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

from sonic_dashboard.processing.cross_dipole import (
    STEP,
    VELOCITY_RANGE,
    WINDOW_LENGTH,
    build_stc,
    pick_velocity,
    read_ldeo_binary,
)
from sonic_dashboard.utils.session_state import initialize_session_state


UPLOAD_DIR = Path("temp_cross_dipole")
UPLOAD_DIR.mkdir(exist_ok=True)
REQUIRED_FILES = {"XX.bin", "XY.bin", "YX.bin", "YY.bin"}
PIPELINE_KEYS = [
    "raw",
    "filt",
    "norm",
    "preprocessed",
    "FF",
    "SS",
    "theta_log",
    "score_log",
    "dts_fast",
    "dts_slow",
    "vs_fast",
    "vs_slow",
    "peak_fast",
    "peak_slow",
    "dts_st",
    "vs_st",
    "peak_sem_st",
    "picks",
]
COMPONENT_LABELS = ["XX", "XY", "YX", "YY"]
F_LOW = 200
F_HIGH = 3000
USE_TIME_GATE = False
T0_SAMPLE = 80
T_WIN = 75


def wiggle_plot(data, depth, dt, receiver_idx):
    nz, _, ns = data.shape
    time_axis = np.arange(ns) * dt
    fig, ax = plt.subplots(figsize=(6, 10))

    for idx in range(0, nz, max(1, nz // 40)):
        trace = data[idx, receiver_idx]
        max_value = np.max(np.abs(trace))
        if max_value > 0:
            trace = trace / max_value
        ax.plot(time_axis, trace + depth[idx], color="black", linewidth=0.8)

    ax.invert_yaxis()
    ax.set_title(f"Wiggle Plot - Receiver {receiver_idx}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Depth")
    fig.tight_layout()
    return fig


def plot_component_trace(data, dt, depth_idx, receiver_idx, title):
    time_axis = np.arange(data.shape[2]) * dt
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(time_axis, data[depth_idx, receiver_idx], color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def plot_stc_slice(stc_slice, velocities, time_axis, picked_velocity=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        stc_slice.T,
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], velocities[0], velocities[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity")
    ax.set_title("STC Slice")
    if picked_velocity is not None:
        ax.axhline(picked_velocity, color="white", linestyle="--", linewidth=1.5)
    fig.colorbar(image, ax=ax, label="Semblance")
    fig.tight_layout()
    return fig


def plot_velocity_log(depth, picks):
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.plot(picks, depth, color="tab:blue")
    ax.invert_yaxis()
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Depth")
    ax.set_title("Velocity Log")
    fig.tight_layout()
    return fig


def clear_downstream(*keys):
    seen_target = False
    for key in ["components", "depth", "meta", *PIPELINE_KEYS]:
        if key in keys:
            seen_target = True
            continue
        if seen_target:
            st.session_state.pop(key, None)


def available_component_files():
    return {
        name.replace(".bin", ""): UPLOAD_DIR / name
        for name in sorted(REQUIRED_FILES)
        if (UPLOAD_DIR / name).exists()
    }


def load_components_from_disk():
    loaded = {}
    depth_ref = None
    meta_ref = None

    for component, path in available_component_files().items():
        depth, waveforms, meta = read_ldeo_binary(path)
        loaded[component] = waveforms
        if depth_ref is None:
            depth_ref = depth
            meta_ref = meta

    return loaded, depth_ref, meta_ref


def select_component_cube(component_bundle, selection):
    return component_bundle[selection]


def symmetrize_components(components):
    out = {name: np.array(data, copy=True) for name, data in components.items()}
    sym = 0.5 * (out["XY"] + out["YX"])
    out["XY"] = sym
    out["YX"] = np.array(sym, copy=True)
    return out


def detrend_traces(data):
    return data - np.mean(data, axis=-1, keepdims=True)


def bandpass_traces(data, dt_us):
    fs = 1e6 / dt_us
    nyq = fs / 2.0
    b, a = butter(4, [F_LOW / nyq, F_HIGH / nyq], btype="band")

    nz, nrec, _ = data.shape
    out = np.empty_like(data)
    for depth_idx in range(nz):
        for rec_idx in range(nrec):
            out[depth_idx, rec_idx] = filtfilt(b, a, data[depth_idx, rec_idx])

    return out


def normalize_traces(data):
    nz, nrec, _ = data.shape
    out = np.empty_like(data)
    for depth_idx in range(nz):
        for rec_idx in range(nrec):
            trace = data[depth_idx, rec_idx]
            scale = np.max(np.abs(trace)) + 1e-10
            out[depth_idx, rec_idx] = trace / scale

    return out


def apply_time_gate(data):
    if not USE_TIME_GATE:
        return data

    _, _, ns = data.shape
    out = np.zeros_like(data)
    lo = max(0, T0_SAMPLE - T_WIN)
    hi = min(ns, T0_SAMPLE + T_WIN)
    out[:, :, lo:hi] = data[:, :, lo:hi]
    return out


def preprocess_components(components, meta):
    dt = float(meta["dt"])
    copied = symmetrize_components(components)
    detrended = {name: detrend_traces(data) for name, data in copied.items()}
    filtered = {name: bandpass_traces(data, dt) for name, data in detrended.items()}
    normalized = {name: normalize_traces(data) for name, data in filtered.items()}
    normalized = {name: apply_time_gate(data) for name, data in normalized.items()}
    return filtered, normalized


def alford_rotate(XX, XY, YX, YY, theta_deg):
    theta = np.radians(theta_deg)
    c_val = np.cos(theta)
    s_val = np.sin(theta)

    rotation = np.array([[c_val, s_val], [-s_val, c_val]])
    matrix = np.array([[XX, XY], [YX, YY]])
    rotated = np.einsum("ik,klrt,jl->ijrt", rotation, matrix, rotation)

    FF = rotated[0, 0]
    FS = rotated[0, 1]
    SF = rotated[1, 0]
    SS = rotated[1, 1]
    return FF, FS, SF, SS


def semblance(traces, t0, win):
    nrec, ns = traces.shape
    lo = max(0, int(t0) - int(win))
    hi = min(ns, int(t0) + int(win))
    window = traces[:, lo:hi]
    stack = np.sum(window, axis=0)

    num = np.sum(stack**2)
    den = nrec * np.sum(window**2) + 1e-10
    return num / den


def energy(traces, t0, win):
    _, ns = traces.shape
    lo = max(0, int(t0) - int(win))
    hi = min(ns, int(t0) + int(win))
    return np.sum(traces[:, lo:hi] ** 2)


def find_theta(XX, XY, YX, YY, theta_step, t0_sample, win, return_scan=False):
    theta_values = np.arange(0.0, 90.0, float(theta_step))
    scores = np.empty(len(theta_values))

    for idx, theta in enumerate(theta_values):
        FF, FS, SF, SS = alford_rotate(XX, XY, YX, YY, theta)

        sFF = semblance(FF, t0_sample, win)
        sSS = semblance(SS, t0_sample, win)
        eFF = energy(FF, t0_sample, win)
        eSS = energy(SS, t0_sample, win)
        eFS = energy(FS, t0_sample, win)
        eSF = energy(SF, t0_sample, win)

        denom = eFF + eSS + 1e-10
        eFS_norm = eFS / denom
        eSF_norm = eSF / denom
        scores[idx] = (sFF + sSS) - 2.0 * (eFS_norm + eSF_norm)

    best_idx = int(np.argmax(scores))
    best_theta = theta_values[best_idx]

    if return_scan:
        return best_theta, scores, theta_values
    return best_theta, scores


@st.cache_data(show_spinner=True)
def cached_alford_rotation(comps, theta_step, t0_sample, win, depth_min, depth_max):
    XX = comps["XX"]
    XY = comps["XY"]
    YX = comps["YX"]
    YY = comps["YY"]
    nz = XX.shape[0]

    theta_log = np.zeros(nz)
    score_log = np.zeros(nz)
    FF_all = np.zeros_like(XX)
    SS_all = np.zeros_like(XX)

    for depth_idx in range(int(depth_min), int(depth_max) + 1):
        XX_i = XX[depth_idx]
        XY_i = XY[depth_idx]
        YX_i = YX[depth_idx]
        YY_i = YY[depth_idx]

        theta, scores = find_theta(
            XX_i,
            XY_i,
            YX_i,
            YY_i,
            theta_step=theta_step,
            t0_sample=t0_sample,
            win=win,
        )
        FF, _, _, SS = alford_rotate(XX_i, XY_i, YX_i, YY_i, theta)

        theta_log[depth_idx] = theta
        score_log[depth_idx] = float(scores.max())
        FF_all[depth_idx] = FF
        SS_all[depth_idx] = SS

    return FF_all, SS_all, theta_log, score_log


@njit(cache=True)
def _build_window_bounds(ns, win):
    lo_idx = np.empty(ns, dtype=np.int64)
    hi_idx = np.empty(ns, dtype=np.int64)

    for time_idx in range(ns):
        lo = time_idx - win
        if lo < 0:
            lo = 0

        hi = time_idx + win
        if hi > ns:
            hi = ns

        lo_idx[time_idx] = lo
        hi_idx[time_idx] = hi

    return lo_idx, hi_idx


@njit(cache=True)
def _build_delay_table(nrec, dt, rec_spacing, slownesses):
    ns_grid = len(slownesses)
    delay_table = np.empty((ns_grid, nrec), dtype=np.int64)

    for slowness_idx in range(ns_grid):
        delay_base = slownesses[slowness_idx] * rec_spacing / dt
        for rec_idx in range(nrec):
            delay_table[slowness_idx, rec_idx] = int(np.rint(delay_base * rec_idx))

    return delay_table


@njit(parallel=True, fastmath=True, cache=True)
def stc_panel_numba(traces, delay_table, lo_idx, hi_idx):
    nrec, ns = traces.shape
    ns_grid = delay_table.shape[0]
    panel = np.zeros((ns_grid, ns), dtype=np.float64)

    for slowness_idx in prange(ns_grid):
        for time_idx in range(ns):
            num = 0.0
            den = 0.0

            for sample_idx in range(lo_idx[time_idx], hi_idx[time_idx]):
                stack = 0.0

                for rec_idx in range(nrec):
                    src_idx = sample_idx + delay_table[slowness_idx, rec_idx]
                    value = 0.0
                    if 0 <= src_idx < ns:
                        value = traces[rec_idx, src_idx]

                    stack += value
                    den += value * value

                num += stack * stack

            if den > 0.0:
                panel[slowness_idx, time_idx] = num / (nrec * den)

    return panel


@njit(parallel=True, fastmath=True, cache=True)
def stc_pick_one_numba(traces, delay_table, lo_idx, hi_idx, slownesses):
    nrec, ns = traces.shape
    ns_grid = delay_table.shape[0]
    best_s_by_slowness = np.empty(ns_grid, dtype=np.float64)
    best_peak_by_slowness = np.empty(ns_grid, dtype=np.float64)

    for slowness_idx in prange(ns_grid):
        best_peak = -1.0

        for time_idx in range(ns):
            num = 0.0
            den = 0.0

            for sample_idx in range(lo_idx[time_idx], hi_idx[time_idx]):
                stack = 0.0

                for rec_idx in range(nrec):
                    src_idx = sample_idx + delay_table[slowness_idx, rec_idx]
                    value = 0.0
                    if 0 <= src_idx < ns:
                        value = traces[rec_idx, src_idx]

                    stack += value
                    den += value * value

                num += stack * stack

            if den > 0.0:
                peak = num / (nrec * den)
                if peak > best_peak:
                    best_peak = peak

        best_s_by_slowness[slowness_idx] = slownesses[slowness_idx]
        best_peak_by_slowness[slowness_idx] = best_peak

    best_idx = 0
    best_peak = best_peak_by_slowness[0]
    for slowness_idx in range(1, ns_grid):
        if best_peak_by_slowness[slowness_idx] > best_peak:
            best_peak = best_peak_by_slowness[slowness_idx]
            best_idx = slowness_idx

    return best_s_by_slowness[best_idx], best_peak


@njit(parallel=True, fastmath=True, cache=True)
def stc_logs_numba(FF, SS, delay_table, lo_idx, hi_idx, slownesses):
    nz = FF.shape[0]
    dts_fast = np.empty(nz, dtype=np.float64)
    dts_slow = np.empty(nz, dtype=np.float64)
    peak_fast = np.empty(nz, dtype=np.float64)
    peak_slow = np.empty(nz, dtype=np.float64)

    for depth_idx in prange(nz):
        s_f, pk_f = stc_pick_one_numba(
            FF[depth_idx],
            delay_table,
            lo_idx,
            hi_idx,
            slownesses,
        )
        dts_fast[depth_idx] = s_f
        peak_fast[depth_idx] = pk_f

        s_s, pk_s = stc_pick_one_numba(
            SS[depth_idx],
            delay_table,
            lo_idx,
            hi_idx,
            slownesses,
        )
        dts_slow[depth_idx] = s_s
        peak_slow[depth_idx] = pk_s

    return dts_fast, dts_slow, peak_fast, peak_slow


def stc_panel_python(traces, dt, rec_spacing, slownesses, win):
    nrec, ns = traces.shape
    offsets = np.arange(nrec, dtype=np.float64) * rec_spacing
    panel = np.zeros((len(slownesses), ns), dtype=np.float64)

    for slowness_idx, s_val in enumerate(slownesses):
        delays = np.rint(s_val * offsets / dt).astype(np.int64)
        corrected = np.zeros((nrec, ns), dtype=np.float64)

        for rec_idx, delay in enumerate(delays):
            if delay == 0:
                corrected[rec_idx] = traces[rec_idx]
            elif 0 < delay < ns:
                corrected[rec_idx, : ns - delay] = traces[rec_idx, delay:]
            elif delay < 0:
                shift = -delay
                corrected[rec_idx, shift:] = traces[rec_idx, : ns - shift]

        for time_idx in range(ns):
            lo = max(0, time_idx - win)
            hi = min(ns, time_idx + win)
            window = corrected[:, lo:hi]
            stack = np.sum(window, axis=0)
            num = np.sum(stack**2)
            den = nrec * np.sum(window**2)
            if den > 0.0:
                panel[slowness_idx, time_idx] = num / den

    return panel


def pick_slowness(panel, slownesses, dt):
    idx = np.unravel_index(np.argmax(panel), panel.shape)
    s_pick = slownesses[idx[0]]
    t_pick = idx[1] * dt
    peak = panel[idx]
    return s_pick, t_pick, peak


def smooth(log_values, win=7):
    return uniform_filter1d(np.asarray(log_values, dtype=float), size=win, mode="nearest")


@st.cache_data(show_spinner=True)
def run_stc_cached(FF, SS, params_tuple):
    dt, rec_spacing, s_min, s_max, s_step, win, use_numba = params_tuple
    slownesses = np.arange(s_min, s_max + s_step, s_step, dtype=np.float64)
    nz = FF.shape[0]
    lo_idx, hi_idx = _build_window_bounds(int(FF.shape[2]), int(win))
    delay_table = _build_delay_table(int(FF.shape[1]), float(dt), float(rec_spacing), slownesses)

    if use_numba:
        dts_fast, dts_slow, peak_fast, peak_slow = stc_logs_numba(
            FF.astype(np.float64),
            SS.astype(np.float64),
            delay_table,
            lo_idx,
            hi_idx,
            slownesses,
        )
    else:
        dts_fast = np.full(nz, np.nan)
        dts_slow = np.full(nz, np.nan)
        peak_fast = np.full(nz, np.nan)
        peak_slow = np.full(nz, np.nan)

        for depth_idx in range(nz):
            panel_f = stc_panel_python(
                FF[depth_idx].astype(np.float64),
                float(dt),
                float(rec_spacing),
                slownesses,
                int(win),
            )
            s_f, _, pk_f = pick_slowness(panel_f, slownesses, dt)
            dts_fast[depth_idx] = s_f
            peak_fast[depth_idx] = pk_f

            panel_s = stc_panel_python(
                SS[depth_idx].astype(np.float64),
                float(dt),
                float(rec_spacing),
                slownesses,
                int(win),
            )
            s_s, _, pk_s = pick_slowness(panel_s, slownesses, dt)
            dts_slow[depth_idx] = s_s
            peak_slow[depth_idx] = pk_s

    vs_fast = 1e6 / dts_fast
    vs_slow = 1e6 / dts_slow
    vs_fast = smooth(vs_fast)
    vs_slow = smooth(vs_slow)

    return dts_fast, dts_slow, vs_fast, vs_slow, peak_fast, peak_slow


def stoneley_preprocess(waveforms, dt, f_low, f_high):
    nz, nrec, _ = waveforms.shape
    fs = 1e6 / dt
    nyq = fs / 2.0
    b, a = butter(4, [f_low / nyq, f_high / nyq], btype="band")

    out = np.zeros_like(waveforms, dtype=np.float32)
    for depth_idx in range(nz):
        for rec_idx in range(nrec):
            trace = filtfilt(b, a, waveforms[depth_idx, rec_idx])
            trace /= np.max(np.abs(trace)) + 1e-10
            out[depth_idx, rec_idx] = trace

    return out


def build_phase_matrix(slowness, offsets, ns, dt):
    freqs = np.fft.rfftfreq(ns, d=dt * 1e-6)
    delays = (slowness[:, None] * 1e-6) * offsets[None, :]
    phase = np.exp(-1j * 2 * np.pi * delays[:, :, None] * freqs[None, None, :])
    return phase.astype(np.complex64)


def stc_panel(traces, dt, phase_matrix, slowness, ns, t_min_us, t_max_us, win_stc):
    nrec = traces.shape[0]
    traces_f = np.fft.rfft(traces, axis=-1)
    shifted_f = phase_matrix * traces_f[None, :, :]
    shifted = np.fft.irfft(shifted_f, n=ns, axis=-1)

    stack = shifted.sum(axis=1)
    stack_sq = stack**2
    shifted_sq = (shifted**2).sum(axis=1)

    c1 = np.cumsum(stack_sq, axis=1)
    c2 = np.cumsum(shifted_sq, axis=1)

    t0 = int(t_min_us / dt)
    t1 = int(t_max_us / dt)
    t_axis = np.arange(t0, t1)

    lo = np.clip(t_axis - win_stc, 0, ns - 1)
    hi = np.clip(t_axis + win_stc, 0, ns - 1)

    num = c1[:, hi] - c1[:, lo]
    den = nrec * (c2[:, hi] - c2[:, lo]) + 1e-6
    panel = num / den

    return panel, t_axis * dt


def extract_features(panel, slowness):
    slow_marginal = panel.max(axis=1)
    peak_sem = slow_marginal.max()
    weights = slow_marginal / (slow_marginal.sum() + 1e-12)

    centroid = np.sum(slowness * weights)
    variance = np.sum(weights * (slowness - centroid) ** 2)
    entropy = -np.sum(weights * np.log(weights + 1e-12))

    return centroid, peak_sem, variance, entropy


def align_log(arr, depth_len):
    if arr is None:
        return np.full(depth_len, np.nan)

    arr = np.asarray(arr)

    if len(arr) == depth_len:
        return arr

    if len(arr) < depth_len:
        out = np.full(depth_len, np.nan)
        out[: len(arr)] = arr
        return out

    return arr[:depth_len]


def fill_nan(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    x = np.arange(n)
    mask = ~np.isnan(arr)

    if mask.sum() == 0:
        return np.full(n, np.nan)

    if mask.sum() == 1:
        out = np.full(n, np.nan)
        out[:] = arr[mask][0]
        return out

    return np.interp(x, x[mask], arr[mask])


def masked(mask, val):
    return np.where(mask, np.nan, val)


def align_to_depth(arr, src_depth, target_depth):
    """Interpolate arr (defined on src_depth) onto target_depth.

    Unlike bare np.interp, values of target_depth that fall *outside* the
    range of src_depth are set to NaN rather than being clamped to the edge
    value.  This prevents elastic logs from appearing as flat lines beyond
    the well-log depth range, which was the root cause of the truncated
    elastic track in P7.
    """
    if arr is None or src_depth is None:
        return np.full(len(target_depth), np.nan, dtype=float)

    arr = np.asarray(arr, dtype=float)
    src_depth = np.asarray(src_depth, dtype=float)
    target_depth = np.asarray(target_depth, dtype=float)

    if len(arr) != len(src_depth):
        return np.full(len(target_depth), np.nan, dtype=float)

    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return np.full(len(target_depth), np.nan, dtype=float)

    src_d_valid = src_depth[valid]
    arr_valid = arr[valid]

    # Interpolate without clamping — use left=nan, right=nan so that
    # target depths outside [src_d_valid.min(), src_d_valid.max()] are NaN.
    out = np.interp(
        target_depth,
        src_d_valid,
        arr_valid,
        left=np.nan,
        right=np.nan,
    )
    return out


initialize_session_state()

if "depth_full" not in st.session_state and "depth" in st.session_state:
    st.session_state["depth_full"] = st.session_state["depth"]

st.title("Cross Dipole Sonic")

with st.expander("P1 - Data Loading & QC", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload XX, XY, YX, YY (.bin)",
        type=["bin"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as handle:
                handle.write(uploaded_file.read())

    existing_files = {path.name for path in UPLOAD_DIR.glob("*.bin")}
    missing_files = sorted(REQUIRED_FILES - existing_files)

    if missing_files:
        st.info(f"Upload all 4 files to proceed. Missing: {', '.join(missing_files)}")
        st.stop()

    raw_components, raw_depth, raw_meta = load_components_from_disk()
    component_selection = st.selectbox("Component", COMPONENT_LABELS, key="P1_component")
    depth_idx = st.slider("Depth slider", 0, int(raw_meta["nz"]) - 1, 0, key="P1_depth_slider")
    receiver_idx = st.selectbox(
        "Receiver",
        list(range(int(raw_meta["nrec"]))),
        key="P1_receiver",
    )

    selected_cube = select_component_cube(raw_components, component_selection)

    qc_col1, qc_col2 = st.columns(2)
    with qc_col1:
        fig = wiggle_plot(selected_cube, raw_depth, float(raw_meta["dt"]), int(receiver_idx))
        st.pyplot(fig)
        plt.close(fig)

    with qc_col2:
        fig = plot_component_trace(
            selected_cube,
            float(raw_meta["dt"]),
            int(depth_idx),
            int(receiver_idx),
            f"{component_selection} Trace Preview",
        )
        st.pyplot(fig)
        plt.close(fig)

    metadata_table = pd.DataFrame(
        [
            {"Field": "Selected component", "Value": component_selection},
            {"Field": "Depth samples", "Value": int(raw_meta["nz"])},
            {"Field": "Time samples", "Value": int(raw_meta["ns"])},
            {"Field": "Receivers", "Value": int(raw_meta["nrec"])},
            {"Field": "Depth interval", "Value": float(raw_meta["dz"])},
            {"Field": "Sample interval", "Value": float(raw_meta["dt"])},
        ]
    )
    st.table(metadata_table)

    if st.button("Store P1 Output", use_container_width=True):
        clear_downstream("components")
        st.session_state["components"] = {
            name: np.array(data, copy=True) for name, data in raw_components.items()
        }
        st.session_state["depth"] = raw_depth
        if "depth_full" not in st.session_state:
            st.session_state["depth_full"] = raw_depth
        st.session_state["meta"] = {
            **raw_meta,
            "component_selection": component_selection,
        }
        st.success("P1 output stored in session state.")

if st.session_state.get("components") is None:
    st.stop()

with st.expander("P2 - Preprocessing", expanded=False):
    normalize = st.toggle("Normalize", value=True, key="P2_normalize")
    st.checkbox("Filter placeholder", value=False, disabled=True, key="P2_filter_placeholder")
    st.checkbox("Gate placeholder", value=False, disabled=True, key="P2_gate_placeholder")

    depth = st.session_state["depth_full"]
    nz = depth.shape[0]
    nrec = st.session_state["components"]["XX"].shape[1]
    depth_idx = st.slider("Depth index", 0, nz - 1, nz // 2, key="P2_depth_index")
    rec_idx = st.selectbox("Receiver", list(range(nrec)), key="P2_receiver")

    if st.button("Apply Preprocessing", use_container_width=True):
        clear_downstream("raw")
        components = {
            name: np.array(data, copy=True) for name, data in st.session_state["components"].items()
        }
        filt_components, norm_components = preprocess_components(components, st.session_state["meta"])

        st.session_state["raw"] = components
        st.session_state["filt"] = filt_components
        st.session_state["norm"] = norm_components
        st.session_state["preprocessed"] = norm_components if normalize else filt_components
        st.success("P2 output stored in session state.")

    if (
        st.session_state.get("raw") is not None
        and st.session_state.get("filt") is not None
        and st.session_state.get("norm") is not None
    ):
        dt = st.session_state["meta"]["dt"]
        ns = st.session_state["raw"]["XX"].shape[2]
        t = np.arange(ns) * dt

        row_data = [
            st.session_state["raw"],
            st.session_state["filt"],
            st.session_state["norm"],
        ]
        row_tags = ["raw (ADC counts)", "bandpass (ADC counts)", "normalized (+/-1)"]
        colors = ["black", "steelblue", "crimson"]

        fig, axes = plt.subplots(3, 4, figsize=(14, 7), sharex=True)

        for row in range(3):
            for col, comp in enumerate(COMPONENT_LABELS):
                trace = row_data[row][comp][depth_idx, rec_idx]
                ax = axes[row, col]
                ax.plot(t, trace, color=colors[row], linewidth=0.8)
                ax.axhline(0, linestyle="--", linewidth=0.3)

                if row == 0:
                    ax.set_title(comp)

                if col == 0:
                    ax.set_ylabel(row_tags[row], fontsize=8)

                if row == 2:
                    ax.set_xlabel("Time (us)", fontsize=8)

        st.pyplot(fig)
        plt.close(fig)

with st.expander("P3 - Alford Rotation", expanded=False):
    if "norm" not in st.session_state:
        st.warning("Complete preprocessing first")
        st.stop()

    theta_step = st.number_input(
        "Theta step (degrees)",
        value=1.0,
        min_value=0.1,
        step=0.1,
        key="P3_theta_step",
    )
    t0_sample = st.number_input(
        "Semblance window center (samples)",
        value=100,
        min_value=0,
        step=1,
        key="P3_t0_sample",
    )
    win = st.number_input(
        "Window half-width (samples)",
        value=60,
        min_value=1,
        step=1,
        key="P3_win",
    )
    depth_min, depth_max = st.slider(
        "Depth range to process",
        0,
        len(st.session_state["depth_full"]) - 1,
        (0, len(st.session_state["depth_full"]) - 1),
        key="P3_depth_range",
    )
    preview_depth = st.slider(
        "Preview depth",
        0,
        len(st.session_state["depth_full"]) - 1,
        len(st.session_state["depth_full"]) // 2,
        key="P3_preview_depth",
    )
    show_theta_scan = st.checkbox("Show theta scan at preview depth", value=False, key="P3_theta_scan")

    if st.button("Run Alford Rotation", use_container_width=True):
        clear_downstream("FF")
        comps = st.session_state["norm"]
        FF_all, SS_all, theta_log, score_log = cached_alford_rotation(
            comps,
            float(theta_step),
            int(t0_sample),
            int(win),
            int(depth_min),
            int(depth_max),
        )

        st.session_state["FF"] = FF_all
        st.session_state["SS"] = SS_all
        st.session_state["theta_log"] = theta_log
        st.session_state["score_log"] = score_log
        st.success("Alford rotation finished.")

    if "theta_log" in st.session_state or "FF" in st.session_state:
        with st.container():
            col1, col2 = st.columns([1, 1])

            if "theta_log" in st.session_state:
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 6))
                    fig.set_dpi(100)
                    ax.plot(st.session_state["theta_log"], st.session_state["depth_full"])
                    ax.invert_yaxis()
                    ax.set_xlabel("Theta (deg)")
                    ax.set_ylabel("Depth (m)")
                    ax.set_title("Theta Log")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)

            if "FF" in st.session_state:
                with col2:
                    FF = st.session_state["FF"]
                    SS = st.session_state["SS"]

                    fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
                    fig.set_dpi(100)
                    ax[0].plot(FF[preview_depth, 0])
                    ax[0].set_title("FF")
                    ax[1].plot(SS[preview_depth, 0])
                    ax[1].set_title("SS")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)

        if show_theta_scan and "FF" in st.session_state:
            comps = st.session_state["norm"]
            theta, scores, theta_values = find_theta(
                comps["XX"][preview_depth],
                comps["XY"][preview_depth],
                comps["YX"][preview_depth],
                comps["YY"][preview_depth],
                theta_step=float(theta_step),
                t0_sample=int(t0_sample),
                win=int(win),
                return_scan=True,
            )

            fig, ax = plt.subplots(figsize=(4, 3))
            fig.set_dpi(100)
            ax.plot(theta_values, scores, color="tab:blue")
            ax.axvline(theta, color="tab:red", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Theta (deg)")
            ax.set_ylabel("Score")
            ax.set_title("Theta Scan")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

if st.session_state.get("FF") is None:
    st.stop()

with st.expander("P4 - STC (Slowness-Time Coherence)", expanded=False):
    if "FF" not in st.session_state:
        st.warning("Run Alford rotation first")
        st.stop()

    meta = st.session_state["meta"]
    dt = float(meta["dt"])

    rec_spacing = st.number_input(
        "Receiver spacing (m)",
        value=float(meta.get("rec_spacing", 0.1524)),
        key="P4_rec_spacing",
    )
    s_min = st.number_input("S_min (us/m)", value=300, key="P4_s_min")
    s_max = st.number_input("S_max (us/m)", value=2500, key="P4_s_max")
    s_step = st.number_input("S_step (us/m)", value=8, key="P4_s_step")
    win = st.number_input("Semblance half-window (samples)", value=40, key="P4_win")
    nz = len(st.session_state["depth_full"])
    depth_min, depth_max = st.slider(
        "Depth range",
        0,
        nz - 1,
        (0, nz - 1),
        key="P4_depth_range",
    )

    slownesses = np.arange(s_min, s_max + s_step, s_step, dtype=np.float64)
    preview_depths = st.multiselect(
        "Preview depths (for STC panel)",
        options=list(range(len(st.session_state["depth_full"]))),
        default=[len(st.session_state["depth_full"]) // 2],
        key="P4_preview_depths",
    )
    use_numba = st.checkbox("Use Numba acceleration", value=True, key="P4_numba_toggle")

    if st.button("Run STC", use_container_width=True):
        clear_downstream("dts_fast")
        FF = st.session_state["FF"]
        SS = st.session_state["SS"]

        dts_fast, dts_slow, vs_fast, vs_slow, peak_fast, peak_slow = run_stc_cached(
            FF,
            SS,
            (
                float(dt),
                float(rec_spacing),
                float(s_min),
                float(s_max),
                float(s_step),
                int(win),
                bool(use_numba),
            ),
        )

        depth = st.session_state["depth_full"]
        assert len(vs_fast) == len(depth)
        assert len(vs_slow) == len(depth)

        st.session_state["dts_fast"] = dts_fast
        st.session_state["dts_slow"] = dts_slow
        st.session_state["vs_fast"] = vs_fast
        st.session_state["vs_slow"] = vs_slow
        st.session_state["peak_fast"] = peak_fast
        st.session_state["peak_slow"] = peak_slow
        st.success("STC logs computed.")

    if "vs_fast" in st.session_state:
        depth = st.session_state["depth_full"]
        vs_fast = st.session_state["vs_fast"]
        vs_slow = st.session_state["vs_slow"]
        peak_fast = st.session_state["peak_fast"]
        peak_slow = st.session_state["peak_slow"]
        plot_slice = slice(int(depth_min), int(depth_max) + 1)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 6))
            fig.set_dpi(100)
            ax.plot(vs_fast[plot_slice], depth[plot_slice], label="Vs_fast")
            ax.plot(vs_slow[plot_slice], depth[plot_slice], label="Vs_slow")
            ax.invert_yaxis()
            ax.legend()
            ax.set_title("Velocity Logs")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 6))
            fig.set_dpi(100)
            ax.plot(peak_fast[plot_slice], depth[plot_slice], label="fast")
            ax.plot(peak_slow[plot_slice], depth[plot_slice], label="slow")
            ax.axvline(0.5, linestyle="--")
            ax.invert_yaxis()
            ax.legend()
            ax.set_title("Semblance QC")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        preview_depths_limited = preview_depths[:3]
        if preview_depths_limited:
            preview_columns = st.columns(len(preview_depths_limited))
            lo_idx, hi_idx = _build_window_bounds(
                int(st.session_state["FF"].shape[2]),
                int(win),
            )
            delay_table = _build_delay_table(
                int(st.session_state["FF"].shape[1]),
                float(dt),
                float(rec_spacing),
                slownesses,
            )

            for col, preview_depth in zip(preview_columns, preview_depths_limited):
                with col:
                    if use_numba:
                        panel = stc_panel_numba(
                            st.session_state["FF"][preview_depth].astype(np.float64),
                            delay_table,
                            lo_idx,
                            hi_idx,
                        )
                    else:
                        panel = stc_panel_python(
                            st.session_state["FF"][preview_depth].astype(np.float64),
                            float(dt),
                            float(rec_spacing),
                            slownesses,
                            int(win),
                        )
                    fig, ax = plt.subplots(figsize=(4, 3))
                    fig.set_dpi(100)
                    ax.imshow(panel, aspect="auto", origin="lower")
                    ax.set_title(f"STC Depth {preview_depth}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Slowness")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)

with st.expander("P5 - Stoneley Wave Processing", expanded=False):
    uploaded_st = st.file_uploader(
        "Upload Stoneley .bin file",
        type=["bin"],
        key="P5_upload",
    )

    if uploaded_st is None:
        st.info("Upload Stoneley file to proceed")
        st.stop()

    f_low = st.number_input("F_LOW Hz", value=400, key="P5_flow")
    f_high = st.number_input("F_HIGH Hz", value=2500, key="P5_fhigh")
    s_min = st.number_input("S_MIN (us/m)", value=250, key="P5_smin")
    s_max = st.number_input("S_MAX (us/m)", value=1200, key="P5_smax")
    s_step = st.number_input("S_STEP", value=2, key="P5_sstep")
    t_min = st.number_input("T_MIN (us)", value=3000, key="P5_tmin")
    t_max = st.number_input("T_MAX (us)", value=12000, key="P5_tmax")
    win = st.number_input("WIN_STC", value=20, key="P5_win")
    preview_depths = st.multiselect(
        "Preview depths",
        options=list(range(len(st.session_state["depth_full"]))),
        default=[len(st.session_state["depth_full"]) // 2],
        key="P5_preview",
    )

    stoneley_upload_bytes = uploaded_st.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(stoneley_upload_bytes)
        tmp_path = tmp.name

    depth_loaded, waveforms_raw, meta_loaded = read_ldeo_binary(tmp_path)
    waveforms = waveforms_raw
    depth = depth_loaded
    dt = float(meta_loaded["dt"])
    nz, nrec, ns = waveforms.shape

    depth_min, depth_max = st.slider(
        "Depth range to process",
        0,
        nz - 1,
        (0, nz - 1),
        key="P5_depth_range",
    )

    if st.button("Run Stoneley Processing", use_container_width=True):
        clear_downstream("dts_st")
        waveforms_processed = stoneley_preprocess(waveforms, dt, float(f_low), float(f_high))

        slowness = np.arange(s_min, s_max, s_step)
        offsets = np.arange(nrec) * 0.1524
        phase_matrix = build_phase_matrix(slowness, offsets, ns, dt)

        features = []
        velocity_subset = []
        raw_slowness = []
        peak_sem = []

        for depth_idx in range(int(depth_min), int(depth_max) + 1):
            panel, _ = stc_panel(
                waveforms_processed[depth_idx],
                dt,
                phase_matrix,
                slowness,
                ns,
                float(t_min),
                float(t_max),
                int(win),
            )

            centroid, sem, var, ent = extract_features(panel, slowness)
            features.append([centroid, sem, var, ent])
            raw_slowness.append(centroid)
            peak_sem.append(sem)

        features = np.array(features)
        raw_slowness = np.array(raw_slowness)
        peak_sem = np.array(peak_sem)

        good = peak_sem > 0.6

        if good.sum() < 30:
            predicted_slowness = gaussian_filter1d(raw_slowness, 5)
        else:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("gbr", GradientBoostingRegressor()),
                ]
            )
            model.fit(features[good], raw_slowness[good])
            predicted_slowness = model.predict(features)

        predicted_slowness = gaussian_filter1d(predicted_slowness, 3)
        velocity_subset = 1e6 / predicted_slowness

        depth = st.session_state["depth_full"]
        nz = len(depth)

        vs_st_full = np.full(nz, np.nan)
        dts_st_full = np.full(nz, np.nan)
        peak_st_full = np.full(nz, np.nan)

        vs_st_full[int(depth_min) : int(depth_max) + 1] = velocity_subset
        dts_st_full[int(depth_min) : int(depth_max) + 1] = predicted_slowness
        peak_st_full[int(depth_min) : int(depth_max) + 1] = peak_sem

        st.session_state["dts_st"] = dts_st_full
        st.session_state["vs_st"] = vs_st_full
        st.session_state["peak_sem_st"] = peak_st_full
        st.success("Stoneley processing finished.")

    if "vs_st" in st.session_state:
        col1, col2 = st.columns(2)
        vs_st = st.session_state["vs_st"]
        peak_sem_st = st.session_state["peak_sem_st"]
        depth = st.session_state["depth_full"]
        valid_vs = ~np.isnan(vs_st)
        valid_peak = ~np.isnan(peak_sem_st)

        st.write("depth_full:", len(st.session_state["depth_full"]))
        st.write("vs_st:", len(vs_st))

        with col1:
            fig, ax = plt.subplots(figsize=(4, 6))
            fig.set_dpi(100)
            ax.plot(vs_st[valid_vs], depth[valid_vs])
            ax.invert_yaxis()
            ax.set_title("Stoneley Velocity")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 6))
            fig.set_dpi(100)
            ax.plot(peak_sem_st[valid_peak], depth[valid_peak])
            ax.axvline(0.6, linestyle="--")
            ax.invert_yaxis()
            ax.set_title("Stoneley QC")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        preview_depths_limited = preview_depths[:2]
        if preview_depths_limited:
            waveforms_processed = stoneley_preprocess(waveforms, dt, float(f_low), float(f_high))
            slowness = np.arange(s_min, s_max, s_step)
            offsets = np.arange(nrec) * 0.1524
            phase_matrix = build_phase_matrix(slowness, offsets, ns, dt)
            preview_columns = st.columns(len(preview_depths_limited))

            for col, preview_depth in zip(preview_columns, preview_depths_limited):
                with col:
                    panel, _ = stc_panel(
                        waveforms_processed[preview_depth],
                        dt,
                        phase_matrix,
                        slowness,
                        ns,
                        float(t_min),
                        float(t_max),
                        int(win),
                    )
                    fig, ax = plt.subplots(figsize=(4, 3))
                    fig.set_dpi(100)
                    ax.imshow(panel, aspect="auto", origin="lower")
                    ax.set_title(f"Stoneley STC {preview_depth}")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)

with st.expander("P6 - VTI Stiffness & Elastic Properties", expanded=False):
    depth = st.session_state.get("depth_full")
    vs_fast = st.session_state.get("vs_fast")
    vs_slow = st.session_state.get("vs_slow")
    vs_st = st.session_state.get("vs_st")

    if depth is None or vs_fast is None or vs_slow is None or vs_st is None:
        st.info("Run shear STC and Stoneley processing to populate logs.")
        st.stop()

    uploaded_log = st.file_uploader(
        "Upload well log (.xlsx with RHOB, DTCO)",
        type=["xlsx"],
        key="P6_upload",
    )

    if uploaded_log is None:
        st.info("Upload well log to proceed")
        st.stop()

    df = pd.read_excel(uploaded_log)

    required = ["Depth_m", "RHOB", "DTCO"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    rho_mud = st.number_input("RHO_MUD (kg/m3)", value=1200.0, key="P6_rho_mud")
    vp_mud = st.number_input("VP_MUD (m/s)", value=1500.0, key="P6_vp_mud")
    delta_v = st.number_input("DELTA_V", value=20.0, key="P6_delta")
    dtco_unit = st.selectbox("DTCO unit", ["us/ft", "us/m"], key="P6_dtco_unit")
    st.session_state["vp_mud"] = float(vp_mud)

    well_depth = df["Depth_m"].values
    rho_b = df["RHOB"].values * 1000.0

    if dtco_unit == "us/ft":
        Vp = 0.3048e6 / df["DTCO"].values
    else:
        Vp = 1e6 / df["DTCO"].values

    if "VS_fast_m_per_s" in df.columns:
        Vs_fast = df["VS_fast_m_per_s"].values
    else:
        Vs_fast = np.interp(well_depth, depth, vs_fast)

    if "VS_slow_m_per_s" in df.columns:
        Vs_slow = df["VS_slow_m_per_s"].values
    else:
        Vs_slow = np.interp(well_depth, depth, vs_slow)

    vs_st_smooth = gaussian_filter1d(vs_st, sigma=5)
    st.session_state["vs_st_smooth"] = vs_st_smooth

    V_st = np.interp(well_depth, depth, st.session_state.get("vs_st_smooth"))

    near_mud = np.abs(V_st - vp_mud) < delta_v
    V_st_clean = V_st.copy()
    V_st_clean[near_mud] = np.nan

    valid = ~np.isnan(V_st_clean)
    if np.sum(valid) >= 2:
        V_st_interp = np.interp(
            well_depth,
            well_depth[valid],
            V_st_clean[valid],
        )
    else:
        V_st_interp = np.full_like(well_depth, np.nan, dtype=float)

    st.write("Stoneley valid %:", np.sum(~np.isnan(V_st_interp)) / len(V_st_interp))

    eps = 1e-8
    denom = (1 / V_st_interp**2) - (1 / vp_mud**2)
    denom[np.abs(denom) < eps] = np.nan

    bad_rho = (rho_b <= 0) | np.isnan(rho_b)
    bad_vp = (Vp <= 0) | np.isnan(Vp)
    bad_vs1 = (Vs_fast <= 0) | np.isnan(Vs_fast)
    bad_vs2 = (Vs_slow <= 0) | np.isnan(Vs_slow)
    bad_st = np.isnan(V_st_interp) | np.isnan(denom)

    C33 = masked(bad_rho | bad_vp, rho_b * Vp**2)
    C44 = masked(bad_rho | bad_vs1, rho_b * Vs_fast**2)
    C55 = masked(bad_rho | bad_vs2, rho_b * Vs_slow**2)
    C66 = masked(bad_st, rho_mud / denom)

    C13 = C33 - 2 * C44
    C11 = C13 + 2 * C66

    C11 /= 1e9
    C13 /= 1e9
    C33 /= 1e9
    C44 /= 1e9
    C55 /= 1e9
    C66 /= 1e9

    C11[C11 < 0] = np.nan
    C33[C33 < 0] = np.nan
    C44[C44 < 0] = np.nan
    C55[C55 < 0] = np.nan

    C12 = C11 - 2 * C66

    denom_vert = C11 + C12
    denom_vert[np.abs(denom_vert) < 1e-6] = np.nan
    denom_E = C11 * C33 - C13**2
    denom_E[np.abs(denom_E) < 1e-6] = np.nan

    E33 = C33 - (2 * (C13**2) / denom_vert)
    nu_vert = C13 / denom_vert

    E11 = (
        ((C11 - C12) * (C11 * C33 - 2 * (C13**2) + C12 * C33))
        / denom_E
    )
    nu_horiz = (C12 * C33 - C13**2) / denom_E

    E11[(E11 < 5) | (E11 > 200)] = np.nan
    nu_horiz[(nu_horiz < -1) | (nu_horiz > 0.5)] = np.nan
    nu_vert[(nu_vert < -1) | (nu_vert > 0.5)] = np.nan

    print("Denom valid %:", np.sum(~np.isnan(denom)) / len(denom))
    print("E11 valid %:", np.sum(~np.isnan(E11)) / len(E11))

    # Re-interpolate every elastic array from well_depth onto depth_full so
    # that P7 receives full-length arrays already aligned to the seismic
    # depth axis.  This eliminates the depth-coordinate mismatch that caused
    # the truncated elastic track in P7 when well_depth (Excel) and
    # depth_full (binary) had different length / range.
    depth_full = st.session_state["depth_full"]

    def _to_full(arr_well):
        return align_to_depth(arr_well, well_depth, depth_full)

    st.session_state["C11"] = _to_full(C11)
    st.session_state["C33"] = _to_full(C33)
    st.session_state["C44"] = _to_full(C44)
    st.session_state["C66"] = _to_full(C66)
    st.session_state["E11"] = _to_full(E11)
    st.session_state["E33"] = _to_full(E33)
    st.session_state["nu_horiz"] = _to_full(nu_horiz)
    st.session_state["nu_vert"] = _to_full(nu_vert)
    # vti_depth now matches depth_full so align_to_depth in P7 is a no-op
    # identity interpolation rather than a cross-coordinate remapping.
    st.session_state["vti_depth"] = depth_full

    st.write("Valid E11 %:", np.sum(~np.isnan(E11)) / len(E11))
    st.write("Valid C66 %:", np.sum(~np.isnan(C66)) / len(C66))

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(4, 6))
        fig.set_dpi(100)
        ax.plot(C33, well_depth, label="C33")
        ax.plot(C11, well_depth, label="C11")
        ax.invert_yaxis()
        ax.legend()
        ax.set_title("Stiffness")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4, 6))
        fig.set_dpi(100)
        ax.plot(E11, well_depth, label="E11")
        ax.plot(E33, well_depth, label="E33")
        ax.invert_yaxis()
        ax.legend()
        ax.set_title("Young Modulus")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    df_out = pd.DataFrame(
        {
            "Depth": well_depth,
            "Vp": Vp,
            "Vs_fast": Vs_fast,
            "Vs_slow": Vs_slow,
            "V_st": V_st,
            "C11": C11,
            "C33": C33,
            "C44": C44,
            "C66": C66,
            "E11": E11,
            "E33": E33,
            "nu_h": nu_horiz,
            "nu_v": nu_vert,
        }
    )

    st.dataframe(df_out.head(50), use_container_width=True)

with st.expander("P7 - Composite Log Viewer & Export", expanded=True):
    depth = st.session_state.get("depth_full")

    if depth is None:
        st.info("Load data first to view composite logs.")
        st.stop()

    nz = len(depth)
    vti_depth = st.session_state.get("vti_depth")

    theta_source = st.session_state.get("theta")
    if theta_source is None:
        theta_source = st.session_state.get("theta_log")

    vs_fast = align_log(st.session_state.get("vs_fast"), nz)
    vs_slow = align_log(st.session_state.get("vs_slow"), nz)
    vs_st = align_to_depth(st.session_state.get("vs_st"), depth, depth)
    theta = align_log(theta_source, nz)
    E11 = align_to_depth(st.session_state.get("E11"), vti_depth, depth)
    E33 = align_to_depth(st.session_state.get("E33"), vti_depth, depth)
    C11 = align_to_depth(st.session_state.get("C11"), vti_depth, depth)
    C33 = align_to_depth(st.session_state.get("C33"), vti_depth, depth)
    C44 = align_to_depth(st.session_state.get("C44"), vti_depth, depth)
    C66 = align_to_depth(st.session_state.get("C66"), vti_depth, depth)
    nu_h = align_to_depth(st.session_state.get("nu_horiz"), vti_depth, depth)
    peak_f = align_log(st.session_state.get("peak_fast"), nz)
    peak_s = align_log(st.session_state.get("peak_slow"), nz)

    # Validate that all aligned logs are the same length as depth before
    # doing any slicing.  A mismatch here means a previous panel wrote an
    # array of the wrong length — surface it clearly rather than silently
    # plotting misaligned data.
    _shapes = {
        "depth": len(depth),
        "vs_fast": len(vs_fast),
        "vs_slow": len(vs_slow),
        "vs_st": len(vs_st),
        "theta": len(theta),
        "E11": len(E11),
        "E33": len(E33),
        "C11": len(C11),
        "C33": len(C33),
        "C44": len(C44),
        "C66": len(C66),
        "nu_h": len(nu_h),
        "peak_f": len(peak_f),
        "peak_s": len(peak_s),
    }
    print("P7 array lengths:", _shapes)
    bad = {k: v for k, v in _shapes.items() if v != len(depth)}
    if bad:
        st.error(f"Array length mismatch vs depth ({len(depth)}): {bad}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        show_stoneley = st.checkbox("Show Stoneley", True, key="P7_st")
        show_theta = st.checkbox("Show Theta", True, key="P7_theta")

    with col2:
        depth_min, depth_max = st.slider(
            "Depth zoom",
            0,
            len(depth) - 1,
            (0, len(depth) - 1),
            key="P7_depth_zoom",
        )

    # ── Depth zoom ────────────────────────────────────────────────────────
    # Slice ALL arrays with the *same* integer index range so every track
    # in the composite plot shares an identical depth axis.  Do NOT slice
    # inside individual ax.plot() calls — that was the pattern that caused
    # elastic tracks to appear vertically compressed.
    sl = slice(depth_min, depth_max + 1)

    d      = depth[sl]
    vf     = vs_fast[sl]
    vs_    = vs_slow[sl]
    vst    = vs_st[sl]
    th     = theta[sl]
    pf     = peak_f[sl]
    ps     = peak_s[sl]
    _E11   = E11[sl]
    _E33   = E33[sl]
    _nu_h  = nu_h[sl]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(10, 6),
        sharey=True,
    )
    fig.set_dpi(100)

    # Track 0 — Velocity
    axes[0].plot(vf, d, label="Vs_fast")
    axes[0].plot(vs_, d, label="Vs_slow")
    if show_stoneley:
        axes[0].plot(vst, d, label="Stoneley")
    axes[0].invert_yaxis()
    axes[0].set_title("Velocity")
    axes[0].legend(fontsize=7)

    # Track 1 — Semblance QC
    axes[1].plot(pf, d, label="fast")
    axes[1].plot(ps, d, label="slow")
    axes[1].axvline(0.5, linestyle="--")
    axes[1].set_title("Semblance")

    # Track 2 — Theta
    if show_theta:
        axes[2].plot(th, d)
    axes[2].set_title("Theta")

    # Track 3 — Elastic  (uses the same d — no separate masking)
    axes[3].plot(_E11,  d, label="E11")
    axes[3].plot(_E33,  d, label="E33")
    axes[3].plot(_nu_h, d, label="nu")
    axes[3].set_title("Elastic")
    axes[3].legend(fontsize=7)

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    export_df = pd.DataFrame(
        {
            "Depth": depth,
            "Vs_fast": vs_fast,
            "Vs_slow": vs_slow,
            "Vs_st": vs_st,
            "Theta": theta,
            "C11": C11,
            "C33": C33,
            "C44": C44,
            "C66": C66,
            "E11": E11,
            "E33": E33,
            "nu": nu_h,
        }
    )
    export_df = export_df.dropna(axis=1, how="all")

    csv = export_df.to_csv(index=False).encode()
    st.download_button(
        "Download CSV",
        csv,
        "sonic_results.csv",
        "text/csv",
        key="P7_download",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        "Download Plot",
        buf.getvalue(),
        "log_plot.png",
        "image/png",
        key="P7_img",
    )
    plt.close(fig)
