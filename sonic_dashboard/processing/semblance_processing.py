# sonic_dashboard/processing/semblance_processing.py

import numpy as np


# ── Slowness axis ────────────────────────────────────────────
def build_slowness_axis(spr_array, slowness_from_dlis=None):
    n_slow = spr_array.shape[1]
    if slowness_from_dlis is not None:
        if len(slowness_from_dlis) == n_slow:
            return np.array(slowness_from_dlis)
    return np.linspace(40, 240, n_slow)   # µs/ft, matches notebook


# ── Semblance utilities ──────────────────────────────────────
def clip_semblance(spr, vmin=0.0, vmax=1.0):
    return np.clip(spr, vmin, vmax)


def get_semblance_at_depth(spr, depth_arr, target_depth_raw):
    idx = int(np.argmin(np.abs(depth_arr - target_depth_raw)))
    return spr[idx, :]


# ── Picking ──────────────────────────────────────────────────
def pick_slowness(spr, slowness_axis):
    """Peak semblance per depth row → slowness (µs/ft)."""
    peak_indices = np.argmax(spr, axis=1)
    return slowness_axis[peak_indices]


def slowness_to_velocity_ms(slowness_us_ft):
    """
    slowness (µs/ft) → velocity (m/s)
    v = 0.3048e6 / slowness
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(slowness_us_ft > 0, 0.3048e6 / slowness_us_ft, np.nan)


def compute_velocities(spr4, spr2, slowness_axis):
    """
    Pick Vp from SPR4 and Vs from SPR2.
    Returns dict with keys: Vp, Vp_slowness, Vs, Vs_slowness (all np.ndarray).
    """
    result = {}

    if spr4 is not None:
        sl_p = pick_slowness(spr4, slowness_axis)
        result["Vp_slowness"] = sl_p
        result["Vp"]          = slowness_to_velocity_ms(sl_p)

    if spr2 is not None:
        sl_s = pick_slowness(spr2, slowness_axis)
        result["Vs_slowness"] = sl_s
        result["Vs"]          = slowness_to_velocity_ms(sl_s)

    return result