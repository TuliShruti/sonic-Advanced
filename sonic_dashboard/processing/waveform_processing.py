import numpy as np


def normalize_waveform(arr, receiver=0):
    """
    Convert raw waveform array to 2-D (depth, samples).

    DLIS stores waveforms as (depth, receivers, samples).
    We select a single receiver to get (depth, samples).
    If already 2-D, return as-is.
    """
    if arr is None:
        return None
    if arr.ndim == 3:
        return arr[:, receiver, :]
    if arr.ndim == 2:
        return arr
    return None


def get_waveform_at_depth(waveform, depth_arr, target_depth_raw):
    """
    Return the 1-D trace (samples,) closest to target_depth_raw.
    depth_arr and target_depth_raw must be in the same raw DLIS units.
    """
    if waveform is None or depth_arr is None:
        return None, None
    idx   = int(np.argmin(np.abs(depth_arr - target_depth_raw)))
    trace = waveform[idx, :]
    return trace, idx


def get_time_axis(n_samples, dt_us=10.0):
    """
    Build a time axis in microseconds.
    dt_us: sample interval in µs (default 10 µs → 100 kHz).
    """
    return np.arange(n_samples) * dt_us
