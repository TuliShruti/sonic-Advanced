import numpy as np

from . import RECEIVER_SPACING, STEP, WINDOW_LENGTH
from .semblance import compute_semblance
from .windowing import get_windows


def build_stc(data, velocities, dt, progress_callback=None):
    nz, nrec, ns = data.shape
    offsets = np.arange(nrec) * RECEIVER_SPACING

    stc_all = []

    for z in range(nz):
        gather = data[z]

        windows, t_idx = get_windows(gather, WINDOW_LENGTH, STEP)

        stc_depth = []

        for w in windows:
            s = compute_semblance(w, velocities, offsets, dt)
            stc_depth.append(s)

        stc_all.append(np.array(stc_depth))
        if progress_callback is not None:
            progress_callback((z + 1) / nz)

    return np.array(stc_all), t_idx
