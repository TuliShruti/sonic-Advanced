import numpy as np


def compute_semblance(window, velocities, offsets, dt):
    nrec, ns = window.shape
    semblance = []

    for v in velocities:
        stack = np.zeros(ns)

        for i in range(nrec):
            shift = offsets[i] / v
            shift_samples = int(shift / dt)
            shifted = np.roll(window[i], -shift_samples)
            stack += shifted

        num = np.sum(stack)**2
        den = nrec * np.sum(stack**2) + 1e-6

        semblance.append(num / den)

    return np.array(semblance)
