import numpy as np


def preprocess(data):
    nz, nrec, ns = data.shape
    out = np.zeros_like(data)

    for z in range(nz):
        d = data[z]
        d = d - np.mean(d, axis=1, keepdims=True)
        d = d / (np.max(np.abs(d)) + 1e-6)
        out[z] = d

    return out
