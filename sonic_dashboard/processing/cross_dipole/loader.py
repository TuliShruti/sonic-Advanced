import numpy as np


def read_ldeo_binary(filepath):
    with open(filepath, "rb") as f:
        nz    = np.fromfile(f, dtype=">i4", count=1)[0]
        ns    = np.fromfile(f, dtype=">i4", count=1)[0]
        nrec  = np.fromfile(f, dtype=">i4", count=1)[0]
        _     = np.fromfile(f, dtype=">i4", count=2)

        dz    = np.fromfile(f, dtype=">f4", count=1)[0]
        _     = np.fromfile(f, dtype=">f4", count=1)
        dt    = np.fromfile(f, dtype=">f4", count=1)[0]

        record_len = 1 + nrec * ns
        raw = np.fromfile(f, dtype=">f4")

    data = raw[: nz * record_len].reshape(nz, record_len)

    depth = np.arange(nz) * dz
    waveforms = data[:, 1:].reshape(nz, nrec, ns)

    meta = dict(nz=nz, ns=ns, nrec=nrec, dz=dz, dt=dt)

    return depth, waveforms, meta
