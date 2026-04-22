import numpy as np


def pick_velocity(stc_volume, velocities):
    nz, nt, nv = stc_volume.shape
    picks = np.zeros(nz)
    pick_indices = np.zeros(nz, dtype=int)

    for iz in range(nz):
        stack = np.zeros(nv)

        for it in range(nt):
            stack = stack + stc_volume[iz, it]

        if nt != 0:
            stack = stack / nt

        pick_index = int(np.argmax(stack))
        pick_indices[iz] = pick_index
        picks[iz] = velocities[pick_index]

    return picks, pick_indices
