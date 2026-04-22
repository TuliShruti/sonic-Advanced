import numpy as np


def compute_elastic(velocity, density=2300.0, vp_vs_ratio=1.8):
    nz = len(velocity)

    C33 = np.zeros(nz)
    C44 = np.zeros(nz)
    young = np.zeros(nz)
    poisson = np.zeros(nz)

    for iz in range(nz):
        vs = velocity[iz]
        vp = vs * vp_vs_ratio

        C33[iz] = density * vp * vp
        C44[iz] = density * vs * vs

        numerator = (3 * vp * vp) - (4 * vs * vs)
        denominator = (vp * vp) - (vs * vs)

        if denominator != 0:
            young[iz] = density * vs * vs * numerator / denominator
            poisson[iz] = ((vp * vp) - (2 * vs * vs)) / (2 * denominator)
        else:
            young[iz] = 0.0
            poisson[iz] = 0.0

    return {
        "C33": C33,
        "C44": C44,
        "Young": young,
        "Poisson": poisson,
    }
