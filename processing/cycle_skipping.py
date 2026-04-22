import numpy as np
import pandas as pd


def detect_bad_zones(df, dt_ch, cali_ch, bit_size_inch, washout_factor,
                     spike_threshold, rolling_window, min_bad_run, logic='OR'):

    d = df.copy()

    # -- DT spike detection -------------------------------------------------
    d["DT_ROLLING"] = d[dt_ch].rolling(rolling_window, center=True, min_periods=1).median()
    d["DT_DIFF"] = d[dt_ch].diff().abs()
    d["DT_DEVIATION"] = (d[dt_ch] - d["DT_ROLLING"]).abs()

    d["FLAG_SPIKE"] = (
        (d["DT_DIFF"] > spike_threshold) |
        (d["DT_DEVIATION"] > spike_threshold)
    )

    # -- Caliper / washout flag --------------------------------------------
    if cali_ch in d.columns:
        d["FLAG_WASHOUT"] = d[cali_ch] > (bit_size_inch * washout_factor)
    else:
        d["FLAG_WASHOUT"] = False

    # -- Combine logic ------------------------------------------------------
    if logic == 'OR':
        d["FLAG_BAD_POINT"] = d["FLAG_SPIKE"] | d["FLAG_WASHOUT"]
    else:
        d["FLAG_BAD_POINT"] = d["FLAG_SPIKE"] & d["FLAG_WASHOUT"]

    # -- Run-length filtering ----------------------------------------------
    s = d["FLAG_BAD_POINT"].astype(int)
    groups = (s != s.shift()).cumsum()
    run_lengths = s.groupby(groups).transform('sum')
    d["FLAG_BAD"] = (s == 1) & (run_lengths >= min_bad_run)

    # -- Extract zone intervals --------------------------------------------
    depth = d["DEPTH_M"].values
    bad = d["FLAG_BAD"].values

    zone_intervals = []
    in_zone = False

    for i, b in enumerate(bad):
        if b and not in_zone:
            zone_start = depth[i]
            in_zone = True
        elif not b and in_zone:
            zone_intervals.append((zone_start, depth[i-1]))
            in_zone = False

    if in_zone:
        zone_intervals.append((zone_start, depth[-1]))

    n_zones = len(zone_intervals)

    return d, zone_intervals, n_zones
