
import tempfile
import os
import numpy as np
from dlisio import dlis

# ── channel name sets ────────────────────────────────────────
DEPTH_CHANNELS     = {"TDEP"}
WAVEFORM_CHANNELS  = {"PWF1", "PWF2", "PWF3", "PWF4", "PWFT", "PWFS"}
SEMBLANCE_CHANNELS = {"SPR4", "SPR2", "SPS4", "SPS2", "SPRM", "SPSM"}
SLOWNESS_CHANNELS  = {"DTCO", "DTSM", "DT4P", "DT4S"}
def load_dlis(file_obj):

    # ── 1. Resolve path ──────────────────────────────────────
    if isinstance(file_obj, (str, os.PathLike)):
        path = str(file_obj)
        _tmp_path = None
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dlis")
        tmp.write(file_obj.read())
        tmp.flush()
        tmp.close()           # close our handle immediately
        path = tmp.name
        _tmp_path = tmp.name

    # ── 2. Load + extract everything inside a try block ──────
    try:
        files = dlis.load(path)
        lf = files[0]

        result = {"frame_names": [], "frames": {}, "metadata": {}}

        # metadata
        try:
            origin = next(iter(lf.origins))
            result["metadata"] = {
                "well_name":     getattr(origin, "well_name", ""),
                "field_name":    getattr(origin, "field_name", ""),
                "company":       getattr(origin, "company", ""),
                "creation_time": str(getattr(origin, "creation_time", "")),
            }
        except StopIteration:
            pass

        # frames
        for frame in lf.frames:
            name   = frame.name
            curves = frame.curves()

            depth     = None
            waveforms = {}
            semblance = {}
            slowness  = None
            rest      = {}

            for ch in frame.channels:
                ch_name = ch.name
                try:
                    data = np.array(curves[ch_name])  # copy into numpy — releases dlisio ref
                except Exception:
                    data = None

                if ch_name in DEPTH_CHANNELS:
                    depth = data
                elif ch_name in WAVEFORM_CHANNELS:
                    waveforms[ch_name] = data
                elif ch_name in SEMBLANCE_CHANNELS:
                    semblance[ch_name] = data
                elif ch_name in SLOWNESS_CHANNELS:
                    if slowness is None:
                        slowness = data
                    rest[ch_name] = data
                else:
                    rest[ch_name] = data

            result["frame_names"].append(name)
            result["frames"][name] = {
                "depth":     depth,
                "waveforms": waveforms,
                "semblance": semblance,
                "slowness":  slowness,
                "channels":  rest,
            }

        # ── dlisio done, all data is in plain numpy arrays now
        for f in files:
            f.close()          # explicitly close dlisio handles

    finally:
        # ── 3. Now safe to delete temp file on Windows ───────
        if _tmp_path and os.path.exists(_tmp_path):
            try:
                os.unlink(_tmp_path)
            except PermissionError:
                pass  # worst case: OS cleans up temp dir on reboot

    return result