import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

BIN_DIR = Path(".")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

BIN_FILES = {
    "XX": BIN_DIR / "XX.bin",
    "XY": BIN_DIR / "XY.bin",
    "YX": BIN_DIR / "YX.bin",
    "YY": BIN_DIR / "YY.bin",
}

VELOCITY_RANGE = np.linspace(1000, 5000, 40)
WINDOW_LENGTH = 40
STEP = 10
RECEIVER_SPACING = 0.1524

from .loader import read_ldeo_binary
from .preprocessing import preprocess
from .windowing import get_windows
from .semblance import compute_semblance
from .stc import build_stc
from .picking import pick_velocity
from .elastic import compute_elastic

__all__ = [
    "BIN_DIR",
    "DATA_DIR",
    "BIN_FILES",
    "VELOCITY_RANGE",
    "WINDOW_LENGTH",
    "STEP",
    "RECEIVER_SPACING",
    "read_ldeo_binary",
    "preprocess",
    "get_windows",
    "compute_semblance",
    "build_stc",
    "pick_velocity",
    "compute_elastic",
]
