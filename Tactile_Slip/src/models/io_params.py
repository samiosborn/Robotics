# src/models/io_params.py
import numpy as np
from typing import Dict

Array = np.ndarray
Params = Dict[str, Array]
_KEYS = ("conv1_w","conv1_b","conv2_w","conv2_b","lin_w","lin_b")

# Save parameters
def save_params(params: Params, path: str):
    np.savez(path, **{k: params[k] for k in _KEYS})

# Load parameters
def load_params(path: str) -> Params: 
    # Load
    z = np.load(path, allow_pickle=False)
    return {k: z[k].astype(np.float64, copy=False) for k in _KEYS}
