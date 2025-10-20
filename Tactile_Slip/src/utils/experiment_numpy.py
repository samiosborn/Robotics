# src/utils/experiment_numpy.py
import glob, os
from pathlib import Path
import json
import numpy as np

# Load configuration
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Resolve data_dir relative to the config
def resolve_data_dir(config_path: str, data_dir_str: str) -> Path:
    config_dir = Path(config_path).resolve().parent
    data_dir = Path(data_dir_str)
    return data_dir if data_dir.is_absolute() else (config_dir / data_dir).resolve()

# List .mat paths
def list_mat_paths(base_dir: str, split: str):
    pat = os.path.join(base_dir, split, "**", "*.mat")
    return sorted(glob.glob(pat, recursive=True))

# Split by unique group IDs so that all samples from the same sequence stay in the same split
def sequence_level_split(X: np.ndarray, y: np.ndarray, g: np.ndarray, val_frac: float = 0.1, seed: int = 0):
    
    rng = np.random.default_rng(seed)
    groups = np.unique(g)
    rng.shuffle(groups)

    n_val = max(1, int(round(len(groups) * val_frac)))
    val_groups = set(groups[:n_val])
    tr_groups  = set(groups[n_val:])

    tr_mask  = np.isin(g, list(tr_groups))
    val_mask = np.isin(g, list(val_groups))

    return X[tr_mask], y[tr_mask], X[val_mask], y[val_mask]

# Logging banner
def banner(msg: str):
    print("=" * 8, msg)
