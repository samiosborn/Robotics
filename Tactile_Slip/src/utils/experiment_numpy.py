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
def sequence_level_split(X, y, g, val_frac=0.1, seed=0):
    # Split (X,y) into train-sub/val by groups 'g' so that no group appears in both
    rng = np.random.default_rng(seed)
    # g: 1D array of length N with group IDs (e.g., sequence index per example)
    g = np.asarray(g).reshape(-1)
    assert X.shape[0] == y.shape[0] == g.shape[0], "Mismatched lengths for X, y, g"

    # unique groups -> shuffle -> split
    uniq = np.unique(g)
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * float(val_frac)))
    g_val = set(uniq[:n_val])
    g_tr  = set(uniq[n_val:])

    # boolean masks
    m_val = np.array([gg in g_val for gg in g], dtype=bool)
    m_tr  = ~m_val

    Xtr, ytr = X[m_tr], y[m_tr]
    Xval, yval = X[m_val], y[m_val]
    return Xtr, ytr, Xval, yval

# Logging banner
def banner(msg: str):
    print("=" * 8, msg)
