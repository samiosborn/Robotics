# src/preprocessing/rasterise_numpy.py
import os, glob
import numpy as np
from typing import Tuple, List, Optional
from scipy.io import loadmat

Array = np.ndarray

# Load .mat file 
def load_mat_table(path: str):
    m = loadmat(path)
    for k, v in m.items():
        # Skip any key whose name starts with "__"
        if k.startswith("__"): continue
        # Check v is a numpy array, is 2D, and is numeric
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            # Cast to float64
            return v.astype(np.float64)
    raise ValueError(f"No 2D numeric array in {path}")

# Load split of data
def load_split(base_dir: str, split: str = "train"):
    # Sort paths in data split (e.g. train folder)
    paths = sorted(glob.glob(os.path.join(base_dir, split, "**", "*.mat"), recursive = True))
    # For all paths, load .mat file and return together
    return [load_mat_table(p) for p in paths]

# Normalise coordinates of sequence
def normalise_coordinate_sequence(seq: Array):
    # Dimensions
    T, D = seq.shape
    assert D % 2 == 0
    M = D // 2
    # seq: (T, 2*M) with [x1,y1,x2,y2,...]
    xs = seq[:, 0::2]
    ys = seq[:, 1::2]
    # Center
    xs = xs - xs.mean(axis = 0, keepdims = True)
    ys = ys - ys.mean(axis = 0, keepdims = True)
    # Normalise
    std = np.std(np.concatenate([xs, ys], axis = 1))
    std = std if std > 1e-8 else 1.0
    xs /= std
    ys /= std
    # Combine
    out = np.empty_like(seq)
    out[:, 0::2] = xs
    out[:, 1::2] = ys
    return out, M

# Bilinear splat to image
def bilinear_splat(points_xy: Array, out_hw: Tuple[int, int], scale: float, center: Tuple[float, float]):
    # Dimensions
    H, W = out_hw
    # Image center
    cx, cy = center
    # Scaled points
    u = points_xy[:, 0] * scale + cx
    v = points_xy[:, 1] * scale + cy
    # Pre allocate image
    img = np.zeros((H, W), dtype = np.float64)
    # Convert to int (pixel positions)
    i0 = np.floor(u).astype(int)
    j0 = np.floor(v).astype(int)
    # Proportion in 4 surrounding pixels
    alpha = u - i0
    beta = v - j0
    # Loop over each point 
    for k in range(points_xy.shape[0]):
        # Index
        i, j = i0[k], j0[0]
        # If outside
        if j < 0 or j >= H or i < 0 or i >= W: 
            continue
        # Alias
        a, b = alpha[k], beta[k]
        # Add (check boundary first)
        img[j, i] += (1-a)*(1-b)
        if i+1 < W: img[j, i+1] += a*(1-b)
        if j+1 < H: img[j+1, i] += (1-a)*b
        if (i+1) < W and (j+1) > H: img[j+1, i+1] += a*b

    return img


