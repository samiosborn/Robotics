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
        # Skip metadata
        if k.startswith("__"): continue
        # Check v is a numpy array, is 2D, and is numeric
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            # Cast to float64
            return v.astype(np.float64)
        # If shape (T, M, 2) -> reshape to (T, 2*M)
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 2 and np.issubdtype(v.dtype, np.number):
            T, M, _ = v.shape
            v2 = v.reshape(T, M * 2)
            return v2.astype(np.float64)
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
        i, j = i0[k], j0[k]
        # If outside
        if j < 0 or j >= H or i < 0 or i >= W: 
            continue
        # Alias
        a, b = alpha[k], beta[k]
        # Add (check boundary first)
        img[j, i] += (1-a)*(1-b)
        if i+1 < W: img[j, i+1] += a*(1-b)
        if j+1 < H: img[j+1, i] += (1-a)*b
        if (i+1) < W and (j+1) < H: img[j+1, i+1] += a*b

    return img

# Label data for supervised training examples
def create_label_for_sequence(seq: Array, out_hw: Tuple[int, int], scale: float, 
                                step: int, channels: str = "pcd", 
                                slip_thresh: Optional[float] = None):
    # Normalise coordinates in sequence (xy)
    seq_norm, M = normalise_coordinate_sequence(seq)
    # Dimensions
    T = seq_norm.shape[0]
    H, W = out_hw
    # Center point
    center = (W/2.0, H/2.0)
    
    # Estimage slip threshold
    if slip_thresh is None and T>1:
        # Mean pin speed
        v = []
        # Loop over points
        for t in range(1, T):
            # Change in directions
            dx = seq_norm[t, 0::2] - seq_norm[t-1, 0::2]
            dy = seq_norm[t, 1::2] - seq_norm[t-1, 1::2]
            # Distance moved (L2 norm)
            v.append(np.mean(np.sqrt(dx*dx + dy*dy)))
        # Median
        med = np.median(v)
        # Median absolute deviation
        mad = np.median(np.abs(v - med))
        # Define robust slip threshold
        slip_thresh = med + 3.0*mad
    
    # Initialise
    X_list, y_list = [], []
    # Loop over list
    for t in range(1, T, step):
        # Current sequence
        curr_xy = np.stack([seq_norm[t, 0::2], seq_norm[t, 1::2]], axis=1)
        # Previous sequence (shifted 1 back)
        prev_xy = np.stack([seq_norm[t-1, 0::2], seq_norm[t-1, 1::2]], axis=1)
        # Current image
        im_curr = bilinear_splat(curr_xy, out_hw, scale, center)
        # Previous image
        im_prev = bilinear_splat(prev_xy, out_hw, scale, center)
        # Channel: "pcd"=prev,curr,diff
        if channels == "pcd":
            diff = np.abs(im_curr - im_prev)
            # Size (3, H, W)
            Xc = np.stack([im_prev, im_curr, diff], axis=0)
        # Else: Channel - "pc"=prev,curr
        else: 
            Xc = np.stack([im_prev, im_curr], axis=0)
        
        # Pin displacement
        dx = curr_xy[:, 0] - prev_xy[:, 0]
        dy = curr_xy[:, 1] - prev_xy[:, 1]
        # L2 norm average
        vmean = float(np.mean(np.sqrt(dx*dx + dy*dy)))
        # Label from mean pin displacement
        if slip_thresh is not None and vmean >= slip_thresh:
            y = 1.0
        else: 
            y = 0.0
        # Append
        X_list.append(Xc); y_list.append(y)

    # Create X and y
    if not X_list:
        raise ValueError(f"No examples produced: need T>=2 and step>=1 (T={T}, step={step})")
    X = np.stack(X_list, axis=0).astype(np.float64)
    y = np.array(y_list, dtype=np.float64)

    # Dataset-level normalisation
    m = X.mean()
    s = X.std() + 1e-8
    X = (X - m) / s

    return X, y

# Build dataset across a split
def build_dataset(base_dir: str, split: str = "train", out_hw: Tuple[int, int] = (96, 96),
                  scale: float = 10.0, step: int = 1, channels: str = "pcd"):
    # Load sequences
    seqs = load_split(base_dir, split)
    Xs, ys = [], []
    for seq in seqs:
        try:
            Xi, yi = create_label_for_sequence(seq, out_hw, scale, step, channels)
            if Xi.shape[0] > 0:
                Xs.append(Xi); ys.append(yi)
        except Exception:
            # Skip sequences that fail preprocessing
            pass
    if not Xs:
        # Return empty arrays with the right shapes
        C = 3 if channels == "pcd" else 2
        return np.zeros((0, C, out_hw[0], out_hw[1])), np.zeros((0,), dtype=np.float64)
    # Concatenate
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y