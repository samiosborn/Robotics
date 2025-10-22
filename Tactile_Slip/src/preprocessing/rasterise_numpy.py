# src/preprocessing/rasterise_numpy.py
import os, glob
import numpy as np
from typing import Tuple, List, Optional
from scipy.io import loadmat
from scipy.ndimage import median_filter

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
    paths = sorted(glob.glob(os.path.join(base_dir, split, "**", "*.mat"), recursive=True))
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
    xs = xs - xs.mean(axis=0, keepdims=True)
    ys = ys - ys.mean(axis=0, keepdims=True)
    # Normalise
    std = np.std(np.concatenate([xs, ys], axis=1))
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
    img = np.zeros((H, W), dtype=np.float64)
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
        img[j, i] += (1 - a) * (1 - b)
        if i + 1 < W: img[j, i + 1] += a * (1 - b)
        if j + 1 < H: img[j + 1, i] += (1 - a) * b
        if (i + 1) < W and (j + 1) < H: img[j + 1, i + 1] += a * b
    return img

# Compute per-frame mean pin speeds (for global threshold estimation)
def compute_pin_speeds(seq_norm: Array) -> np.ndarray:
    dx = np.diff(seq_norm[:, 0::2], axis=0)
    dy = np.diff(seq_norm[:, 1::2], axis=0)
    v = np.sqrt(dx * dx + dy * dy).mean(axis=1)
    return v

# Label data for supervised training examples
def create_label_for_sequence(seq: Array,
                              out_hw: Tuple[int, int],
                              scale: float,
                              step: int,
                              slip_thresh: float,
                              channels: str = "pcd"):
    # Normalise coordinates in sequence (x, y)
    seq_norm, M = normalise_coordinate_sequence(seq)
    # Dimensions
    T = seq_norm.shape[0]
    H, W = out_hw
    # Center point
    center = (W / 2.0, H / 2.0)

    # Initialise
    X_list, y_list = [], []
    # Loop over list
    for t in range(1, T, step):
        # Current sequence
        curr_xy = np.stack([seq_norm[t, 0::2], seq_norm[t, 1::2]], axis=1)
        # Previous sequence (shifted 1 back)
        prev_xy = np.stack([seq_norm[t - 1, 0::2], seq_norm[t - 1, 1::2]], axis=1)
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
        vmean = float(np.mean(np.sqrt(dx * dx + dy * dy)))
        # Label from mean pin displacement
        y = 1.0 if vmean >= slip_thresh else 0.0
        X_list.append(Xc)
        y_list.append(y)

    # Create X and y
    if not X_list:
        raise ValueError(f"No examples produced: need T>=2 and step>=1 (T={T}, step={step})")
    X = np.stack(X_list, axis=0).astype(np.float64)
    y = np.array(y_list, dtype=np.float64)

    # Apply short temporal median filter to smooth noisy single-frame spikes
    y = median_filter(y, size=5)

    # Dataset-level normalisation
    m = X.mean()
    s = X.std() + 1e-8
    X = (X - m) / s
    return X, y

# Build dataset across a split
def build_dataset(base_dir: str,
                  split: str = "train",
                  out_hw: Tuple[int, int] = (96, 96),
                  scale: float = 10.0,
                  step: int = 1,
                  channels: str = "pcd",
                  return_groups: bool = False):
    # Load sequences
    seqs = load_split(base_dir, split)

    # Collect all speeds from sequences to derive a global threshold (median + 3*mad)
    all_speeds = []
    for seq in seqs:
        try:
            seq_norm, _ = normalise_coordinate_sequence(seq)
            v = compute_pin_speeds(seq_norm)
            if v.size > 0:
                all_speeds.append(v)
        except Exception:
            pass
    if all_speeds:
        all_speeds_concat = np.concatenate(all_speeds)
        med_global = np.median(all_speeds_concat)
        mad_global = np.median(np.abs(all_speeds_concat - med_global))
        slip_thresh_global = med_global + 3.0 * mad_global
        print(f"[INFO] Global slip threshold (computed over {len(seqs)} sequences): {slip_thresh_global:.5f}")
    else:
        # If there are no usable sequences, return empty arrays as before
        C = 3 if channels == "pcd" else 2
        H, W = out_hw
        X = np.zeros((0, C, H, W), dtype=np.float64)
        y = np.zeros((0,), dtype=np.float64)
        if return_groups:
            g = np.zeros((0,), dtype=np.int64)
            print("[WARN] No sequences found while computing global threshold.")
            return X, y, g
        print("[WARN] No sequences found while computing global threshold.")
        return X, y

    # gids are Group IDs: per-example group index (sequence id) if want to use sequence-level splits later
    Xs, ys, gids = [], [], []
    for si, seq in enumerate(seqs):
        try:
            Xi, yi = create_label_for_sequence(seq, tuple(out_hw), scale, step, slip_thresh_global, channels)
            if Xi.shape[0] > 0:
                Xs.append(Xi)
                ys.append(yi)
                if return_groups:
                    gids.append(np.full((Xi.shape[0],), si, dtype=np.int64))
        except Exception as e:
            print(f"[WARN] Skipping sequence {si}: {e}")
            continue

    C = 3 if channels == "pcd" else 2
    H, W = out_hw

    if not Xs:
        # Return empty arrays
        X = np.zeros((0, C, H, W), dtype=np.float64)
        y = np.zeros((0,), dtype=np.float64)
        if return_groups:
            g = np.zeros((0,), dtype=np.int64)
            return X, y, g
        return X, y

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if return_groups:
        # Safe even if some sequences were skipped
        g = np.concatenate(gids, axis=0) if len(gids) > 0 else np.zeros((X.shape[0],), dtype=np.int64)
        return X, y, g
    return X, y
