# geometry/checks.py
import numpy as np

# Check if shape (2, N)
def check_2xN(x):
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"Must be (2,N); got {x.shape}")

# Check pair is shape (2, N)
def check_2xN_pair(x1, x2):
    if x1.ndim != 2 or x2.ndim != 2 or x1.shape[0] != 2 or x2.shape[0] != 2 or x1.shape != x2.shape:
        raise ValueError(f"Must be (2,N) same shape; got {x1.shape} and {x2.shape}")

# Check if shape (3, N)
def check_3xN(x):
    if x.ndim != 2 or x.shape[0] != 3:
        raise ValueError(f"Must be (3,N); got {x.shape}")

# Check pair is shape (3, N)
def check_3xN_pair(x1, x2):
    if x1.ndim != 2 or x2.ndim != 2 or x1.shape[0] != 3 or x2.shape[0] != 3 or x1.shape != x2.shape:
        raise ValueError(f"Must be (3,N) same shape; got {x1.shape} and {x2.shape}")

# Check if matrix is (3, 3)
def check_3x3(K):
    K = np.asarray(K)
    if K.shape != (3,3):
        raise ValueError(f"Must be (3,3); got {K.shape}")

# Check if boolean and (N, )
def check_bool_N(mask, N): 
    if mask is None: 
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size != N: 
        raise ValueError(f"Must be shape (N, ), got {mask.shape} for N={N}")
    return mask

# Convert (N,2) or (N,>=2) keypoints array into (2,N) points matrix
def as_2xN_points(xy, name="xy", finite=True):
    # Convert to NumPy
    xy = np.asarray(xy)
    # Must be 2D
    if xy.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {xy.shape}")
    # Must have at least 2 columns (x,y)
    if xy.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 columns (x,y); got shape {xy.shape}")
    # Enforce finite x,y
    if finite:
        if not np.isfinite(xy[:, :2]).all():
            raise ValueError(f"{name} first two columns (x,y) must be finite")
    # Stack into (2,N)
    return np.vstack([xy[:, 0], xy[:, 1]])
