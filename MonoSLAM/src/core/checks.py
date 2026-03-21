# src/core/checks.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


# --- GENERIC ARRAY CHECKS ---

# Check NumPy array properties
def check_array(x, name="x", *, ndim=None, shape=None, dtype=None, finite=False) -> np.ndarray:
    # Convert to array and optionally cast dtype
    arr = np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    # Check ndim if requested
    if ndim is not None and arr.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={int(ndim)}; got ndim={arr.ndim} with shape {arr.shape}")

    # Check exact shape pattern if requested
    if shape is not None:
        if len(shape) != arr.ndim:
            raise ValueError(f"{name} must have shape {tuple(shape)}; got {arr.shape}")
        for i, exp in enumerate(shape):
            if exp is None:
                continue
            if arr.shape[i] != int(exp):
                raise ValueError(f"{name} must have shape {tuple(shape)}; got {arr.shape}")

    # Check finiteness if requested
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")

    return arr


# Check object is a dict
def check_dict(x, name="obj") -> dict:
    if not isinstance(x, dict):
        raise ValueError(f"{name} must be a dict")
    return x


# Check dict contains required keys
def check_required_keys(d, keys, name="dict") -> dict:
    # Check dict
    d = check_dict(d, name=name)

    # Find missing keys
    missing = [str(k) for k in keys if k not in d]
    if len(missing) > 0:
        raise ValueError(f"{name} is missing required keys: {missing}")

    return d


# --- SCALAR CHECKS ---

# Check finite scalar
def check_finite_scalar(x, name="value") -> float:
    try:
        xf = float(x)
    except Exception:
        raise ValueError(f"{name} must be a real number; got {type(x)}") from None

    if not np.isfinite(xf):
        raise ValueError(f"{name} must be finite; got {x}")

    return xf


# Check integer >= 0
def check_int_ge0(x, name="value") -> int:
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(x)}")

    xi = int(x)
    if xi < 0:
        raise ValueError(f"{name} must be >= 0; got {xi}")

    return xi


# Check integer > 0
def check_int_gt0(x, name="value") -> int:
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(x)}")

    xi = int(x)
    if xi <= 0:
        raise ValueError(f"{name} must be > 0; got {xi}")

    return xi


# Check odd integer >= 1
def check_int_odd_ge1(x, name="value") -> int:
    xi = check_int_gt0(x, name)
    if (xi % 2) != 1:
        raise ValueError(f"{name} must be odd; got {xi}")
    return xi


# Check positive scalar with epsilon guard
def check_positive(x, name="value", eps=1e-8) -> float:
    xf = check_finite_scalar(x, name)
    if xf <= float(eps):
        raise ValueError(f"{name} must be > {eps}; got {x}")
    return xf


# Check values lie in [0,1]
def check_in_01(x, name="value", eps=1e-8):
    a = np.asarray(x)
    vmin = float(np.min(a))
    vmax = float(np.max(a))

    if vmin < -float(eps) or vmax > (1.0 + float(eps)):
        raise ValueError(f"{name} must be in [0,1] (within eps={eps}); got min={vmin}, max={vmax}")

    return x


# --- CHOICE CHECKS ---

# Check case-insensitive string choice
def check_choice_str(x, choices, name="value") -> str:
    s = str(x)
    sl = s.lower()
    allowed_map = {str(c).lower(): str(c) for c in choices}

    if sl not in allowed_map:
        raise ValueError(f"{name} must be one of {sorted(list(allowed_map.keys()))}; got '{x}'")

    return sl


# Check value is one of allowed choices
def check_choice(x, choices, name="value") -> str:
    return check_choice_str(x, choices, name)


# Check supported border mode
def check_border_mode(mode, name="border_mode") -> str:
    return check_choice(mode, {"reflect", "constant", "edge"}, name=name)


# Check axis is 0 or 1
def check_axis_01(axis, name="axis") -> int:
    if not isinstance(axis, (int, np.integer)):
        raise ValueError(f"{name} must be 0 or 1; got {type(axis)}")

    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1; got {axis}")

    return axis


# --- VECTORS AND MATRICES ---

# Check vector has length 3
def check_vector_3(x, name="x", dtype=None, finite=True) -> np.ndarray:
    # Check raw array
    arr = check_array(x, name=name, dtype=dtype, finite=False)

    # Accept (3,), (3,1), or (1,3)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"{name} must have shape (3,); got {arr.shape}")
    elif arr.ndim == 2:
        if arr.shape not in {(3, 1), (1, 3)}:
            raise ValueError(f"{name} must have shape (3,), (3,1), or (1,3); got {arr.shape}")
    else:
        raise ValueError(f"{name} must have shape (3,), (3,1), or (1,3); got {arr.shape}")

    # Flatten to canonical shape
    arr = arr.reshape(3,)

    # Check finiteness if requested
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")

    return arr


# Check matrix is 3x3
def check_matrix_3x3(M, name="M", dtype=None, finite=True) -> np.ndarray:
    # Check array and optionally cast dtype
    arr = check_array(M, name=name, ndim=2, dtype=dtype, finite=False)

    # Require shape (3,3)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be (3,3); got {arr.shape}")

    # Check finiteness if requested
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")

    return arr


# Check 1D integer array
def check_int_array_1d(x, name="x", dtype=np.int64) -> np.ndarray:
    # Check array and cast to integer dtype
    arr = check_array(x, name=name, dtype=dtype, finite=False)

    # Require 1D
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.shape}")

    return arr


# Check 1D integer index array
def check_index_array_1d(x, name="x", dtype=np.int64, allow_negative=True) -> np.ndarray:
    # Check integer array
    arr = check_int_array_1d(x, name=name, dtype=dtype)

    # Optionally forbid negative indices
    if not bool(allow_negative) and np.any(arr < 0):
        raise ValueError(f"{name} must contain only nonnegative indices")

    return arr


# --- POINTS IN D x N FORM ---

# Internal helper for point sets with fixed row count
def _check_points_rows(x, rows, name="x", dtype=None, finite=True) -> np.ndarray:
    # Check array and optionally cast dtype
    arr = check_array(x, name=name, ndim=2, dtype=dtype, finite=False)

    # Require fixed number of rows
    if arr.shape[0] != int(rows):
        raise ValueError(f"{name} must be ({int(rows)},N); got {arr.shape}")

    # Check finiteness if requested
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")

    return arr


# Check points are (2,N)
def check_points_2xN(x, name="x", dtype=None, finite=True) -> np.ndarray:
    return _check_points_rows(x, 2, name=name, dtype=dtype, finite=finite)


# Check points are (3,N)
def check_points_3xN(x, name="x", dtype=None, finite=True) -> np.ndarray:
    return _check_points_rows(x, 3, name=name, dtype=dtype, finite=finite)


# --- X,Y POINTS IN N x D FORM ---

# Check (x,y) points with shape (N,2)
def check_points_xy_N2(xy, name="xy", dtype=None, finite=True) -> np.ndarray:
    # Check array and optionally cast dtype
    arr = check_array(xy, name=name, ndim=2, dtype=dtype, finite=False)

    # Require exactly 2 columns
    if arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2); got {arr.shape}")

    # Check finiteness if requested
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")

    return arr


# Check arrays with at least x,y columns
def check_points_xy_N2plus(kps, name="kps", dtype=None, finite=True) -> np.ndarray:
    # Check array and optionally cast dtype
    arr = check_array(kps, name=name, ndim=2, dtype=dtype, finite=False)

    # Require at least 2 columns
    if arr.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 columns (x,y); got {arr.shape}")

    # Check finiteness of x,y columns if requested
    if bool(finite) and not np.isfinite(arr[:, :2]).all():
        raise ValueError(f"{name} first two columns (x,y) must be finite")

    return arr


# Check (N,2) points with expected row count
def check_points_xy_N2_rows(xy, N, name="xy", dtype=None, finite=True) -> np.ndarray:
    # Check points
    arr = check_points_xy_N2(xy, name=name, dtype=dtype, finite=finite)

    # Check expected row count
    N = check_int_ge0(N, name="N")
    if arr.shape[0] != N:
        raise ValueError(f"{name} must have shape ({N},2); got {arr.shape}")

    return arr


# Convert (N,2) or (N,2+) points to (2,N)
def as_2xN_points(xy, name="xy", finite=True, dtype=None):
    # Check x,y point array
    xy = check_points_xy_N2plus(xy, name=name, dtype=dtype, finite=finite)

    # Convert to (2,N)
    return np.vstack([xy[:, 0], xy[:, 1]])


# --- PAIRS OF POINT SETS ---

# Check two 1D arrays have same length
def check_1d_pair_same_length(a, b, nameA="a", nameB="b"):
    # Check arrays
    a = check_array(a, name=nameA, ndim=1, finite=False)
    b = check_array(b, name=nameB, ndim=1, finite=False)

    # Require same length
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"{nameA} and {nameB} must have same length; got {a.shape[0]} and {b.shape[0]}")

    return a, b


# Check shape-matched (2,N) point pair
def check_2xN_pair(x1, x2, dtype=None, finite=True):
    # Check point sets
    a = check_points_2xN(x1, name="x1", dtype=dtype, finite=finite)
    b = check_points_2xN(x2, name="x2", dtype=dtype, finite=finite)

    # Require same shape
    if a.shape != b.shape:
        raise ValueError(f"x1 and x2 must both be (2,N) with same shape; got {a.shape} and {b.shape}")

    return a, b


# Check shape-matched (3,N) point pair
def check_3xN_pair(x1, x2, dtype=None, finite=True):
    # Check point sets
    a = check_points_3xN(x1, name="x1", dtype=dtype, finite=finite)
    b = check_points_3xN(x2, name="x2", dtype=dtype, finite=finite)

    # Require same shape
    if a.shape != b.shape:
        raise ValueError(f"x1 and x2 must both be (3,N) with same shape; got {a.shape} and {b.shape}")

    return a, b


# --- MIX OF (3,N) AND (2,N) ---

# Check 3D points and 2D points match column count
def check_3xN_2xN_cols(X, x, nameX="X", namex="x", dtype=None, finite=True):
    # Check (3,N) points
    X = check_points_3xN(X, name=nameX, dtype=dtype, finite=finite)

    # Check (2,N) points
    x = check_points_2xN(x, name=namex, dtype=dtype, finite=finite)

    # Require same number of columns
    if X.shape[1] != x.shape[1]:
        raise ValueError(f"{nameX} and {namex} must share N; got {X.shape} and {x.shape}")

    return X, x


# --- IMAGES AND PATCHES ---

# Check 2D image
def check_2d_image(im, name="im", dtype=None, finite=False):
    return check_array(im, name=name, ndim=2, dtype=dtype, finite=finite)


# Check 1D kernel with odd length
def check_kernel_1d_odd(k, name="k", dtype=None, finite=False):
    # Check 1D array
    k = check_array(k, name=name, ndim=1, dtype=dtype, finite=finite)

    # Reject empty kernel
    if k.size == 0:
        raise ValueError(f"{name} is empty")

    # Require odd length
    if (k.size % 2) == 0:
        raise ValueError(f"{name} length must be odd; got {k.size}")

    return k


# Check numeric 2D score map
def check_score_map(score, name="score", dtype=None):
    # Check array and finiteness
    score = check_array(score, name=name, ndim=2, dtype=dtype, finite=True)

    # Reject boolean score maps
    if score.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric, not bool")

    return score


# Check 3D square patch tensor (N,P,P)
def check_3d_patches(patches, name="patches", dtype=None, finite=False):
    # Check tensor
    patches = check_array(patches, name=name, ndim=3, dtype=dtype, finite=finite)

    # Require square patches
    if patches.shape[1] != patches.shape[2]:
        raise ValueError(f"{name} must have square patches; got shape {patches.shape}")

    return patches


# --- MASKS ---

# Check boolean mask shape (N,)
def check_mask_bool_1d(mask, name="mask") -> np.ndarray | None:
    # Allow None
    if mask is None:
        return None

    # Convert to boolean array
    arr = np.asarray(mask, dtype=bool)

    # Require 1D
    if arr.ndim != 1:
        raise ValueError(f"{name} must have shape (N,); got {arr.shape}")

    return arr


# Check boolean mask shape (N,) with explicit expected length
def check_mask_bool_N(mask, N, name="mask") -> np.ndarray | None:
    # Allow None
    if mask is None:
        return None

    # Check expected length
    n = check_int_ge0(N, "N")

    # Convert to boolean array
    arr = np.asarray(mask, dtype=bool)

    # Require exact length
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(f"{name} must have shape (N,); got {arr.shape} for N={n}")

    return arr


# --- PATHS ---

# Check path exists
def check_path_exists(p, name="path") -> Path:
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


# Check directory exists
def check_dir(p, name="dir") -> Path:
    path = check_path_exists(p, name=name)
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory: {path}")
    return path


# Check file exists
def check_file(p, name="file") -> Path:
    path = check_path_exists(p, name=name)
    if not path.is_file():
        raise ValueError(f"{name} must be a file: {path}")
    return path


# Check path suffix
def check_suffix(p, allowed: Iterable[str], name="file") -> Path:
    path = Path(p)
    allowed_norm = {str(s).lower() for s in allowed}
    suffix = path.suffix.lower()

    if suffix not in allowed_norm:
        raise ValueError(f"{name} suffix must be one of {sorted(list(allowed_norm))}; got '{suffix}'")

    return path

