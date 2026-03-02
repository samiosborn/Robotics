# src/core/checks.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


# NumPy array check
def check_array(x, name, *, ndim=None, shape=None, dtype=None, finite=False) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

    if ndim is not None and arr.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={int(ndim)}; got ndim={arr.ndim} with shape {arr.shape}")

    if shape is not None:
        if len(shape) != arr.ndim:
            raise ValueError(f"{name} must have shape {tuple(shape)}; got {arr.shape}")
        for i, exp in enumerate(shape):
            if exp is None:
                continue
            if arr.shape[i] != int(exp):
                raise ValueError(f"{name} must have shape {tuple(shape)}; got {arr.shape}")

    if bool(finite):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")

    return arr


# Finite scalar check
def check_finite_scalar(x, name) -> float:
    try:
        xf = float(x)
    except Exception:
        raise ValueError(f"{name} must be a real number; got {type(x)}") from None
    if not np.isfinite(xf):
        raise ValueError(f"{name} must be finite; got {x}")
    return xf


# Integer checks
def check_int_ge0(x, name) -> int:
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(x)}")
    xi = int(x)
    if xi < 0:
        raise ValueError(f"{name} must be >= 0; got {xi}")
    return xi

def check_int_gt0(x, name) -> int:
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(x)}")
    xi = int(x)
    if xi <= 0:
        raise ValueError(f"{name} must be > 0; got {xi}")
    return xi


# Case-insensitive string choice check returning lowercase text
def check_choice_str(x, choices, name) -> str:
    s = str(x)
    sl = s.lower()
    allowed_map = {str(c).lower(): str(c) for c in choices}
    if sl not in allowed_map:
        raise ValueError(f"{name} must be one of {sorted(list(allowed_map.keys()))}; got '{x}'")
    return sl


# Check (x,y) points with shape (N, 2)
def check_points_xy_N2(xy, name="xy", finite=True) -> np.ndarray:
    arr = check_array(xy, name, ndim=2, finite=False)
    if arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2); got shape {arr.shape}")
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr

# Check (x,y) points with shape (N, 2+)
def check_points_xy_N2plus(kps, name="kps", finite=True) -> np.ndarray:
    arr = check_array(kps, name, ndim=2, finite=False)
    if arr.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 columns (x,y); got shape {arr.shape}")
    if bool(finite) and not np.isfinite(arr[:, :2]).all():
        raise ValueError(f"{name} first two columns (x,y) must be finite")
    return arr

# Check matrix is 3x3
def check_matrix_3x3(M, name="M", finite=True) -> np.ndarray:
    arr = check_array(M, name, ndim=2, finite=False)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be (3,3); got {arr.shape}")
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr

# Check points are (2, N)
def check_points_2xN(x, name="x", finite=True) -> np.ndarray:
    arr = check_array(x, name, ndim=2, finite=False)
    if arr.shape[0] != 2:
        raise ValueError(f"{name} must be (2,N); got {arr.shape}")
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr

# Check points are (3, N)
def check_points_3xN(x, name="x", finite=True) -> np.ndarray:
    arr = check_array(x, name, ndim=2, finite=False)
    if arr.shape[0] != 3:
        raise ValueError(f"{name} must be (3,N); got {arr.shape}")
    if bool(finite) and not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr

# Check boolean mask
def check_mask_bool_N(mask, N, name="mask") -> np.ndarray | None:
    if mask is None:
        return None
    n = check_int_ge0(N, "N")
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(f"{name} must be shape (N, ); got {arr.shape} for N={n}")
    return arr


# Path exists check
def check_path_exists(p, name="path") -> Path:
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path

# Check director
def check_dir(p, name="dir") -> Path:
    path = check_path_exists(p, name=name)
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory: {path}")
    return path

# Check file
def check_file(p, name="file") -> Path:
    path = check_path_exists(p, name=name)
    if not path.is_file():
        raise ValueError(f"{name} must be a file: {path}")
    return path

# Check suffix
def check_suffix(p, allowed: Iterable[str], name="file") -> Path:
    path = Path(p)
    allowed_norm = {str(s).lower() for s in allowed}
    suffix = path.suffix.lower()
    if suffix not in allowed_norm:
        raise ValueError(f"{name} suffix must be one of {sorted(list(allowed_norm))}; got '{suffix}'")
    return path
