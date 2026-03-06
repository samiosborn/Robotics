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
def check_finite_scalar(x, name="value") -> float:
    try:
        xf = float(x)
    except Exception:
        raise ValueError(f"{name} must be a real number; got {type(x)}") from None
    if not np.isfinite(xf):
        raise ValueError(f"{name} must be finite; got {x}")
    return xf


# Integer checks
def check_int_ge0(x, name="value") -> int:
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(x)}")
    xi = int(x)
    if xi < 0:
        raise ValueError(f"{name} must be >= 0; got {xi}")
    return xi


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


# Case-insensitive string choice check returning lowercase text
def check_choice_str(x, choices, name) -> str:
    s = str(x)
    sl = s.lower()
    allowed_map = {str(c).lower(): str(c) for c in choices}
    if sl not in allowed_map:
        raise ValueError(f"{name} must be one of {sorted(list(allowed_map.keys()))}; got '{x}'")
    return sl


# Check a value is one of allowed choices
def check_choice(x, choices, name="value") -> str:
    return check_choice_str(x, choices, name)


# Check border mode is supported
def check_border_mode(mode, name="border_mode") -> str:
    return check_choice(mode, {"reflect", "constant", "edge"}, name=name)


# Check axis label is 0 (y-axis) or 1 (x-axis)
def check_axis_01(axis, name="axis") -> int:
    if not isinstance(axis, (int, np.integer)):
        raise ValueError(f"{name} must be 0 or 1; got {type(axis)}")
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1; got {axis}")
    return axis


# Check strictly positive scalar with epsilon guard
def check_positive(x, name="value", eps=1e-8) -> float:
    try:
        xf = float(x)
    except Exception:
        raise ValueError(f"{name} must be a real number; got {type(x)}") from None
    if xf <= eps:
        raise ValueError(f"{name} must be > {eps}; got {x}")
    return xf


# Check values are in [0, 1]
def check_in_01(x, name="value", eps=1e-8):
    a = np.asarray(x)
    vmin = float(np.min(a))
    vmax = float(np.max(a))
    if vmin < -float(eps) or vmax > (1.0 + float(eps)):
        raise ValueError(f"{name} must be in [0,1] (within eps={eps}); got min={vmin}, max={vmax}")
    return x


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


# Check for shape-matched (2, N) point pairs
def check_2xN_pair(x1, x2):
    a = check_points_2xN(x1, name="x1", finite=False)
    b = check_points_2xN(x2, name="x2", finite=False)
    if a.shape != b.shape:
        raise ValueError(f"Must be (2,N) same shape; got {a.shape} and {b.shape}")


# Check shape-matched (3, N) point pairs
def check_3xN_pair(x1, x2):
    a = check_points_3xN(x1, name="x1", finite=False)
    b = check_points_3xN(x2, name="x2", finite=False)
    if a.shape != b.shape:
        raise ValueError(f"Must be (3,N) same shape; got {a.shape} and {b.shape}")


# Convert keypoints (N,2) / (N,>=2) to a (2,N) matrix
def as_2xN_points(xy, name="xy", finite=True):
    xy = check_points_xy_N2plus(xy, name=name, finite=finite)
    return np.vstack([xy[:, 0], xy[:, 1]])


# Check strict (N,2) xy array
def check_xy(points, name="points", finite=True):
    return check_points_xy_N2(points, name=name, finite=finite)


# Check keypoints array shape (N,2) or (N,>=2)
def check_keypoints_xy(kps, name="kps", finite=True):
    return check_points_xy_N2plus(kps, name=name, finite=finite)


# Check 2D image
def check_2d_image(im, name="im", finite=False):
    return check_array(im, name, ndim=2, finite=finite)


# Check 1D kernel is 1D and odd length
def check_kernel_1d_odd(k, name="k", finite=False):
    k = check_array(k, name, ndim=1, finite=finite)
    if k.size == 0:
        raise ValueError(f"{name} is empty")
    if (k.size % 2) == 0:
        raise ValueError(f"{name} length must be odd; got {k.size}")
    return k


# Check two 2D arrays with same shape
def check_2d_pair_same_shape(A, B, nameA="A", nameB="B", finite=False):
    A = check_array(A, nameA, ndim=2, finite=False)
    B = check_array(B, nameB, ndim=2, finite=False)
    if A.shape != B.shape:
        raise ValueError(f"{nameA} and {nameB} must have same shape; got {A.shape} and {B.shape}")
    if finite:
        if not np.isfinite(A).all():
            raise ValueError(f"{nameA} must contain only finite values")
        if not np.isfinite(B).all():
            raise ValueError(f"{nameB} must contain only finite values")
    return A, B


# Check a 2D numeric score map
def check_score_map(score, name="score"):
    score = check_array(score, name, ndim=2, finite=True)
    if score.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric, not bool")
    return score


# Check 3D patches tensor (N, P, P)
def check_3d_patches(patches, name="patches", finite=False):
    patches = check_array(patches, name, ndim=3, finite=finite)
    if patches.shape[1] != patches.shape[2]:
        raise ValueError(f"{name} must have square patches; got shape {patches.shape}")
    return patches


# Check two arrays share the same first dimension
def check_same_first_dim(A, B, nameA="A", nameB="B"):
    A = check_array(A, nameA)
    B = check_array(B, nameB)
    if A.ndim < 1 or B.ndim < 1:
        raise ValueError(f"{nameA} and {nameB} must be at least 1D")
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"{nameA} and {nameB} must have same first dim; got {A.shape[0]} and {B.shape[0]}")
    return A, B


# Check boolean mask
def check_mask_bool_1d(mask, name="mask") -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be shape (N, ); got {arr.shape}")
    return arr


# Check boolean mask with expected length
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
