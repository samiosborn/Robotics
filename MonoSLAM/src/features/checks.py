# src/features/checks.py

import numpy as np

from core.checks import (
    check_array,
    check_choice_str,
    check_finite_scalar as check_finite_scalar_core,
    check_int_ge0 as check_int_ge0_core,
    check_int_gt0 as check_int_gt0_core,
    check_points_xy_N2plus,
    check_points_xy_N2,
)


# Check 2D image (wrapper)
def check_2d_image(im, name="im", finite=False):
    return check_array(im, name, ndim=2, finite=finite)


# Check axis label is 0 (y-axis) or 1 (x-axis)
def check_axis_01(axis, name="axis"):
    if not isinstance(axis, (int, np.integer)):
        raise ValueError(f"{name} must be 0 or 1; got {type(axis)}")
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1; got {axis}")
    return axis


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


# Check strictly positive scalar with epsilon guard
def check_positive(x, name="value", eps=1e-8):
    try:
        xf = float(x)
    except Exception:
        raise ValueError(f"{name} must be a real number; got {type(x)}") from None
    if xf <= eps:
        raise ValueError(f"{name} must be > {eps}; got {x}")
    return xf


# Integer checks (wrapper)
def check_int_ge0(x, name="value"):
    return check_int_ge0_core(x, name)

def check_int_gt0(x, name="value"):
    return check_int_gt0_core(x, name)


# Check odd integer >= 1
def check_int_odd_ge1(x, name="value"):
    xi = check_int_gt0_core(x, name)
    if (xi % 2) != 1:
        raise ValueError(f"{name} must be odd; got {xi}")
    return xi


# Finite scalar (wrapper)
def check_finite_scalar(x, name="value"):
    return check_finite_scalar_core(x, name)


# Check a 2D numeric score map
def check_score_map(score, name="score"):
    score = check_array(score, name, ndim=2, finite=True)
    if score.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric, not bool")
    return score


# Check keypoints array shape (N,2) or (N,>=2) (wrapper)
def check_keypoints_xy(kps, name="kps", finite=True):
    return check_points_xy_N2plus(kps, name=name, finite=finite)


# Check strict (N,2) xy array (wrapper)
def check_xy(points, name="points", finite=True):
    return check_points_xy_N2(points, name=name, finite=finite)


# Check a value is one of allowed choices (wrapper)
def check_choice(x, choices, name="value"):
    return check_choice_str(x, choices, name)


# Check border mode is supported (wrapper)
def check_border_mode(mode, name="border_mode"):
    return check_choice(mode, {"reflect", "constant", "edge"}, name=name)


# Check values are in [0, 1]
def check_in_01(x, name="value", eps=1e-8):
    a = np.asarray(x)
    vmin = float(np.min(a))
    vmax = float(np.max(a))
    if vmin < -float(eps) or vmax > (1.0 + float(eps)):
        raise ValueError(f"{name} must be in [0,1] (within eps={eps}); got min={vmin}, max={vmax}")
    return x


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
