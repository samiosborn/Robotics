# src/features/checks.py
import numpy as np

# Check 2D image
def check_2d_image(im, name="im"):
    im = np.asarray(im)
    if im.ndim != 2:
        raise ValueError(f"{name} must be a 2D greyscale image; got shape {im.shape}")
    return im

# Check axis label is 0 (y-axis) or 1 (x-axis)
def check_axis_01(axis, name="axis"):
    if axis not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1; got {axis}")
    return axis

# Check 1D kernel is 1D and odd length
def check_kernel_1d_odd(k, name="k"):
    k = np.asarray(k)
    if k.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got ndim={k.ndim} with shape {k.shape}")
    if k.size == 0:
        raise ValueError(f"{name} is empty")
    if (k.size % 2) == 0:
        raise ValueError(f"{name} length must be odd; got {k.size}")
    return k

# Check two 2D arrays, same shape
def check_2d_pair_same_shape(A, B, nameA="A", nameB="B"):
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{nameA} and {nameB} must be 2D; got {A.ndim}D and {B.ndim}D")
    if A.shape != B.shape:
        raise ValueError(f"{nameA} and {nameB} must have same shape; got {A.shape} and {B.shape}")
    return A, B

# Check strictly positive (with epsilon)
def check_positive(x, name="value", eps=1e-8):
    try:
        xf = float(x)
    except Exception:
        raise ValueError(f"{name} must be a real number; got {type(x)}") from None
    if xf <= eps:
        raise ValueError(f"{name} must be > {eps}; got {x}")
    return xf
