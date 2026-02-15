# src/features/checks.py

import numpy as np


# Check 2D image
def check_2d_image(im, name="im", finite=False):
    # Convert to NumPy
    im = np.asarray(im)
    # Must be 2D
    if im.ndim != 2:
        raise ValueError(f"{name} must be a 2D greyscale image; got shape {im.shape}")
    # Enforce finite values
    if finite:
        if not np.isfinite(im).all():
            raise ValueError(f"{name} must contain only finite values")
    return im


# Check axis label is 0 (y-axis) or 1 (x-axis)
def check_axis_01(axis, name="axis"):
    # Accept Python int or NumPy int
    if not isinstance(axis, (int, np.integer)):
        raise ValueError(f"{name} must be 0 or 1; got {type(axis)}")
    # Cast to int
    axis = int(axis)
    # Axis must be 0 or 1
    if axis not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1; got {axis}")
    return axis


# Check 1D kernel is 1D and odd length
def check_kernel_1d_odd(k, name="k", finite=False):
    # Convert to NumPy
    k = np.asarray(k)
    # Must be 1D
    if k.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got ndim={k.ndim} with shape {k.shape}")
    # Must not be empty
    if k.size == 0:
        raise ValueError(f"{name} is empty")
    # Must be odd length
    if (k.size % 2) == 0:
        raise ValueError(f"{name} length must be odd; got {k.size}")
    # Enforce finite entries
    if finite:
        if not np.isfinite(k).all():
            raise ValueError(f"{name} must contain only finite values")
    return k


# Check two 2D arrays, same shape
def check_2d_pair_same_shape(A, B, nameA="A", nameB="B", finite=False):
    # Convert to NumPy
    A = np.asarray(A)
    B = np.asarray(B)
    # Must both be 2D
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{nameA} and {nameB} must be 2D; got {A.ndim}D and {B.ndim}D")
    # Must match shape
    if A.shape != B.shape:
        raise ValueError(f"{nameA} and {nameB} must have same shape; got {A.shape} and {B.shape}")
    # Optionally enforce finite values
    if finite:
        if not np.isfinite(A).all():
            raise ValueError(f"{nameA} must contain only finite values")
        if not np.isfinite(B).all():
            raise ValueError(f"{nameB} must contain only finite values")
    return A, B


# Check strictly positive (with epsilon)
def check_positive(x, name="value", eps=1e-8):
    # Convert to float if possible
    try:
        xf = float(x)
    except Exception:
        raise ValueError(f"{name} must be a real number; got {type(x)}") from None
    # Must be > eps
    if xf <= eps:
        raise ValueError(f"{name} must be > {eps}; got {x}")
    return xf


# Check non-negative integer
def check_int_ge0(x, name="value"):
    # Must be int-like
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(x)}")
    # Cast to int
    xi = int(x)
    # Must be >= 0
    if xi < 0:
        raise ValueError(f"{name} must be >= 0; got {xi}")
    return xi


# Check positive integer
def check_int_gt0(x, name="value"):
    # Must be int-like
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(x)}")
    # Cast to int
    xi = int(x)
    # Must be > 0
    if xi <= 0:
        raise ValueError(f"{name} must be > 0; got {xi}")
    return xi


# Check odd integer >= 1
def check_int_odd_ge1(x, name="value"):
    # Must be int-like
    if not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer; got {type(x)}")
    # Cast to int
    xi = int(x)
    # Must be >= 1
    if xi < 1:
        raise ValueError(f"{name} must be >= 1; got {xi}")
    # Must be odd
    if (xi % 2) != 1:
        raise ValueError(f"{name} must be odd; got {xi}")
    return xi


# Check finite scalar
def check_finite_scalar(x, name="value"):
    # Convert to float
    try:
        xf = float(x)
    except Exception:
        raise ValueError(f"{name} must be a real number; got {type(x)}") from None
    # Must be finite
    if not np.isfinite(xf):
        raise ValueError(f"{name} must be finite; got {x}")
    return xf


# Check a 2D score map
def check_score_map(score, name="score"):
    # Must be 2D
    score = check_2d_image(score, name=name)
    # Must be finite
    if not np.isfinite(score).all():
        raise ValueError(f"{name} must contain only finite values")
    # Must not be boolean
    if score.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric, not bool")
    return score


# Check keypoints array shape (N,2) or (N,3)+
def check_keypoints_xy(kps, name="kps", finite=True):
    # Must be numpy array
    kps = np.asarray(kps)
    # Must be 2D
    if kps.ndim != 2:
        raise ValueError(f"{name} must be 2D array; got shape {kps.shape}")
    # Must have at least x,y columns
    if kps.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 columns (x,y); got shape {kps.shape}")
    # Optionally enforce finite values in x, y
    if finite:
        if not np.isfinite(kps[:, :2]).all():
            raise ValueError(f"{name} first two columns (x,y) must be finite")
    return kps


# Check a value is one of allowed choices
def check_choice(x, choices, name="value"):
    # Convert to string
    s = str(x)
    # Normalise to lowercase
    sl = s.lower()
    # Build normalised set
    allowed = {str(c).lower() for c in choices}
    # Validate membership
    if sl not in allowed:
        raise ValueError(f"{name} must be one of {sorted(list(allowed))}; got '{x}'")
    return sl


# Check border mode is supported
def check_border_mode(mode, name="border_mode"):
    # Validate supported modes
    return check_choice(mode, {"reflect", "constant", "edge"}, name=name)


# Check values are in [0, 1]
def check_in_01(x, name="value", eps=1e-8):
    # Convert to NumPy
    a = np.asarray(x)
    # Get min/max
    vmin = float(np.min(a))
    vmax = float(np.max(a))
    # Validate range with small tolerance
    if vmin < -float(eps) or vmax > (1.0 + float(eps)):
        raise ValueError(f"{name} must be in [0,1] (within eps={eps}); got min={vmin}, max={vmax}")
    return x


# Check 3D patches tensor (N, P, P)
def check_3d_patches(patches, name="patches", finite=False):
    # Convert to NumPy
    patches = np.asarray(patches)
    # Must be 3D
    if patches.ndim != 3:
        raise ValueError(f"{name} must be 3D with shape (N,P,P); got shape {patches.shape}")
    # Must be square patches
    if patches.shape[1] != patches.shape[2]:
        raise ValueError(f"{name} must have square patches; got shape {patches.shape}")
    # Optionally enforce finite values
    if finite:
        if not np.isfinite(patches).all():
            raise ValueError(f"{name} must contain only finite values")
    return patches


# Check two arrays share the same first dimension
def check_same_first_dim(A, B, nameA="A", nameB="B"):
    # Convert to NumPy
    A = np.asarray(A)
    B = np.asarray(B)
    # Must be at least 1D
    if A.ndim < 1 or B.ndim < 1:
        raise ValueError(f"{nameA} and {nameB} must be at least 1D")
    # First dimension must match
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"{nameA} and {nameB} must have same first dim; got {A.shape[0]} and {B.shape[0]}")
    return A, B
