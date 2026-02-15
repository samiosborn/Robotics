# src/features/patches.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from features.checks import check_2d_image, check_positive

# Patch result
@dataclass(frozen=True)
class PatchResult:
    # Extracted patches, shape (N, P, P)
    patches: np.ndarray
    # Keypoints aligned to patches, shape (N, 2) or (N, 3) depending on input
    kps: np.ndarray
    # Integer pixel centres used for extraction, shape (N, 2) as (x, y)
    centres_xy: np.ndarray


# Convert keypoints to integer pixel centres (x, y)
def keypoints_to_centres_xy(kps, *, rounding="round"):

    # --- CHECKS ---
    # Require numpy array
    if not isinstance(kps, np.ndarray):
        raise ValueError("kps must be a numpy array")
    # Require 2D array
    if kps.ndim != 2:
        raise ValueError(f"kps must be 2D array of shape (N,2) or (N,3); got {kps.shape}")
    # Require at least x,y columns
    if kps.shape[1] < 2:
        raise ValueError(f"kps must have at least 2 columns (x,y); got {kps.shape}")
    
    # Extract x and y as float
    xs = np.asarray(kps[:, 0], dtype=np.float64)
    ys = np.asarray(kps[:, 1], dtype=np.float64)

    # Get rounding strategy
    mode = str(rounding).lower().strip()

    # --- Rounding ---
    # Round to nearest integer
    if mode == "round":
        xi = np.rint(xs)
        yi = np.rint(ys)
    # Floor to pixel (top-left bias)
    elif mode == "floor":
        xi = np.floor(xs)
        yi = np.floor(ys)
    # Ceil to pixel (bottom-right bias)
    elif mode == "ceil":
        xi = np.ceil(xs)
        yi = np.ceil(ys)
    # Reject unknown rounding modes
    else:
        raise ValueError("rounding must be one of: 'round', 'floor', 'ceil'")

    # Stack into (N,2) as (x,y)
    centres_xy = np.stack([xi, yi], axis=1)

    # Cast to int64
    return centres_xy.astype(np.int64, copy=False)


# Filter keypoints so a (P x P) patch fits fully inside the image
def filter_keypoints_in_bounds(centres_xy, H, W, *, patch_radius, border_margin=0):

    # --- Checks ---
    # Require integer centres
    if not isinstance(centres_xy, np.ndarray) or centres_xy.ndim != 2 or centres_xy.shape[1] != 2:
        raise ValueError("centres_xy must be a numpy array of shape (N,2)")
    # Require positive sizes
    if not isinstance(H, (int, np.integer)) or H <= 0:
        raise ValueError("H must be positive int")
    if not isinstance(W, (int, np.integer)) or W <= 0:
        raise ValueError("W must be positive int")
    # Require non-negative margin
    if not isinstance(border_margin, (int, np.integer)) or int(border_margin) < 0:
        raise ValueError("border_margin must be int >= 0")
    border_margin = int(border_margin)
    # Require non-negative radius
    if not isinstance(patch_radius, (int, np.integer)) or int(patch_radius) < 0:
        raise ValueError("patch_radius must be int >= 0")
    patch_radius = int(patch_radius)

    # Compute effective margin needed so the full patch stays in-bounds
    m = border_margin + patch_radius

    # Extract x,y columns
    xs = centres_xy[:, 0]
    ys = centres_xy[:, 1]

    # In-bounds mask for full patch support
    keep = (
        (xs >= m) &
        (xs < (W - m)) &
        (ys >= m) &
        (ys < (H - m))
    )

    return keep


# Extract a single patch (P x P) at integer centre (x,y)
def extract_patch(im, x, y, *, patch_size):

    # --- Checks ---
    # Validate 2D image
    im = check_2d_image(im, "im")
    # Require odd patch size
    if not isinstance(patch_size, (int, np.integer)) or int(patch_size) < 1 or (int(patch_size) % 2) != 1:
        raise ValueError(f"patch_size must be odd int >= 1; got {patch_size}")
    patch_size = int(patch_size)

    # Compute radius
    r = patch_size // 2

    # Compute slicing bounds in (row,col) space
    y0 = int(y) - r
    y1 = int(y) + r + 1
    x0 = int(x) - r
    x1 = int(x) + r + 1

    # Extract patch
    patch = im[y0:y1, x0:x1]

    # Check size
    if patch.shape != (patch_size, patch_size):
        raise RuntimeError(f"Patch out of bounds or wrong shape; got {patch.shape}")

    return patch


# Normalise a patch to zero-mean, unit-std
def normalise_patch(patch, *, eps=1e-8, dtype=np.float64):

    # --- Checks ---
    # Require 2D patch
    patch = check_2d_image(patch, "patch")
    # Validate eps
    eps = float(eps)
    if eps <= 0.0 or (not np.isfinite(eps)):
        raise ValueError(f"eps must be finite and > 0; got {eps}")
    
    # Cast
    p = np.asarray(patch, dtype=dtype)

    # Compute mean
    mu = float(np.mean(p))

    # Subtract mean
    z = p - mu

    # Compute std
    sigma = float(np.std(z))

    # Standardise
    denom = max(sigma, eps)
    z = z / denom

    return z


# Extract patches around keypoints
def extract_patches(
    im,
    kps,
    *,
    patch_size=11,
    border_margin=0,
    rounding="round",
    normalise=True,
    dtype=np.float64,
    eps=1e-8,
) -> PatchResult:

    # --- Checks ---
    # Validate 2D image
    im = check_2d_image(im, "im")
    # Cast to dtype
    im = np.asarray(im, dtype=dtype)
    # Require kps as numpy array
    if not isinstance(kps, np.ndarray):
        raise ValueError("kps must be a numpy array")
    # Require kps size (N,2) or (N,3)+
    if kps.ndim != 2 or kps.shape[1] < 2:
        raise ValueError(f"kps must be shape (N,2) or (N,3); got {kps.shape}")
    # Require odd patch size
    if not isinstance(patch_size, (int, np.integer)) or int(patch_size) < 1 or (int(patch_size) % 2) != 1:
        raise ValueError(f"patch_size must be odd int >= 1; got {patch_size}")
    patch_size = int(patch_size)

    # Compute patch radius
    r = patch_size // 2

    # Read image shape
    H, W = im.shape

    # Convert keypoints to integer centres
    centres_xy = keypoints_to_centres_xy(kps, rounding=rounding)

    # Compute which centres can support a full patch in-bounds
    keep = filter_keypoints_in_bounds(centres_xy, H, W, patch_radius=r, border_margin=border_margin)

    # Filter down keypoints and centres
    kps_kept = kps[keep]
    centres_kept = centres_xy[keep]

    # If nothing remains, return empty arrays
    if centres_kept.shape[0] == 0:
        empty_patches = np.zeros((0, patch_size, patch_size), dtype=dtype)
        empty_kps = np.zeros((0, kps.shape[1]), dtype=kps.dtype)
        empty_centres = np.zeros((0, 2), dtype=np.int64)
        return PatchResult(patches=empty_patches, kps=empty_kps, centres_xy=empty_centres)

    # Pre-allocate patch tensor
    patches = np.empty((centres_kept.shape[0], patch_size, patch_size), dtype=dtype)

    # Loop over centres and slice patches
    for i in range(centres_kept.shape[0]):

        # Read integer centre
        x = int(centres_kept[i, 0])
        y = int(centres_kept[i, 1])

        # Extract patch
        p = extract_patch(im, x, y, patch_size=patch_size)

        # Normalise
        if normalise:
            p = normalise_patch(p, eps=eps, dtype=dtype)

        # Store
        patches[i] = p

    return PatchResult(patches=patches, kps=kps_kept, centres_xy=centres_kept)
