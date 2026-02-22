# src/features/pyramid.py

from __future__ import annotations

import math

import numpy as np
from PIL import Image


# Build a simple Gaussian pyramid using PIL downsampling
def build_pil_pyramid(img: Image.Image, *, levels: int, scale: float = 0.8333333, min_size: int = 48, resample: str = "lanczos"):

    # --- Checks ---
    # Require PIL image
    if not isinstance(img, Image.Image):
        raise ValueError("img must be a PIL.Image")
    # Levels must be int >= 1
    if not isinstance(levels, int) or levels < 1:
        raise ValueError(f"levels must be int >= 1; got {levels}")
    # Scale must be in (0,1)
    try:
        s = float(scale)
    except Exception:
        raise ValueError(f"scale must be a float in (0,1); got {type(scale)}") from None
    if not (0.0 < s < 1.0):
        raise ValueError(f"scale must be in (0,1); got {s}")
    # Min size must be int >= 8
    if not isinstance(min_size, int) or min_size < 8:
        raise ValueError(f"min_size must be int >= 8; got {min_size}")

    # Choose PIL resampler
    rname = str(resample).lower().strip()

    # Handle PIL version differences
    try:
        Resampling = Image.Resampling
    except Exception:
        Resampling = Image

    # Map string -> resample enum
    if rname == "nearest":
        r = Resampling.NEAREST
    elif rname == "bilinear":
        r = Resampling.BILINEAR
    elif rname == "bicubic":
        r = Resampling.BICUBIC
    elif rname == "lanczos":
        r = Resampling.LANCZOS
    else:
        raise ValueError("resample must be one of: nearest, bilinear, bicubic, lanczos")

    # Start pyramid with level 0
    pyr = [img]

    # Build subsequent levels
    for _ in range(1, int(levels)):

        # Read previous level size (PIL uses (W,H))
        w, h = pyr[-1].size

        # Compute next size
        wn = int(max(1, round(w * s)))
        hn = int(max(1, round(h * s)))

        # Stop if too small
        if wn < int(min_size) or hn < int(min_size):
            break

        # Downsample with antialiasing resampler
        nxt = pyr[-1].resize((wn, hn), resample=r)

        # Append
        pyr.append(nxt)

    return pyr


# Return pixel scale factors for each pyramid level relative to level 0
def pyramid_level_scales(*, levels: int, scale: float):

    # --- Checks ---
    if not isinstance(levels, int) or levels < 1:
        raise ValueError(f"levels must be int >= 1; got {levels}")

    try:
        s = float(scale)
    except Exception:
        raise ValueError(f"scale must be float; got {type(scale)}") from None
    if not (0.0 < s < 1.0):
        raise ValueError(f"scale must be in (0,1); got {s}")

    # Scale at level l is s^l
    return [float(s ** l) for l in range(int(levels))]


# Convert keypoints from level coords -> original coords
def kps_level_to_original(kps: np.ndarray, *, level_scale: float):

    # --- Checks ---
    kps = np.asarray(kps)
    if kps.ndim != 2 or kps.shape[1] < 2:
        raise ValueError(f"kps must have shape (N,2)+; got {kps.shape}")
    # Convert scale
    try:
        s = float(level_scale)
    except Exception:
        raise ValueError(f"level_scale must be float; got {type(level_scale)}") from None
    if s <= 0.0:
        raise ValueError(f"level_scale must be > 0; got {s}")
    # Copy so we don't mutate caller
    out = np.array(kps, copy=True)

    # Scale x,y back to original
    out[:, 0] = out[:, 0] / s
    out[:, 1] = out[:, 1] / s

    return out


# Convert keypoints from original coords -> level coords
def kps_original_to_level(kps: np.ndarray, *, level_scale: float):

    # --- Checks ---
    kps = np.asarray(kps)
    if kps.ndim != 2 or kps.shape[1] < 2:
        raise ValueError(f"kps must have shape (N,2)+; got {kps.shape}")
    # Convert scale
    try:
        s = float(level_scale)
    except Exception:
        raise ValueError(f"level_scale must be float; got {type(level_scale)}") from None
    if s <= 0.0:
        raise ValueError(f"level_scale must be > 0; got {s}")

    # Copy so we don't mutate caller
    out = np.array(kps, copy=True)

    # Scale x,y to this level
    out[:, 0] = out[:, 0] * s
    out[:, 1] = out[:, 1] * s

    return out
