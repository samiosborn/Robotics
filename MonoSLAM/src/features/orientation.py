# src/features/orientation.py

from __future__ import annotations

import numpy as np

from features.checks import check_2d_image
from features.checks import check_finite_scalar
from features.checks import check_int_ge0
from features.checks import check_int_gt0
from features.checks import check_keypoints_xy

from features.gradients import gradients_dog


# Compute per-keypoint orientation using local gradient evidence
def keypoint_orientations(
    im: np.ndarray,
    kps_xy: np.ndarray,
    *,
    sigma_d: float = 1.0,
    truncate: float = 3.0,
    border_mode: str = "reflect",
    constant_value: float = 0.0,
    window_radius: int = 8,
    dtype=np.float64,
    eps: float = 1e-8,
) -> np.ndarray:

    # --- Checks ---
    # Validate image
    im = check_2d_image(im, name="im", finite=True)
    # Validate keypoints (expects (N,2)+ with x,y in first two cols)
    kps_xy = check_keypoints_xy(kps_xy, name="kps_xy", finite=True)
    # Validate sigma
    sigma_d = check_finite_scalar(sigma_d, name="sigma_d")
    if sigma_d <= 0.0:
        raise ValueError(f"sigma_d must be > 0; got {sigma_d}")
    # Validate truncate
    truncate = check_finite_scalar(truncate, name="truncate")
    if truncate <= 0.0:
        raise ValueError(f"truncate must be > 0; got {truncate}")
    # Validate window radius
    window_radius = check_int_ge0(window_radius, name="window_radius")
    # Validate eps
    eps = check_finite_scalar(eps, name="eps")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0; got {eps}")

    # Cast image to working dtype
    im = np.asarray(im, dtype=dtype)

    # Compute image gradients (DoG)
    Ix, Iy = gradients_dog(
        im,
        sigma_d=float(sigma_d),
        truncate=float(truncate),
        border_mode=str(border_mode),
        constant_value=float(constant_value),
        dtype=dtype,
        eps=float(eps),
    )

    # Compute gradient magnitude (used as weights)
    mag = np.hypot(Ix, Iy)

    # Read image shape
    H, W = im.shape

    # Read keypoint coords
    xs = np.asarray(kps_xy[:, 0], dtype=float)
    ys = np.asarray(kps_xy[:, 1], dtype=float)

    # Convert to integer centres for a stable window sum
    xi = np.rint(xs).astype(np.int64)
    yi = np.rint(ys).astype(np.int64)

    # Allocate output angles
    ang = np.zeros((kps_xy.shape[0],), dtype=dtype)

    # If there are no keypoints, return empty
    if ang.size == 0:
        return ang

    # Window radius
    r = int(window_radius)

    # Compute orientation per keypoint as atan2(sum(w*Iy), sum(w*Ix))
    for i in range(int(ang.size)):

        # Centre pixel
        cx = int(xi[i])
        cy = int(yi[i])

        # Clamp centre (just in case caller gave out-of-bounds coords)
        cx = int(np.clip(cx, 0, W - 1))
        cy = int(np.clip(cy, 0, H - 1))

        # Compute window bounds (clipped to image)
        x0 = max(cx - r, 0)
        x1 = min(cx + r + 1, W)
        y0 = max(cy - r, 0)
        y1 = min(cy + r + 1, H)

        # Slice gradients and weights
        gx = Ix[y0:y1, x0:x1]
        gy = Iy[y0:y1, x0:x1]
        w = mag[y0:y1, x0:x1]

        # Weighted sums (more robust than a single-pixel angle)
        sx = float(np.sum(gx * w))
        sy = float(np.sum(gy * w))

        # If gradients are tiny, fall back to 0 angle
        if (sx * sx + sy * sy) <= float(eps) * float(eps):
            ang[i] = 0.0
        else:
            ang[i] = float(np.arctan2(sy, sx))

    return ang
