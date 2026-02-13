# src/features/nms.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from features.checks import check_2d_image

# Result of NMS container
@dataclass(frozen=True)
class NMSResult:
    # x coordinates (cols)
    xs: np.ndarray
    # y coordinates (rows)
    ys: np.ndarray
    # scores at the selected points
    scores: np.ndarray
    # boolean mask of selected local maxima before top-K truncation
    mask: np.ndarray


# Non-maximum suppression (NMS) for a 2D score map
def nms_2d(score_map: np.ndarray, radius: int = 1, threshold: float, border_margin: int = 0, max_points: int = None) -> NMSResult:
    
    # --- CHECKS ---
    # Validate shape
    check_2d_image(score_map)
    # Enforce finite score map
    if not np.isfinite(score_map).all():
        raise ValueError("score_map must contain only finite values")
    # Radius of window (>= 1 integer)
    if not isinstance(radius, int) or radius < 1:
        raise ValueError(f"radius must be int >= 1, got {radius}")
    # Threshold (finite)
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")
    # Border margin (non-negative int)
    if not isinstance(border_margin, int) or border_margin < 0:
        raise ValueError(f"border_margin must be int >= 0, got {border_margin}")
    # Max points (positive int)
    if max_points is not None:
        if (not isinstance(max_points, int)) or (max_points <= 0):
            raise ValueError(f"max_points must be int > 0, got {max_points}")

    # Read score map dimensions
    H, W = score_map.shape

    # Convert radius into an odd window size
    win = 2 * radius + 1

    # Pad with -inf so outside-image values cannot win local maxima
    padded = np.pad(score_map, pad_width=((radius, radius), (radius, radius)), mode="constant", constant_values=-np.inf)

    # Create a sliding view of shape (H, W, win, win)
    windows = sliding_window_view(padded, (win, win))

    # Local maximum per pixel over the last two axes -> shape (H, W)
    local_max = windows.max(axis=(-2, -1))

    # Keep pixels that equal the max in their neighbourhood window
    mask = (score_map == local_max)

    # Apply threshold
    if threshold is not None:
        mask &= (score_map > threshold)

    # Remove maxima too close to image borders
    if border_margin > 0:
        mask[:border_margin, :] = False
        mask[-border_margin:, :] = False
        mask[:, :border_margin] = False
        mask[:, -border_margin:] = False

    # Extract y (row) and x (col) indices where mask = True
    ys, xs = np.nonzero(mask)

    # Gather corresponding scores
    scores = score_map[ys, xs]

    # If nothing is left, return empty arrays with correct dtypes
    if scores.size == 0:
        return NMSResult(
            xs=xs.astype(np.int64, copy=False),
            ys=ys.astype(np.int64, copy=False),
            scores=scores.astype(score_map.dtype, copy=False),
            mask=mask)

    # Sort by score (descending)
    order = np.argsort(scores)[::-1]

    # Truncate to top-K
    if max_points is not None:
        order = order[:max_points]

    # Reorder coordinates and scores
    xs = xs[order].astype(np.int64, copy=False)
    ys = ys[order].astype(np.int64, copy=False)
    scores = scores[order]

    return NMSResult(xs=xs, ys=ys, scores=scores, mask=mask)
