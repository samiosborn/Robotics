# geometry/parallax.py
import numpy as np
from geometry.checks import check_2xN_pair, check_3x3, check_bool_N
from geometry.homogeneous import homogenise
from geometry.checks import check_2xN_pair

# Bearing vectors
def bearing_vectors(x, K, eps=1e-12): 
    # Homogenise
    x_h = homogenise(x)
    # Bearings: Solve K b = x_h
    b = np.linalg.solve(K, x_h)
    # Normalise
    norms = np.maximum(np.linalg.norm(b, axis=0, keepdims=True), eps)
    b = b / norms
    # Filter non-finite columns
    valid = np.isfinite(b).all(axis=0)
    return b, valid

# Parallax angles between points (radians)
def parallax_angles_rad(R, K1, K2, x1, x2, mask=None, eps=1e-12):
    # Checks
    check_2xN_pair(x1, x2)
    check_3x3(R)
    check_3x3(K1)
    check_3x3(K2)
    N = x1.shape[1]
    mask = check_bool_N(mask, N)
    if mask is None:
        mask = np.ones(N, dtype=bool)
    # Bearing vectors
    b1, valid1 = bearing_vectors(x1, K1, eps)
    b2, valid2 = bearing_vectors(x2, K2, eps)
    # Intersect valid masks
    valid = mask & valid1 & valid2
    # Rotate into cam 1 frame
    b2_in_1 = R.T @ b2
    # Dot product
    cos_theta = np.sum(b1 * b2_in_1, axis=0)
    # Validity
    valid &= np.isfinite(cos_theta)
    # Clip
    cos_theta = np.clip(cos_theta, -1.0, 1.0) 
    # Apply validity
    cos_theta = cos_theta[valid]
    if cos_theta.size == 0:
        return np.array([], dtype=float)
    # Theta
    return np.arccos(cos_theta)

# Parallax angle stats (degrees)
def parallax_angle_stats_deg(R, K1, K2, x1, x2, mask=None, quartile_trim=(0.1, 0.9), min_points=8): 
    # Parallax angles
    a_rad = parallax_angles_rad(R, K1, K2, x1, x2, mask)
    # Default
    stats = {"n": int(a_rad.size), "n_trim": 0, "p50": 0.0, "p25": 0.0, "p75": 0.0, "reason": None}
    # Too few angles
    if a_rad.size < min_points: 
        stats.update({"reason": "too_few_angles"})
        return stats
    # Convert to degrees
    a_deg = a_rad * (180.0 / np.pi)
    # Trim extremes for robustness
    lo_q, hi_q = quartile_trim
    lo = np.quantile(a_deg, lo_q)
    hi = np.quantile(a_deg, hi_q)
    a_trim = a_deg[(a_deg >= lo) & (a_deg <= hi)]
    # Don't trim if too small
    if a_trim.size < min_points:
        a_trim = a_deg
    # Update stats
    stats.update({
        "n_trim": int(a_trim.size),
        "p25": float(np.percentile(a_trim, 25)),
        "p50": float(np.percentile(a_trim, 50)),
        "p75": float(np.percentile(a_trim, 75))})
    # Return with quartiles
    return stats
