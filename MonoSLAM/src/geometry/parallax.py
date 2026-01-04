# geometry/parallax.py
import numpy as np
from geometry.homogeneous import homogenise
from geometry.checks import check_2xN_pair

# Bearing vectors
def bearing_vectors(x, K): 
    # Homogenise
    x_h = homogenise(x)
    # Bearings
    b = np.linalg.inv(K) @ x_h
    # Normalise
    norms = np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-12)
    b = b / norms
    return b 

# Parallax angles between points (radians)
def parallax_angles(R, K1, K2, x1, x2):
    # Bearing vectors
    b1 = bearing_vectors(x1, K1)
    b2 = bearing_vectors(x2, K2)
    # Rotate into cam 1 frame
    b2_in_1 = R.T @ b2
    # Dot product
    cos_theta = np.sum(b1 * b2_in_1, axis=0)
    # Clip
    cos_theta = np.clip(cos_theta, -1.0, 1.0) 
    # Angles
    theta = np.arccos(cos_theta)
    return theta

# Median parallax angle (degrees)
def median_parallax_angle_deg(R, K1, K2, x1, x2, mask=None): 
    # Parallax angles
    angles = parallax_angles(R, K1, K2, x1, x2)
    # Mask
    if mask is not None: 
        angles = angles[mask]
    # Error check
    if angles.size == 0:
        return 0.0
    # Median (in deg)
    return float(np.median(angles) * 180.0 / np.pi)
