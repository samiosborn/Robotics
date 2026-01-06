# geometry/homography.py
import numpy as np
from geometry.checks import check_2xN_pair
from geometry.homogeneous import homogenise, dehomogenise
from geometry.normalisation import hartley_norm, denormalise_point_mapping, normalise_projective_scale
from epipolar import point_to_point_distances_sq

# Apply homography
def apply_homography(H, x): 
    # Homogenise
    x_h = homogenise(x)
    # Transform
    y_h = H @ x_h
    # Dehomogenise
    y = dehomogenise(y_h)
    return y

# Direct Linear Transform for homography (assuming Hartley normalised points)
def dlt_homography(x1, x2): 
    # Check dims
    check_2xN_pair(x1, x2)
    # No. of correspondences
    N = x1.shape[1]
    if N < 4:
        raise ValueError(f"Need at least 4 correspondences; got {N}")
    # Homogenise
    x1_h = homogenise(x1)
    x2_h = homogenise(x2)
    # Initialise A
    A = np.zeros((2 * N, 9), dtype=float)
    # Loop for all pair of correspondences
    for i in range(N): 
        # X1
        X = x1_h[:, i]
        # X2
        u, v, _ = x2_h[:, i]
        # (h1^T X) - u (h3^T X) = 0
        A[2*i, 0:3] =  X
        A[2*i, 6:9] = -u * X
        # (h2^T X) - v (h3^T X) = 0
        A[2*i+1, 3:6] = X
        A[2*i+1, 6:9] = -v * X
    # Solve: Ah = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    # Smallest singular value
    h = Vt[-1, :]
    # Reshape into square matrix
    H = h.reshape(3, 3)
    # Fix scale convention
    return normalise_projective_scale(H)

# Estimate homography
def estimate_homography(x1, x2, normalise=True): 
    # Check dims
    check_2xN_pair(x1, x2)
    # Already normalised
    if not normalise: 
        return dlt_homography(x1, x2)
    # Normalise
    x1n, T1 = hartley_norm(x1)
    x2n, T2 = hartley_norm(x2)
    # Homography
    Hn = dlt_homography(x1n, x2n)
    # Denormalise homography
    H = denormalise_point_mapping(Hn, T1, T2)
    # Fix scale convention
    return normalise_projective_scale(H)

# Transfer errors (squared)
def transfer_errors_sq(x1, x2, H):
    # Check dims
    check_2xN_pair(x1, x2)
    # Apply homography
    x2_hat = apply_homography(H, x1)
    # Point-to-point distance (squared)
    return point_to_point_distances_sq(x1, x2)

# Symmetric transfer errors (squared)
def symmetric_transfer_errors_sq(x1, x2, H): 
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(H)
    # Forward
    e12 = transfer_errors_sq(x1, x2, H)
    # Backwards
    Hinv = np.linalg.inv(H)
    e21 = transfer_errors_sq(x2, x1, Hinv)
    # Sum
    return e12 + e21
