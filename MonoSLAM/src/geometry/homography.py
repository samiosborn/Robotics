# geometry/homography.py
import numpy as np
from geometry.checks import check_2xN, check_2xN_pair, check_3x3
from geometry.homogeneous import homogenise, dehomogenise
from geometry.normalisation import hartley_norm, denormalise_point_mapping, normalise_projective_scale
from geometry.distances import point_to_point_distances_sq

# Apply homography
def apply_homography(H, x): 
    # Checks
    check_2xN(x)
    check_3x3(H)
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
    return point_to_point_distances_sq(x2, x2_hat)

# Symmetric transfer errors (squared)
def symmetric_transfer_errors_sq(x1, x2, H): 
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(H)
    # Near singular H
    det = np.linalg.det(H)
    if abs(det) < 1e-12:
        return np.full(x1.shape[1], np.inf, dtype=float)
    # Forward
    e12 = transfer_errors_sq(x1, x2, H)
    # Backwards
    Hinv = np.linalg.inv(H)
    e21 = transfer_errors_sq(x2, x1, Hinv)
    # Sum
    return e12 + e21

# Estimate homography RANSAC
def estimate_homography_ransac(x1, x2, num_trials, threshold, normalise=True, seed=None): 
    # Check dims
    check_2xN_pair(x1, x2)
    N = x1.shape[1]
    # Initialise
    rng = np.random.default_rng(seed)
    best_H = None
    best_mask = None
    best_count = 0
    best_err = np.inf
    # Fail
    if N < 4: 
        reason = "requires_more_than_4_correspondences"
        return best_H, best_mask, reason
    # Loop
    for _ in range(num_trials): 
        # Subset
        idx = rng.choice(N, size=4, replace=False)
        # Estimate homography
        try: 
            H = estimate_homography(x1[:, idx], x2[:, idx], normalise)
        except Exception: 
            continue
        # Symmetric transfer errors (sq)
        d_sq = symmetric_transfer_errors_sq(x1, x2, H)
        # Inlier mask
        mask = d_sq < threshold**2
        # Count
        count = int(mask.sum())
        if count > 0: 
            err = float(np.mean(d_sq[mask]))
            # Take higher count
            if count > best_count: 
                best_count = count
                best_H = H
                best_mask = mask
                best_err = err
            # Tie
            elif count == best_count: 
                if err < best_err: 
                    best_err = err
                    best_H = H
                    best_mask = mask
    # Failure
    if best_mask is None: 
        reason = "homography_mask_missing"
    else: 
        reason = None
    # Return
    return best_H, best_mask, reason
