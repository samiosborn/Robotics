# src/geometry/fundamental.py
import numpy as np
from geometry.hartley_normalisation import hartley_norm
from geometry.essential import essential_from_pose
from geometry.epipolar import sampson_distances_sq

# Fundamental from essential matrix
def fundamental_from_essential(E, K1, K2):
    # F = K2^-T E K1^-1
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

# Fundamental matrix from pose
def fundamental_from_pose(R, t, K1, K2):
    # Essential matrix
    E = essential_from_pose(R, t)
    # Fundamental matrix
    return fundamental_from_essential(E, K1, K2)

# Estimate fundamental matrix (normalised 8-point algo)
def estimate_fundamental(x1, x2): 
    # Check dimensions
    if x1.shape[0] != 2 or x1.shape != x2.shape: 
        raise ValueError("x1, x2 must have shape (2, N)")
    N = x1.shape[1]
    # Minimum points
    if N < 8: 
        raise ValueError("At least 8 correspondences required")
    # Hartley normalisation
    x1h, T1 = hartley_norm(x1)
    x2h, T2 = hartley_norm(x2)
    # Build A
    u1, v1 = x1h[0], x1h[1]
    u2, v2 = x2h[0], x2h[1]
    A = np.stack([
        u2 * u1, u2 * v1, u2, 
        v2 * u1, v2 * v1, v2, 
        u1, v1, np.ones(N)
    ], axis=1)
    # Solve Af = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    # Estimate normalised F 
    F_hat_0 = Vt[-1].reshape(3, 3)
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F_hat_0)
    S[-1] = 0.0
    # Update normalised F
    F_hat = U @ np.diag(S) @ Vt
    # Un-normalised F
    F = T2.T @ F_hat @ T1
    # Deterministic fix scale
    F /= np.linalg.norm(F)
    return F

# Estimate fundamental matrix via RANSAC
def estimate_fundamental_ransac(x1, x2, num_trials, threshold_sq, seed=42):
    # Random number generator
    rng = np.random.default_rng(seed)
    # Check dimensions
    if x1.shape[0] != 2 or x1.shape != x2.shape: 
        raise ValueError("x1, x2 must have shape (2, N)")
    N = x1.shape[1]
    # Minimum points
    sample_size = 8
    if N < sample_size: 
        raise ValueError("At least 8 correspondences required")
    # Initialise inliers
    best_inlier_mask = np.zeros(N, dtype=bool)
    best_count = 0
    best_F = None
    # Trial iteration
    for _ in range(num_trials): 
        # Random subset index
        s_idx = rng.choice(N, size=sample_size, replace=False)
        # Random subset of points
        x1_t = x1[:, s_idx]
        x2_t = x2[:, s_idx]
        # Try to fit
        try: 
            # 8-point algorithm
            F_t = estimate_fundamental(x1_t, x2_t)
            # Sampson distances squared
            distances_sq_t = sampson_distances_sq(x1, x2, F_t)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        # Degenerate sample / numerical failure (skip)
            continue
        # Below threshold
        inlier_mask = distances_sq_t < threshold_sq
        # Update best count
        count = int(np.sum(inlier_mask))
        if count > best_count: 
            best_count = count
            if best_count == N: 
                break
            best_inlier_mask = inlier_mask
            best_F = F_t

    if best_F is None or best_count < sample_size: 
        raise ValueError("RANSAC failed: not enough inliers to estimate F below threshold")

    # Refit using all inliers
    best_F = estimate_fundamental(x1[:, best_inlier_mask], x2[:, best_inlier_mask])

    return best_F, best_inlier_mask, best_count
