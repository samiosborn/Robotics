# geometry/fundamental.py
import numpy as np
from geometry.checks import check_2xN_pair, check_3x3, check_bool_N
from geometry.normalisation import hartley_norm
from geometry.essential import essential_from_pose
from geometry.distances import sampson_distances_sq

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
    check_2xN_pair(x1, x2)
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
def estimate_fundamental_ransac(x1, x2, num_trials=1000, sample_size = 8, threshold=3, seed=42):
    # Random number generator
    rng = np.random.default_rng(seed)
    # Check dimensions
    check_2xN_pair(x1, x2)
    N = x1.shape[1]
    # Too few points vs sample size
    if sample_size > N: 
        raise ValueError(f"Sample size of {sample_size} is more than number of points: {N}")
    # Sample size too small
    if sample_size < 8: 
        raise ValueError("Sample size of at least 8 correspondences is required")
    # Initialise inliers
    best_inlier_mask = np.zeros(N, dtype=bool)
    best_count = 0
    F_best = None
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
        inlier_mask = distances_sq_t < threshold**2
        # Update best count
        count = int(np.sum(inlier_mask))
        if count > best_count: 
            best_count = count
            best_inlier_mask = inlier_mask
            F_best = F_t
            # Early return
            if best_count == N: 
                break

    if F_best is None or best_count < sample_size: 
        raise ValueError("RANSAC failed: not enough inliers to estimate F below threshold")

    return F_best, best_inlier_mask

# Refit fundamental matrix on inliers
def refit_fundamental_on_inliers(x1, x2, F, inlier_mask, min_inliers=8, threshold=3, shrink_guard=0.8, recompute_mask=True):
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(F)
    N = x1.shape[1]
    mask = check_bool_N(inlier_mask, N)
    # No mask
    if mask is None:
        return F, None, {"refit": False, "reason": "mask_is_none"}
    n0 = int(mask.sum())
    # Initialise statistics
    stats = {"n0": n0, "reason": None}
    # Not enough points to refit
    if n0 < min_inliers:
        # Return original
        stats.update({"refit": False, "reason": "too_few_inliers"})
        return F, mask, stats
    # Refit on inliers
    F_refit = estimate_fundamental(x1[:, mask], x2[:, mask])
    stats.update({"refit": True})
    # Don't recompute mask
    if not recompute_mask:
        stats.update({"n1": n0, "used_mask": "original"})
        return F_refit, mask, stats
    # Recompute mask from refit model
    d_sq = sampson_distances_sq(x1, x2, F_refit)
    mask_new = d_sq < float(threshold**2)
    n1 = int(mask_new.sum())
    # Shrink ratio
    stats.update({"n1": n1, "shrink_ratio": (n1 / max(n0, 1))})
    # Shrink guard
    if n1 < shrink_guard * n0:
        stats.update({"used_mask": "original", "reason": "shrink_guard_triggered"})
        return F_refit, mask, stats
    # Use new mask
    stats.update({"used_mask": "recomputed"})
    return F_refit, mask_new, stats
