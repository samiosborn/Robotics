# src/geometry/fundamental.py
import numpy as np
from geometry.hartley_normalisation import hartley_norm
from geometry.essential import essential_from_pose

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
