# geometry/triangulation.py
import numpy as np
from geometry.homogeneous import homogenise, dehomogenise

# Triangulate a point
def triangulate_point(P1, P2, x1, x2):
    # Homogenise and reshape
    x1 = homogenise(x1).reshape(3, )
    x2 = homogenise(x2).reshape(3, )
    # AX = 0
    A = np.vstack([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1],
    ])
    # SVD
    _, _, Vt = np.linalg.svd(A)
    # X
    X_h = Vt[-1]
    # Avoid near degenerate points
    w = X_h[-1]
    if abs(w) < 1e-12: 
        return np.array([np.nan, np.nan, np.nan])
    # Dehomogenise
    X = dehomogenise(X_h).reshape(3, )
    return X

# Triangulate points
def triangulate_points(P1, P2, x1, x2):
    # Assert
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    # Dimension check
    if x1.ndim != 2 or x2.ndim != 2 or x1.shape[0] != 2 or x2.shape[0] != 2 or x1.shape[1] != x2.shape[1]:
        raise ValueError(f"x1 and x2 must be (2, N); got {x1.shape} and {x2.shape}")
    # Number of points
    n_points = x1.shape[1]
    # Pre-allocate
    X = np.zeros((3, n_points))
    # Loop over points
    for i in range(n_points):
        # Corresponding point
        x1i = x1[:, i]
        x2i = x2[:, i]
        # Triangulated point
        Xi = triangulate_point(P1, P2, x1i, x2i)
        # Append
        X[:, i] = Xi
    return X

# Depths in the two cameras for Euclidean X
def depths_two_view(R, t, X):
    # Z-dim (cam1)
    z1 = X[2, :]
    # Project (cam2)
    X2 = R @ X + t.reshape(3, 1)
    # Z-dim (cam2)
    z2 = X2[2, :]
    return z1, z2

# Cheirality mask from X
def cheirality_mask_from_X(R, t, X):
    # Assert
    X = np.asarray(X, dtype=float)
    # Check dimensions
    if X.ndim != 2 or X.shape[0] != 3:
        raise ValueError(f"X must be (3, N) Euclidean; got {X.shape}")
    # Depths
    z1, z2 = depths_two_view(R, t, X)
    # Both positive depth
    eps = 1e-9
    return (z1 > eps) & (z2 > eps)

# Cheirality mask for pose
def cheirality_mask_for_pose(R, t, P1, K2, x1, x2): 
    # Projection matrix 2
    P2 = K2 @ np.hstack((R, t.reshape(3, 1)))
    # Triangulate points
    X = triangulate_points(P1, P2, x1, x2)
    # Cheirality mask
    mask = cheirality_mask_from_X(R, t, X)
    return mask, X

# Disambiguate pose candidates using cheirality
def disambiguate_pose_cheirality(candidates, K1, K2, x1, x2):
    # Assert
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    # Check dimensions
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError(f"Shape of K1 and K2 should be (3, 3); got K1 {K1.shape}, K2 {K2.shape}")
    if x1.ndim != 2 or x2.ndim != 2 or x1.shape != x2.shape or x1.shape[0] != 2:
        raise ValueError(f"Points should be (2, N); got x1 {x1.shape}, x2 {x2.shape}")
    # Initialise
    best = None
    best_count = 0
    best_mask = None
    # Projection matrix 1 (at origin)
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # Loop over candidates
    for i, (R, t) in enumerate(candidates): 
        # Cheirality count for pose
        mask, _ = cheirality_mask_for_pose(R, t, P1, K2, x1, x2)
        count = int(np.sum(mask))
        # Counts
        if count > best_count: 
            best = R, t
            best_count = count
            best_mask = mask
    # Return best pose
    if best_count > 0:
        return best[0], best[1], best_mask
    # Failure
    raise RuntimeError("No valid pose found")
