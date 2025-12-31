# src/geometry/triangulation.py
import numpy as np
from geometry.homogeneous import homogenise, dehomogenise

# Triangulate a point
def triangulate_point(P1, P2, x1, x2):
    # Homogenise
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
    # Dehomogenise
    X = dehomogenise(X_h).reshape(3, )
    return X

# Triangulate points
def triangulate_points(P1, P2, x1s, x2s):
    # Number of points
    n_points = x1s.shape[1]
    # Pre-allocate
    Xs = np.zeros((3, n_points))
    # Loop over points
    for i in range(n_points):
        # Corresponding point
        x1i = x1s[:, i]
        x2i = x2s[:, i]
        # Triangulated point
        Xi = triangulate_point(P1, P2, x1i, x2i)
        # Append
        Xs[:, i] = Xi
    return Xs

# Check if a point is in front of camera
def is_in_front_of_camera(R, t, X):
    # Camera model
    X_cam = R @ X + t
    return X_cam[2] > 0

# Cheirality check for two views
def cheirality_check(R, t, X):
    # Camera 1 (at origin)
    in_front_cam_1 = X[2] > 0
    # Camera 2 with (R, t)
    in_front_cam_2 = is_in_front_of_camera(R, t, X)
    # Check both
    return (in_front_cam_1 and in_front_cam_2)

# Select the valid pose from candidates
def select_valid_pose(candidates, K1, K2, x1s, x2s):
    # Assert
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    # Check dimensions
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("Shape of K1 and K2 should be (3, 3)")
    if x1s.shape[0] != 2 or x1s.shape != x2s.shape: 
        raise ValueError("Points should be (2, N)")
    # Initialise
    n_points = x1s.shape[1]
    best_idx = None
    best_count = 0
    # Projection matrix 1 (at origin)
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # Loop over candidates
    for i, (R, t) in enumerate(candidates): 
        # Projection matrix (assuming no intrinsics)
        P2 = K2 @ np.hstack((R, t.reshape(3, 1)))
        # Initialise cheirality check count
        count = 0
        # Loop over corresponding points
        for j in range(n_points): 
            # Corresponding point
            x1j = x1s[:, j]
            x2j = x2s[:, j]
            X = triangulate_point(P1, P2, x1j, x2j)
            # Check for cheirality
            if cheirality_check(R, t, X): 
                count += 1
        # Counts
        if count > best_count: 
            best_idx = i
            best_count = count
    # Return best pose
    if best_count > 0:
        return candidates[best_idx]
    # Failure
    raise RuntimeError("No valid pose found")
