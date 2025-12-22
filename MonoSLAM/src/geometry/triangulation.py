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
    return dehomogenise(X_h).reshape(3, )

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
def select_valid_pose(candidates, P1, x1s, x2s):
    # Loop over candidates
    for R, t in candidates: 
        # Projection matrix (assuming no intrinsics)
        P2 = np.hstack((R, t.reshape(3, 1)))
        # Initialise cheirality check count
        count = 0
        # Loop over corresponding points
        for x1, x2 in zip(x1s, x2s): 
            X = triangulate_point(P1, P2, x1, x2)
            # Check for cheirality
            if cheirality_check(R, t, X): 
                count += 1
        # Return valid pose
        if count > 0: 
            return R, t
    # Failure
    raise RuntimeError("No valid pose found")
