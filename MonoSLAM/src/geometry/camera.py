# src/geometry/camera.py
from __future__ import annotations

import numpy as np

from core.checks import check_3xN_2xN_cols, check_matrix_3x3, check_points_2xN, check_points_3xN, check_vector_3
from geometry.distances import point_to_point_distances_sq, rmse_from_sq
from geometry.homogeneous import dehomogenise, homogenise


# Build projection matrix P = K [R | t]
def projection_matrix(K, R, t):
    # --- Checks ---
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    # Check rotation
    R = check_matrix_3x3(R, name="R", dtype=float, finite=False)
    # Check translation
    t = check_vector_3(t, name="t", dtype=float, finite=False)

    # Build projection matrix
    P = K @ np.hstack((R, t.reshape(3, 1)))

    return P


# Compute camera centre in world coordinates
def camera_center(R, t):
    # --- Checks ---
    # Check rotation
    R = check_matrix_3x3(R, name="R", dtype=float, finite=False)
    # Check translation
    t = check_vector_3(t, name="t", dtype=float, finite=False)

    # Compute camera centre
    C = -R.T @ t

    return C


# Transform world points into camera coordinates
def world_to_camera_points(R, t, X_w):
    # --- Checks ---
    # Check rotation
    R = check_matrix_3x3(R, name="R", dtype=float, finite=False)
    # Check translation
    t = check_vector_3(t, name="t", dtype=float, finite=False)
    # Check world points
    X_w = check_points_3xN(X_w, name="X_w", dtype=float, finite=False)

    # Transform into camera frame X_c = R X_w + t
    X_c = R @ X_w + t.reshape(3, 1)

    return X_c


# Project world points into inhomogeneous pixel coordinates
def project_points(K, R, t, X_w):
    # --- Checks ---
    # Check world points
    X_w = check_points_3xN(X_w, name="X_w", dtype=float, finite=False)

    # Build projection matrix
    P = projection_matrix(K, R, t)

    # Homogenise world points
    X_h = homogenise(X_w)

    # Project into image
    x_h = P @ X_h

    # Dehomogenise to pixel coordinates
    x = dehomogenise(x_h)

    return x


# Compute reprojection residuals in pixels
def reprojection_residuals(K, R, t, X_w, x_obs):
    # --- Checks ---
    # Check 3D to 2D correspondence count and shapes
    X_w, x_obs = check_3xN_2xN_cols(X_w, x_obs, nameX="X_w", namex="x_obs", dtype=float, finite=False)

    # Predict pixels
    x_pred = project_points(K, R, t, X_w)

    # Residuals are observed minus predicted
    r = x_obs - x_pred

    return r


# Compute squared reprojection errors per point
def reprojection_errors_sq(K, R, t, X_w, x_obs):
    # --- Checks ---
    # Check 3D to 2D correspondence count and shapes
    X_w, x_obs = check_3xN_2xN_cols(X_w, x_obs, nameX="X_w", namex="x_obs", dtype=float, finite=False)

    # Predict pixels
    x_pred = project_points(K, R, t, X_w)

    # Squared Euclidean image errors
    d_sq = point_to_point_distances_sq(x_obs, x_pred)

    return np.asarray(d_sq, dtype=float)


# Compute reprojection RMSE in pixels
def reprojection_rmse(K, R, t, X_w, x_obs):
    # Compute squared reprojection errors
    d_sq = reprojection_errors_sq(K, R, t, X_w, x_obs)

    # Compute RMSE in pixels
    return rmse_from_sq(d_sq)


# Convert inhomogeneous pixel coordinates to normalised homogeneous image coordinates
def pixel_to_normalised(K, x):
    # --- Checks ---
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    # Check pixel coordinates
    x = check_points_2xN(x, name="x", dtype=float, finite=False)

    # Homogenise pixel coordinates
    x_h = homogenise(x)

    # Convert to normalised image coordinates
    x_hat = np.linalg.inv(K) @ x_h

    return x_hat


# Convert normalised homogeneous image coordinates to homogeneous pixel coordinates
def normalised_to_pixel(K, x_hat):
    # --- Checks ---
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    # Check normalised homogeneous points
    x_hat = check_points_3xN(x_hat, name="x_hat", dtype=float, finite=False)

    # Convert to homogeneous pixels
    x_h = K @ x_hat

    return x_h


class Camera:
    def __init__(self, K, R, t):
        # --- Checks ---
        # Check intrinsics
        self.K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
        # Check rotation
        self.R = check_matrix_3x3(R, name="R", dtype=float, finite=False)
        # Check translation
        self.t = check_vector_3(t, name="t", dtype=float, finite=False)

        # Cache projection matrix
        self.P = projection_matrix(self.K, self.R, self.t)

        # Cache camera centre
        self.C = camera_center(self.R, self.t)

    # Project world points to image pixels
    def project(self, X_w):
        return project_points(self.K, self.R, self.t, X_w)

    # Compute reprojection residuals in pixels
    def reprojection_residuals(self, X_w, x_obs):
        return reprojection_residuals(self.K, self.R, self.t, X_w, x_obs)

    # Compute squared reprojection errors per point
    def reprojection_errors_sq(self, X_w, x_obs):
        return reprojection_errors_sq(self.K, self.R, self.t, X_w, x_obs)

    # Compute reprojection RMSE in pixels
    def reprojection_rmse(self, X_w, x_obs):
        return reprojection_rmse(self.K, self.R, self.t, X_w, x_obs)

    # Convert pixels to normalised image coordinates
    def pixel_to_normalised(self, x):
        return pixel_to_normalised(self.K, x)

    # Convert normalised image coordinates to homogeneous pixels
    def normalised_to_pixel(self, x_hat):
        return normalised_to_pixel(self.K, x_hat)