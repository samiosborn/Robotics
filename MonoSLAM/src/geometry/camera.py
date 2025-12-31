# src/geometry/camera.py
import numpy as np
from geometry.homogeneous import homogenise, dehomogenise

class Camera:
    def __init__(self, K, R, t): 
        # --- Class Variables ---

        # Intrinsic Matrix (K)
        self.K = K
        # Rotation Matrix (R)
        self.R = R
        # Translation Vector (t)
        self.t = t
        # Projection Matrix (P)
        self.P = self.K @ np.hstack((self.R, self.t.reshape(3,1)))
        # Camera Center (C)
        self.C = -self.R.T @ self.t

    # --- Public API ---

    # Project to camera frame
    def project(self, X): 
        # Assert 
        X = np.asarray(X, dtype=float)
        # Raise error
        if X.shape[0] != 3: 
            raise ValueError(f"X should have shape (3, N), got {X.shape}")
        # Convert to homogeneous 
        X_h = homogenise(X)
        # Project
        x_h = self.P @ X_h
        # Dehomogenise
        x_camera = dehomogenise(x_h)
        # Return pixel coordinates
        return x_camera

    # Reprojection residuals
    def reprojection_residuals(self, X, x_obs):
        # Assert
        x_obs = np.asarray(x_obs, dtype=float)
        # Project
        x_cam = self.project(X)
        # Raise errror
        if x_cam.shape[1] != x_obs.shape[1]:
            raise ValueError(f"Shape mismatch: projected {x_cam.shape} vs observed {x_obs.shape}")
        # Residuals
        return x_obs - x_cam

    # Reprojection RMSE (scalar)
    def reprojection_rmse(self, X, x_obs): 
        # Reprojection residuals 
        residuals = self.reprojection_residuals(X, x_obs)
        # RMSE
        return float(np.sqrt(np.mean(np.sum(residuals **2, axis=0))))

    # Inhomogeneous pixel coordinates to normalised homogeneous image coordinates
    def pixel_to_normalised(self, x): 
        x_h = homogenise(x)
        return np.linalg.inv(self.K) @ x_h

    # Normalised homogeneous image coordinates to homogeneous pixel coordinates
    def normalised_to_pixel(self, x_hat):
        return self.K @ x_hat
