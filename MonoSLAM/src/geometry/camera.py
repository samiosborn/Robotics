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

    # --- Properties ---
    
    # Projection Matrix (P)
    @property
    def P(self):
        # P = K [R | t]
        return self.K @ np.hstack((self.R, self.t.reshape(3,1)))

    # Camera Center (C)
    @property
    def C(self): 
        # C = - R^T t
        return - self.R.T @ self.t

    # --- Public API ---

    # Project to camera frame
    def project(self, X_world): 
        # Convert to homogeneous 
        X_h = homogenise(X_world)
        # Project
        x_h = self.P @ X_h
        # Dehomogenise
        x_camera = dehomogenise(x_h)
        # Return pixel coordinates
        return x_camera

    # Inhomogeneous pixel coordinates to normalised homogeneous image coordinates
    def pixel_to_normalised(self, x): 
        x_h = homogenise(x)
        return np.linalg.inv(self.K) @ x_h

    # Normalised homogeneous image coordinates to homogeneous pixel coordinates
    def normalised_to_pixel(self, x_hat):
        return self.K @ x_hat
