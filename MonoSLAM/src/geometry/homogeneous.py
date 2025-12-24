# src/geometry/homogeneous.py
import numpy as np

# Homogenise
def homogenise(X):
    # Convert
    X = np.asarray(X, dtype=float)
    # Reshape
    if X.ndim == 1: 
        X = X.reshape(-1, 1)
    # Error
    if X.ndim != 2:
        raise ValueError(f"Expected array of shape (d, N), got {X.shape}")
    # Homogenise
    ones = np.ones((1, X.shape[1]))
    return np.vstack((X, ones))

# Dehomogenise
def dehomogenise(X_h): 
    # Convert
    X_h = np.asarray(X_h, dtype=float)
    # Reshape
    if X_h.ndim == 1: 
        X_h = X_h.reshape(-1, 1)
    # Error
    if X_h.ndim != 2 or X_h.shape[0] < 2:
        raise ValueError(f"Expected array of shape (d+1, N), got {X_h.shape}")
    # Weight
    w = X_h[-1]
    # Error
    if np.any(abs(w)) < 1e-12: 
        raise ValueError("Cannot dehomogenise point at infinity")
    # Divide all but last by weight
    return X_h[:-1] / w
