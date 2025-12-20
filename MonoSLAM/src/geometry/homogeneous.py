# src/geometry/homogeneous.py
import numpy as np

# Homogenise
def homogenise(X):
    # Convert
    X = np.asarray(X)
    # Reshape
    if X.ndim == 1: 
        X = X.reshape(-1, 1)
    # Error
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(f"Expected column vector, got shape {X.shape}")
    # Homogenise
    return np.vstack((X, np.ones((1, 1))))

# Dehomogenise
def dehomogenise(X_h): 
    # Convert
    X_h = np.asarray(X_h)
    # Reshape
    if X_h.ndim == 1: 
        X_h = X_h.reshape(-1, 1)
    # Error
    if X_h.ndim != 2 or X_h.shape[1] != 1:
        raise ValueError(f"Expected column vector, got shape {X_h.shape}")
    # Weight
    w = X_h[-1, 0]
    if abs(w) < 1e-12: 
        raise ValueError("Cannot dehomogenise point at infinity")
    # Divide all but last by weight
    return X_h[:-1] / w
