# src/geometry/hartley_normalisation.py
import numpy as np
from geometry.homogeneous import homogenise

# Hartley normalisation
def hartley_norm(x): 
    # Assert
    x = np.asarray(x, dtype=float).copy()
    # Error
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError("x must have shape (2, N)")
    # Centroid
    c = np.mean(x, axis=1)
    # Translate
    x -= c[:, None]
    # Distance
    d = np.mean(np.sqrt(np.sum(x**2, axis=0)))
    # Error
    if d < 1e-12:
        raise ValueError("Degenerate configuration: points are coincident")
    # Scale factor
    s = np.sqrt(2) / d
    # Similarity transform
    T = np.array([
        [s, 0, -s * c[0]],
        [0, s, -s * c[1]],
        [0, 0, 1]
    ])
    # Homogenise
    x_h = homogenise(x)
    # Guard
    if x_h.shape[0] != 3:
        raise RuntimeError("Homogenise must return shape (3, N)")
    # Transform
    x_transformed = T @ x_h
    return x_transformed, T
