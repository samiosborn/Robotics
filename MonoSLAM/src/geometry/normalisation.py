# src/geometry/normalisation.py
import numpy as np
from geometry.homogeneous import homogenise, dehomogenise

# Normalise projective scale
def normalise_projective_scale(M, eps=1e-12):
    # Assert
    M = np.asarray(M, dtype=float)
    # Check dims
    if M.shape != (3, 3):
        raise ValueError(f"M must be (3,3); got {M.shape}")
    # Normalise
    if abs(M[2, 2]) > eps:
        return M / M[2, 2]
    return M / (np.linalg.norm(M) + eps)

# Denormalise point mapping (x2 ~ M x1)
def denormalise_point_mapping(Mn, T1, T2):
    return np.linalg.inv(T2) @ Mn @ T1

# Denormalise bilinear form (x1n = T1 x1)
def denormalise_bilinear_form(Fn, T1, T2):
    return T2.T @ Fn @ T1

# Hartley normalisation
def hartley_norm(x): 
    # Assert
    x = np.asarray(x, dtype=float).copy()
    # Error
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError("x must have shape (2, N)")
    # Homogenise
    x_h = homogenise(x)
    # Guard
    if x_h.shape[0] != 3:
        raise RuntimeError("Homogenise must return shape (3, N)")
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
    # Similarity Transform
    T = np.array([
        [s, 0, -s * c[0]],
        [0, s, -s * c[1]],
        [0, 0, 1]
    ])
    # Transformed (homogeneous coords)
    x_n_h = T @ x_h
    # Dehomogenise
    x_n = dehomogenise(x_n_h)
    return x_n, T
