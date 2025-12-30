# src/geometry/epipolar.py
import numpy as np
from geometry.homogeneous import homogenise

# Algebraic residuals
def algebraic_residuals(x1, x2, F):
    # Assert
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    # Shape error
    if x1.shape[0] != 2:
        raise ValueError(f"Shape of x should be (2, N)")
    # Homogenise
    x1h = homogenise(x1)
    x2h = homogenise(x2)
    # Epipolar lines in image 2
    Fx1 = F @ x1h
    # Residual r_i = x2_i^T F x1_i
    return np.sum(x2h * Fx1, axis=0)

# Algebraic residual RMSE
def algebraic_residual_rmse(x1, x2, F):
    # Algebraic residuals
    r = algebraic_residuals(x1, x2, F)
    # Root mean squared residual
    return np.sqrt(np.mean(r**2))

# Euclidean point-to-line distances
def point_to_line_distances(x, l): 
    # Assert 
    x = np.asarray(x, dtype = float)
    l = np.asarray(l, dtype = float)
    # Shape error
    if x.shape[0] != 2: 
        raise ValueError(f"Shape of x should be (2, N)")
    if l.shape[0] != 3: 
        raise ValueError(f"Shape of l should be (3, N)")
    # Homogenise
    x_h = homogenise(x)
    # Distance
    num = np.abs(np.sum(l * x_h, axis=0))
    den = np.maximum(np.sqrt(l[0]**2 + l[1]**2), 1e-12)
    return num / den

# Symmetric distances squared
def symmetric_distances_sq(x1, x2, F):
    # Assert 
    x1 = np.asarray(x1, dtype = float)
    x2 = np.asarray(x2, dtype = float)
    # Shape error
    if x1.shape[0] != 2: 
        raise ValueError(f"Shape of x should be (2, N)")
    # Homogenise
    x1_h = homogenise(x1)
    x2_h = homogenise(x2)
    # Epipolar lines
    l1 = F.T @ x2_h
    l2 = F @ x1_h
    # Euclidean point-to-line distances 
    p2l1 = point_to_line_distances(x1, l1)
    p2l2 = point_to_line_distances(x2, l2)
    # Square and sum
    return p2l1**2 + p2l2**2
    
# Sampson distances squared
def sampson_distances_sq(x1, x2, F):
    # Assert 
    x1 = np.asarray(x1, dtype = float)
    x2 = np.asarray(x2, dtype = float)
    # Shape error
    if x1.shape[0] != 2: 
        raise ValueError(f"Shape of x should be (2, N)")
    # Algebraic residuals
    r = algebraic_residuals(x1, x2, F)
    # Numerator
    num = r**2
    # Homogenise
    x1_h = homogenise(x1)
    x2_h = homogenise(x2)
    # Epipolar lines
    l1 = F.T @ x2_h
    l2 = F @ x1_h
    # Denominator
    den = np.maximum(l1[0]**2 + l1[1]**2 + l2[0]**2 + l2[1]**2, 1e-12)
    return num / den
