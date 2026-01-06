# geometry/distances.py
import numpy as np
from geometry.checks import check_2xN, check_3xN, check_2xN_pair
from geometry.homogeneous import homogenise
from geometry.epipolar import algebraic_residuals

# Euclidean point-to-line distances
def point_to_line_distances(x, l): 
    # Assert 
    x = np.asarray(x, dtype = float)
    l = np.asarray(l, dtype = float)
    # Check dims
    check_2xN(x)
    check_3xN(l)
    if l.shape[1] != x.shape[1]:
        raise ValueError(f"x and l must share N; got x {x.shape}, l {l.shape}")
    # Homogenise
    x_h = homogenise(x)
    # Distance
    num = np.abs(np.sum(l * x_h, axis=0))
    den = np.maximum(np.sqrt(l[0]**2 + l[1]**2), 1e-12)
    return num / den

# Euclidean point-to-point distances (squared)
def point_to_point_distances_sq(x1, x2): 
    # Assert 
    x1 = np.asarray(x1, dtype = float)
    x2 = np.asarray(x2, dtype = float)
    # Check dims
    check_2xN_pair(x1, x2)
    # Difference
    d = x1 - x2
    # Square
    return np.sum(d * d, axis=0)

# Symmetric epipolar distances squared
def symmetric_epipolar_distances_sq(x1, x2, F):
    # Assert 
    x1 = np.asarray(x1, dtype = float)
    x2 = np.asarray(x2, dtype = float)
    # Check dims
    check_2xN_pair(x1, x2)
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
    # Check dims
    check_2xN_pair(x1, x2)
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

# Root mean squared error from squared error
def rmse_from_sq(d_sq):
    # Assert
    d_sq = np.asarray(d_sq, dtype=float)
    # In case none
    if d_sq.size == 0:
        raise ValueError("d_sq is empty; cannot compute RMSE")
    return float(np.sqrt(np.mean(d_sq)))

# Sampson RMSE
def sampson_rmse(x1, x2, F): 
    # Sampson distances squared
    d_sq = sampson_distances_sq(x1, x2, F)
    # RMSE
    return rmse_from_sq(d_sq)