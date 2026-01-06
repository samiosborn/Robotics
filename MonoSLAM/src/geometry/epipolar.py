# src/geometry/epipolar.py
import numpy as np
from geometry.checks import check_2xN_pair
from geometry.homogeneous import homogenise

# Algebraic residuals
def algebraic_residuals(x1, x2, F):
    # Assert
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    F = np.asarray(F, dtype=float)
    # Shape error
    check_2xN_pair(x1, x2)
    check_3x3(F)
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
