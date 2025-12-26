# src/geometry/epipolar.py
import numpy as np
from geometry.homogeneous import homogenise

# Algebraic residual
def algebraic_residual(x1, x2, F):
    # Assert
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    # Shape error
    if x1.shape[0] != 2:
        raise ValueError(f"Shape should be (2, N)")
    # Homogenise
    x1h = homogenise(x1)
    x2h = homogenise(x2)
    # Epipolar lines in image 2
    Fx1 = F @ x1h
    # Residual r_i = x2_i^T F x1_i
    return np.sum(x2h * Fx1, axis=0)

# Algebratic residual RMSE
def algebraic_residual_RMSE(x1, x2, F):
    # Algebraic residuals
    r = algebraic_residual(x1, x2, F)
    # Root mean squared residual
    return np.sqrt(np.mean(r ** 2))
