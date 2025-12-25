# src/geometry/epipolar.py
import numpy as np
from geometry.homogeneous import homogenise

# Epipolar constraint
def epipolar_constraint(x1, x2, F): 
    # x2^T F x1 = 0
    return float(x2.T @ F @ x1)

# Normalised epipolar constraint
def normalised_epipolar_constraint(x1_hat, x2_hat, E): 
    # x2_hat^T E x1_hat = 0
    return float(x2_hat.T @ E @ x1_hat)

# Mean epipolar residual
def mean_epipolar_residual(x1, x2, F):
    # Homogenise
    x1h = homogenise(x1)
    x2h = homogenise(x2)

    # Epipolar lines in image 2
    Fx1 = F @ x1h

    # Residual r_i = x2_i^T F x1_i
    r = np.sum(x2h * Fx1, axis=0)

    # Root mean squared residual
    return np.sqrt(np.mean(r ** 2))
