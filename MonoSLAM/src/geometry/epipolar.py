# src/geometry/epipolar.py
import numpy as np
from geometry.lie import hat

# Epipolar constraint
def epipolar_constraint(x1, x2, F): 
    # x2^T F x1 = 0
    return float(x2.T @ F @ x1)

# Normalised epipolar constraint
def normalised_epipolar_constraint(x1_hat, x2_hat, E): 
    # x2_hat^T E x1_hat = 0
    return float(x2_hat.T @ E @ x1_hat)

# Essential matrix from pose
def essential_from_pose(R, t): 
    # Enforce shape
    t = np.asarray(t).reshape(3,)
    # E = t x R
    return hat (t) @ R

# Essential from fundamental matrix
def essential_from_fundamental(F, K1, K2): 
    # E = K2^T F K1
    return K2.T @ F @ K1

# Fundamental from essential matrix
def fundamental_from_essential(E, K1, K2):
    # F = K2^-T E K1^-1
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

# Fundamental matrix from pose
def fundamental_from_pose(R, t, K1, K2):
    # Essential matrix
    E = essential_from_pose(R, t)
    # Fundamental matrix
    return fundamental_from_essential(E, K1, K2)
