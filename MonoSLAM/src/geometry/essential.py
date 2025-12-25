# src/geometry/essential.py
import numpy as np
from geometry.lie import hat

# Pi/2 rotation about z-axis used in essential matrix decomposition
_W = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1],
])

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

# Enforce essential constraints
def enforce_essential_constraints(E_raw): 
    # SVD
    U, S, Vt = np.linalg.svd(E_raw)
    # Correct determinant
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1, :] *= -1
    # Enforce singular values (mean of first two singular values)
    s = 0.5 * (S[0] + S[1])
    S_new = np.diag([s, s, 0])
    # Reconstruct
    E = U @ S_new @ Vt
    return E

# Decompose essential matrix
def decompose_essential(E): 
    # SVD
    U, _, Vt = np.linalg.svd(E)
    # Correct determinant
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1, :] *= -1
    # Candidate rotations
    R1 = U @ _W @ Vt
    R2 = U @ _W.T @ Vt
    # Translation
    t = U[:, 2]
    # Return 4 candidate solutions
    return [
        (R1, t), 
        (R1, -t), 
        (R2, t),
        (R2, -t)
    ]
