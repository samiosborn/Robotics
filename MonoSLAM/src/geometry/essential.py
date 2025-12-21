# src/geometry/essential.py
import numpy as np

# Pi/2 rotation about z-axis used in essential matrix decomposition
_W = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1],
])

# Enforce essential constraints
def enforce_essential_constraints(E_raw): 
    # SVD
    U, _, Vt = np.linalg.svd(E_raw)
    # Correct determinant
    if np.linalg.det(U) < 0: U[:, -1] *= -1
    if np.linalg.det(Vt) < 0: Vt[-1, :] *= -1
    # Enforce singular values
    S = np.diag([1, 1, 0])
    # Reconstruct
    E = U @ S @ Vt
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
