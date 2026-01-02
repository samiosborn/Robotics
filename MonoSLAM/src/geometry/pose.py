# src/geometry/pose.py 
import numpy as np
from geometry.triangulation import disambiguate_pose_cheirality
from geometry.essential import essential_from_fundamental, enforce_essential_constraints, decompose_essential

# Angle (radians) between translations
def angle_between_translations(t_1, t_2): 
    # Assert
    t_1 = np.asarray(t_1, dtype=float).reshape(-1)
    t_2 = np.asarray(t_2, dtype=float).reshape(-1)
    # Check dimensions
    if t_1.shape[0] != 3 or t_2.shape[0] != 3:
        raise ValueError("Translations must be length-3 vectors")
    # Normal
    n_1 = np.linalg.norm(t_1)
    n_2 = np.linalg.norm(t_2)
    # Guard
    if n_1 < 1e-12 or n_2 < 1e-12:
        raise ValueError("Translation vectors must be non-zero")
    # Normalise
    t_1_n = t_1 / n_1
    t_2_n = t_2 / n_2
    # Cosine(theta)
    cos_theta = np.clip(np.dot(t_1_n, t_2_n), -1.0, 1.0)
    # Angle
    return np.arccos(cos_theta)

# Recover pose from fundamental matrix estimate
def recover_pose_from_fundamental(F, K1, K2, x1, x2, enforce_constraints=True):
    # Assert
    F  = np.asarray(F, dtype=float)
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    # Check dims
    if x1.ndim != 2 or x2.ndim != 2 or x1.shape != x2.shape or x1.shape[0] != 2:
        raise ValueError(f"Points must be (2, N); got x1 {x1.shape}, x2 {x2.shape}")
    # Essential from fundamental
    E = essential_from_fundamental(F, K1, K2)
    # Enforce essential constraints (rank 2)
    if enforce_constraints: 
        E = enforce_essential_constraints(E)
    # Decompose essential matrix 
    candidate_poses = decompose_essential(E)
    # Disambiguate pose via cheirality
    R, t = disambiguate_pose_cheirality(candidate_poses, K1, K2, x1, x2)
    return R, t, E
