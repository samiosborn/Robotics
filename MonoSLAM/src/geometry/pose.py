# src/geometry/pose.py 
import numpy as np
from geometry.checks import check_2xN_pair, check_3x3, check_bool_N
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
def pose_from_fundamental(F, K1, K2, x1, x2, F_mask=None, enforce_constraints=True):
    # Check dims
    check_2xN_pair(x1, x2)
    N_full = x1.shape[1]
    check_3x3(K1)
    check_3x3(K2)
    check_bool_N(F_mask, N_full)
    # Apply mask
    if F_mask is not None: 
        x1 = x1[:, F_mask]
        x2 = x2[:, F_mask]
    # Essential from fundamental
    E = essential_from_fundamental(F, K1, K2)
    # Enforce essential constraints (rank 2)
    if enforce_constraints: 
        E = enforce_essential_constraints(E)
    # Decompose essential matrix 
    candidate_poses = decompose_essential(E)
    # Disambiguate pose via cheirality
    R, t, best_cheir_mask = disambiguate_pose_cheirality(candidate_poses, K1, K2, x1, x2)
    # Cheirality ratio
    cheir_ratio = float(best_cheir_mask.mean())
    # If mask provided
    if F_mask is not None: 
        # Full mask 
        full_best_mask = np.zeros(N_full, dtype=bool)
        full_best_mask[F_mask] = best_cheir_mask
        return R, t, E, cheir_ratio, full_best_mask
    else: 
        return R, t, E, cheir_ratio, best_cheir_mask
