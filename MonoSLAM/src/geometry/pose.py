# src/geometry/pose.py
import numpy as np

from core.checks import check_mask_bool_N, check_matrix_3x3, check_2xN_pair, check_vector_3
from geometry.lie import hat
from geometry.rotation import axis_angle_to_rotmat
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


# Apply a left-multiplicative increment to a world-to-camera pose
def apply_left_pose_increment_wc(R, t, delta, eps=1e-12):
    # --- Checks ---
    # Check rotation
    R = check_matrix_3x3(R, name="R", dtype=float, finite=False)
    # Check translation
    t = check_vector_3(t, name="t", dtype=float, finite=False)
    # Check increment
    delta = np.asarray(delta, dtype=float).reshape(6,)

    # Split translational and rotational parts
    rho = delta[:3]
    omega = delta[3:]

    # Rotation magnitude
    theta = float(np.linalg.norm(omega))

    # Small-angle update
    if theta <= float(eps):
        dR = np.eye(3, dtype=float) + hat(omega)
    else:
        axis = omega / theta
        dR = axis_angle_to_rotmat(axis, theta)

    # Left-multiplicative world-to-camera update
    R_new = dR @ R
    t_new = dR @ t + rho

    return np.asarray(R_new, dtype=float), np.asarray(t_new, dtype=float).reshape(3,)


# Recover pose from fundamental matrix estimate
def pose_from_fundamental(F, K1, K2, x1, x2, F_mask=None, enforce_constraints=True):
    # Check dims
    check_2xN_pair(x1, x2)
    N_full = x1.shape[1]
    check_matrix_3x3(K1, name="K1", finite=False)
    check_matrix_3x3(K2, name="K2", finite=False)
    F_mask = check_mask_bool_N(F_mask, N_full, name="F_mask")
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

