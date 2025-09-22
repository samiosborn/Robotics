# kinematics/inverse.py
import numpy as np
from typing import List, Optional, Tuple
import config
from .forward import fk_pose
from .jacobian import geometric_jacobian
from .pose import rpy_to_R, so3_log

# Degrees of freedom
DOF = 6

# Tolerances
POS_TOL_M = config.IK_POS_TOL_M
ANG_TOL_RAD = config.IK_ANG_TOL_RAD

# DLS parameters
ALPHA = 1.0 
DLS_LAMBDA = config.IK_DAMPING_LAMBDA
ROT_WEIGHT = config.IK_ROT_WEIGHT
MAX_ITERS = config.IK_MAX_ITERS

# Joint limits (helper)
def _limits_rad() -> List[Tuple[float, float]]:
    # Vector of joint limits
    out = []
    for j in config.JOINT_ORDER:
        lo_deg, hi_deg = config.JOINT_LIMITS_DEG[j]
        out.append((np.deg2rad(lo_deg), np.deg2rad(hi_deg)))
    return out

# Clamp a joint vector (rad) to limits
def _clamp_to_limits(q: np.ndarray) -> np.ndarray:
    lims = _limits_rad()
    # Clamped joint angles
    qc = np.array(q, dtype=float)
    for i, (lo, hi) in enumerate(lims):
        qc[i] = np.clip(qc[i], lo, hi)
    return qc

# Build pose errors and weighted Jacobian
def _pose_errors_and_weighted_J(q: np.ndarray,
                                target_pos: np.ndarray,
                                target_rpy: Optional[np.ndarray],
                                rot_weight: float) -> tuple[
                                    float,
                                    float,
                                    np.ndarray,
                                    np.ndarray
                                ]:
    # Current pose (position and roll-pitch-yaw)
    p_cur, rpy_cur = fk_pose(q)

    # Raw position error vector and its norm
    e_p = target_pos - p_cur
    pos_err = float(np.linalg.norm(e_p))

    # Base-frame Jacobian (unweighted)
    J_base = geometric_jacobian(q)

    # Position-only tracking
    if target_rpy is None or rot_weight <= 0.0:
        # Error vector (no angular RPY error term)
        e = np.hstack([e_p, np.zeros(3)])
        # No angular error
        return pos_err, 0.0, e, J_base

    # Convert current and target RPY to rotation matrices
    R_cur = rpy_to_R(rpy_cur[0], rpy_cur[1], rpy_cur[2])
    R_tgt = rpy_to_R(target_rpy[0], target_rpy[1], target_rpy[2])

    # Relative rotation error matrix
    R_err = R_cur.T @ R_tgt

    # Orientation error as rotation vector (axis * angle)
    e_r = so3_log(R_err)
    ang_err = float(np.linalg.norm(e_r))

    # Weight matrix for weighted error and weighted Jacobian 
    W = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])

    # Error vector
    e = np.hstack([e_p, e_r])
    # Weighted error vector
    e_w = W @ e
    # Weighted Jacobian
    J_w = W @ J_base

    return pos_err, ang_err, e_w, J_w


# 1 DLS update step: dq = (J^T J + lambda^2 I)^-1 J^T e
def _dls_step(J: np.ndarray, e: np.ndarray, lam: float) -> np.ndarray:
    # Regularised Hessian approximation: J^T J + lambda^2 I
    H = J.T @ J + (lam**2) * np.eye(J.shape[1])
    # Solve H * dq = J^T * e - without forming the inverse explicitly
    return np.linalg.solve(H, (J.T @ e))

# Simple backtracking on position error only
def _backtrack_if_worse(q: np.ndarray,
                        dq: np.ndarray,
                        target_pos: np.ndarray,
                        max_halves: int) -> np.ndarray:
    # Current position error
    p_cur, _ = fk_pose(q)
    err0 = np.linalg.norm(target_pos - p_cur)

    # Try full step first
    step = 1.0
    q_trial = _clamp_to_limits(q + dq)
    p_trial, _ = fk_pose(q_trial)
    err = np.linalg.norm(target_pos - p_trial)

    # If worse, shrink the step by 1/2 up to max_halves times
    k = 0
    while err > err0 and k < max_halves:
        step *= 0.5
        q_trial = _clamp_to_limits(q + step * dq)
        p_trial, _ = fk_pose(q_trial)
        err = np.linalg.norm(target_pos - p_trial)
        k += 1

    return q_trial

# IK via DLS
def inverse_via_DLS(target_pose: np.ndarray,
                    theta_initial: Optional[List[float]] = None,
                    pos_tol: float = POS_TOL_M,
                    ang_tol: float = ANG_TOL_RAD,
                    rot_weight: float = ROT_WEIGHT,
                    lambda_damp: float = DLS_LAMBDA,
                    max_iterations: int = MAX_ITERS,
                    alpha: float = ALPHA) -> np.ndarray:
    # Target position
    target_pos = np.array(target_pose[:3], dtype=float)
    # Target RPY (optional)
    target_rpy = np.array(target_pose[3:], dtype=float) if len(target_pose) == 6 else None

    # Seed q
    if theta_initial is None:
        q = np.zeros(DOF, dtype=float)
    else:
        q = np.array(theta_initial, dtype=float)
    # Ensure initial guess is within limits
    q = _clamp_to_limits(q)

    # Iterate until converged or max iterations
    for _ in range(max_iterations):
        # Get raw errors and weighted (e, J)
        pos_err, ang_err, e_w, J_w = _pose_errors_and_weighted_J(q, target_pos, target_rpy, rot_weight)

        # Check convergence on raw errors
        if pos_err <= pos_tol and ang_err <= ang_tol:
            return _clamp_to_limits(q)

        # DLS update step with damping and step scale alpha
        dq = _dls_step(J_w, e_w, lambda_damp) * alpha

        # Backtracking function if step makes position error worse
        q = _backtrack_if_worse(q, dq, target_pos, max_halves = 3)

    # Did not converge
    raise RuntimeError(f"IK did not converge within {max_iterations} iterations.")

# Position-only IK via DLS
def inverse_via_DLS_position_only(target_pos: np.ndarray,
                                  theta_initial: Optional[List[float]] = None,
                                  pos_tol: float = POS_TOL_M,
                                  lambda_damp: float = DLS_LAMBDA,
                                  max_iterations: int = MAX_ITERS,
                                  alpha: float = ALPHA) -> np.ndarray:
    pose = np.hstack([np.array(target_pos, dtype=float), np.zeros(3)])
    return inverse_via_DLS(pose,
                           theta_initial=theta_initial,
                           pos_tol=pos_tol,
                           ang_tol=1e9,
                           rot_weight=0.0,
                           lambda_damp=lambda_damp,
                           max_iterations=max_iterations,
                           alpha=alpha)
