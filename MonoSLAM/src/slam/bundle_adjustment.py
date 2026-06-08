# src/slam/bundle_adjustment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.checks import check_int_gt0, check_matrix_3x3, check_positive, check_vector_3
from geometry.lie import hat
from geometry.pose import apply_left_pose_increment_wc
from slam.keyframe_state import get_active_keyframe_kf, get_keyframe_store, get_pose_for_kf, set_pose_for_kf
from slam.landmark_state import build_landmark_id_index, get_landmarks, iter_landmark_observations


@dataclass(frozen=True)
class _LocalBAProblem:
    kf_ids: list[int]
    anchor_kf: int
    pose_var_kfs: list[int]
    landmark_ids: list[int]
    observations: list[tuple[int, int, np.ndarray]]
    pose_col_by_kf: dict[int, int]
    landmark_col_by_id: dict[int, int]
    n_vars: int
    n_residuals: int


# Build a complete stats dictionary for one local BA call
def _new_stats() -> dict[str, Any]:
    return {
        "attempted": True,
        "skipped": False,
        "succeeded": False,
        "skip_reason": None,
        "reason": None,
        "acceptance_reason": None,
        "rejection_reason": None,
        "n_local_keyframes": 0,
        "local_keyframes": [],
        "anchor_kf": None,
        "optimised_keyframes": [],
        "n_local_landmarks": 0,
        "n_observations": 0,
        "n_residuals": 0,
        "n_variables": 0,
        "initial_mean_reproj_error_px": None,
        "initial_median_reproj_error_px": None,
        "final_mean_reproj_error_px": None,
        "final_median_reproj_error_px": None,
        "initial_cost": None,
        "final_cost": None,
        "iterations": 0,
        "accepted_iterations": 0,
        "initial_damping": None,
        "final_damping": None,
    }


# Pack a skipped setup result
def _skip_stats(reason: str, *, stats: dict[str, Any] | None = None) -> dict[str, Any]:
    out = _new_stats() if stats is None else dict(stats)
    out["skipped"] = True
    out["succeeded"] = False
    out["skip_reason"] = str(reason)
    out["reason"] = str(reason)
    return out


# Convert any accepted project pose representation to R, t blocks
def _pose_to_rt(pose, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pose, dict) and ("R" in pose or "t" in pose):
        if "R" not in pose or "t" not in pose:
            raise ValueError(f"{name} must contain both 'R' and 't'")
        R = check_matrix_3x3(pose["R"], name=f"{name}['R']", dtype=float, finite=True)
        t = check_vector_3(pose["t"], name=f"{name}['t']", dtype=float, finite=True)
        return np.asarray(R, dtype=np.float64).copy(), np.asarray(t, dtype=np.float64).reshape(3).copy()

    if isinstance(pose, (tuple, list)) and len(pose) == 2:
        R = check_matrix_3x3(pose[0], name=f"{name}[0]", dtype=float, finite=True)
        t = check_vector_3(pose[1], name=f"{name}[1]", dtype=float, finite=True)
        return np.asarray(R, dtype=np.float64).copy(), np.asarray(t, dtype=np.float64).reshape(3).copy()

    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be a pose tuple, dict, or 4x4 matrix; got {arr.shape}")
    R = check_matrix_3x3(arr[:3, :3], name=f"{name} rotation block", dtype=float, finite=True)
    t = check_vector_3(arr[:3, 3], name=f"{name} translation block", dtype=float, finite=True)
    return np.asarray(R, dtype=np.float64).copy(), np.asarray(t, dtype=np.float64).reshape(3).copy()


# Project one point and return camera-frame coordinates for Jacobians
def _project_one(K: np.ndarray, R: np.ndarray, t: np.ndarray, X_w: np.ndarray, *, eps: float):
    X_c = R @ X_w.reshape(3) + t.reshape(3)
    if not np.isfinite(X_c).all():
        return None, None
    Z = float(X_c[2])
    if Z <= float(eps):
        return None, None

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = fx * (float(X_c[0]) / Z) + cx
    v = fy * (float(X_c[1]) / Z) + cy
    xy = np.asarray([u, v], dtype=np.float64)
    if not np.isfinite(xy).all():
        return None, None

    return xy, X_c


# Gather active and recent connected keyframes
def _select_local_keyframes(seed: dict, max_keyframes: int) -> tuple[list[int], str | None]:
    active_kf = int(get_active_keyframe_kf(seed))
    keyframes = get_keyframe_store(seed)
    keyframe_ids = sorted(int(kf) for kf in keyframes.keys())
    if active_kf not in keyframe_ids:
        return [], "active_keyframe_missing"

    landmark_kfs: dict[int, set[int]] = {}
    for lm_id, obs_kf, _, _ in iter_landmark_observations(seed, context="local BA observations"):
        if int(obs_kf) not in keyframes:
            continue
        if int(lm_id) not in landmark_kfs:
            landmark_kfs[int(lm_id)] = set()
        landmark_kfs[int(lm_id)].add(int(obs_kf))

    active_landmarks = {int(lm_id) for lm_id, kfs in landmark_kfs.items() if int(active_kf) in kfs}
    if len(active_landmarks) == 0:
        return [active_kf], "no_active_landmark_observations"

    connected: list[int] = []
    for kf in sorted((int(kf) for kf in keyframe_ids if int(kf) != int(active_kf)), reverse=True):
        shared = 0
        for lm_id in active_landmarks:
            if int(kf) in landmark_kfs.get(int(lm_id), set()):
                shared += 1
        if shared > 0:
            connected.append(int(kf))

    if len(connected) == 0:
        return [active_kf], "no_connected_keyframes"

    selected = [int(active_kf)] + connected[: max(0, int(max_keyframes) - 1)]
    return sorted(selected), None


# Read selected poses into owned arrays
def _initial_pose_state(seed: dict, kf_ids: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    R_by_kf: dict[int, np.ndarray] = {}
    t_by_kf: dict[int, np.ndarray] = {}
    for kf in kf_ids:
        R, t = _pose_to_rt(get_pose_for_kf(seed, int(kf), context="local BA pose"), name=f"pose {int(kf)}")
        R_by_kf[int(kf)] = R
        t_by_kf[int(kf)] = t
    return R_by_kf, t_by_kf


# Build a local optimisation problem from checked observations
def _build_problem(
    K: np.ndarray,
    seed: dict,
    *,
    max_keyframes: int,
    min_keyframes: int,
    min_landmarks: int,
    min_observations: int,
    eps: float,
) -> tuple[_LocalBAProblem | None, dict[str, Any]]:
    stats = _new_stats()

    kf_ids, select_reason = _select_local_keyframes(seed, int(max_keyframes))
    stats["n_local_keyframes"] = int(len(kf_ids))
    stats["local_keyframes"] = [int(kf) for kf in kf_ids]
    stats["anchor_kf"] = None if len(kf_ids) == 0 else int(kf_ids[0])

    if select_reason is not None and len(kf_ids) < int(min_keyframes):
        return None, _skip_stats(select_reason, stats=stats)
    if len(kf_ids) < int(min_keyframes):
        return None, _skip_stats("too_few_keyframes", stats=stats)

    R_by_kf, t_by_kf = _initial_pose_state(seed, kf_ids)
    local_kfs = set(int(kf) for kf in kf_ids)
    landmark_by_id = build_landmark_id_index(seed, context="local BA landmarks")

    obs_by_landmark: dict[int, list[tuple[int, np.ndarray]]] = {}
    for lm_id, obs_kf, _, xy in iter_landmark_observations(seed, context="local BA observations"):
        if int(obs_kf) not in local_kfs:
            continue
        if int(lm_id) not in landmark_by_id:
            continue

        lm = landmark_by_id[int(lm_id)]
        X_w = np.asarray(lm.get("X_w", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue

        projection, _ = _project_one(
            K,
            R_by_kf[int(obs_kf)],
            t_by_kf[int(obs_kf)],
            X_w,
            eps=float(eps),
        )
        if projection is None:
            continue

        if int(lm_id) not in obs_by_landmark:
            obs_by_landmark[int(lm_id)] = []
        obs_by_landmark[int(lm_id)].append((int(obs_kf), np.asarray(xy, dtype=np.float64).reshape(2).copy()))

    landmark_ids: list[int] = []
    observations: list[tuple[int, int, np.ndarray]] = []
    for lm_id in sorted(obs_by_landmark.keys()):
        local_obs = obs_by_landmark[int(lm_id)]
        distinct_kfs = {int(obs_kf) for obs_kf, _ in local_obs}
        if len(distinct_kfs) < 2:
            continue
        landmark_ids.append(int(lm_id))
        for obs_kf, xy in local_obs:
            observations.append((int(lm_id), int(obs_kf), xy))

    anchor_kf = int(kf_ids[0])
    pose_var_kfs = [int(kf) for kf in kf_ids if int(kf) != int(anchor_kf)]
    pose_col_by_kf: dict[int, int] = {}
    col = 0
    for kf in pose_var_kfs:
        pose_col_by_kf[int(kf)] = int(col)
        col += 6

    landmark_col_by_id: dict[int, int] = {}
    for lm_id in landmark_ids:
        landmark_col_by_id[int(lm_id)] = int(col)
        col += 3

    n_residuals = int(2 * len(observations))
    n_vars = int(col)
    stats.update(
        {
            "anchor_kf": int(anchor_kf),
            "optimised_keyframes": [int(kf) for kf in pose_var_kfs],
            "n_local_landmarks": int(len(landmark_ids)),
            "n_observations": int(len(observations)),
            "n_residuals": int(n_residuals),
            "n_variables": int(n_vars),
        }
    )

    if len(landmark_ids) < int(min_landmarks):
        return None, _skip_stats("too_few_landmarks", stats=stats)
    if len(observations) < int(min_observations):
        return None, _skip_stats("too_few_observations", stats=stats)
    if n_vars <= 0:
        return None, _skip_stats("no_optimisation_variables", stats=stats)
    if n_residuals < n_vars:
        return None, _skip_stats("underdetermined", stats=stats)

    problem = _LocalBAProblem(
        kf_ids=[int(kf) for kf in kf_ids],
        anchor_kf=int(anchor_kf),
        pose_var_kfs=[int(kf) for kf in pose_var_kfs],
        landmark_ids=[int(lm_id) for lm_id in landmark_ids],
        observations=observations,
        pose_col_by_kf=pose_col_by_kf,
        landmark_col_by_id=landmark_col_by_id,
        n_vars=int(n_vars),
        n_residuals=int(n_residuals),
    )
    return problem, stats


# Read initial landmark positions into owned arrays
def _initial_landmark_state(seed: dict, landmark_ids: list[int]) -> dict[int, np.ndarray]:
    landmark_by_id = build_landmark_id_index(seed, context="local BA landmarks")
    X_by_id: dict[int, np.ndarray] = {}
    for lm_id in landmark_ids:
        lm = landmark_by_id[int(lm_id)]
        X = np.asarray(lm.get("X_w", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
        X_by_id[int(lm_id)] = X.copy()
    return X_by_id


# Copy mutable optimisation state
def _copy_state(
    R_by_kf: dict[int, np.ndarray],
    t_by_kf: dict[int, np.ndarray],
    X_by_id: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    R_copy = {int(kf): np.asarray(R, dtype=np.float64).copy() for kf, R in R_by_kf.items()}
    t_copy = {int(kf): np.asarray(t, dtype=np.float64).reshape(3).copy() for kf, t in t_by_kf.items()}
    X_copy = {int(lm_id): np.asarray(X, dtype=np.float64).reshape(3).copy() for lm_id, X in X_by_id.items()}
    return R_copy, t_copy, X_copy


# Apply one LM update to local state
def _apply_delta(
    problem: _LocalBAProblem,
    R_by_kf: dict[int, np.ndarray],
    t_by_kf: dict[int, np.ndarray],
    X_by_id: dict[int, np.ndarray],
    delta: np.ndarray,
    *,
    eps: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    R_new, t_new, X_new = _copy_state(R_by_kf, t_by_kf, X_by_id)

    for kf in problem.pose_var_kfs:
        col = int(problem.pose_col_by_kf[int(kf)])
        d_pose = np.asarray(delta[col : col + 6], dtype=np.float64).reshape(6)
        R_i, t_i = apply_left_pose_increment_wc(R_new[int(kf)], t_new[int(kf)], d_pose, eps=float(eps))
        R_new[int(kf)] = np.asarray(R_i, dtype=np.float64)
        t_new[int(kf)] = np.asarray(t_i, dtype=np.float64).reshape(3)

    for lm_id in problem.landmark_ids:
        col = int(problem.landmark_col_by_id[int(lm_id)])
        X_new[int(lm_id)] = X_new[int(lm_id)] + np.asarray(delta[col : col + 3], dtype=np.float64).reshape(3)

    return R_new, t_new, X_new


# Evaluate residuals and optionally the dense local Jacobian
def _evaluate_problem(
    K: np.ndarray,
    problem: _LocalBAProblem,
    R_by_kf: dict[int, np.ndarray],
    t_by_kf: dict[int, np.ndarray],
    X_by_id: dict[int, np.ndarray],
    *,
    build_jacobian: bool,
    eps: float,
):
    n_obs = int(len(problem.observations))
    residual = np.zeros((2 * n_obs,), dtype=np.float64)
    errors = np.zeros((n_obs,), dtype=np.float64)
    J = None if not bool(build_jacobian) else np.zeros((2 * n_obs, problem.n_vars), dtype=np.float64)

    fx = float(K[0, 0])
    fy = float(K[1, 1])

    for i, (lm_id, kf, xy_obs) in enumerate(problem.observations):
        R = R_by_kf[int(kf)]
        t = t_by_kf[int(kf)]
        X_w = X_by_id[int(lm_id)]
        xy_pred, X_c = _project_one(K, R, t, X_w, eps=float(eps))
        if xy_pred is None or X_c is None:
            return None, None, None, "invalid_geometry"

        row = int(2 * i)
        r_i = np.asarray(xy_obs, dtype=np.float64).reshape(2) - xy_pred.reshape(2)
        residual[row : row + 2] = r_i
        errors[i] = float(np.linalg.norm(r_i))

        if J is None:
            continue

        X = float(X_c[0])
        Y = float(X_c[1])
        Z = float(X_c[2])
        J_proj = np.asarray(
            [
                [fx / Z, 0.0, -fx * X / (Z ** 2)],
                [0.0, fy / Z, -fy * Y / (Z ** 2)],
            ],
            dtype=np.float64,
        )

        if int(kf) in problem.pose_col_by_kf:
            pose_col = int(problem.pose_col_by_kf[int(kf)])
            J_pose = np.hstack([np.eye(3, dtype=np.float64), -hat(X_c)])
            J[row : row + 2, pose_col : pose_col + 6] = -J_proj @ J_pose

        lm_col = int(problem.landmark_col_by_id[int(lm_id)])
        J[row : row + 2, lm_col : lm_col + 3] = -J_proj @ R

    return residual, errors, J, None


# Score residual vector as a sum of squared pixel residuals
def _cost_from_residual(residual: np.ndarray) -> float:
    return float(np.dot(residual, residual))


# Summarise scalar reprojection errors
def _error_summary(errors: np.ndarray) -> tuple[float | None, float | None]:
    if errors is None or int(errors.size) == 0:
        return None, None
    return float(np.mean(errors)), float(np.median(errors))


# Write accepted optimisation state back into the mutable seed
def _write_back_state(
    seed: dict,
    problem: _LocalBAProblem,
    R_by_kf: dict[int, np.ndarray],
    t_by_kf: dict[int, np.ndarray],
    X_by_id: dict[int, np.ndarray],
) -> None:
    for kf in problem.pose_var_kfs:
        pose = (
            np.asarray(R_by_kf[int(kf)], dtype=np.float64).copy(),
            np.asarray(t_by_kf[int(kf)], dtype=np.float64).reshape(3).copy(),
        )
        set_pose_for_kf(seed, int(kf), pose, copy=True, context="local BA pose")

    landmark_by_id = build_landmark_id_index(seed, context="local BA writeback landmarks")
    for lm_id in problem.landmark_ids:
        landmark_by_id[int(lm_id)]["X_w"] = np.asarray(X_by_id[int(lm_id)], dtype=np.float64).reshape(3).copy()


# Run one compact local Levenberg-Marquardt bundle adjustment pass
def run_local_bundle_adjustment(
    K,
    seed: dict,
    *,
    max_keyframes: int = 3,
    min_keyframes: int = 2,
    min_landmarks: int = 6,
    min_observations: int = 12,
    max_iters: int = 5,
    initial_damping: float = 1e-3,
    max_damping: float = 1e9,
    step_tol: float = 1e-7,
    improvement_tol: float = 1e-6,
    eps: float = 1e-12,
) -> dict[str, Any]:
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")

    max_keyframes = check_int_gt0(max_keyframes, name="max_keyframes")
    min_keyframes = check_int_gt0(min_keyframes, name="min_keyframes")
    min_landmarks = check_int_gt0(min_landmarks, name="min_landmarks")
    min_observations = check_int_gt0(min_observations, name="min_observations")
    max_iters = check_int_gt0(max_iters, name="max_iters")
    initial_damping = check_positive(initial_damping, name="initial_damping", eps=0.0)
    max_damping = check_positive(max_damping, name="max_damping", eps=0.0)
    step_tol = check_positive(step_tol, name="step_tol", eps=0.0)
    improvement_tol = check_positive(improvement_tol, name="improvement_tol", eps=0.0)
    eps = check_positive(eps, name="eps", eps=0.0)

    if int(min_keyframes) > int(max_keyframes):
        raise ValueError(f"min_keyframes must be <= max_keyframes; got {min_keyframes} > {max_keyframes}")

    problem, stats = _build_problem(
        K,
        seed,
        max_keyframes=int(max_keyframes),
        min_keyframes=int(min_keyframes),
        min_landmarks=int(min_landmarks),
        min_observations=int(min_observations),
        eps=float(eps),
    )
    stats["initial_damping"] = float(initial_damping)
    stats["final_damping"] = float(initial_damping)
    if problem is None:
        seed["last_local_ba_stats"] = stats
        return stats

    R_by_kf, t_by_kf = _initial_pose_state(seed, problem.kf_ids)
    X_by_id = _initial_landmark_state(seed, problem.landmark_ids)

    residual, errors, _, eval_reason = _evaluate_problem(
        K,
        problem,
        R_by_kf,
        t_by_kf,
        X_by_id,
        build_jacobian=False,
        eps=float(eps),
    )
    if residual is None or errors is None:
        stats = _skip_stats(eval_reason or "initial_evaluation_failed", stats=stats)
        seed["last_local_ba_stats"] = stats
        return stats

    current_cost = _cost_from_residual(residual)
    initial_mean, initial_median = _error_summary(errors)
    stats.update(
        {
            "initial_mean_reproj_error_px": initial_mean,
            "initial_median_reproj_error_px": initial_median,
            "final_mean_reproj_error_px": initial_mean,
            "final_median_reproj_error_px": initial_median,
            "initial_cost": float(current_cost),
            "final_cost": float(current_cost),
        }
    )

    damping = float(initial_damping)
    accepted_steps = 0
    iterations = 0
    rejection_reason = None

    for it in range(int(max_iters)):
        iterations = int(it + 1)
        residual, _, J, eval_reason = _evaluate_problem(
            K,
            problem,
            R_by_kf,
            t_by_kf,
            X_by_id,
            build_jacobian=True,
            eps=float(eps),
        )
        if residual is None or J is None:
            rejection_reason = eval_reason or "linearisation_failed"
            break

        H = J.T @ J
        g = J.T @ residual
        diag = np.maximum(np.diag(H), 1e-12)

        accepted_this_iter = False
        for _ in range(6):
            if damping > float(max_damping):
                rejection_reason = "damping_limit_reached"
                break

            H_lm = H + float(damping) * np.diag(diag)
            try:
                delta = -np.linalg.solve(H_lm, g)
            except Exception:
                damping *= 10.0
                rejection_reason = "normal_equations_failed"
                continue

            if not np.isfinite(delta).all():
                damping *= 10.0
                rejection_reason = "nonfinite_step"
                continue

            step_norm = float(np.linalg.norm(delta))
            R_prop, t_prop, X_prop = _apply_delta(
                problem,
                R_by_kf,
                t_by_kf,
                X_by_id,
                delta,
                eps=float(eps),
            )

            residual_prop, errors_prop, _, prop_reason = _evaluate_problem(
                K,
                problem,
                R_prop,
                t_prop,
                X_prop,
                build_jacobian=False,
                eps=float(eps),
            )
            if residual_prop is None or errors_prop is None:
                damping *= 10.0
                rejection_reason = prop_reason or "invalid_proposed_geometry"
                continue

            proposed_cost = _cost_from_residual(residual_prop)
            if not np.isfinite(proposed_cost) or proposed_cost >= current_cost:
                damping *= 10.0
                rejection_reason = "worse_reprojection_error"
                continue

            R_by_kf, t_by_kf, X_by_id = R_prop, t_prop, X_prop
            improvement = float(current_cost - proposed_cost)
            current_cost = float(proposed_cost)
            damping = max(float(initial_damping) * 1e-6, damping / 3.0)
            accepted_steps += 1
            accepted_this_iter = True
            rejection_reason = None

            final_mean, final_median = _error_summary(errors_prop)
            stats.update(
                {
                    "final_mean_reproj_error_px": final_mean,
                    "final_median_reproj_error_px": final_median,
                    "final_cost": float(current_cost),
                    "iterations": int(iterations),
                    "accepted_iterations": int(accepted_steps),
                    "final_damping": float(damping),
                }
            )

            if step_norm <= float(step_tol):
                stats["acceptance_reason"] = "step_tol_reached"
                break
            if improvement <= float(improvement_tol):
                stats["acceptance_reason"] = "improvement_tol_reached"
                break
            stats["acceptance_reason"] = "accepted"
            break

        if not bool(accepted_this_iter):
            break
        if stats.get("acceptance_reason") in {"step_tol_reached", "improvement_tol_reached"}:
            break

    stats["iterations"] = int(iterations)
    stats["accepted_iterations"] = int(accepted_steps)
    stats["final_damping"] = float(damping)

    if accepted_steps <= 0:
        stats["succeeded"] = False
        stats["rejection_reason"] = rejection_reason or "no_accepted_step"
        stats["reason"] = stats["rejection_reason"]
        seed["last_local_ba_stats"] = stats
        return stats

    _write_back_state(seed, problem, R_by_kf, t_by_kf, X_by_id)
    stats["succeeded"] = True
    stats["skipped"] = False
    stats["reason"] = stats.get("acceptance_reason", None) or "accepted"
    seed["last_local_ba_stats"] = stats
    return stats


# Build caller-side stats when BA is deliberately not invoked
def local_bundle_adjustment_not_run_stats(reason: str) -> dict[str, Any]:
    stats = _new_stats()
    stats["attempted"] = False
    return _skip_stats(str(reason), stats=stats)
