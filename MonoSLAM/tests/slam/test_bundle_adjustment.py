from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from geometry.camera import camera_centre
from geometry.lie import hat
from geometry.rotation import angle_between_rotmats, axis_angle_to_rotmat
from slam.bundle_adjustment import (
    _apply_delta,
    _build_problem,
    _evaluate_problem,
    _initial_landmark_state,
    _initial_pose_state,
    _project_one,
    run_local_bundle_adjustment,
)
from slam.invariants import audit_seed_invariants
from slam.keyframe_state import get_pose_for_kf


# Build a feature stub with keypoints
def _features(kps_xy):
    return SimpleNamespace(kps_xy=np.asarray(kps_xy, dtype=np.float64))


# Build a simple world-to-camera pose from a camera centre
def _pose_from_centre(C):
    R = np.eye(3, dtype=np.float64)
    t = -np.asarray(C, dtype=np.float64).reshape(3)
    return R, t


# Project one point into a calibrated camera
def _project(K, R, t, X_w):
    X_c = R @ np.asarray(X_w, dtype=np.float64).reshape(3) + np.asarray(t, dtype=np.float64).reshape(3)
    x_h = K @ X_c
    return np.asarray(x_h[:2] / x_h[2], dtype=np.float64)


# Ground-truth 3-camera / 8-landmark synthetic scene shared by all BA tests
def _scene_ground_truth():
    K = np.asarray(
        [
            [120.0, 0.0, 80.0],
            [0.0, 120.0, 60.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    centres = [
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.25, 0.0, 0.02], dtype=np.float64),
        np.asarray([0.50, 0.04, 0.01], dtype=np.float64),
    ]
    poses_true = [_pose_from_centre(C) for C in centres]
    points_true = np.asarray(
        [
            [-0.8, -0.3, 4.2],
            [-0.4, 0.2, 4.6],
            [0.0, -0.1, 5.0],
            [0.4, 0.3, 4.8],
            [0.8, -0.2, 5.4],
            [-0.2, 0.5, 5.6],
            [0.6, 0.4, 4.4],
            [1.0, 0.1, 5.2],
        ],
        dtype=np.float64,
    )
    return K, poses_true, points_true


# Build a local BA seed from ground truth plus optional pose/point state noise
# Observations are always projected from poses_true/points_true (perfect pixels);
# only the initial R/t/X_w state stored in the seed is perturbed
def _build_ba_seed(
    K,
    poses_true,
    points_true,
    *,
    pose_noise_by_kf=None,
    rot_noise_by_kf=None,
    point_noise=None,
    active_kf=None,
):
    n_points = int(points_true.shape[0])
    pose_noise_by_kf = {} if pose_noise_by_kf is None else pose_noise_by_kf
    rot_noise_by_kf = {} if rot_noise_by_kf is None else rot_noise_by_kf
    point_noise = np.zeros((n_points, 3), dtype=np.float64) if point_noise is None else point_noise
    active_kf = int(len(poses_true) - 1) if active_kf is None else int(active_kf)

    keyframes = {}
    poses = {}
    lookup = np.arange(n_points, dtype=np.int64)

    for kf, pose_true in enumerate(poses_true):
        R_true, t_true = pose_true
        xy = np.vstack([_project(K, R_true, t_true, X) for X in points_true])

        t_noise = pose_noise_by_kf.get(int(kf), np.zeros(3, dtype=np.float64))
        rot_noise = rot_noise_by_kf.get(int(kf), None)
        if rot_noise is None:
            R_init = R_true.copy()
        else:
            axis = np.asarray(rot_noise, dtype=np.float64)
            angle = float(np.linalg.norm(axis))
            dR = np.eye(3, dtype=np.float64) if angle < 1e-12 else axis_angle_to_rotmat(axis / angle, angle)
            R_init = dR @ R_true

        pose = (R_init, t_true + t_noise)
        poses[int(kf)] = pose
        keyframes[int(kf)] = {
            "kf": int(kf),
            "pose": pose,
            "feats": _features(xy),
            "landmark_id_by_feat": lookup.copy(),
        }

    landmarks = []
    for lm_id, X_true in enumerate(points_true):
        obs = []
        for kf, pose_true in enumerate(poses_true):
            R_true, t_true = pose_true
            obs.append(
                {
                    "kf": int(kf),
                    "feat": int(lm_id),
                    "xy": _project(K, R_true, t_true, X_true),
                }
            )
        landmarks.append(
            {
                "id": int(lm_id),
                "X_w": np.asarray(X_true + point_noise[int(lm_id)], dtype=np.float64),
                "obs": obs,
            }
        )

    return {"poses": poses, "keyframes": keyframes, "active_keyframe_kf": int(active_kf), "landmarks": landmarks}


# Build a small local BA fixture with three connected keyframes
def _local_ba_seed():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise = {
        1: np.asarray([0.025, -0.015, 0.010], dtype=np.float64),
        2: np.asarray([-0.020, 0.018, -0.012], dtype=np.float64),
    }
    point_noise = np.asarray(
        [
            [0.020, -0.010, 0.030],
            [-0.015, 0.025, -0.020],
            [0.010, 0.015, 0.025],
            [-0.020, -0.010, 0.015],
            [0.025, 0.005, -0.020],
            [-0.010, 0.020, 0.020],
            [0.015, -0.020, -0.015],
            [-0.025, 0.010, 0.010],
        ],
        dtype=np.float64,
    )
    seed = _build_ba_seed(K, poses_true, points_true, pose_noise_by_kf=pose_noise, point_noise=point_noise, active_kf=2)
    return K, seed


# Shared deterministic noise used by every noisy-init BA test (kept identical
# across Phase 3/4/5 for direct before/after comparability)
def _noisy_test_perturbations():
    pose_noise = {
        1: np.asarray([0.06, -0.04, 0.03], dtype=np.float64),
        2: np.asarray([-0.05, 0.05, -0.04], dtype=np.float64),
    }
    rot_noise = {
        1: np.asarray([0.02, -0.01, 0.03], dtype=np.float64),
        2: np.asarray([-0.015, 0.025, -0.02], dtype=np.float64),
    }
    point_noise = np.asarray(
        [
            [0.030, -0.020, 0.040],
            [-0.025, 0.035, -0.030],
            [0.020, 0.025, 0.035],
            [-0.030, -0.020, 0.025],
            [0.035, 0.010, -0.030],
            [-0.020, 0.030, 0.030],
            [0.025, -0.030, -0.025],
            [-0.035, 0.020, 0.020],
        ],
        dtype=np.float64,
    )
    return pose_noise, rot_noise, point_noise


# Best-fit scalar s minimising ||s(C-C_a) - (C_true-C_a)||^2 over cameras and
# landmarks - the same scale-alignment procedure used throughout Phase 3
def _best_fit_scale(C_by_kf, X_by_id, C_a, C_true_by_kf, X_true_by_id):
    num = 0.0
    den = 0.0
    for kf in (1, 2):
        d = C_by_kf[kf] - C_a
        d_true = C_true_by_kf[kf] - C_a
        num += float(np.dot(d, d_true))
        den += float(np.dot(d, d))
    for lm_id, X in X_by_id.items():
        d = X - C_a
        d_true = X_true_by_id[lm_id] - C_a
        num += float(np.dot(d, d_true))
        den += float(np.dot(d, d))
    return num / den


# Build the pre-fix Jacobian with six variables for every optimised pose
def _unconstrained_jacobian(K, problem, R_by_kf, t_by_kf, X_by_id):
    pose_col_by_kf = {}
    col = 0
    for kf in problem.pose_var_kfs:
        pose_col_by_kf[kf] = col
        col += 6

    landmark_col_by_id = {}
    for lm_id in problem.landmark_ids:
        landmark_col_by_id[lm_id] = col
        col += 3

    J = np.zeros((problem.n_residuals, col), dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])

    for i, (lm_id, kf, _) in enumerate(problem.observations):
        R = R_by_kf[kf]
        _, X_c = _project_one(K, R, t_by_kf[kf], X_by_id[lm_id], eps=1e-12)
        X, Y, Z = (float(value) for value in X_c)
        J_proj = np.asarray(
            [
                [fx / Z, 0.0, -fx * X / (Z ** 2)],
                [0.0, fy / Z, -fy * Y / (Z ** 2)],
            ],
            dtype=np.float64,
        )
        row = 2 * i
        if kf in pose_col_by_kf:
            pose_col = pose_col_by_kf[kf]
            J_pose = np.hstack([np.eye(3, dtype=np.float64), -hat(X_c)])
            J[row : row + 2, pose_col : pose_col + 6] = -J_proj @ J_pose
        landmark_col = landmark_col_by_id[lm_id]
        J[row : row + 2, landmark_col : landmark_col + 3] = -J_proj @ R

    return J, pose_col_by_kf, landmark_col_by_id


# Select the unique member of the true scale orbit satisfying the fixed gauge
def _gauge_selected_truth(problem, C_true_by_kf, X_true_by_id):
    C_a = C_true_by_kf[problem.anchor_kf]
    true_projection = float(problem.gauge_n @ (C_true_by_kf[problem.gauge_kf] - C_a))
    scale = float(problem.gauge_b0 / true_projection)
    C_target = {kf: C_a + scale * (C - C_a) for kf, C in C_true_by_kf.items()}
    X_target = {lm_id: C_a + scale * (X - C_a) for lm_id, X in X_true_by_id.items()}
    return scale, C_target, X_target


# Measure camera-centre and landmark errors against a specified state
def _state_errors(C_by_kf, X_by_id, C_target_by_kf, X_target_by_id):
    pose_errors = {
        kf: float(np.linalg.norm(C_by_kf[kf] - C_target_by_kf[kf]))
        for kf in C_by_kf
        if kf in C_target_by_kf and kf != 0
    }
    landmark_errors = np.asarray(
        [float(np.linalg.norm(X - X_target_by_id[lm_id])) for lm_id, X in X_by_id.items()],
        dtype=np.float64,
    )
    return pose_errors, landmark_errors


# Local BA reduces reprojection error and preserves canonical state ownership
def test_local_bundle_adjustment_refines_small_keyframe_window():
    K, seed = _local_ba_seed()
    anchor_before = get_pose_for_kf(seed, 0)

    stats = run_local_bundle_adjustment(K, seed, max_iters=8)

    assert stats["attempted"] is True
    assert stats["skipped"] is False
    assert stats["succeeded"] is True
    assert stats["local_keyframes"] == [0, 1, 2]
    assert stats["anchor_kf"] == 0
    assert stats["optimised_keyframes"] == [1, 2]
    assert stats["final_mean_reproj_error_px"] < stats["initial_mean_reproj_error_px"]
    assert stats["final_median_reproj_error_px"] < stats["initial_median_reproj_error_px"]
    np.testing.assert_allclose(get_pose_for_kf(seed, 0)[0], anchor_before[0])
    np.testing.assert_allclose(get_pose_for_kf(seed, 0)[1], anchor_before[1])
    assert audit_seed_invariants(seed)["errors"] == []


# Test G: exact ground-truth initialisation must remain a fixed point
def test_ba_exact_init_is_a_fixed_point():
    K, poses_true, points_true = _scene_ground_truth()
    seed = _build_ba_seed(K, poses_true, points_true, active_kf=2)

    problem, _ = _build_problem(
        K,
        seed,
        max_keyframes=3,
        min_keyframes=2,
        min_landmarks=6,
        min_observations=12,
        eps=1e-12,
    )

    R0 = {kf: get_pose_for_kf(seed, kf)[0].copy() for kf in range(3)}
    t0 = {kf: get_pose_for_kf(seed, kf)[1].copy() for kf in range(3)}
    X0 = {lm["id"]: lm["X_w"].copy() for lm in seed["landmarks"]}
    C_anchor = camera_centre(R0[problem.anchor_kf], t0[problem.anchor_kf])
    C_gauge = camera_centre(R0[problem.gauge_kf], t0[problem.gauge_kf])
    gauge_before = float(problem.gauge_n @ (C_gauge - C_anchor))

    stats = run_local_bundle_adjustment(K, seed, max_iters=10)

    assert stats["initial_cost"] < 1e-18
    assert stats["final_cost"] < 1e-18

    max_pose_translation_change = 0.0
    max_pose_rotation_change = 0.0
    for kf in (1, 2):
        R1, t1 = get_pose_for_kf(seed, kf)
        max_pose_translation_change = max(max_pose_translation_change, float(np.linalg.norm(t1 - t0[kf])))
        max_pose_rotation_change = max(max_pose_rotation_change, float(angle_between_rotmats(R1, R0[kf])))
    max_landmark_change = max(float(np.linalg.norm(lm["X_w"] - X0[lm["id"]])) for lm in seed["landmarks"])

    assert max_pose_translation_change < 1e-9
    assert max_pose_rotation_change < 1e-9
    assert max_landmark_change < 1e-9

    R_anchor, t_anchor = get_pose_for_kf(seed, 0)
    np.testing.assert_allclose(R_anchor, R0[0])
    np.testing.assert_allclose(t_anchor, t0[0])

    R_g, t_g = get_pose_for_kf(seed, problem.gauge_kf)
    gauge_after = float(problem.gauge_n @ (camera_centre(R_g, t_g) - C_anchor))
    initial_rmse = float(np.sqrt(stats["initial_cost"] / len(problem.observations)))
    final_rmse = float(np.sqrt(stats["final_cost"] / len(problem.observations)))

    print(f"\n[Test G] reprojection RMSE initial/final: {initial_rmse:.3e} / {final_rmse:.3e}")
    print(f"[Test G] max translation/rotation/landmark movement: "
          f"{max_pose_translation_change:.3e} / {max_pose_rotation_change:.3e} / {max_landmark_change:.3e}")
    print(f"[Test G] gauge scalar before/after: {gauge_before:.16e} / {gauge_after:.16e}")

    np.testing.assert_allclose(gauge_before, problem.gauge_b0, atol=1e-12)
    np.testing.assert_allclose(gauge_after, problem.gauge_b0, atol=1e-12)


# Test E: the gauge fix must remove exactly one scalar parameter, not zero and not
# three - the gauge camera keeps 2 translational DOF plus its full 3 rotational DOF
def test_ba_gauge_reduces_variable_count_by_exactly_one():
    K, poses_true, points_true = _scene_ground_truth()
    seed = _build_ba_seed(K, poses_true, points_true, active_kf=2)

    problem, _ = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)
    assert problem is not None

    naive_uniform_n_vars = 6 * len(problem.pose_var_kfs) + 3 * len(problem.landmark_ids)
    assert problem.n_vars == naive_uniform_n_vars - 1
    assert problem.n_vars == 35
    assert naive_uniform_n_vars == 36

    assert problem.gauge_kf == problem.pose_var_kfs[0]
    assert problem.pose_dof_by_kf[problem.gauge_kf] == 5
    for kf in problem.pose_var_kfs:
        if kf != problem.gauge_kf:
            assert problem.pose_dof_by_kf[kf] == 6
    assert problem.gauge_U.shape == (3, 2)
    np.testing.assert_allclose(problem.gauge_U.T @ problem.gauge_U, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(problem.gauge_U.T @ problem.gauge_n, np.zeros(2), atol=1e-12)


# Test F: compare the undamped Jacobian before and after removing one variable
def test_ba_full_rank_without_damping():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise, rot_noise, point_noise = _noisy_test_perturbations()
    seed_gt = _build_ba_seed(K, poses_true, points_true, active_kf=2)
    seed_noisy = _build_ba_seed(
        K, poses_true, points_true, pose_noise_by_kf=pose_noise, rot_noise_by_kf=rot_noise, point_noise=point_noise, active_kf=2
    )

    for label, seed in (("ground_truth", seed_gt), ("noisy", seed_noisy)):
        problem, _ = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)
        R_by_kf, t_by_kf = _initial_pose_state(seed, problem.kf_ids)
        X_by_id = _initial_landmark_state(seed, problem.landmark_ids)
        _, _, J, reason = _evaluate_problem(K, problem, R_by_kf, t_by_kf, X_by_id, build_jacobian=True, eps=1e-12)
        assert reason is None

        J_old, _, _ = _unconstrained_jacobian(K, problem, R_by_kf, t_by_kf, X_by_id)
        S_old = np.linalg.svd(J_old, compute_uv=False)
        S = np.linalg.svd(J, compute_uv=False)
        old_rank = int(np.sum(S_old > 1e-8 * S_old[0]))
        rank = int(np.sum(S > 1e-8 * S[0]))

        print(f"\n[Test F/{label}] before: n_vars={J_old.shape[1]} rank={old_rank} "
              f"smallest={S_old[-1]:.6e} next={S_old[-2]:.6e} "
              f"condition={S_old[0] / S_old[-1]:.3e} "
              f"effective_condition={S_old[0] / S_old[-2]:.3e}")
        print(f"[Test F/{label}] after:  n_vars={problem.n_vars} rank={rank} "
              f"smallest={S[-1]:.6e} next={S[-2]:.6e} condition={S[0] / S[-1]:.3e}")

        assert J_old.shape[1] == problem.n_vars + 1
        assert old_rank == J_old.shape[1] - 1
        assert rank == problem.n_vars
        assert S[-1] > 1e-4


# Test H: noisy initialisation must recover the fixed-slice truth directly
def test_ba_noisy_init_recovers_fixed_gauge_representative():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise, rot_noise, point_noise = _noisy_test_perturbations()
    seed = _build_ba_seed(
        K,
        poses_true,
        points_true,
        pose_noise_by_kf=pose_noise,
        rot_noise_by_kf=rot_noise,
        point_noise=point_noise,
        active_kf=2,
    )

    R_true_by_kf = {kf: poses_true[kf][0] for kf in range(3)}
    t_true_by_kf = {kf: poses_true[kf][1] for kf in range(3)}
    C_true_by_kf = {kf: camera_centre(R_true_by_kf[kf], t_true_by_kf[kf]) for kf in range(3)}
    X_true_by_id = {i: points_true[i].copy() for i in range(points_true.shape[0])}

    R0 = {kf: get_pose_for_kf(seed, kf)[0].copy() for kf in range(3)}
    t0 = {kf: get_pose_for_kf(seed, kf)[1].copy() for kf in range(3)}
    C0 = {kf: camera_centre(R0[kf], t0[kf]) for kf in range(3)}
    X0 = {lm["id"]: lm["X_w"].copy() for lm in seed["landmarks"]}

    problem = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)[0]
    gauge_scale, C_target, X_target = _gauge_selected_truth(problem, C_true_by_kf, X_true_by_id)
    gauge_b0_initial = problem.gauge_b0

    stats = run_local_bundle_adjustment(
        K,
        seed,
        max_iters=100,
        improvement_tol=1e-12,
        step_tol=1e-10,
    )
    assert stats["succeeded"] is True
    assert stats["final_cost"] < 1e-16

    R1 = {kf: get_pose_for_kf(seed, kf)[0].copy() for kf in range(3)}
    t1 = {kf: get_pose_for_kf(seed, kf)[1].copy() for kf in range(3)}
    C1 = {kf: camera_centre(R1[kf], t1[kf]) for kf in range(3)}
    X1 = {lm["id"]: lm["X_w"].copy() for lm in seed["landmarks"]}

    np.testing.assert_allclose(R1[0], R0[0])
    np.testing.assert_allclose(t1[0], t0[0])
    C_a = C1[0]

    gauge_value_final = float(np.dot(problem.gauge_n, C1[problem.gauge_kf] - C_a))
    np.testing.assert_allclose(gauge_value_final, gauge_b0_initial, atol=1e-12)

    metric_align_initial = _best_fit_scale(C0, X0, C_a, C_true_by_kf, X_true_by_id)
    metric_align_final = _best_fit_scale(C1, X1, C_a, C_true_by_kf, X_true_by_id)
    gauge_align_final = _best_fit_scale(C1, X1, C_a, C_target, X_target)

    rot_err0 = {kf: float(angle_between_rotmats(R0[kf], R_true_by_kf[kf])) for kf in (1, 2)}
    rot_err1 = {kf: float(angle_between_rotmats(R1[kf], R_true_by_kf[kf])) for kf in (1, 2)}

    pose_initial, lm_initial = _state_errors(C0, X0, C_target, X_target)
    pose_raw, lm_raw = _state_errors(C1, X1, C_target, X_target)
    C_gauge_aligned = {kf: C_a + gauge_align_final * (C - C_a) for kf, C in C1.items()}
    X_gauge_aligned = {lm_id: C_a + gauge_align_final * (X - C_a) for lm_id, X in X1.items()}
    pose_aligned, lm_aligned = _state_errors(C_gauge_aligned, X_gauge_aligned, C_target, X_target)
    pose_metric_raw, lm_metric_raw = _state_errors(C1, X1, C_true_by_kf, X_true_by_id)
    C_metric_aligned = {kf: C_a + metric_align_final * (C - C_a) for kf, C in C1.items()}
    X_metric_aligned = {lm_id: C_a + metric_align_final * (X - C_a) for lm_id, X in X1.items()}
    pose_metric_aligned, lm_metric_aligned = _state_errors(
        C_metric_aligned,
        X_metric_aligned,
        C_true_by_kf,
        X_true_by_id,
    )

    initial_rmse = float(np.sqrt(stats["initial_cost"] / len(problem.observations)))
    final_rmse = float(np.sqrt(stats["final_cost"] / len(problem.observations)))
    print(f"\n[Test H] reprojection RMSE initial/final: {initial_rmse:.6e} / {final_rmse:.6e}")
    print(f"[Test H] fixed-slice scale: {gauge_scale:.12f}")
    print(f"[Test H] post-hoc scale to metric truth initial/final: "
          f"{metric_align_initial:.12f} / {metric_align_final:.12f}")
    print(f"[Test H] post-hoc scale within fixed slice: {gauge_align_final:.12f}")
    print("[Test H] rotation error initial/final (deg):",
          {kf: np.degrees(value) for kf, value in rot_err0.items()},
          {kf: np.degrees(value) for kf, value in rot_err1.items()})
    print("[Test H] fixed-slice camera error initial/final/aligned:", pose_initial, pose_raw, pose_aligned)
    print("[Test H] fixed-slice landmark initial median/p90/max:",
          np.median(lm_initial), np.percentile(lm_initial, 90), np.max(lm_initial))
    print("[Test H] fixed-slice landmark final raw median/p90/max:",
          np.median(lm_raw), np.percentile(lm_raw, 90), np.max(lm_raw))
    print("[Test H] fixed-slice landmark final aligned median/p90/max:",
          np.median(lm_aligned), np.percentile(lm_aligned, 90), np.max(lm_aligned))
    print("[Test H] metric-truth camera error raw/aligned:", pose_metric_raw, pose_metric_aligned)
    print("[Test H] metric-truth landmark raw median/p90/max:",
          np.median(lm_metric_raw), np.percentile(lm_metric_raw, 90), np.max(lm_metric_raw))
    print("[Test H] metric-truth landmark aligned median/p90/max:",
          np.median(lm_metric_aligned), np.percentile(lm_metric_aligned, 90), np.max(lm_metric_aligned))

    assert max(rot_err1.values()) < 1e-6
    assert max(rot_err1.values()) < 1e-4 * max(rot_err0.values())
    assert max(pose_raw.values()) < 1e-8
    assert float(np.max(lm_raw)) < 1e-6
    assert abs(gauge_align_final - 1.0) < 1e-8
    assert float(np.max(lm_aligned)) < 1e-6

    # This fixture's noisy baseline does not encode the metric truth scale
    assert abs(gauge_scale - 1.0) > 0.1
    assert float(np.median(lm_metric_raw)) > 0.5
    assert float(np.max(lm_metric_aligned)) < 1e-6

    # no landmark may end up behind any camera after optimisation
    for lm in seed["landmarks"]:
        X_w = lm["X_w"]
        for kf in (0, 1, 2):
            R, t = get_pose_for_kf(seed, kf)
            assert float((R @ X_w + t)[2]) > 0.0


# Supplementary to Test H: when the gauge camera's initial pose IS accurate
# (n and b0 measured correctly), raw recovery should closely approach the
# scale-aligned quality, with no post-hoc scale fitting needed for camera
# position specifically. Only kf 1 (the gauge camera) is left unperturbed;
# kf 2 and all landmarks are still noisy, so this is not a trivial fixed point
def test_ba_gauge_reference_accuracy_determines_raw_recovery():
    K, poses_true, points_true = _scene_ground_truth()
    _, _, point_noise = _noisy_test_perturbations()
    pose_noise = {2: np.asarray([-0.05, 0.05, -0.04], dtype=np.float64)}
    rot_noise = {2: np.asarray([-0.015, 0.025, -0.02], dtype=np.float64)}
    seed = _build_ba_seed(
        K, poses_true, points_true, pose_noise_by_kf=pose_noise, rot_noise_by_kf=rot_noise, point_noise=point_noise, active_kf=2
    )

    R_true_by_kf = {kf: poses_true[kf][0] for kf in range(3)}
    t_true_by_kf = {kf: poses_true[kf][1] for kf in range(3)}
    C_true_by_kf = {kf: camera_centre(R_true_by_kf[kf], t_true_by_kf[kf]) for kf in range(3)}
    X_true_by_id = {i: points_true[i].copy() for i in range(points_true.shape[0])}

    problem = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)[0]
    b0_true = float(np.linalg.norm(C_true_by_kf[problem.gauge_kf] - C_true_by_kf[0]))
    np.testing.assert_allclose(problem.gauge_b0, b0_true, atol=1e-12)

    stats = run_local_bundle_adjustment(K, seed, max_iters=100)
    assert stats["succeeded"] is True

    R1 = {kf: get_pose_for_kf(seed, kf)[0].copy() for kf in range(3)}
    t1 = {kf: get_pose_for_kf(seed, kf)[1].copy() for kf in range(3)}
    C1 = {kf: camera_centre(R1[kf], t1[kf]) for kf in range(3)}
    X1 = {lm["id"]: lm["X_w"].copy() for lm in seed["landmarks"]}
    C_a = C1[0]

    pose_raw = {kf: float(np.linalg.norm(C1[kf] - C_true_by_kf[kf])) for kf in (1, 2)}
    print("\n[gauge-accurate] pose RAW error:", pose_raw)

    # the gauge camera's own position needs no post-hoc scale fit: it is pinned
    # directly by the (accurate) constraint
    assert pose_raw[1] < 0.01


# Test I: the fixed gauge scalar n.(C_gauge - C_anchor) must be exactly
# preserved through every code path that touches state - a rejected trial, and
# the final accepted writeback after a full optimisation run
def test_ba_gauge_scalar_invariant_across_updates():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise, rot_noise, point_noise = _noisy_test_perturbations()
    seed = _build_ba_seed(
        K, poses_true, points_true, pose_noise_by_kf=pose_noise, rot_noise_by_kf=rot_noise, point_noise=point_noise, active_kf=2
    )

    problem, _ = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)
    R_by_kf, t_by_kf = _initial_pose_state(seed, problem.kf_ids)
    X_by_id = _initial_landmark_state(seed, problem.landmark_ids)

    def _gauge_value(R_by_kf, t_by_kf):
        C_a = camera_centre(R_by_kf[problem.anchor_kf], t_by_kf[problem.anchor_kf])
        C_g = camera_centre(R_by_kf[problem.gauge_kf], t_by_kf[problem.gauge_kf])
        return float(np.dot(problem.gauge_n, C_g - C_a))

    gauge_initial = _gauge_value(R_by_kf, t_by_kf)
    np.testing.assert_allclose(gauge_initial, problem.gauge_b0, atol=1e-12)

    # a large, deliberately bad delta (the kind of trial LM would reject) must
    # still land exactly on the constraint plane - the constraint is structural,
    # not something that depends on the step being small or being accepted
    bad_delta = np.zeros(problem.n_vars, dtype=np.float64)
    lm_col = problem.landmark_col_by_id[problem.landmark_ids[0]]
    bad_delta[lm_col : lm_col + 3] = -1000.0
    pose_col = problem.pose_col_by_kf[problem.pose_var_kfs[-1]]
    bad_delta[pose_col : pose_col + 6] = 7.0
    gauge_col = problem.pose_col_by_kf[problem.gauge_kf]
    bad_delta[gauge_col : gauge_col + 5] = np.asarray([3.0, -4.0, 0.5, -0.3, 0.2])

    R_prop, t_prop, _ = _apply_delta(problem, R_by_kf, t_by_kf, X_by_id, bad_delta, eps=1e-12)
    gauge_rejected = _gauge_value(R_prop, t_prop)
    np.testing.assert_allclose(gauge_rejected, problem.gauge_b0, atol=1e-12)

    stats = run_local_bundle_adjustment(K, seed, max_iters=100)
    assert stats["succeeded"] is True
    R_final = {kf: get_pose_for_kf(seed, kf)[0] for kf in problem.kf_ids}
    t_final = {kf: get_pose_for_kf(seed, kf)[1] for kf in problem.kf_ids}
    gauge_final = _gauge_value(R_final, t_final)
    np.testing.assert_allclose(gauge_final, problem.gauge_b0, atol=1e-12)

    print(f"\n[Test I] gauge target/initial/rejected/final: {problem.gauge_b0:.16e} / "
          f"{gauge_initial:.16e} / {gauge_rejected:.16e} / {gauge_final:.16e}")


# Test C / Test J: finite-difference the actual residual function against the
# analytical Jacobian _evaluate_problem builds, using the production
# perturbation code path (_apply_delta) so the comparison respects the real
# left-SE(3)/Euclidean convention for ordinary cameras AND the new constrained
# 2-DOF-translation/3-DOF-rotation retraction for the gauge camera specifically
def test_ba_jacobian_matches_finite_differences():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise, rot_noise, point_noise = _noisy_test_perturbations()
    seed = _build_ba_seed(
        K,
        poses_true,
        points_true,
        pose_noise_by_kf=pose_noise,
        rot_noise_by_kf=rot_noise,
        point_noise=point_noise,
        active_kf=2,
    )

    problem, _ = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)
    assert problem is not None
    R_by_kf, t_by_kf = _initial_pose_state(seed, problem.kf_ids)
    X_by_id = _initial_landmark_state(seed, problem.landmark_ids)

    _, _, J_analytic, reason = _evaluate_problem(K, problem, R_by_kf, t_by_kf, X_by_id, build_jacobian=True, eps=1e-12)
    assert reason is None

    def _fd_jacobian(eps):
        n_vars = problem.n_vars
        J_fd = np.zeros((problem.n_residuals, n_vars), dtype=np.float64)
        for k in range(n_vars):
            dp = np.zeros(n_vars, dtype=np.float64)
            dp[k] = eps
            dm = np.zeros(n_vars, dtype=np.float64)
            dm[k] = -eps
            R_p, t_p, X_p = _apply_delta(problem, R_by_kf, t_by_kf, X_by_id, dp, eps=1e-12)
            R_m, t_m, X_m = _apply_delta(problem, R_by_kf, t_by_kf, X_by_id, dm, eps=1e-12)
            r_p, _, _, reason_p = _evaluate_problem(K, problem, R_p, t_p, X_p, build_jacobian=False, eps=1e-12)
            r_m, _, _, reason_m = _evaluate_problem(K, problem, R_m, t_m, X_m, build_jacobian=False, eps=1e-12)
            assert reason_p is None and reason_m is None
            J_fd[:, k] = (r_p - r_m) / (2.0 * eps)
        return J_fd

    gauge_col = problem.pose_col_by_kf[problem.gauge_kf]

    def _block_for_column(col):
        for kf in problem.pose_var_kfs:
            start = problem.pose_col_by_kf[kf]
            dof = problem.pose_dof_by_kf[kf]
            if start <= col < start + dof:
                offset = col - start
                split = 2 if kf == problem.gauge_kf else 3
                kind = "translation" if offset < split else "rotation"
                return f"pose[{kf}].{kind}"
        for lm_id in problem.landmark_ids:
            start = problem.landmark_col_by_id[lm_id]
            if start <= col < start + 3:
                return f"landmark[{lm_id}]"
        raise AssertionError(f"unmapped Jacobian column {col}")

    best_abs_err = None
    best_rel_err = None
    best_gauge_block_err = None
    for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        J_fd = _fd_jacobian(eps)
        abs_err = np.abs(J_fd - J_analytic)
        max_abs_err = float(np.max(abs_err))
        max_rel_err = float(np.max(abs_err / np.maximum(np.abs(J_analytic), 1e-6)))
        gauge_block_err = float(np.max(abs_err[:, gauge_col : gauge_col + 5]))
        worst_row, worst_col = np.unravel_index(int(np.argmax(abs_err)), abs_err.shape)
        worst_block = _block_for_column(int(worst_col))
        print(f"[Test C/J] eps={eps:.0e} max_abs={max_abs_err:.3e} max_rel={max_rel_err:.3e} "
              f"gauge_block={gauge_block_err:.3e} worst={worst_block} row={worst_row}")
        best_abs_err = max_abs_err if best_abs_err is None else min(best_abs_err, max_abs_err)
        best_rel_err = max_rel_err if best_rel_err is None else min(best_rel_err, max_rel_err)
        best_gauge_block_err = gauge_block_err if best_gauge_block_err is None else min(best_gauge_block_err, gauge_block_err)

    assert best_abs_err < 1e-5
    assert best_rel_err < 1e-4
    assert best_gauge_block_err < 1e-5


# Test K: the old scale generator is null before reduction and excluded after it
def test_ba_old_scale_generator_no_longer_null():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise, rot_noise, point_noise = _noisy_test_perturbations()
    seed_gt = _build_ba_seed(K, poses_true, points_true, active_kf=2)
    seed_noisy = _build_ba_seed(
        K, poses_true, points_true, pose_noise_by_kf=pose_noise, rot_noise_by_kf=rot_noise, point_noise=point_noise, active_kf=2
    )

    for label, seed in (("ground_truth", seed_gt), ("noisy", seed_noisy)):
        problem, _ = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)
        assert problem is not None
        R_by_kf, t_by_kf = _initial_pose_state(seed, problem.kf_ids)
        X_by_id = _initial_landmark_state(seed, problem.landmark_ids)

        _, _, J, reason = _evaluate_problem(
            K,
            problem,
            R_by_kf,
            t_by_kf,
            X_by_id,
            build_jacobian=True,
            eps=1e-12,
        )
        assert reason is None

        np.testing.assert_allclose(problem.gauge_U.T @ problem.gauge_n, np.zeros(2), atol=1e-9)

        C_a = camera_centre(R_by_kf[problem.anchor_kf], t_by_kf[problem.anchor_kf])
        J_old, old_pose_cols, old_landmark_cols = _unconstrained_jacobian(
            K,
            problem,
            R_by_kf,
            t_by_kf,
            X_by_id,
        )
        v_old = np.zeros(J_old.shape[1], dtype=np.float64)
        v_projected = np.zeros(problem.n_vars, dtype=np.float64)

        for kf in problem.pose_var_kfs:
            C_j = camera_centre(R_by_kf[kf], t_by_kf[kf])
            dC_old = C_j - C_a
            rho_old = -R_by_kf[kf] @ dC_old
            old_col = old_pose_cols[kf]
            v_old[old_col : old_col + 3] = rho_old

            reduced_col = problem.pose_col_by_kf[kf]
            if kf == problem.gauge_kf:
                v_projected[reduced_col : reduced_col + 2] = problem.gauge_U.T @ dC_old
            else:
                v_projected[reduced_col : reduced_col + 3] = rho_old

        for lm_id in problem.landmark_ids:
            dX_old = X_by_id[lm_id] - C_a
            old_col = old_landmark_cols[lm_id]
            reduced_col = problem.landmark_col_by_id[lm_id]
            v_old[old_col : old_col + 3] = dX_old
            v_projected[reduced_col : reduced_col + 3] = dX_old

        v_old_unit = v_old / np.linalg.norm(v_old)
        v_projected_unit = v_projected / np.linalg.norm(v_projected)

        old_response = float(np.linalg.norm(J_old @ v_old_unit))
        projected_response = float(np.linalg.norm(J @ v_projected_unit))
        C_gauge = camera_centre(R_by_kf[problem.gauge_kf], t_by_kf[problem.gauge_kf])
        constraint_rate = float(problem.gauge_n @ (C_gauge - C_a))
        S_old = np.linalg.svd(J_old, compute_uv=False)
        S = np.linalg.svd(J, compute_uv=False)

        print(f"\n[Test K/{label}] old full ||Jv||={old_response:.6e} "
              f"constraint_rate={constraint_rate:.6e} smallest_sv={S_old[-1]:.6e}")
        print(f"[Test K/{label}] reduced projection ||Jv||={projected_response:.6e} "
              f"smallest_sv={S[-1]:.6e}")

        assert old_response < 1e-10
        np.testing.assert_allclose(constraint_rate, problem.gauge_b0, atol=1e-12)
        assert abs(constraint_rate) > 1e-4
        assert projected_response > 0.1
        assert S[-1] > 1e-4


# Phase 4 state-safety: a deliberately bad (cheirality-violating) proposed step
# must not mutate the base state that was passed in to _apply_delta - the LM
# loop relies on this to discard rejected trials by simply not reassigning
def test_ba_apply_delta_does_not_mutate_input_state():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise = {
        1: np.asarray([0.06, -0.04, 0.03], dtype=np.float64),
        2: np.asarray([-0.05, 0.05, -0.04], dtype=np.float64),
    }
    seed = _build_ba_seed(K, poses_true, points_true, pose_noise_by_kf=pose_noise, active_kf=2)

    problem, _ = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)
    assert problem is not None
    R_by_kf, t_by_kf = _initial_pose_state(seed, problem.kf_ids)
    X_by_id = _initial_landmark_state(seed, problem.landmark_ids)

    R_before = {kf: R.copy() for kf, R in R_by_kf.items()}
    t_before = {kf: t.copy() for kf, t in t_by_kf.items()}
    X_before = {lm_id: X.copy() for lm_id, X in X_by_id.items()}

    # deliberately bad configuration: drive one landmark far behind every camera
    bad_delta = np.zeros(problem.n_vars, dtype=np.float64)
    lm_col = problem.landmark_col_by_id[problem.landmark_ids[0]]
    bad_delta[lm_col : lm_col + 3] = -1000.0

    R_prop, t_prop, X_prop = _apply_delta(problem, R_by_kf, t_by_kf, X_by_id, bad_delta, eps=1e-12)

    # the bad proposal is indeed geometrically invalid (confirms it would be rejected)
    residual_prop, _, _, reason = _evaluate_problem(K, problem, R_prop, t_prop, X_prop, build_jacobian=False, eps=1e-12)
    assert residual_prop is None
    assert reason == "invalid_geometry"

    # the base state passed into _apply_delta must be untouched
    for kf in problem.kf_ids:
        np.testing.assert_array_equal(R_by_kf[kf], R_before[kf])
        np.testing.assert_array_equal(t_by_kf[kf], t_before[kf])
    for lm_id in problem.landmark_ids:
        np.testing.assert_array_equal(X_by_id[lm_id], X_before[lm_id])

    # the proposal must be independent objects, not aliases of the base state
    assert all(R_prop[kf] is not R_by_kf[kf] for kf in problem.kf_ids)
    assert all(X_prop[lm_id] is not X_by_id[lm_id] for lm_id in problem.landmark_ids)


# Phase 4 state-safety: if the damping ceiling is already below the initial
# damping, the very first inner-loop check must reject every trial before it is
# attempted, so zero steps are accepted and canonical seed state (poses and
# landmark X_w) must be left completely untouched - only bookkeeping is written
def test_ba_zero_accepted_steps_leaves_canonical_state_untouched():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise = {
        1: np.asarray([0.06, -0.04, 0.03], dtype=np.float64),
        2: np.asarray([-0.05, 0.05, -0.04], dtype=np.float64),
    }
    rot_noise = {
        1: np.asarray([0.02, -0.01, 0.03], dtype=np.float64),
        2: np.asarray([-0.015, 0.025, -0.02], dtype=np.float64),
    }
    point_noise = np.asarray(
        [
            [0.030, -0.020, 0.040],
            [-0.025, 0.035, -0.030],
            [0.020, 0.025, 0.035],
            [-0.030, -0.020, 0.025],
            [0.035, 0.010, -0.030],
            [-0.020, 0.030, 0.030],
            [0.025, -0.030, -0.025],
            [-0.035, 0.020, 0.020],
        ],
        dtype=np.float64,
    )
    seed = _build_ba_seed(
        K,
        poses_true,
        points_true,
        pose_noise_by_kf=pose_noise,
        rot_noise_by_kf=rot_noise,
        point_noise=point_noise,
        active_kf=2,
    )

    poses_before = {kf: (R.copy(), t.copy()) for kf, (R, t) in seed["poses"].items()}
    X_before = {lm["id"]: lm["X_w"].copy() for lm in seed["landmarks"]}

    # max_damping below initial_damping forces the first damping check to fire
    # before any trial is attempted, on every outer iteration
    stats = run_local_bundle_adjustment(K, seed, max_iters=3, initial_damping=1.0, max_damping=0.5)

    assert stats["succeeded"] is False
    assert stats["accepted_iterations"] == 0
    assert stats["rejection_reason"] == "damping_limit_reached"

    for kf, (R_true, t_true) in poses_before.items():
        R_after, t_after = seed["poses"][kf]
        np.testing.assert_array_equal(R_after, R_true)
        np.testing.assert_array_equal(t_after, t_true)
    for lm in seed["landmarks"]:
        np.testing.assert_array_equal(lm["X_w"], X_before[lm["id"]])


# Phase 5: the converged fixed-slice solution must not depend on initial damping
def test_ba_gauge_removes_damping_dependence():
    K, poses_true, points_true = _scene_ground_truth()
    pose_noise, rot_noise, point_noise = _noisy_test_perturbations()
    R_true_by_kf = {kf: poses_true[kf][0] for kf in range(3)}
    t_true_by_kf = {kf: poses_true[kf][1] for kf in range(3)}
    C_true_by_kf = {kf: camera_centre(R_true_by_kf[kf], t_true_by_kf[kf]) for kf in range(3)}
    X_true_by_id = {i: points_true[i].copy() for i in range(points_true.shape[0])}

    result_by_damping = {}

    for damping in (1e-4, 1e-3, 1e-2):
        seed = _build_ba_seed(
            K, poses_true, points_true, pose_noise_by_kf=pose_noise, rot_noise_by_kf=rot_noise, point_noise=point_noise, active_kf=2
        )
        problem, _ = _build_problem(K, seed, max_keyframes=3, min_keyframes=2, min_landmarks=6, min_observations=12, eps=1e-12)
        gauge_kf, gauge_n, gauge_b0 = problem.gauge_kf, problem.gauge_n, problem.gauge_b0
        gauge_scale, C_target, X_target = _gauge_selected_truth(problem, C_true_by_kf, X_true_by_id)

        stats = run_local_bundle_adjustment(
            K,
            seed,
            max_iters=100,
            initial_damping=damping,
            improvement_tol=1e-12,
            step_tol=1e-10,
        )
        assert stats["succeeded"] is True

        R1 = {kf: get_pose_for_kf(seed, kf)[0].copy() for kf in range(3)}
        t1 = {kf: get_pose_for_kf(seed, kf)[1].copy() for kf in range(3)}
        C1 = {kf: camera_centre(R1[kf], t1[kf]) for kf in range(3)}
        X1 = {lm["id"]: lm["X_w"].copy() for lm in seed["landmarks"]}

        pose_errors, landmark_errors = _state_errors(C1, X1, C_target, X_target)
        scale_to_metric = _best_fit_scale(C1, X1, C1[0], C_true_by_kf, X_true_by_id)
        state_vector = np.concatenate(
            [
                *(C1[kf] for kf in (1, 2)),
                *(X1[lm_id] for lm_id in sorted(X1)),
            ]
        )
        result_by_damping[damping] = {
            "rmse": float(np.sqrt(stats["final_cost"] / len(problem.observations))),
            "pose_max": max(pose_errors.values()),
            "landmark_median": float(np.median(landmark_errors)),
            "landmark_p90": float(np.percentile(landmark_errors, 90)),
            "landmark_max": float(np.max(landmark_errors)),
            "gauge": float(gauge_n @ (C1[gauge_kf] - C1[0])),
            "gauge_target": gauge_b0,
            "recovered_scale": float(1.0 / scale_to_metric),
            "target_scale": gauge_scale,
            "state": state_vector,
        }

    print("\n[Test lambda] damping rmse pose_max lm_median lm_p90 lm_max gauge recovered_scale")
    for damping, result in result_by_damping.items():
        print(f"[Test lambda] {damping:.0e} {result['rmse']:.3e} {result['pose_max']:.3e} "
              f"{result['landmark_median']:.3e} {result['landmark_p90']:.3e} "
              f"{result['landmark_max']:.3e} {result['gauge']:.16e} "
              f"{result['recovered_scale']:.12f}")

    for result in result_by_damping.values():
        np.testing.assert_allclose(result["gauge"], result["gauge_target"], atol=1e-12)
        np.testing.assert_allclose(result["recovered_scale"], result["target_scale"], atol=1e-7)
        assert result["pose_max"] < 1e-8
        assert result["landmark_max"] < 1e-6

    states = np.vstack([result["state"] for result in result_by_damping.values()])
    max_state_spread = float(np.max(np.ptp(states, axis=0)))
    print(f"[Test lambda] maximum coordinate spread across damping values: {max_state_spread:.3e}")
    assert max_state_spread < 1e-6
