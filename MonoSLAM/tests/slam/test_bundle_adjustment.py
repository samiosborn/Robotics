from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from slam.bundle_adjustment import run_local_bundle_adjustment
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


# Build a small local BA fixture with three connected keyframes
def _local_ba_seed():
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

    keyframes = {}
    poses = {}
    landmarks = []
    n_points = int(points_true.shape[0])
    lookup = np.arange(n_points, dtype=np.int64)

    for kf, pose_true in enumerate(poses_true):
        R_true, t_true = pose_true
        xy = np.vstack([_project(K, R_true, t_true, X) for X in points_true])
        if kf == 0:
            pose = (R_true.copy(), t_true.copy())
        else:
            pose = (R_true.copy(), t_true + pose_noise[int(kf)])
        poses[int(kf)] = pose
        keyframes[int(kf)] = {
            "kf": int(kf),
            "pose": pose,
            "feats": _features(xy),
            "landmark_id_by_feat": lookup.copy(),
        }

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

    seed = {
        "poses": poses,
        "keyframes": keyframes,
        "active_keyframe_kf": 2,
        "landmarks": landmarks,
    }
    return K, seed


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
