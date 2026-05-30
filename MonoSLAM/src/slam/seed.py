# src/slam/seed.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.checks import check_2xN_pair, check_matrix_3x3, check_vector_3
from slam.keyframe_state import (
    get_active_keyframe_pose,
    set_active_keyframe_record,
    set_keyframe_record,
    set_pose_for_kf,
)


# Build a minimal feature bundle from bootstrap point columns
def _feature_bundle_from_2xN(x, *, name: str):
    points = np.asarray(x, dtype=np.float64)
    if points.ndim != 2 or int(points.shape[0]) != 2:
        raise ValueError(f"{name} must be (2,N); got {points.shape}")
    return SimpleNamespace(kps_xy=np.asarray(points.T, dtype=np.float64).copy())


# Build a two-view seed map from validated bootstrap outputs
def build_two_view_seed(x1, x2, *, idx_init, X_valid, R1, t1) -> dict:
    # Checks
    check_2xN_pair(x1, x2)
    R1 = check_matrix_3x3(R1, name="R1", finite=False)
    t1 = np.asarray(t1, dtype=np.float64).reshape(3)
    idx_init = np.asarray(idx_init, dtype=np.int64).reshape(-1)
    X_valid = np.asarray(X_valid, dtype=np.float64)
    if X_valid.ndim != 2 or X_valid.shape[0] != 3:
        raise ValueError(f"X_valid must be (3,N); got {X_valid.shape}")
    if X_valid.shape[1] != idx_init.size:
        raise ValueError(f"Mismatch: X_valid has {X_valid.shape[1]} cols but idx_init has {idx_init.size}")

    # Validate index bounds
    n_cols = int(x1.shape[1])
    if idx_init.size > 0 and (int(idx_init.min()) < 0 or int(idx_init.max()) >= n_cols):
        raise ValueError(f"idx_init contains values outside [0,{n_cols - 1}]")

    # Poses
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.zeros(3, dtype=np.float64)

    # Landmarks
    landmarks = []
    for lm_id, (j, X_w) in enumerate(zip(idx_init, X_valid.T)):
        j = int(j)
        landmarks.append(
            {
                "id": int(lm_id),
                "X_w": np.asarray(X_w, dtype=np.float64).reshape(3).copy(),
                "birth_source": "bootstrap",
                "birth_kf": 1,
                "obs": [
                    {"kf": 0, "feat": j, "xy": np.asarray(x1[:, j], dtype=np.float64).reshape(2).copy()},
                    {"kf": 1, "feat": j, "xy": np.asarray(x2[:, j], dtype=np.float64).reshape(2).copy()},
                ],
                "descriptor": None,
                "quality": {"reproj0_px": None, "reproj1_px": None},
            }
        )

    # Initialised correspondence mask
    mask_init = np.zeros(n_cols, dtype=bool)
    if idx_init.size > 0:
        mask_init[idx_init] = True

    seed = {
        "poses": {},
        "keyframes": {},
        "active_keyframe_kf": 1,
        "landmarks": landmarks,
        "mask_init": mask_init,
        "idx_init": idx_init,
    }

    lookup0 = np.full((n_cols,), -1, dtype=np.int64)
    lookup1 = np.full((n_cols,), -1, dtype=np.int64)
    for lm_id, j in enumerate(idx_init):
        lookup0[int(j)] = int(lm_id)
        lookup1[int(j)] = int(lm_id)

    set_pose_for_kf(seed, 0, (R0, t0), context="bootstrap pose 0")
    set_keyframe_record(
        seed,
        0,
        (R0, t0),
        _feature_bundle_from_2xN(x1, name="x1"),
        lookup0,
        context="bootstrap keyframe 0",
    )
    set_active_keyframe_record(
        seed,
        1,
        (R1, t1),
        _feature_bundle_from_2xN(x2, name="x2"),
        lookup1,
    )

    return seed


# Read the frozen keyframe pose stored inside the seed
def seed_keyframe_pose(seed: dict) -> tuple[np.ndarray, np.ndarray]:
    # Read stored active pose tuple
    pose = get_active_keyframe_pose(seed)
    if not isinstance(pose, (tuple, list)) or len(pose) != 2:
        raise ValueError("active keyframe pose must be a length-2 tuple/list (R, t)")

    # Check pose blocks
    R = check_matrix_3x3(pose[0], name="active keyframe pose[0]", dtype=float, finite=False)
    t = check_vector_3(pose[1], name="active keyframe pose[1]", dtype=float, finite=False)

    return np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3,)


# Attach descriptor and feature-index bookkeeping for initial seed landmarks
def attach_feature_bookkeeping_to_seed(seed, feats0, feats1, matches):
    if not isinstance(seed, dict):
        return seed

    landmarks_raw = seed.get("landmarks", [])
    landmarks = landmarks_raw if isinstance(landmarks_raw, list) else list(landmarks_raw)

    idx_init = np.asarray(seed.get("idx_init", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    ia = np.asarray(getattr(matches, "ia", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    ib = np.asarray(getattr(matches, "ib", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)

    kps0 = np.asarray(getattr(feats0, "kps_xy", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    kps1 = np.asarray(getattr(feats1, "kps_xy", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    desc1 = np.asarray(getattr(feats1, "desc", np.zeros((0,), dtype=np.float64)))

    n_feat0 = int(kps0.shape[0]) if kps0.ndim == 2 else 0
    n_feat1 = int(kps1.shape[0]) if kps1.ndim == 2 else 0

    for lm_id, lm in enumerate(landmarks):
        if not isinstance(lm, dict):
            continue
        if lm_id >= idx_init.size:
            continue

        j = int(idx_init[lm_id])
        lm["match_idx"] = j

        if j < 0 or j >= ia.size or j >= ib.size:
            continue

        feat0 = int(ia[j])
        feat1 = int(ib[j])

        if feat0 < 0 or feat0 >= n_feat0 or feat1 < 0 or feat1 >= n_feat1:
            continue

        obs = lm.get("obs")
        if not isinstance(obs, list):
            obs = []
        while len(obs) < 2:
            obs.append({"kf": len(obs), "feat": -1, "xy": np.zeros((2,), dtype=np.float64)})

        obs[0]["feat"] = feat0
        obs[1]["feat"] = feat1
        obs[0]["xy"] = np.asarray(kps0[feat0, :2], dtype=np.float64).reshape(2).copy()
        obs[1]["xy"] = np.asarray(kps1[feat1, :2], dtype=np.float64).reshape(2).copy()
        lm["obs"] = obs

        if desc1.ndim >= 1 and feat1 < int(desc1.shape[0]):
            lm["descriptor"] = np.asarray(desc1[feat1]).copy()

    lookup0 = np.full((n_feat0,), -1, dtype=np.int64)
    lookup1 = np.full((n_feat1,), -1, dtype=np.int64)
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        lm_id = int(lm.get("id", -1))
        obs = lm.get("obs")
        if not isinstance(obs, list) or len(obs) < 2:
            continue
        feat0 = int(obs[0].get("feat", -1))
        feat1 = int(obs[1].get("feat", -1))
        if 0 <= feat0 < n_feat0:
            lookup0[feat0] = lm_id
        if 0 <= feat1 < n_feat1:
            lookup1[feat1] = lm_id

    pose0 = seed["poses"][0]
    pose1 = seed["poses"][1]

    # Pack canonical keyframe records
    seed["landmarks"] = landmarks
    seed["matches01"] = matches
    set_keyframe_record(seed, 0, pose0, feats0, lookup0, context="bootstrap keyframe 0")
    set_active_keyframe_record(seed, 1, pose1, feats1, lookup1)

    return seed
