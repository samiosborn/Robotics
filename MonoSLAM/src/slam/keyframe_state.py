# src/slam/keyframe_state.py
from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import check_int_ge0_no_bool


# Check a mutable seed container
def _check_seed_dict(seed) -> dict:
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    return seed


# Read a feature row count when available
def _feature_count(feats) -> int:
    kps_xy = getattr(feats, "kps_xy", None)
    if kps_xy is None:
        return 0

    arr = np.asarray(kps_xy)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return 0

    return int(arr.shape[0])


# Build a lookup array from landmark observations
def _build_landmark_id_by_feat_for_kf(seed: dict, n_feat: int, kf: int) -> np.ndarray:
    n_feat = check_int_ge0_no_bool(n_feat, name="n_feat")
    kf = check_int_ge0_no_bool(kf, name="kf")
    lookup = np.full((int(n_feat),), -1, dtype=np.int64)

    landmarks = seed.get("landmarks", [])
    if not isinstance(landmarks, list):
        return lookup

    for landmark in landmarks:
        if not isinstance(landmark, dict) or "id" not in landmark:
            continue

        landmark_id = int(landmark["id"])
        obs = landmark.get("obs", None)
        if not isinstance(obs, list):
            continue

        for observation in obs:
            if not isinstance(observation, dict):
                continue
            if int(observation.get("kf", -1)) != int(kf):
                continue

            feat = int(observation.get("feat", -1))
            if feat < 0 or feat >= int(n_feat):
                continue

            lookup[feat] = int(landmark_id)

    return lookup


# Store one keyframe record
def _set_keyframe_record(seed: dict, kf: int, pose, feats, landmark_id_by_feat) -> dict:
    keyframes = ensure_keyframe_store(seed)
    keyframes[int(kf)] = {
        "kf": int(kf),
        "pose": pose,
        "feats": feats,
        "landmark_id_by_feat": landmark_id_by_feat,
    }
    return keyframes[int(kf)]


# Ensure the canonical pose dictionary exists
def ensure_pose_store(seed) -> dict:
    seed = _check_seed_dict(seed)
    poses = seed.get("poses", None)
    if poses is None:
        poses = {}
        seed["poses"] = poses
    if not isinstance(poses, dict):
        raise ValueError("seed['poses'] must be a dict")
    return poses


# Ensure the canonical keyframe dictionary exists
def ensure_keyframe_store(seed) -> dict:
    seed = _check_seed_dict(seed)
    keyframes = seed.get("keyframes", None)
    if keyframes is None:
        keyframes = {}
        seed["keyframes"] = keyframes
    if not isinstance(keyframes, dict):
        raise ValueError("seed['keyframes'] must be a dict")
    return keyframes


# Initialise canonical stores from legacy seed fields
def initialise_canonical_keyframe_state(seed) -> dict:
    seed = _check_seed_dict(seed)
    poses = ensure_pose_store(seed)

    if "T_WC0" in seed:
        poses[0] = seed["T_WC0"]

    if "T_WC1" in seed:
        poses[1] = seed["T_WC1"]
        if "keyframe_kf" not in seed:
            seed["keyframe_kf"] = 1

    if "T_WC0" in seed and "feats0" in seed:
        lookup0 = _build_landmark_id_by_feat_for_kf(seed, _feature_count(seed["feats0"]), 0)
        _set_keyframe_record(seed, 0, seed["T_WC0"], seed["feats0"], lookup0)

    has_active_mirrors = all(key in seed for key in ("T_WC1", "feats1", "landmark_id_by_feat1", "keyframe_kf"))
    if bool(has_active_mirrors):
        sync_active_keyframe_mirrors(seed)

    return seed


# Read the active keyframe id
def get_active_keyframe_kf(seed) -> int:
    seed = _check_seed_dict(seed)
    active_raw = seed.get("active_keyframe_kf", None)
    legacy_raw = seed.get("keyframe_kf", None)

    if active_raw is None and legacy_raw is None:
        raise ValueError("seed must contain 'active_keyframe_kf' or 'keyframe_kf'")

    if active_raw is None:
        return check_int_ge0_no_bool(legacy_raw, name="seed['keyframe_kf']")

    active_kf = check_int_ge0_no_bool(active_raw, name="seed['active_keyframe_kf']")
    if legacy_raw is not None:
        legacy_kf = check_int_ge0_no_bool(legacy_raw, name="seed['keyframe_kf']")
        if int(active_kf) != int(legacy_kf):
            raise ValueError("seed['active_keyframe_kf'] must match seed['keyframe_kf']")

    return int(active_kf)


# Read the active keyframe record
def get_active_keyframe_record(seed) -> dict:
    seed = _check_seed_dict(seed)
    active_kf = get_active_keyframe_kf(seed)
    keyframes = seed.get("keyframes", None)
    if not isinstance(keyframes, dict):
        raise ValueError("seed['keyframes'] must be a dict")
    if int(active_kf) not in keyframes:
        raise ValueError(f"seed['active_keyframe_kf']={active_kf} is missing from seed['keyframes']")

    record = keyframes[int(active_kf)]
    if not isinstance(record, dict):
        raise ValueError(f"seed['keyframes'][{active_kf}] must be a dict")

    return record


# Sync canonical active state from legacy mirrors
def sync_active_keyframe_mirrors(seed) -> dict:
    seed = _check_seed_dict(seed)
    if "keyframe_kf" not in seed:
        raise ValueError("seed is missing 'keyframe_kf' for active keyframe sync")
    for key in ("T_WC1", "feats1", "landmark_id_by_feat1"):
        if key not in seed:
            raise ValueError(f"seed is missing '{key}' for active keyframe sync")

    active_kf = check_int_ge0_no_bool(seed["keyframe_kf"], name="seed['keyframe_kf']")
    seed["active_keyframe_kf"] = int(active_kf)

    poses = ensure_pose_store(seed)
    poses[int(active_kf)] = seed["T_WC1"]
    _set_keyframe_record(
        seed,
        int(active_kf),
        seed["T_WC1"],
        seed["feats1"],
        seed["landmark_id_by_feat1"],
    )

    return seed


# Sync active state only when all legacy mirrors exist
def sync_active_keyframe_mirrors_if_present(seed) -> dict:
    seed = _check_seed_dict(seed)
    has_active_mirrors = all(key in seed for key in ("T_WC1", "feats1", "landmark_id_by_feat1", "keyframe_kf"))
    if bool(has_active_mirrors):
        sync_active_keyframe_mirrors(seed)
    return seed


# Store an accepted current-frame pose
def store_current_pose(seed, current_kf: int, R_cur, t_cur) -> dict:
    seed = _check_seed_dict(seed)
    current_kf = check_int_ge0_no_bool(current_kf, name="current_kf")

    poses = ensure_pose_store(seed)
    active_kf = seed.get("active_keyframe_kf", seed.get("keyframe_kf", None))
    if active_kf is not None and int(active_kf) == int(current_kf) and "T_WC1" in seed:
        poses[int(current_kf)] = seed["T_WC1"]
        return seed

    poses[int(current_kf)] = (
        np.asarray(R_cur, dtype=np.float64).copy(),
        np.asarray(t_cur, dtype=np.float64).reshape(3).copy(),
    )

    return seed


# Validate canonical keyframe state through invariants
def validate_keyframe_store(seed, *, context="keyframe_store") -> dict:
    from slam.invariants import audit_seed_invariants

    return audit_seed_invariants(seed, context=str(context))
