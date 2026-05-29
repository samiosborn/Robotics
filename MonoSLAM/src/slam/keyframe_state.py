# src/slam/keyframe_state.py
from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import check_int_ge0_no_bool
from slam.landmark_state import build_observation_indexes


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

    indexes = build_observation_indexes(seed, context="seed['landmarks']")
    for (obs_kf, feat), landmark_id in indexes["landmark_id_by_feature"].items():
        if int(obs_kf) != int(kf):
            continue
        if int(feat) < 0 or int(feat) >= int(n_feat):
            continue
        lookup[int(feat)] = int(landmark_id)

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


# Read one field from the active keyframe record
def _get_active_record_field(seed, field: str, legacy_key: str):
    seed = _check_seed_dict(seed)
    has_canonical_active = "active_keyframe_kf" in seed or "keyframes" in seed
    if not bool(has_canonical_active):
        if legacy_key not in seed:
            raise ValueError(f"seed is missing '{legacy_key}' for active keyframe access")
        return seed[legacy_key]

    active_kf = get_active_keyframe_kf(seed)
    record = get_active_keyframe_record(seed)
    if field not in record:
        raise ValueError(f"seed['keyframes'][{active_kf}] is missing '{field}'")

    return record[field]


# Read the active keyframe pose
def get_active_keyframe_pose(seed):
    return _get_active_record_field(seed, "pose", "T_WC1")


# Read the active keyframe feature bundle
def get_active_keyframe_features(seed):
    return _get_active_record_field(seed, "feats", "feats1")


# Read the active keyframe landmark lookup
def get_active_landmark_lookup(seed):
    return _get_active_record_field(seed, "landmark_id_by_feat", "landmark_id_by_feat1")


# Store a new active keyframe through legacy mirrors
def set_active_keyframe_record(seed, kf, pose, feats, landmark_id_by_feat) -> dict:
    seed = _check_seed_dict(seed)
    kf = check_int_ge0_no_bool(kf, name="kf")

    seed["T_WC1"] = pose
    seed["feats1"] = feats
    seed["landmark_id_by_feat1"] = landmark_id_by_feat
    seed["keyframe_kf"] = int(kf)
    sync_active_keyframe_mirrors(seed)

    return get_active_keyframe_record(seed)


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
    try:
        active_kf = get_active_keyframe_kf(seed)
    except ValueError:
        active_kf = None
    if active_kf is not None and int(active_kf) == int(current_kf) and "T_WC1" in seed:
        poses[int(current_kf)] = get_active_keyframe_pose(seed)
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


# Validate the active keyframe mirrors through invariants
def validate_active_keyframe_state(seed, *, context="active_keyframe") -> dict:
    from slam.invariants import audit_active_keyframe_lookup

    return audit_active_keyframe_lookup(seed, context=str(context))
