# src/slam/keyframe_state.py
from __future__ import annotations

import numpy as np

from core.checks import check_int_ge0_no_bool, check_matrix_3x3, check_vector_3
from slam.landmark_state import build_observation_indexes


# Canonical stores own state; legacy active fields are mirrors.


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


# Read a checked feature row count
def _checked_feature_count(feats, *, name: str) -> int:
    if not hasattr(feats, "kps_xy"):
        raise ValueError(f"{name} must have attribute 'kps_xy'")

    kps_xy = np.asarray(getattr(feats, "kps_xy"), dtype=np.float64)
    if kps_xy.ndim != 2 or int(kps_xy.shape[1]) < 2:
        raise ValueError(f"{name}.kps_xy must have shape (N,2+); got {kps_xy.shape}")
    if not np.isfinite(kps_xy[:, :2]).all():
        raise ValueError(f"{name}.kps_xy first two columns must be finite")

    return int(kps_xy.shape[0])


# Copy a pose value for canonical storage
def _copy_pose_value(pose, *, name: str):
    if isinstance(pose, dict) and ("R" in pose or "t" in pose):
        if "R" not in pose:
            raise ValueError(f"{name} is missing required key 'R'")
        if "t" not in pose:
            raise ValueError(f"{name} is missing required key 't'")
        return {
            "R": check_matrix_3x3(pose["R"], name=f"{name}['R']", dtype=float, finite=False).copy(),
            "t": check_vector_3(pose["t"], name=f"{name}['t']", dtype=float, finite=False).copy(),
        }

    if isinstance(pose, (tuple, list)) and len(pose) == 2:
        return (
            check_matrix_3x3(pose[0], name=f"{name}[0]", dtype=float, finite=False).copy(),
            check_vector_3(pose[1], name=f"{name}[1]", dtype=float, finite=False).copy(),
        )

    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be a pose tuple/dict or 4x4 matrix; got {arr.shape}")
    return arr.copy()


# Convert a pose value to a homogeneous matrix
def _pose_value_to_matrix(pose, *, name: str) -> np.ndarray:
    if isinstance(pose, dict) and ("R" in pose or "t" in pose):
        if "R" not in pose:
            raise ValueError(f"{name} is missing required key 'R'")
        if "t" not in pose:
            raise ValueError(f"{name} is missing required key 't'")
        R = check_matrix_3x3(pose["R"], name=f"{name}['R']", dtype=float, finite=False)
        t = check_vector_3(pose["t"], name=f"{name}['t']", dtype=float, finite=False)
    elif isinstance(pose, (tuple, list)) and len(pose) == 2:
        R = check_matrix_3x3(pose[0], name=f"{name}[0]", dtype=float, finite=False)
        t = check_vector_3(pose[1], name=f"{name}[1]", dtype=float, finite=False)
    else:
        arr = np.asarray(pose, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(f"{name} must be a pose tuple/dict or 4x4 matrix; got {arr.shape}")
        return arr

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3,)
    return T


# Compare two pose values without mutating either
def _poses_agree(pose_a, pose_b, *, name_a: str, name_b: str) -> bool:
    T_a = _pose_value_to_matrix(pose_a, name=name_a)
    T_b = _pose_value_to_matrix(pose_b, name=name_b)
    return bool(np.allclose(T_a, T_b))


# Copy an active landmark lookup for canonical storage
def _copy_landmark_lookup(landmark_id_by_feat, *, name: str):
    if isinstance(landmark_id_by_feat, dict):
        lookup: dict[int, int] = {}
        for feat_raw, landmark_id_raw in landmark_id_by_feat.items():
            feat = check_int_ge0_no_bool(feat_raw, name=f"{name} feature index")
            landmark_id = check_int_ge0_no_bool(landmark_id_raw, name=f"{name}[{int(feat)}]")
            lookup[int(feat)] = int(landmark_id)
        return lookup

    arr_raw = np.asarray(landmark_id_by_feat)
    if arr_raw.ndim != 1:
        raise ValueError(f"{name} must be a dict or 1D integer array; got shape {arr_raw.shape}")
    if arr_raw.size > 0 and arr_raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integer landmark ids; got dtype {arr_raw.dtype}")

    arr = np.asarray(arr_raw, dtype=np.int64).reshape(-1)
    if np.any(arr < -1):
        raise ValueError(f"{name} must contain landmark ids >= -1; got min={int(np.min(arr))}")

    return arr.copy()


# Build a lookup array from landmark observations
def build_landmark_lookup_for_kf(seed: dict, n_feat: int, kf: int, *, context: str = "active_lookup") -> np.ndarray:
    n_feat = check_int_ge0_no_bool(n_feat, name="n_feat")
    kf = check_int_ge0_no_bool(kf, name="kf")
    lookup = np.full((int(n_feat),), -1, dtype=np.int64)

    indexes = build_observation_indexes(seed, context="seed['landmarks']")
    for (obs_kf, feat), landmark_id in indexes["landmark_id_by_feature"].items():
        if int(obs_kf) != int(kf):
            continue
        if int(feat) >= int(n_feat):
            raise ValueError(
                f"{context} observation for keyframe {int(kf)} has feature index {int(feat)} outside feature length {int(n_feat)}"
            )
        lookup[int(feat)] = int(landmark_id)

    return lookup


# Build a lookup array from landmark observations
def _build_landmark_id_by_feat_for_kf(seed: dict, n_feat: int, kf: int) -> np.ndarray:
    return build_landmark_lookup_for_kf(seed, n_feat, kf, context="keyframe lookup")


# Store one keyframe record
def _set_keyframe_record(seed: dict, kf: int, pose, feats, landmark_id_by_feat) -> dict:
    return set_keyframe_record(seed, kf, pose, feats, landmark_id_by_feat, sync_active=False)


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


# Read the canonical pose dictionary
def get_pose_store(seed) -> dict:
    return ensure_pose_store(seed)


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


# Read the canonical keyframe dictionary
def get_keyframe_store(seed) -> dict:
    return ensure_keyframe_store(seed)


# Check whether any active keyframe state is available
def has_active_keyframe_state(seed) -> bool:
    seed = _check_seed_dict(seed)
    has_canonical = "active_keyframe_kf" in seed and "keyframes" in seed
    has_legacy = all(key in seed for key in ("T_WC1", "feats1", "landmark_id_by_feat1", "keyframe_kf"))
    return bool(has_canonical or has_legacy)


# Read the canonical active keyframe id without consulting legacy mirrors
def _get_canonical_active_keyframe_kf(seed) -> int:
    seed = _check_seed_dict(seed)
    if "active_keyframe_kf" not in seed:
        raise ValueError("seed is missing 'active_keyframe_kf'")
    return int(check_int_ge0_no_bool(seed["active_keyframe_kf"], name="seed['active_keyframe_kf']"))


# Read one canonical pose by keyframe id
def get_pose_for_kf(seed, kf, *, context: str = "pose"):
    seed = _check_seed_dict(seed)
    kf = check_int_ge0_no_bool(kf, name="kf")
    poses = get_pose_store(seed)
    if int(kf) not in poses:
        raise ValueError(f"{context}: seed['poses'] is missing keyframe {int(kf)}")
    return poses[int(kf)]


# Store one canonical pose by keyframe id
def set_pose_for_kf(seed, kf, pose, *, copy: bool = True, context: str = "pose") -> dict:
    seed = _check_seed_dict(seed)
    kf = check_int_ge0_no_bool(kf, name="kf")
    poses = get_pose_store(seed)
    stored_pose = _copy_pose_value(pose, name=f"{context} for keyframe {int(kf)}") if bool(copy) else pose
    poses[int(kf)] = stored_pose

    keyframes = seed.get("keyframes", None)
    if isinstance(keyframes, dict) and int(kf) in keyframes and isinstance(keyframes[int(kf)], dict):
        keyframes[int(kf)]["pose"] = poses[int(kf)]

    return poses


# Read one canonical keyframe record by id
def get_keyframe_record(seed, kf, *, context: str = "keyframe") -> dict:
    seed = _check_seed_dict(seed)
    kf = check_int_ge0_no_bool(kf, name="kf")
    keyframes = get_keyframe_store(seed)
    if int(kf) not in keyframes:
        raise ValueError(f"{context}: seed['keyframes'] is missing keyframe {int(kf)}")

    record = keyframes[int(kf)]
    if not isinstance(record, dict):
        raise ValueError(f"{context}: seed['keyframes'][{int(kf)}] must be a dict")
    if "kf" in record:
        record_kf = check_int_ge0_no_bool(record["kf"], name=f"{context}['kf']")
        if int(record_kf) != int(kf):
            raise ValueError(f"{context}: seed['keyframes'][{int(kf)}]['kf'] must match keyframe {int(kf)}")

    return record


# Store one canonical keyframe record
def set_keyframe_record(
    seed,
    kf,
    pose,
    feats,
    landmark_id_by_feat,
    *,
    sync_active: bool = True,
    context: str = "keyframe",
) -> dict:
    seed = _check_seed_dict(seed)
    kf = check_int_ge0_no_bool(kf, name="kf")
    set_pose_for_kf(seed, int(kf), pose, copy=True, context=f"{context} pose")
    lookup = _copy_landmark_lookup(landmark_id_by_feat, name=f"{context} landmark lookup")

    keyframes = get_keyframe_store(seed)
    keyframes[int(kf)] = {
        "kf": int(kf),
        "pose": get_pose_for_kf(seed, int(kf), context=context),
        "feats": feats,
        "landmark_id_by_feat": lookup,
    }

    if bool(sync_active):
        try:
            active_kf = _get_canonical_active_keyframe_kf(seed)
        except ValueError:
            active_kf = None
        if active_kf is not None and int(active_kf) == int(kf):
            sync_legacy_active_fields_from_canonical(seed)

    return keyframes[int(kf)]


# Initialise canonical stores from legacy seed fields
def initialise_canonical_keyframe_state(seed) -> dict:
    seed = _check_seed_dict(seed)

    if "T_WC0" in seed:
        set_pose_for_kf(seed, 0, seed["T_WC0"], copy=True, context="bootstrap T_WC0")
        seed["T_WC0"] = get_pose_for_kf(seed, 0, context="bootstrap T_WC0")

    if "T_WC1" in seed:
        if "keyframe_kf" not in seed:
            seed["keyframe_kf"] = 1

    if "T_WC0" in seed and "feats0" in seed:
        lookup0 = _build_landmark_id_by_feat_for_kf(seed, _feature_count(seed["feats0"]), 0)
        set_keyframe_record(seed, 0, seed["T_WC0"], seed["feats0"], lookup0, sync_active=False, context="bootstrap keyframe 0")
        seed["T_WC0"] = get_pose_for_kf(seed, 0, context="bootstrap keyframe 0")

    has_active_mirrors = all(key in seed for key in ("T_WC1", "feats1", "landmark_id_by_feat1", "keyframe_kf"))
    if bool(has_active_mirrors):
        sync_canonical_active_from_legacy(seed)

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
    keyframes = get_keyframe_store(seed)
    if int(active_kf) not in keyframes:
        raise ValueError(f"seed['active_keyframe_kf']={active_kf} is missing from seed['keyframes']")
    return get_keyframe_record(seed, int(active_kf), context="active keyframe")


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


# Store a new canonical active keyframe and sync legacy mirrors
def set_active_keyframe_record(seed, kf, pose, feats, landmark_id_by_feat) -> dict:
    seed = _check_seed_dict(seed)
    kf = check_int_ge0_no_bool(kf, name="kf")

    seed["active_keyframe_kf"] = int(kf)
    set_keyframe_record(
        seed,
        int(kf),
        pose,
        feats,
        landmark_id_by_feat,
        sync_active=False,
        context="active keyframe",
    )
    sync_legacy_active_fields_from_canonical(seed)

    return get_active_keyframe_record(seed)


# Sync legacy active fields from canonical state
def sync_legacy_active_fields_from_canonical(seed) -> dict:
    seed = _check_seed_dict(seed)
    active_kf = _get_canonical_active_keyframe_kf(seed)
    record = get_keyframe_record(seed, int(active_kf), context="active keyframe")
    pose = get_pose_for_kf(seed, int(active_kf), context="active keyframe pose")

    for field in ("pose", "feats", "landmark_id_by_feat"):
        if field not in record:
            raise ValueError(f"seed['keyframes'][{int(active_kf)}] is missing '{field}' for legacy sync")

    if not _poses_agree(
        pose,
        record["pose"],
        name_a=f"seed['poses'][{int(active_kf)}]",
        name_b=f"seed['keyframes'][{int(active_kf)}]['pose']",
    ):
        raise ValueError(f"active keyframe {int(active_kf)} pose store must match keyframe record before legacy sync")

    seed["keyframe_kf"] = int(active_kf)
    seed["T_WC1"] = pose
    seed["feats1"] = record["feats"]
    seed["landmark_id_by_feat1"] = record["landmark_id_by_feat"]

    return seed


# Sync canonical active state from legacy fields
def sync_canonical_active_from_legacy(seed) -> dict:
    seed = _check_seed_dict(seed)
    if "keyframe_kf" not in seed:
        raise ValueError("seed is missing 'keyframe_kf' for canonical active sync")
    for key in ("T_WC1", "feats1", "landmark_id_by_feat1"):
        if key not in seed:
            raise ValueError(f"seed is missing '{key}' for canonical active sync")

    active_kf = check_int_ge0_no_bool(seed["keyframe_kf"], name="seed['keyframe_kf']")
    seed["active_keyframe_kf"] = int(active_kf)
    set_keyframe_record(
        seed,
        int(active_kf),
        seed["T_WC1"],
        seed["feats1"],
        seed["landmark_id_by_feat1"],
        sync_active=False,
        context="legacy active sync",
    )
    sync_legacy_active_fields_from_canonical(seed)

    return seed


# Sync canonical active state from legacy mirrors
def sync_active_keyframe_mirrors(seed) -> dict:
    return sync_canonical_active_from_legacy(seed)


# Sync active state only when all legacy mirrors exist
def sync_active_keyframe_mirrors_if_present(seed) -> dict:
    seed = _check_seed_dict(seed)
    has_active_mirrors = all(key in seed for key in ("T_WC1", "feats1", "landmark_id_by_feat1", "keyframe_kf"))
    if bool(has_active_mirrors):
        sync_canonical_active_from_legacy(seed)
    return seed


# Store a canonical active lookup cache
def set_active_landmark_lookup(seed, landmark_id_by_feat, *, context: str = "active_lookup"):
    seed = _check_seed_dict(seed)
    active_kf = _get_canonical_active_keyframe_kf(seed)
    record = get_keyframe_record(seed, int(active_kf), context=str(context))
    record["landmark_id_by_feat"] = _copy_landmark_lookup(landmark_id_by_feat, name=str(context))
    sync_legacy_active_fields_from_canonical(seed)
    return get_keyframe_record(seed, int(active_kf), context=str(context))["landmark_id_by_feat"]


# Rebuild active lookup cache from landmark observations
def rebuild_active_landmark_lookup(seed, *, context: str = "active_lookup") -> np.ndarray:
    seed = _check_seed_dict(seed)
    if "active_keyframe_kf" not in seed:
        sync_active_keyframe_mirrors_if_present(seed)
    active_kf = get_active_keyframe_kf(seed)
    feats = get_active_keyframe_features(seed)
    n_feat = _checked_feature_count(feats, name=f"{context} active features")
    lookup = build_landmark_lookup_for_kf(seed, n_feat, int(active_kf), context=str(context))
    return set_active_landmark_lookup(seed, lookup, context=str(context))


# Rebuild and read the active lookup cache for runtime consumers
def get_rebuilt_active_landmark_lookup(seed, *, context: str = "active_lookup") -> np.ndarray:
    return rebuild_active_landmark_lookup(seed, context=str(context))


# Store an accepted current-frame pose
def store_current_pose(seed, current_kf: int, R_cur, t_cur) -> dict:
    seed = _check_seed_dict(seed)
    current_kf = check_int_ge0_no_bool(current_kf, name="current_kf")

    try:
        active_kf = get_active_keyframe_kf(seed)
    except ValueError:
        active_kf = None

    pose = (
        np.asarray(R_cur, dtype=np.float64).copy(),
        np.asarray(t_cur, dtype=np.float64).reshape(3).copy(),
    )
    set_pose_for_kf(seed, int(current_kf), pose, copy=True, context="current pose")

    if active_kf is not None and int(active_kf) == int(current_kf):
        sync_legacy_active_fields_from_canonical(seed)
        return seed

    return seed


# Validate canonical keyframe state through invariants
def validate_keyframe_store(seed, *, context="keyframe_store") -> dict:
    from slam.invariants import audit_seed_invariants

    return audit_seed_invariants(seed, context=str(context))


# Validate the active keyframe mirrors through invariants
def validate_active_keyframe_state(seed, *, context="active_keyframe") -> dict:
    from slam.invariants import audit_active_keyframe_lookup

    return audit_active_keyframe_lookup(seed, context=str(context))
