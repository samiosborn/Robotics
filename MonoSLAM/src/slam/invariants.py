# src/slam/invariants.py
from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import check_array, check_dict, check_int_ge0_no_bool, check_matrix_3x3, check_points_xy_N2plus, check_vector_3
from slam.landmark_state import feature_assignment_key, observation_key


_POSE_FIELDS = ("T_WC0", "T_WC1", "last_accepted_pose")


# Create a fresh invariant audit report
def _new_report(context: str) -> dict[str, Any]:
    return {
        "context": str(context),
        "num_landmarks": 0,
        "num_observations": 0,
        "num_poses": 0,
        "num_keyframes": 0,
        "num_active_lookup_entries": 0,
        "num_duplicate_observations": 0,
        "num_feature_assignment_conflicts": 0,
        "errors": [],
        "warnings": [],
    }


# Add one audit error
def _add_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(str(message))


# Raise when strict auditing fails
def _raise_if_needed(report: dict[str, Any], strict: bool) -> None:
    if bool(strict) and len(report["errors"]) > 0:
        context = str(report.get("context", "seed"))
        raise ValueError(f"{context} invariant audit failed: " + "; ".join(report["errors"]))


# Format a phase-specific missing-key error
def _missing_key_message(name: str, key: str, phase: str) -> str:
    return f"{name} is missing required key '{key}' for {phase}"


# Check one required key with phase context
def _check_required_key(d: dict, key: str, name: str, phase: str) -> None:
    if key not in d:
        raise ValueError(_missing_key_message(name, key, phase))


# Check a homogeneous pose matrix
def _check_pose_matrix(T, name: str) -> np.ndarray:
    try:
        arr_raw = np.asarray(T, dtype=np.float64)
    except Exception:
        raise ValueError(f"{name} must be convertible to a pose matrix with shape (4,4)") from None
    arr = check_array(arr_raw, name=name, shape=(4, 4), dtype=float, finite=True)

    # Check homogeneous pose bottom row
    bottom = arr[3, :]
    expected = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(bottom, expected):
        raise ValueError(f"{name} bottom row must be close to [0, 0, 0, 1]; got {bottom.tolist()}")

    check_matrix_3x3(arr[:3, :3], name=f"{name} rotation block", dtype=float, finite=True)

    return arr


# Build a homogeneous pose matrix
def _pose_from_rt(R, t, name: str) -> np.ndarray:
    R_arr = check_matrix_3x3(R, name=f"{name} rotation block", dtype=float, finite=True)
    t_arr = check_vector_3(t, name=f"{name} translation block", dtype=float, finite=True)

    # Build homogeneous matrix from rotation and translation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_arr
    T[:3, 3] = t_arr

    return _check_pose_matrix(T, name)


# Check a pose in current project conventions
def check_pose_T_WC(pose, name: str = "pose") -> np.ndarray:
    if isinstance(pose, dict) and ("R" in pose or "t" in pose):
        _check_required_key(pose, "R", name, "pose validation")
        _check_required_key(pose, "t", name, "pose validation")
        return _pose_from_rt(pose["R"], pose["t"], name)

    if isinstance(pose, (tuple, list)) and len(pose) == 2:
        return _pose_from_rt(pose[0], pose[1], name)

    return _check_pose_matrix(pose, name)


# Check one landmark observation record
def check_observation_record(observation, *, name: str = "observation") -> tuple[int, int, np.ndarray]:
    observation = check_dict(observation, name=name)

    for key in ("kf", "feat", "xy"):
        _check_required_key(observation, key, name, "observation validation")

    kf = check_int_ge0_no_bool(observation["kf"], name=f"{name}['kf']")
    feat = check_int_ge0_no_bool(observation["feat"], name=f"{name}['feat']")
    xy = check_array(observation["xy"], name=f"{name}['xy']", shape=(2,), dtype=float, finite=True)

    return kf, feat, xy


# Check one landmark record
def check_landmark_record(landmark, *, name: str = "landmark") -> tuple[int, np.ndarray]:
    landmark = check_dict(landmark, name=name)

    for key in ("id", "X_w"):
        _check_required_key(landmark, key, name, "landmark validation")

    landmark_id = check_int_ge0_no_bool(landmark["id"], name=f"{name}['id']")
    X_w = check_array(landmark["X_w"], name=f"{name}['X_w']", shape=(3,), dtype=float, finite=True)

    return landmark_id, X_w


# Read feature keypoints from active features
def _feature_keypoints_array(feats, name: str) -> np.ndarray:
    if hasattr(feats, "kps_xy"):
        arr_name = f"{name}.kps_xy"
        value = getattr(feats, "kps_xy")
    else:
        arr_name = name
        value = feats

    return check_points_xy_N2plus(value, name=arr_name, dtype=float, finite=True)


# Read active lookup entries
def _active_lookup_entries(lookup, name: str) -> list[tuple[int, int]]:
    if isinstance(lookup, dict):
        entries: list[tuple[int, int]] = []
        for feat_raw, landmark_raw in lookup.items():
            feat = check_int_ge0_no_bool(feat_raw, name=f"{name} feature index")
            landmark_id = check_int_ge0_no_bool(landmark_raw, name=f"{name}[{feat}]")
            entries.append((feat, landmark_id))
        return entries

    try:
        arr_raw = np.asarray(lookup)
    except Exception:
        raise ValueError(f"{name} must be a dict or 1D integer array with -1 sentinel values") from None

    if arr_raw.ndim != 1:
        raise ValueError(f"{name} must be a dict or 1D integer array; got shape {arr_raw.shape}")
    if arr_raw.size > 0 and arr_raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integer landmark ids; got dtype {arr_raw.dtype}")

    arr = np.asarray(arr_raw, dtype=np.int64).reshape(-1)
    if np.any(arr < -1):
        raise ValueError(f"{name} must contain landmark ids >= -1; got min={int(np.min(arr))}")

    entries = []
    for feat, landmark_id in enumerate(arr):
        if int(landmark_id) >= 0:
            entries.append((int(feat), int(landmark_id)))

    return entries


# Compare two accepted pose representations
def _poses_agree(pose_a, pose_b, name_a: str, name_b: str) -> bool:
    T_a = check_pose_T_WC(pose_a, name=name_a)
    T_b = check_pose_T_WC(pose_b, name=name_b)
    return bool(np.allclose(T_a, T_b))


# Convert a lookup to an integer array when possible
def _lookup_array_or_none(lookup, name: str) -> np.ndarray | None:
    if isinstance(lookup, dict):
        return None

    try:
        arr_raw = np.asarray(lookup)
    except Exception:
        raise ValueError(f"{name} must be a dict or 1D integer array with -1 sentinel values") from None

    if arr_raw.ndim != 1:
        raise ValueError(f"{name} must be a dict or 1D integer array; got shape {arr_raw.shape}")
    if arr_raw.size > 0 and arr_raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integer landmark ids; got dtype {arr_raw.dtype}")

    arr = np.asarray(arr_raw, dtype=np.int64).reshape(-1)
    if np.any(arr < -1):
        raise ValueError(f"{name} must contain landmark ids >= -1; got min={int(np.min(arr))}")

    return arr


# Compare two keyframe lookup structures
def _lookups_agree(lookup_a, lookup_b, name_a: str, name_b: str) -> bool:
    arr_a = _lookup_array_or_none(lookup_a, name_a)
    arr_b = _lookup_array_or_none(lookup_b, name_b)
    if arr_a is not None and arr_b is not None:
        return bool(arr_a.shape == arr_b.shape and np.array_equal(arr_a, arr_b))

    entries_a = sorted(_active_lookup_entries(lookup_a, name_a))
    entries_b = sorted(_active_lookup_entries(lookup_b, name_b))
    return entries_a == entries_b


# Compare two feature containers by keypoint rows
def _features_agree(feats_a, feats_b, name_a: str, name_b: str) -> bool:
    kps_a = _feature_keypoints_array(feats_a, name_a)
    kps_b = _feature_keypoints_array(feats_b, name_b)
    return bool(kps_a.shape == kps_b.shape and np.allclose(kps_a, kps_b))


# Audit the canonical pose store
def _audit_pose_store(seed: dict[str, Any], report: dict[str, Any], context: str) -> dict[int, Any]:
    if "poses" not in seed:
        return {}

    poses = seed["poses"]
    if not isinstance(poses, dict):
        _add_error(report, f"{context}['poses'] must be a dict")
        return {}

    report["num_poses"] = int(len(poses))
    checked: dict[int, Any] = {}
    for key_raw, pose in poses.items():
        pose_name = f"{context}['poses'][{key_raw!r}]"
        try:
            pose_key = check_int_ge0_no_bool(key_raw, name=f"{context}['poses'] key")
            check_pose_T_WC(pose, name=pose_name)
        except ValueError as exc:
            _add_error(report, str(exc))
            continue

        checked[int(pose_key)] = pose

    return checked


# Audit one keyframe lookup against feature bounds
def _audit_keyframe_lookup_entries(
    entries: list[tuple[int, int]],
    n_feats: int | None,
    report: dict[str, Any],
    lookup_name: str,
) -> None:
    if n_feats is None:
        return

    for feat, _ in entries:
        if int(feat) >= int(n_feats):
            _add_error(report, f"{lookup_name} feature index {feat} must be within keyframe feature length {n_feats}")


# Audit the canonical keyframe store
def _audit_keyframe_store(
    seed: dict[str, Any],
    report: dict[str, Any],
    context: str,
    poses: dict[int, Any],
) -> dict[int, dict]:
    if "keyframes" not in seed:
        return {}

    keyframes = seed["keyframes"]
    if not isinstance(keyframes, dict):
        _add_error(report, f"{context}['keyframes'] must be a dict")
        return {}

    report["num_keyframes"] = int(len(keyframes))
    checked: dict[int, dict] = {}
    for key_raw, record in keyframes.items():
        record_name = f"{context}['keyframes'][{key_raw!r}]"
        try:
            kf = check_int_ge0_no_bool(key_raw, name=f"{context}['keyframes'] key")
        except ValueError as exc:
            _add_error(report, str(exc))
            continue

        if not isinstance(record, dict):
            _add_error(report, f"{record_name} must be a dict")
            continue

        if "kf" not in record:
            _add_error(report, _missing_key_message(record_name, "kf", "keyframe validation"))
        else:
            try:
                record_kf = check_int_ge0_no_bool(record["kf"], name=f"{record_name}['kf']")
                if int(record_kf) != int(kf):
                    _add_error(report, f"{record_name}['kf'] must match dict key {kf}; got {record_kf}")
            except ValueError as exc:
                _add_error(report, str(exc))

        if "pose" in record:
            try:
                check_pose_T_WC(record["pose"], name=f"{record_name}['pose']")
                if int(kf) in poses and not _poses_agree(
                    record["pose"],
                    poses[int(kf)],
                    f"{record_name}['pose']",
                    f"{context}['poses'][{kf}]",
                ):
                    _add_error(report, f"{record_name}['pose'] must match {context}['poses'][{kf}]")
            except ValueError as exc:
                _add_error(report, str(exc))
        elif int(kf) in poses:
            try:
                check_pose_T_WC(poses[int(kf)], name=f"{context}['poses'][{kf}]")
            except ValueError as exc:
                _add_error(report, str(exc))
        else:
            _add_error(report, _missing_key_message(record_name, "pose", "keyframe validation"))

        n_feats = None
        if "feats" not in record:
            _add_error(report, _missing_key_message(record_name, "feats", "keyframe validation"))
        else:
            try:
                feats = _feature_keypoints_array(record["feats"], f"{record_name}['feats']")
                n_feats = int(feats.shape[0])
            except ValueError as exc:
                _add_error(report, str(exc))

        if "landmark_id_by_feat" not in record:
            _add_error(report, _missing_key_message(record_name, "landmark_id_by_feat", "keyframe validation"))
        else:
            try:
                entries = _active_lookup_entries(
                    record["landmark_id_by_feat"],
                    f"{record_name}['landmark_id_by_feat']",
                )
                _audit_keyframe_lookup_entries(
                    entries,
                    n_feats,
                    report,
                    f"{record_name}['landmark_id_by_feat']",
                )
            except ValueError as exc:
                _add_error(report, str(exc))

        checked[int(kf)] = record

    return checked


# Audit canonical active keyframe compatibility
def _audit_active_keyframe_store(
    seed: dict[str, Any],
    report: dict[str, Any],
    context: str,
    poses: dict[int, Any],
    keyframes: dict[int, dict],
) -> None:
    if "active_keyframe_kf" not in seed:
        return

    try:
        active_kf = check_int_ge0_no_bool(seed["active_keyframe_kf"], name=f"{context}['active_keyframe_kf']")
    except ValueError as exc:
        _add_error(report, str(exc))
        return

    if "keyframe_kf" in seed:
        try:
            keyframe_kf = check_int_ge0_no_bool(seed["keyframe_kf"], name=f"{context}['keyframe_kf']")
            if int(active_kf) != int(keyframe_kf):
                _add_error(report, f"{context}['active_keyframe_kf'] must match {context}['keyframe_kf']")
        except ValueError as exc:
            _add_error(report, str(exc))

    if "keyframes" not in seed:
        _add_error(report, f"{context}['active_keyframe_kf'] requires {context}['keyframes']")
        return

    if int(active_kf) not in keyframes:
        _add_error(report, f"{context}['active_keyframe_kf']={active_kf} must point to an existing keyframe")
        return

    record = keyframes[int(active_kf)]
    record_name = f"{context}['keyframes'][{active_kf}]"
    if not isinstance(record, dict):
        return

    if "poses" in seed:
        if int(active_kf) not in poses:
            _add_error(report, f"{context}['poses'] must contain active keyframe {active_kf}")
        elif "T_WC1" in seed:
            try:
                if not _poses_agree(
                    poses[int(active_kf)],
                    seed["T_WC1"],
                    f"{context}['poses'][{active_kf}]",
                    f"{context}['T_WC1']",
                ):
                    _add_error(report, f"{context}['poses'][{active_kf}] must match {context}['T_WC1']")
            except ValueError as exc:
                _add_error(report, str(exc))

    if "pose" in record and "T_WC1" in seed:
        try:
            if not _poses_agree(record["pose"], seed["T_WC1"], f"{record_name}['pose']", f"{context}['T_WC1']"):
                _add_error(report, f"active keyframe record pose must match {context}['T_WC1']")
        except ValueError as exc:
            _add_error(report, str(exc))

    if "feats" in record and "feats1" in seed:
        try:
            if not _features_agree(record["feats"], seed["feats1"], f"{record_name}['feats']", f"{context}['feats1']"):
                _add_error(report, f"active keyframe record feats must match {context}['feats1']")
        except ValueError as exc:
            _add_error(report, str(exc))

    if "landmark_id_by_feat" in record and "landmark_id_by_feat1" in seed:
        try:
            if not _lookups_agree(
                record["landmark_id_by_feat"],
                seed["landmark_id_by_feat1"],
                f"{record_name}['landmark_id_by_feat']",
                f"{context}['landmark_id_by_feat1']",
            ):
                _add_error(report, f"active keyframe record lookup must match {context}['landmark_id_by_feat1']")
        except ValueError as exc:
            _add_error(report, str(exc))


# Audit landmark records and observations
def _audit_landmarks(seed: dict[str, Any], report: dict[str, Any], context: str) -> dict[str, Any]:
    landmark_ids: set[int] = set()
    assignment_by_kf_feat: dict[tuple[int, int], int] = {}
    observation_keys: set[tuple[int, int, int]] = set()

    if "landmarks" not in seed:
        return {
            "landmark_ids": landmark_ids,
            "assignment_by_kf_feat": assignment_by_kf_feat,
            "has_landmarks": False,
        }

    landmarks = seed["landmarks"]
    if not isinstance(landmarks, list):
        _add_error(report, f"{context}['landmarks'] must be a list")
        return {
            "landmark_ids": landmark_ids,
            "assignment_by_kf_feat": assignment_by_kf_feat,
            "has_landmarks": True,
        }

    report["num_landmarks"] = int(len(landmarks))

    for i, landmark in enumerate(landmarks):
        lm_name = f"{context}['landmarks'][{i}]"
        try:
            landmark_id, _ = check_landmark_record(landmark, name=lm_name)
        except ValueError as exc:
            _add_error(report, str(exc))
            continue

        if landmark_id in landmark_ids:
            _add_error(report, f"{lm_name} duplicate landmark id {landmark_id}")
        landmark_ids.add(landmark_id)

        obs = landmark.get("obs", None)
        if obs is None:
            continue
        if not isinstance(obs, list):
            _add_error(report, f"{lm_name}['obs'] must be a list when present")
            continue

        # Track duplicate observations and feature assignments
        for j, observation in enumerate(obs):
            ob_name = f"{lm_name}['obs'][{j}]"
            try:
                kf, feat, _ = check_observation_record(observation, name=ob_name)
            except ValueError as exc:
                _add_error(report, str(exc))
                continue

            report["num_observations"] += 1
            obs_key = observation_key(landmark_id, kf, feat)
            if obs_key in observation_keys:
                report["num_duplicate_observations"] += 1
                _add_error(
                    report,
                    f"{ob_name} duplicates observation for landmark id {landmark_id}, kf {kf}, feat {feat}",
                )
            else:
                observation_keys.add(obs_key)

            feature_key = feature_assignment_key(kf, feat)
            previous_landmark_id = assignment_by_kf_feat.get(feature_key, None)
            if previous_landmark_id is None:
                assignment_by_kf_feat[feature_key] = int(landmark_id)
            elif int(previous_landmark_id) != int(landmark_id):
                report["num_feature_assignment_conflicts"] += 1
                _add_error(
                    report,
                    f"{ob_name} maps keyframe-feature pair (kf={kf}, feat={feat}) to landmark id {landmark_id}, already assigned to landmark id {previous_landmark_id}",
                )

    return {
        "landmark_ids": landmark_ids,
        "assignment_by_kf_feat": assignment_by_kf_feat,
        "has_landmarks": True,
    }


# Audit active keyframe lookup consistency
def _audit_active_keyframe(seed: dict[str, Any], report: dict[str, Any], context: str, landmark_info: dict[str, Any]) -> None:
    keyframe_kf = None
    if "keyframe_kf" in seed:
        try:
            keyframe_kf = check_int_ge0_no_bool(seed["keyframe_kf"], name=f"{context}['keyframe_kf']")
        except ValueError as exc:
            _add_error(report, str(exc))

        if "T_WC1" not in seed:
            _add_error(report, _missing_key_message(context, "T_WC1", "active keyframe validation"))

    n_feats = None
    if "feats1" in seed:
        try:
            feats1 = _feature_keypoints_array(seed["feats1"], f"{context}['feats1']")
            n_feats = int(feats1.shape[0])
        except ValueError as exc:
            _add_error(report, str(exc))

    if "landmark_id_by_feat1" not in seed:
        return

    try:
        entries = _active_lookup_entries(seed["landmark_id_by_feat1"], f"{context}['landmark_id_by_feat1']")
    except ValueError as exc:
        _add_error(report, str(exc))
        return

    report["num_active_lookup_entries"] = int(len(entries))
    landmark_ids = landmark_info["landmark_ids"]
    assignment_by_kf_feat = landmark_info["assignment_by_kf_feat"]
    has_landmarks = bool(landmark_info["has_landmarks"])

    for feat, landmark_id in entries:
        if n_feats is not None and int(feat) >= int(n_feats):
            _add_error(
                report,
                f"{context}['landmark_id_by_feat1'] feature index {feat} must be within seed['feats1'] length {n_feats}",
            )

        if bool(has_landmarks) and int(landmark_id) not in landmark_ids:
            _add_error(
                report,
                f"{context}['landmark_id_by_feat1'][{feat}] references missing landmark id {landmark_id}",
            )

        if keyframe_kf is None:
            continue

        observed_landmark_id = assignment_by_kf_feat.get((int(keyframe_kf), int(feat)), None)
        if observed_landmark_id is None:
            _add_error(
                report,
                f"{context}['landmark_id_by_feat1'][{feat}] maps to landmark id {landmark_id}, but no observation exists for active keyframe-feature pair (kf={keyframe_kf}, feat={feat})",
            )
        elif int(observed_landmark_id) != int(landmark_id):
            _add_error(
                report,
                f"{context}['landmark_id_by_feat1'][{feat}] maps to landmark id {landmark_id}, but observation (kf={keyframe_kf}, feat={feat}) maps to landmark id {observed_landmark_id}",
            )


# Audit only landmark invariants
def audit_landmarks(seed, *, context: str = "seed", strict: bool = True) -> dict[str, Any]:
    report = _new_report(context)
    try:
        seed = check_dict(seed, name=context)
    except ValueError as exc:
        _add_error(report, str(exc))
        _raise_if_needed(report, strict)
        return report

    _audit_landmarks(seed, report, context)
    _raise_if_needed(report, strict)
    return report


# Audit active keyframe lookup invariants
def audit_active_keyframe_lookup(seed, *, context: str = "seed", strict: bool = True) -> dict[str, Any]:
    report = _new_report(context)
    try:
        seed = check_dict(seed, name=context)
    except ValueError as exc:
        _add_error(report, str(exc))
        _raise_if_needed(report, strict)
        return report

    poses = _audit_pose_store(seed, report, context)
    keyframes = _audit_keyframe_store(seed, report, context, poses)
    _audit_active_keyframe_store(seed, report, context, poses, keyframes)

    landmark_info = _audit_landmarks(seed, report, context)
    _audit_active_keyframe(seed, report, context, landmark_info)
    _raise_if_needed(report, strict)
    return report


# Audit the current mutable seed invariants
def audit_seed_invariants(seed, *, context: str = "seed", strict: bool = True) -> dict[str, Any]:
    report = _new_report(context)
    try:
        seed = check_dict(seed, name=context)
    except ValueError as exc:
        _add_error(report, str(exc))
        _raise_if_needed(report, strict)
        return report

    for field in _POSE_FIELDS:
        if field not in seed:
            continue

        try:
            check_pose_T_WC(seed[field], name=f"{context}['{field}']")
        except ValueError as exc:
            _add_error(report, str(exc))

    poses = _audit_pose_store(seed, report, context)
    keyframes = _audit_keyframe_store(seed, report, context, poses)
    _audit_active_keyframe_store(seed, report, context, poses, keyframes)

    landmark_info = _audit_landmarks(seed, report, context)
    _audit_active_keyframe(seed, report, context, landmark_info)

    _raise_if_needed(report, strict)
    return report
