# src/slam/landmark_state.py
from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import check_array, check_dict, check_int_ge0_no_bool


# Check a mutable seed container
def _check_seed_dict(seed) -> dict:
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    return seed


# Check one landmark id value
def _check_landmark_id(landmark_id, *, name: str = "landmark_id") -> int:
    return int(check_int_ge0_no_bool(landmark_id, name=name))


# Check one landmark record id
def _landmark_id_from_record(landmark, *, name: str = "landmark") -> int:
    landmark = check_dict(landmark, name=name)
    if "id" not in landmark:
        raise ValueError(f"{name} is missing required key 'id'")
    return _check_landmark_id(landmark["id"], name=f"{name}['id']")


# Copy one observation image point
def _copy_observation_xy(xy, *, name: str = "xy") -> np.ndarray:
    arr = check_array(xy, name=name, dtype=float, finite=False)
    if int(arr.size) != 2:
        raise ValueError(f"{name} must have exactly 2 values; got shape {arr.shape}")
    arr = np.asarray(arr, dtype=np.float64).reshape(2,)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr.copy()


# Check one observation record
def _check_observation_record(observation, *, name: str = "observation") -> tuple[int, int, np.ndarray]:
    observation = check_dict(observation, name=name)
    for key in ("kf", "feat", "xy"):
        if key not in observation:
            raise ValueError(f"{name} is missing required key '{key}'")

    kf = check_int_ge0_no_bool(observation["kf"], name=f"{name}['kf']")
    feat = check_int_ge0_no_bool(observation["feat"], name=f"{name}['feat']")
    xy = _copy_observation_xy(observation["xy"], name=f"{name}['xy']")

    return int(kf), int(feat), xy


# Read the mutable landmark list
def get_landmarks(seed) -> list:
    seed = _check_seed_dict(seed)
    landmarks = seed.get("landmarks", [])
    if not isinstance(landmarks, list):
        raise ValueError("seed['landmarks'] must be a list")
    return landmarks


# Iterate over landmark records
def iter_landmarks(seed):
    for landmark in get_landmarks(seed):
        yield landmark


# Build a disposable landmark id index
def build_landmark_id_index(seed, *, context: str = "landmarks") -> dict[int, dict]:
    landmarks = get_landmarks(seed)
    index: dict[int, dict] = {}

    for i, landmark in enumerate(landmarks):
        landmark_id = _landmark_id_from_record(landmark, name=f"{context}[{i}]")
        if int(landmark_id) in index:
            raise ValueError(f"{context}[{i}] duplicate landmark id {landmark_id}")
        index[int(landmark_id)] = landmark

    return index


# Read one landmark by id
def get_landmark_by_id(seed, landmark_id, *, context: str = "landmarks") -> dict:
    landmark_id = _check_landmark_id(landmark_id, name="landmark_id")
    index = build_landmark_id_index(seed, context=str(context))
    landmark = index.get(int(landmark_id), None)
    if landmark is None:
        raise ValueError(f"{context} missing landmark id {landmark_id}")
    return landmark


# Check whether a landmark id exists
def has_landmark_id(seed, landmark_id) -> bool:
    landmark_id = _check_landmark_id(landmark_id, name="landmark_id")
    return int(landmark_id) in build_landmark_id_index(seed)


# Allocate the next monotonically increasing landmark id
def next_landmark_id(seed) -> int:
    index = build_landmark_id_index(seed)
    if len(index) == 0:
        return 0
    return int(max(index.keys()) + 1)


# Build the unique observation key
def observation_key(landmark_id, kf, feat) -> tuple[int, int, int]:
    landmark_id = _check_landmark_id(landmark_id, name="landmark_id")
    kf = check_int_ge0_no_bool(kf, name="kf")
    feat = check_int_ge0_no_bool(feat, name="feat")
    return int(landmark_id), int(kf), int(feat)


# Build the feature-assignment key
def feature_assignment_key(kf, feat) -> tuple[int, int]:
    kf = check_int_ge0_no_bool(kf, name="kf")
    feat = check_int_ge0_no_bool(feat, name="feat")
    return int(kf), int(feat)


# Iterate over checked landmark observations
def iter_landmark_observations(seed, *, context: str = "landmarks"):
    for i, landmark in enumerate(get_landmarks(seed)):
        lm_name = f"{context}[{i}]"
        landmark_id = _landmark_id_from_record(landmark, name=lm_name)

        obs = landmark.get("obs", None)
        if obs is None:
            continue
        if not isinstance(obs, list):
            raise ValueError(f"{lm_name}['obs'] must be a list when present")

        for j, observation in enumerate(obs):
            kf, feat, xy = _check_observation_record(observation, name=f"{lm_name}['obs'][{j}]")
            yield int(landmark_id), int(kf), int(feat), xy


# Build disposable observation and feature indexes
def build_observation_indexes(seed, *, context: str = "observations") -> dict[str, dict]:
    landmark_by_id = build_landmark_id_index(seed, context=str(context))
    observation_by_key: dict[tuple[int, int, int], dict] = {}
    landmark_id_by_feature: dict[tuple[int, int], int] = {}

    for i, landmark in enumerate(get_landmarks(seed)):
        lm_name = f"{context}[{i}]"
        landmark_id = _landmark_id_from_record(landmark, name=lm_name)

        obs = landmark.get("obs", None)
        if obs is None:
            continue
        if not isinstance(obs, list):
            raise ValueError(f"{lm_name}['obs'] must be a list when present")

        for j, observation in enumerate(obs):
            ob_name = f"{lm_name}['obs'][{j}]"
            kf, feat, _ = _check_observation_record(observation, name=ob_name)
            obs_key = observation_key(landmark_id, kf, feat)
            if obs_key in observation_by_key:
                raise ValueError(
                    f"{ob_name} duplicates observation for landmark id {landmark_id}, kf {kf}, feat {feat}"
                )

            feat_key = feature_assignment_key(kf, feat)
            previous_landmark_id = landmark_id_by_feature.get(feat_key, None)
            if previous_landmark_id is not None and int(previous_landmark_id) != int(landmark_id):
                raise ValueError(
                    f"{ob_name} feature assignment conflict for kf {kf}, feat {feat}: "
                    f"landmark id {previous_landmark_id} and {landmark_id}"
                )

            observation_by_key[obs_key] = observation
            landmark_id_by_feature[feat_key] = int(landmark_id)

    return {
        "landmark_by_id": landmark_by_id,
        "observation_by_key": observation_by_key,
        "landmark_id_by_feature": landmark_id_by_feature,
    }


# Check whether a landmark already has one observation
def landmark_has_observation(landmark, kf, feat) -> bool:
    landmark = check_dict(landmark, name="landmark")
    kf = check_int_ge0_no_bool(kf, name="kf")
    feat = check_int_ge0_no_bool(feat, name="feat")

    obs = landmark.get("obs", None)
    if not isinstance(obs, list):
        return False

    for observation in obs:
        if not isinstance(observation, dict):
            continue
        if int(observation.get("kf", -1)) != int(kf):
            continue
        if int(observation.get("feat", -1)) != int(feat):
            continue
        return True

    return False


# Add one duplicate-safe landmark observation
def add_landmark_observation(
    landmark,
    kf,
    feat,
    xy,
    *,
    assignment_by_feature: dict[tuple[int, int], int] | None = None,
    context: str = "observation",
) -> bool:
    landmark = check_dict(landmark, name="landmark")
    landmark_id = _landmark_id_from_record(landmark)
    kf = check_int_ge0_no_bool(kf, name="kf")
    feat = check_int_ge0_no_bool(feat, name="feat")
    xy_copy = _copy_observation_xy(xy, name=f"{context}['xy']")

    obs = landmark.get("obs", None)
    if not isinstance(obs, list):
        obs = []

    feat_key = feature_assignment_key(kf, feat)
    if assignment_by_feature is not None:
        previous_landmark_id = assignment_by_feature.get(feat_key, None)
        if previous_landmark_id is not None and int(previous_landmark_id) != int(landmark_id):
            raise ValueError(
                f"{context} feature assignment conflict for kf {int(kf)}, feat {int(feat)}: "
                f"landmark id {previous_landmark_id} and {landmark_id}"
            )

    if landmark_has_observation(landmark, int(kf), int(feat)):
        landmark["obs"] = obs
        if assignment_by_feature is not None:
            assignment_by_feature[feat_key] = int(landmark_id)
        return False

    obs.append(
        {
            "kf": int(kf),
            "feat": int(feat),
            "xy": xy_copy,
        }
    )
    landmark["obs"] = obs
    if assignment_by_feature is not None:
        assignment_by_feature[feat_key] = int(landmark_id)

    return True


# Count dict observation entries on one landmark
def count_landmark_observations(landmark) -> int:
    landmark = check_dict(landmark, name="landmark")
    obs = landmark.get("obs", None)
    if not isinstance(obs, list):
        return 0
    return int(sum(1 for observation in obs if isinstance(observation, dict)))


# Count checked observations on one landmark
def count_valid_landmark_observations(landmark, *, context: str = "landmark") -> int:
    landmark = check_dict(landmark, name=context)
    obs = landmark.get("obs", None)
    if obs is None:
        return 0
    if not isinstance(obs, list):
        raise ValueError(f"{context}['obs'] must be a list when present")

    n_obs = 0
    for j, observation in enumerate(obs):
        _check_observation_record(observation, name=f"{context}['obs'][{j}]")
        n_obs += 1

    return int(n_obs)


# Build a mutable landmark record
def create_landmark_record(
    landmark_id,
    X_w,
    *,
    birth_source: str | None = None,
    birth_kf=None,
    descriptor=None,
    quality: dict[str, Any] | None = None,
    context: str = "landmark",
) -> dict:
    landmark_id = _check_landmark_id(landmark_id, name=f"{context}['id']")
    X_w = check_array(X_w, name=f"{context}['X_w']", shape=(3,), dtype=float, finite=True)

    landmark = {
        "id": int(landmark_id),
        "X_w": np.asarray(X_w, dtype=np.float64).reshape(3,).copy(),
    }
    if birth_source is not None:
        landmark["birth_source"] = str(birth_source)
    if birth_kf is not None:
        landmark["birth_kf"] = int(check_int_ge0_no_bool(birth_kf, name=f"{context}['birth_kf']"))

    landmark["obs"] = []
    landmark["descriptor"] = descriptor
    landmark["quality"] = {} if quality is None else dict(quality)

    return landmark
