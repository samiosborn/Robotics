# tests/slam/test_keyframe_state.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from slam.invariants import audit_seed_invariants
from slam.frame_pipeline import _copy_pose_blocks, process_frame_against_seed
from slam.keyframe import promote_frame_to_keyframe
from slam.keyframe_state import (
    get_active_keyframe_features,
    get_active_keyframe_pose,
    get_active_keyframe_record,
    get_active_landmark_lookup,
    initialise_canonical_keyframe_state,
    set_active_keyframe_record,
    sync_active_keyframe_mirrors,
    validate_active_keyframe_state,
)


# Build a feature stub with keypoints
def _features(kps_xy):
    return SimpleNamespace(kps_xy=np.asarray(kps_xy, dtype=np.float64))


# Build a project pose tuple
def _pose(tx: float = 0.0):
    return np.eye(3, dtype=np.float64), np.asarray([tx, 0.0, 0.0], dtype=np.float64)


# Build a bootstrap-style seed with legacy fields
def _bootstrap_seed():
    return {
        "T_WC0": _pose(0.0),
        "T_WC1": _pose(1.0),
        "landmarks": [
            {
                "id": 0,
                "X_w": np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
                "obs": [
                    {"kf": 0, "feat": 0, "xy": np.asarray([10.0, 20.0], dtype=np.float64)},
                    {"kf": 1, "feat": 1, "xy": np.asarray([30.0, 40.0], dtype=np.float64)},
                ],
            },
            {
                "id": 1,
                "X_w": np.asarray([1.0, 0.0, 5.0], dtype=np.float64),
                "obs": [
                    {"kf": 0, "feat": 1, "xy": np.asarray([50.0, 60.0], dtype=np.float64)},
                    {"kf": 1, "feat": 2, "xy": np.asarray([70.0, 80.0], dtype=np.float64)},
                ],
            },
        ],
        "feats0": _features([[10.0, 20.0], [50.0, 60.0]]),
        "feats1": _features([[15.0, 25.0], [30.0, 40.0], [70.0, 80.0]]),
        "landmark_id_by_feat1": np.asarray([-1, 0, 1], dtype=np.int64),
    }


# Initialise canonical stores from a bootstrap-style seed
def test_bootstrap_seed_initialises_canonical_pose_and_keyframe_stores():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())

    assert seed["keyframe_kf"] == 1
    assert seed["active_keyframe_kf"] == 1
    assert seed["poses"][0] is seed["T_WC0"]
    assert seed["poses"][1] is seed["T_WC1"]
    assert seed["keyframes"][0]["kf"] == 0
    assert seed["keyframes"][1]["kf"] == 1
    assert seed["keyframes"][1]["pose"] is seed["T_WC1"]

    report = audit_seed_invariants(seed)
    assert report["errors"] == []


# Read the active canonical record as a compatibility mirror
def test_active_keyframe_record_mirrors_legacy_active_fields():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())
    record = get_active_keyframe_record(seed)

    assert record["kf"] == seed["keyframe_kf"]
    assert record["pose"] is seed["T_WC1"]
    assert record["feats"] is seed["feats1"]
    assert record["landmark_id_by_feat"] is seed["landmark_id_by_feat1"]


# Read active fields through the explicit helpers
def test_active_keyframe_field_helpers_return_mirrors():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())

    assert get_active_keyframe_pose(seed) is seed["T_WC1"]
    assert get_active_keyframe_features(seed) is seed["feats1"]
    assert get_active_landmark_lookup(seed) is seed["landmark_id_by_feat1"]


# Reject missing active keyframes with a clear error
def test_missing_active_keyframe_fails_clearly():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())
    seed["active_keyframe_kf"] = 7
    seed["keyframe_kf"] = 7

    with pytest.raises(ValueError, match="missing from seed\\['keyframes'\\]"):
        get_active_keyframe_pose(seed)


# Sync legacy active updates back into canonical mirrors
def test_sync_active_keyframe_mirrors_repairs_after_legacy_update():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())
    pose = _pose(3.0)
    feats = _features([[1.0, 2.0], [3.0, 4.0]])
    lookup = np.asarray([4, -1], dtype=np.int64)

    seed["T_WC1"] = pose
    seed["feats1"] = feats
    seed["landmark_id_by_feat1"] = lookup
    seed["keyframe_kf"] = 3
    sync_active_keyframe_mirrors(seed)
    record = get_active_keyframe_record(seed)

    assert seed["active_keyframe_kf"] == 3
    assert seed["poses"][3] is pose
    assert record["pose"] is pose
    assert record["feats"] is feats
    assert record["landmark_id_by_feat"] is lookup


# Validate active mirror mismatches without repairing them
def test_validate_active_keyframe_state_catches_lookup_mismatch():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())
    seed["keyframes"][1]["landmark_id_by_feat"] = np.asarray([-1, 1, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="active keyframe record lookup"):
        validate_active_keyframe_state(seed)


# Store active records through the setter boundary
def test_set_active_keyframe_record_updates_legacy_and_canonical_state():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())
    pose = _pose(4.0)
    feats = _features([[9.0, 8.0], [7.0, 6.0]])
    lookup = np.asarray([-1, 1], dtype=np.int64)

    record = set_active_keyframe_record(seed, 4, pose, feats, lookup)

    assert seed["keyframe_kf"] == 4
    assert seed["active_keyframe_kf"] == 4
    assert seed["T_WC1"] is pose
    assert seed["feats1"] is feats
    assert seed["landmark_id_by_feat1"] is lookup
    assert seed["poses"][4] is pose
    assert record["pose"] is pose
    seed["landmarks"][1]["obs"].append({"kf": 4, "feat": 1, "xy": np.asarray([7.0, 6.0], dtype=np.float64)})
    assert validate_active_keyframe_state(seed)["errors"] == []


# Reject stale active-keyframe frame-pipeline indices
def test_process_frame_rejects_stale_active_keyframe_index_argument():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())

    with pytest.raises(ValueError, match="keyframe_kf argument must match active keyframe state"):
        process_frame_against_seed(
            np.eye(3, dtype=np.float64),
            seed,
            seed["feats1"],
            np.zeros((8, 8), dtype=np.float64),
            feature_cfg={},
            F_cfg={},
            keyframe_kf=2,
            current_kf=2,
        )


# Reject stale active-keyframe frame-pipeline feature bundles
def test_process_frame_rejects_stale_active_keyframe_features_argument():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())
    stale_feats = _features([[99.0, 20.0], [30.0, 40.0], [70.0, 80.0]])

    with pytest.raises(ValueError, match="keyframe_feats argument must match active keyframe features"):
        process_frame_against_seed(
            np.eye(3, dtype=np.float64),
            seed,
            stale_feats,
            np.zeros((8, 8), dtype=np.float64),
            feature_cfg={},
            F_cfg={},
            keyframe_kf=1,
            current_kf=2,
        )


# Copy accepted pose blocks before storing frame-pipeline state
def test_copy_pose_blocks_returns_owned_arrays():
    R = np.eye(3, dtype=np.float64)
    t = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    R_copy, t_copy = _copy_pose_blocks(R, t)
    R[0, 0] = 9.0
    t[0] = 9.0

    assert not np.shares_memory(R_copy, R)
    assert not np.shares_memory(t_copy, t)
    np.testing.assert_allclose(R_copy, np.eye(3, dtype=np.float64))
    np.testing.assert_allclose(t_copy, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))


# Promote a current frame and sync canonical active state
def test_promotion_updates_active_keyframe_store_and_legacy_mirrors():
    seed = initialise_canonical_keyframe_state(_bootstrap_seed())
    seed["landmarks"][0]["obs"].append({"kf": 2, "feat": 0, "xy": np.asarray([11.0, 21.0])})
    seed["landmarks"][1]["obs"].append({"kf": 2, "feat": 2, "xy": np.asarray([51.0, 61.0])})

    cur_feats = _features([[11.0, 21.0], [35.0, 45.0], [51.0, 61.0]])
    R_cur, t_cur = _pose(2.0)
    seed = promote_frame_to_keyframe(seed, cur_feats, R_cur, t_cur, current_kf=2)
    record = get_active_keyframe_record(seed)

    assert seed["keyframe_kf"] == 2
    assert seed["active_keyframe_kf"] == 2
    assert seed["poses"][2] is seed["T_WC1"]
    assert record["pose"] is seed["T_WC1"]
    assert record["feats"] is seed["feats1"]
    assert record["landmark_id_by_feat"] is seed["landmark_id_by_feat1"]
    assert np.array_equal(seed["landmark_id_by_feat1"], np.asarray([0, -1, 1], dtype=np.int64))

    report = audit_seed_invariants(seed)
    assert report["errors"] == []
