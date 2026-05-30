# tests/slam/test_keyframe_state.py
from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

from slam import frame_pipeline as frame_pipeline_module
from slam.frame_pipeline import _copy_pose_blocks, process_frame_against_seed
from slam.invariants import audit_seed_invariants
from slam.keyframe import promote_frame_to_keyframe
from slam.keyframe_state import (
    get_active_keyframe_features,
    get_active_keyframe_pose,
    get_active_keyframe_record,
    get_active_landmark_lookup,
    get_keyframe_record,
    get_pose_for_kf,
    rebuild_active_landmark_lookup,
    set_active_keyframe_record,
    set_keyframe_record,
    set_pose_for_kf,
    validate_active_keyframe_state,
)


LEGACY_ROOT_FIELDS = ("T_WC0", "T_WC1", "feats0", "feats1", "keyframe_kf", "landmark_id_by_feat1")


# Build a feature stub with keypoints
def _features(kps_xy):
    return SimpleNamespace(kps_xy=np.asarray(kps_xy, dtype=np.float64))


# Build a project pose tuple
def _pose(tx: float = 0.0):
    return np.eye(3, dtype=np.float64), np.asarray([tx, 0.0, 0.0], dtype=np.float64)


# Build a canonical seed with one active keyframe
def _canonical_seed():
    pose0 = _pose(0.0)
    pose1 = _pose(1.0)
    feats0 = _features([[10.0, 20.0], [50.0, 60.0]])
    feats1 = _features([[15.0, 25.0], [30.0, 40.0], [70.0, 80.0]])
    lookup0 = np.asarray([0, 1], dtype=np.int64)
    lookup1 = np.asarray([-1, 0, 1], dtype=np.int64)
    return {
        "poses": {0: pose0, 1: pose1},
        "keyframes": {
            0: {"kf": 0, "pose": pose0, "feats": feats0, "landmark_id_by_feat": lookup0},
            1: {"kf": 1, "pose": pose1, "feats": feats1, "landmark_id_by_feat": lookup1},
        },
        "active_keyframe_kf": 1,
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
    }


# Accept canonical keyframe state without root mirrors
def test_valid_canonical_seed_has_no_legacy_root_fields():
    seed = _canonical_seed()

    report = audit_seed_invariants(seed)

    assert report["errors"] == []
    for field in LEGACY_ROOT_FIELDS:
        assert field not in seed


# Read active fields through canonical helpers
def test_active_keyframe_helpers_return_canonical_record_fields():
    seed = _canonical_seed()
    record = get_active_keyframe_record(seed)

    assert record["kf"] == 1
    assert get_active_keyframe_pose(seed) is record["pose"]
    assert get_active_keyframe_features(seed) is record["feats"]
    assert get_active_landmark_lookup(seed) is record["landmark_id_by_feat"]


# Store poses through the canonical pose helper
def test_set_pose_for_kf_updates_pose_store_and_keyframe_record():
    seed = _canonical_seed()
    pose = _pose(8.0)

    set_pose_for_kf(seed, 1, pose, context="unit active pose")

    stored = get_pose_for_kf(seed, 1)
    assert get_keyframe_record(seed, 1)["pose"] is stored
    np.testing.assert_allclose(stored[1], np.asarray([8.0, 0.0, 0.0], dtype=np.float64))
    assert audit_seed_invariants(seed)["errors"] == []


# Reject missing active keyframes with a clear error
def test_missing_active_keyframe_fails_clearly():
    seed = _canonical_seed()
    seed["active_keyframe_kf"] = 7

    with pytest.raises(ValueError, match="missing from seed\\['keyframes'\\]"):
        get_active_keyframe_pose(seed)


# Store active records through the canonical setter boundary
def test_set_active_keyframe_record_updates_canonical_state_only():
    seed = _canonical_seed()
    pose = _pose(4.0)
    feats = _features([[9.0, 8.0], [7.0, 6.0]])
    lookup = np.asarray([-1, 1], dtype=np.int64)
    seed["landmarks"][1]["obs"].append({"kf": 4, "feat": 1, "xy": np.asarray([7.0, 6.0], dtype=np.float64)})

    record = set_active_keyframe_record(seed, 4, pose, feats, lookup)

    assert seed["active_keyframe_kf"] == 4
    assert seed["poses"][4] is record["pose"]
    assert record["pose"] is not pose
    assert record["feats"] is feats
    assert record["landmark_id_by_feat"] is not lookup
    np.testing.assert_array_equal(record["landmark_id_by_feat"], lookup)
    for field in LEGACY_ROOT_FIELDS:
        assert field not in seed
    assert validate_active_keyframe_state(seed)["errors"] == []


# Copy canonical pose and lookup inputs when storing keyframe records
def test_set_keyframe_record_copies_pose_and_lookup_inputs():
    seed = _canonical_seed()
    pose = _pose(5.0)
    lookup = np.asarray([1, -1], dtype=np.int64)
    feats = _features([[1.0, 2.0], [3.0, 4.0]])

    record = set_keyframe_record(seed, 5, pose, feats, lookup, context="unit keyframe")
    pose[1][0] = 99.0
    lookup[0] = 99

    assert record is get_keyframe_record(seed, 5)
    assert not np.shares_memory(record["pose"][1], pose[1])
    assert not np.shares_memory(record["landmark_id_by_feat"], lookup)
    np.testing.assert_allclose(record["pose"][1], np.asarray([5.0, 0.0, 0.0], dtype=np.float64))
    np.testing.assert_array_equal(record["landmark_id_by_feat"], np.asarray([1, -1], dtype=np.int64))


# Rebuild active lookup from active keyframe observations
def test_rebuild_active_lookup_repairs_stale_cache_from_observations():
    seed = _canonical_seed()
    seed["keyframes"][1]["landmark_id_by_feat"] = np.asarray([99, 99, 99], dtype=np.int64)

    lookup = rebuild_active_landmark_lookup(seed, context="unit active lookup")

    np.testing.assert_array_equal(lookup, np.asarray([-1, 0, 1], dtype=np.int64))
    assert seed["keyframes"][1]["landmark_id_by_feat"] is lookup


# Reject conflicting observations during active lookup rebuild
def test_rebuild_active_lookup_rejects_active_feature_conflicts():
    seed = _canonical_seed()
    seed["landmarks"].append(
        {
            "id": 2,
            "X_w": np.asarray([2.0, 0.0, 5.0], dtype=np.float64),
            "obs": [{"kf": 1, "feat": 1, "xy": np.asarray([30.0, 40.0], dtype=np.float64)}],
        }
    )

    with pytest.raises(ValueError, match="feature assignment conflict"):
        rebuild_active_landmark_lookup(seed, context="unit active lookup")


# Keep process-frame active keyframe state canonical
def test_process_frame_signature_has_no_active_keyframe_compat_args():
    params = signature(process_frame_against_seed).parameters

    assert "keyframe_kf" not in params
    assert "keyframe_feats" not in params
    assert list(params.keys())[:3] == ["K", "seed", "cur_im"]


# Derive active frame-pipeline inputs from canonical state
def test_process_frame_derives_active_keyframe_state_from_seed(monkeypatch):
    seed = _canonical_seed()
    canonical_feats = get_active_keyframe_features(seed)
    cur_im = np.zeros((8, 8), dtype=np.float64)
    calls = {}

    def _fake_track_against_keyframe(K, reference_feats, cur_im_arg, **kwargs):
        calls["reference_feats"] = reference_feats
        calls["cur_im"] = cur_im_arg
        return {"stats": {"n_matches": 0, "n_inliers": 0, "reason": "unit_no_inliers"}}

    monkeypatch.setattr(frame_pipeline_module, "track_against_keyframe", _fake_track_against_keyframe)

    out = frame_pipeline_module.process_frame_against_seed(
        np.eye(3, dtype=np.float64),
        seed,
        cur_im,
        feature_cfg={},
        F_cfg={},
        current_kf=2,
    )

    assert calls["reference_feats"] is canonical_feats
    assert calls["cur_im"] is cur_im
    assert out["ok"] is False
    assert out["stats"]["reason"] == "unit_no_inliers"


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


# Promote a current frame and update canonical active state
def test_promotion_updates_active_keyframe_store_only():
    seed = _canonical_seed()
    seed["landmarks"][0]["obs"].append({"kf": 2, "feat": 0, "xy": np.asarray([11.0, 21.0])})
    seed["landmarks"][1]["obs"].append({"kf": 2, "feat": 2, "xy": np.asarray([51.0, 61.0])})

    cur_feats = _features([[11.0, 21.0], [35.0, 45.0], [51.0, 61.0]])
    R_cur, t_cur = _pose(2.0)
    seed = promote_frame_to_keyframe(seed, cur_feats, R_cur, t_cur, current_kf=2)
    record = get_active_keyframe_record(seed)

    assert seed["active_keyframe_kf"] == 2
    assert seed["poses"][2] is record["pose"]
    assert record["feats"] is cur_feats
    np.testing.assert_array_equal(record["landmark_id_by_feat"], np.asarray([0, -1, 1], dtype=np.int64))
    for field in LEGACY_ROOT_FIELDS:
        assert field not in seed
    assert audit_seed_invariants(seed)["errors"] == []
