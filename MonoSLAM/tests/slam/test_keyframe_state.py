# tests/slam/test_keyframe_state.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from slam.invariants import audit_seed_invariants
from slam.keyframe import promote_frame_to_keyframe
from slam.keyframe_state import get_active_keyframe_record, initialise_canonical_keyframe_state


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
