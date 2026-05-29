# tests/slam/test_invariants.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from slam.invariants import audit_seed_invariants


# Build a feature stub with keypoints
def _features(kps_xy):
    return SimpleNamespace(kps_xy=np.asarray(kps_xy, dtype=np.float64))


# Build a project pose tuple
def _pose_rt(tx: float = 0.0):
    return np.eye(3, dtype=np.float64), np.asarray([tx, 0.0, 0.0], dtype=np.float64)


# Build a simple landmark record
def _landmark(landmark_id: int, *, feat: int = 0, kf: int = 1):
    return {
        "id": int(landmark_id),
        "X_w": np.asarray([float(landmark_id), 0.0, 5.0], dtype=np.float64),
        "obs": [
            {"kf": int(kf), "feat": int(feat), "xy": np.asarray([10.0 + feat, 20.0], dtype=np.float64)},
        ],
    }


# Build a valid active seed
def _valid_seed_with_landmarks():
    return {
        "T_WC0": _pose_rt(),
        "T_WC1": _pose_rt(1.0),
        "last_accepted_pose": {"kf": 2, "R": np.eye(3, dtype=np.float64), "t": np.asarray([2.0, 0.0, 0.0])},
        "keyframe_kf": 1,
        "feats1": _features([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        "landmark_id_by_feat1": np.asarray([-1, 10, 11], dtype=np.int64),
        "landmarks": [
            {
                "id": 10,
                "X_w": np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
                "obs": [
                    {"kf": 0, "feat": 4, "xy": np.asarray([1.0, 2.0], dtype=np.float64)},
                    {"kf": 1, "feat": 1, "xy": np.asarray([30.0, 40.0], dtype=np.float64)},
                ],
            },
            {
                "id": 11,
                "X_w": np.asarray([1.0, 0.0, 5.0], dtype=np.float64),
                "obs": [
                    {"kf": 1, "feat": 2, "xy": np.asarray([50.0, 60.0], dtype=np.float64)},
                    {"kf": 2, "feat": 7, "xy": np.asarray([70.0, 80.0], dtype=np.float64)},
                ],
            },
        ],
    }


# Build a valid seed with canonical keyframe stores
def _valid_canonical_seed(*, active_lookup=None):
    seed = _valid_seed_with_landmarks()
    if active_lookup is not None:
        seed["landmark_id_by_feat1"] = active_lookup

    seed["feats0"] = _features([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    seed["poses"] = {
        0: seed["T_WC0"],
        1: seed["T_WC1"],
    }
    seed["keyframes"] = {
        1: {
            "kf": 1,
            "pose": seed["T_WC1"],
            "feats": seed["feats1"],
            "landmark_id_by_feat": seed["landmark_id_by_feat1"],
        }
    }
    seed["active_keyframe_kf"] = 1
    return seed


# Accept a partial seed with no optional structures
def test_valid_minimal_seed():
    report = audit_seed_invariants({}, context="minimal")

    assert report["context"] == "minimal"
    assert report["num_landmarks"] == 0
    assert report["num_observations"] == 0
    assert report["errors"] == []


# Accept a valid seed with landmarks and observations
def test_valid_seed_with_landmarks_and_observations():
    report = audit_seed_invariants(_valid_seed_with_landmarks())

    assert report["num_landmarks"] == 2
    assert report["num_observations"] == 4
    assert report["num_active_lookup_entries"] == 2
    assert report["num_duplicate_observations"] == 0
    assert report["num_feature_assignment_conflicts"] == 0
    assert report["errors"] == []


# Reject pose fields with malformed shapes
def test_invalid_pose_shape():
    with pytest.raises(ValueError, match="T_WC1.*shape"):
        audit_seed_invariants({"T_WC1": np.eye(3, dtype=np.float64)})


# Reject landmark records with malformed world points
def test_invalid_landmark_x_w_shape():
    seed = {"landmarks": [{"id": 0, "X_w": np.asarray([1.0, 2.0], dtype=np.float64)}]}

    with pytest.raises(ValueError, match="X_w.*shape"):
        audit_seed_invariants(seed)


# Reject duplicate landmark identifiers
def test_duplicate_landmark_ids():
    seed = {"landmarks": [_landmark(0, feat=0), _landmark(0, feat=1)]}

    with pytest.raises(ValueError, match="duplicate landmark id 0"):
        audit_seed_invariants(seed)


# Reject bool landmark identifiers
def test_bool_landmark_id_rejected():
    seed = {"landmarks": [{"id": True, "X_w": np.asarray([0.0, 0.0, 5.0], dtype=np.float64)}]}

    with pytest.raises(ValueError, match="not bool"):
        audit_seed_invariants(seed)


# Reject observations with missing required fields
def test_malformed_observation_record():
    seed = {
        "landmarks": [
            {
                "id": 0,
                "X_w": np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
                "obs": [{"kf": 1, "feat": 0}],
            }
        ]
    }

    with pytest.raises(ValueError, match="missing required key 'xy'"):
        audit_seed_invariants(seed)


# Reject duplicate observations on one landmark
def test_duplicate_observation_on_same_landmark_kf_feat():
    # Reuse the same observation twice on one landmark
    ob = {"kf": 1, "feat": 0, "xy": np.asarray([10.0, 20.0], dtype=np.float64)}
    seed = {
        "landmarks": [
            {
                "id": 0,
                "X_w": np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
                "obs": [dict(ob), dict(ob)],
            }
        ]
    }

    with pytest.raises(ValueError, match="duplicates observation"):
        audit_seed_invariants(seed)


# Reject feature assignment conflicts across landmarks
def test_one_keyframe_feature_pair_assigned_to_two_landmarks():
    # Create two landmarks that claim the same keyframe feature
    seed = {
        "landmarks": [
            _landmark(0, feat=3, kf=1),
            _landmark(1, feat=3, kf=1),
        ]
    }

    with pytest.raises(ValueError, match="keyframe-feature pair"):
        audit_seed_invariants(seed)


# Reject active lookup entries for missing landmarks
def test_active_lookup_points_to_missing_landmark_id():
    seed = {
        "feats1": _features([[10.0, 20.0]]),
        "landmark_id_by_feat1": np.asarray([99], dtype=np.int64),
        "landmarks": [],
    }

    with pytest.raises(ValueError, match="missing landmark id 99"):
        audit_seed_invariants(seed)


# Reject active lookup feature indices outside feats1
def test_active_lookup_feature_index_out_of_range():
    seed = {
        "feats1": _features([[10.0, 20.0]]),
        "landmark_id_by_feat1": {2: 0},
        "landmarks": [_landmark(0, feat=0, kf=1)],
    }

    with pytest.raises(ValueError, match="feature index 2"):
        audit_seed_invariants(seed)


# Return audit errors when strict mode is disabled
def test_strict_false_returns_errors_instead_of_raising():
    seed = {
        "landmarks": [
            {
                "id": 0,
                "X_w": np.asarray([0.0, 0.0], dtype=np.float64),
            }
        ]
    }

    report = audit_seed_invariants(seed, strict=False)

    assert len(report["errors"]) == 1
    assert "X_w" in report["errors"][0]


# Accept canonical stores with NumPy -1 sentinel lookup
def test_canonical_numpy_lookup_with_sentinel_is_accepted():
    report = audit_seed_invariants(_valid_canonical_seed())

    assert report["num_poses"] == 2
    assert report["num_keyframes"] == 1
    assert report["errors"] == []


# Accept canonical stores with dict lookup entries
def test_canonical_dict_lookup_is_accepted():
    seed = _valid_canonical_seed(active_lookup={1: 10, 2: 11})
    seed["keyframes"][1]["landmark_id_by_feat"] = seed["landmark_id_by_feat1"]

    report = audit_seed_invariants(seed)

    assert report["num_active_lookup_entries"] == 2
    assert report["errors"] == []


# Reject active keyframe ids that are absent from the store
def test_active_keyframe_missing_from_keyframe_store_fails():
    seed = _valid_canonical_seed()
    seed["active_keyframe_kf"] = 3
    seed["keyframe_kf"] = 3

    with pytest.raises(ValueError, match="existing keyframe"):
        audit_seed_invariants(seed)


# Reject keyframe records whose kf field mismatches the dict key
def test_keyframe_record_kf_mismatch_fails():
    seed = _valid_canonical_seed()
    seed["keyframes"][1]["kf"] = 2

    with pytest.raises(ValueError, match="must match dict key 1"):
        audit_seed_invariants(seed)


# Reject active canonical pose mirrors that diverge from legacy T_WC1
def test_active_keyframe_pose_mirror_mismatch_fails():
    seed = _valid_canonical_seed()
    seed["keyframes"][1]["pose"] = _pose_rt(9.0)

    with pytest.raises(ValueError, match="active keyframe record pose"):
        audit_seed_invariants(seed)


# Reject malformed canonical pose stores
def test_malformed_pose_store_fails_validation():
    seed = _valid_canonical_seed()
    seed["poses"][1] = np.eye(3, dtype=np.float64)

    with pytest.raises(ValueError, match="poses.*shape"):
        audit_seed_invariants(seed)


# Return canonical-store errors when strict mode is disabled
def test_strict_false_reports_canonical_store_errors():
    seed = _valid_canonical_seed()
    seed["keyframes"][1]["kf"] = 2

    report = audit_seed_invariants(seed, strict=False)

    assert len(report["errors"]) >= 1
    assert any("must match dict key 1" in error for error in report["errors"])
