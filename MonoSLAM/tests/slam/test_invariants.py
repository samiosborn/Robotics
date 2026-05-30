# tests/slam/test_invariants.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from slam.invariants import audit_seed_invariants


LEGACY_ROOT_FIELDS = ("T_WC0", "T_WC1", "feats0", "feats1", "keyframe_kf", "landmark_id_by_feat1")


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


# Build a valid canonical seed
def _valid_canonical_seed(*, active_lookup=None):
    pose0 = _pose_rt()
    pose1 = _pose_rt(1.0)
    feats0 = _features([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    feats1 = _features([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    lookup = np.asarray([-1, 10, 11], dtype=np.int64) if active_lookup is None else active_lookup
    return {
        "poses": {
            0: pose0,
            1: pose1,
        },
        "keyframes": {
            0: {
                "kf": 0,
                "pose": pose0,
                "feats": feats0,
                "landmark_id_by_feat": np.asarray([-1, -1, -1, -1, 10], dtype=np.int64),
            },
            1: {
                "kf": 1,
                "pose": pose1,
                "feats": feats1,
                "landmark_id_by_feat": lookup,
            },
        },
        "active_keyframe_kf": 1,
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


# Accept a partial seed with no runtime structures
def test_valid_minimal_seed():
    report = audit_seed_invariants({}, context="minimal")

    assert report["context"] == "minimal"
    assert report["num_landmarks"] == 0
    assert report["num_observations"] == 0
    assert report["errors"] == []


# Accept a canonical seed with landmarks and observations
def test_valid_canonical_seed_passes_and_has_no_legacy_root_fields():
    seed = _valid_canonical_seed()

    report = audit_seed_invariants(seed)

    assert report["num_landmarks"] == 2
    assert report["num_observations"] == 4
    assert report["num_active_lookup_entries"] == 2
    assert report["num_duplicate_observations"] == 0
    assert report["num_feature_assignment_conflicts"] == 0
    assert report["errors"] == []
    for field in LEGACY_ROOT_FIELDS:
        assert field not in seed


# Reject removed root pose mirrors
def test_seed_with_root_legacy_t_wc1_fails():
    seed = _valid_canonical_seed()
    seed["T_WC1"] = _pose_rt(9.0)

    with pytest.raises(ValueError, match="removed legacy root fields.*T_WC1"):
        audit_seed_invariants(seed)


# Reject removed root active lookup mirrors
def test_seed_with_root_legacy_active_lookup_fails():
    seed = _valid_canonical_seed()
    seed["landmark_id_by_feat1"] = np.asarray([-1, 10, 11], dtype=np.int64)

    with pytest.raises(ValueError, match="removed legacy root fields.*landmark_id_by_feat1"):
        audit_seed_invariants(seed)


# Reject missing canonical stores after active state appears
def test_missing_canonical_stores_fail_after_bootstrap_context():
    seed = {"active_keyframe_kf": 1}

    with pytest.raises(ValueError, match="missing required key 'poses'"):
        audit_seed_invariants(seed, context="post_bootstrap")


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
    seed = _valid_canonical_seed(active_lookup=np.asarray([99, -1, -1], dtype=np.int64))

    with pytest.raises(ValueError, match="missing landmark id 99"):
        audit_seed_invariants(seed)


# Reject active lookup feature indices outside active features
def test_active_lookup_feature_index_out_of_range():
    seed = _valid_canonical_seed(active_lookup={3: 10, 1: 10, 2: 11})

    with pytest.raises(ValueError, match="feature index 3"):
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
    assert report["num_keyframes"] == 2
    assert report["errors"] == []


# Accept canonical stores with dict lookup entries
def test_canonical_dict_lookup_is_accepted():
    seed = _valid_canonical_seed(active_lookup={1: 10, 2: 11})

    report = audit_seed_invariants(seed)

    assert report["num_active_lookup_entries"] == 2
    assert report["errors"] == []


# Reject active keyframe ids that are absent from the store
def test_active_keyframe_missing_from_keyframe_store_fails():
    seed = _valid_canonical_seed()
    seed["active_keyframe_kf"] = 3

    with pytest.raises(ValueError, match="existing keyframe"):
        audit_seed_invariants(seed)


# Reject keyframe records whose kf field mismatches the dict key
def test_keyframe_record_kf_mismatch_fails():
    seed = _valid_canonical_seed()
    seed["keyframes"][1]["kf"] = 2

    with pytest.raises(ValueError, match="must match dict key 1"):
        audit_seed_invariants(seed)


# Reject pose/keyframe drift
def test_pose_store_and_keyframe_record_pose_drift_fails():
    seed = _valid_canonical_seed()
    seed["keyframes"][1]["pose"] = _pose_rt(9.0)

    with pytest.raises(ValueError, match="must match seed\\['poses'\\]\\[1\\]"):
        audit_seed_invariants(seed)


# Reject active lookup entries with no active-keyframe observation
def test_active_lookup_without_active_observation_fails():
    seed = _valid_canonical_seed()
    seed["landmarks"][0]["obs"] = [
        ob for ob in seed["landmarks"][0]["obs"] if not (int(ob["kf"]) == 1 and int(ob["feat"]) == 1)
    ]

    with pytest.raises(ValueError, match="no observation exists for active keyframe-feature pair"):
        audit_seed_invariants(seed)


# Reject active observations missing from the active lookup cache
def test_active_observation_missing_from_lookup_cache_fails():
    seed = _valid_canonical_seed()
    seed["keyframes"][1]["landmark_id_by_feat"] = np.asarray([-1, 10, -1], dtype=np.int64)

    with pytest.raises(ValueError, match="missing active observation"):
        audit_seed_invariants(seed)


# Reject active pose stores missing the active keyframe pose
def test_active_pose_missing_from_pose_store_fails():
    seed = _valid_canonical_seed()
    del seed["poses"][1]

    with pytest.raises(ValueError, match="poses.*active keyframe 1"):
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
