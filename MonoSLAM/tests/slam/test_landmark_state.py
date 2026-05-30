# tests/slam/test_landmark_state.py
from __future__ import annotations

import numpy as np
import pytest

from slam.invariants import audit_seed_invariants
from slam.landmark_state import (
    add_landmark_observation,
    build_landmark_id_index,
    build_observation_indexes,
    count_valid_landmark_observations,
    get_landmark_by_id,
    iter_landmark_observations,
    next_landmark_id,
)
from slam.seed import build_two_view_seed


# Build a small landmark record
def _landmark(landmark_id: int, *, kf: int = 1, feat: int = 0, xy=None):
    if xy is None:
        xy = [10.0 + float(feat), 20.0]
    return {
        "id": int(landmark_id),
        "X_w": np.asarray([float(landmark_id), 0.0, 5.0], dtype=np.float64),
        "obs": [
            {"kf": int(kf), "feat": int(feat), "xy": np.asarray(xy, dtype=np.float64)},
        ],
    }


# Build a small landmark seed
def _seed():
    return {
        "landmarks": [
            _landmark(10, kf=0, feat=1, xy=[1.0, 2.0]),
            _landmark(12, kf=1, feat=2, xy=[3.0, 4.0]),
        ]
    }


# Build a landmark id index
def test_build_landmark_id_index_returns_records_by_id():
    seed = _seed()

    index = build_landmark_id_index(seed)

    assert sorted(index.keys()) == [10, 12]
    assert index[10] is seed["landmarks"][0]
    assert index[12] is seed["landmarks"][1]


# Reject duplicate landmark ids clearly
def test_duplicate_landmark_ids_fail_clearly():
    seed = {"landmarks": [_landmark(10, feat=0), _landmark(10, feat=1)]}

    with pytest.raises(ValueError, match="duplicate landmark id 10"):
        build_landmark_id_index(seed)


# Lookup an existing landmark id
def test_get_landmark_by_existing_id_returns_record():
    seed = _seed()

    landmark = get_landmark_by_id(seed, 12)

    assert landmark is seed["landmarks"][1]


# Reject lookup of a missing landmark id
def test_get_landmark_by_missing_id_fails_clearly():
    with pytest.raises(ValueError, match="missing landmark id 99"):
        get_landmark_by_id(_seed(), 99)


# Allocate the next landmark id
def test_next_landmark_id_returns_max_existing_id_plus_one():
    assert next_landmark_id(_seed()) == 13
    assert next_landmark_id({"landmarks": []}) == 0


# Iterate checked landmark observations
def test_iter_landmark_observations_yields_expected_records():
    rows = list(iter_landmark_observations(_seed()))

    assert [(lm_id, kf, feat) for lm_id, kf, feat, _ in rows] == [(10, 0, 1), (12, 1, 2)]
    np.testing.assert_allclose(rows[0][3], np.asarray([1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(rows[1][3], np.asarray([3.0, 4.0], dtype=np.float64))


# Skip duplicate observation insertion on the same landmark
def test_duplicate_observation_insertion_is_skipped():
    seed = _seed()
    indexes = build_observation_indexes(seed)
    assignments = dict(indexes["landmark_id_by_feature"])
    landmark = seed["landmarks"][0]

    added = add_landmark_observation(
        landmark,
        0,
        1,
        np.asarray([9.0, 9.0], dtype=np.float64),
        assignment_by_feature=assignments,
    )

    assert added is False
    assert len(landmark["obs"]) == 1
    np.testing.assert_allclose(landmark["obs"][0]["xy"], np.asarray([1.0, 2.0], dtype=np.float64))


# Detect one feature assigned to two landmarks
def test_feature_assignment_conflict_is_detected_by_observation_index():
    seed = {"landmarks": [_landmark(10, kf=1, feat=3), _landmark(12, kf=1, feat=3)]}

    with pytest.raises(ValueError, match="feature assignment conflict"):
        build_observation_indexes(seed)


# Copy xy arrays when adding observations
def test_added_observation_xy_is_copied():
    landmark = {"id": 7, "X_w": np.asarray([0.0, 0.0, 5.0], dtype=np.float64), "obs": []}
    xy = np.asarray([5.0, 6.0], dtype=np.float64)

    added = add_landmark_observation(landmark, 2, 4, xy)
    xy[:] = 99.0

    assert added is True
    assert not np.shares_memory(landmark["obs"][0]["xy"], xy)
    np.testing.assert_allclose(landmark["obs"][0]["xy"], np.asarray([5.0, 6.0], dtype=np.float64))


# Copy bootstrap landmark arrays from caller-owned inputs
def test_bootstrap_seed_landmark_arrays_are_copied():
    x1 = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)
    x2 = np.asarray([[5.0, 7.0], [6.0, 8.0]], dtype=np.float64)
    X_valid = np.asarray([[0.0], [1.0], [5.0]], dtype=np.float64)

    seed = build_two_view_seed(
        x1,
        x2,
        idx_init=np.asarray([0], dtype=np.int64),
        X_valid=X_valid,
        R1=np.eye(3, dtype=np.float64),
        t1=np.zeros((3,), dtype=np.float64),
    )
    landmark = seed["landmarks"][0]
    x1[:, 0] = 99.0
    x2[:, 0] = 88.0
    X_valid[:, 0] = 77.0

    assert not np.shares_memory(landmark["X_w"], X_valid)
    assert not np.shares_memory(landmark["obs"][0]["xy"], x1)
    assert not np.shares_memory(landmark["obs"][1]["xy"], x2)
    np.testing.assert_allclose(landmark["X_w"], np.asarray([0.0, 1.0, 5.0], dtype=np.float64))
    np.testing.assert_allclose(landmark["obs"][0]["xy"], np.asarray([1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(landmark["obs"][1]["xy"], np.asarray([5.0, 6.0], dtype=np.float64))
    for field in ("T_WC0", "T_WC1", "feats0", "feats1", "keyframe_kf", "landmark_id_by_feat1"):
        assert field not in seed


# Count only checked observation records as valid
def test_count_valid_landmark_observations_rejects_malformed_records():
    landmark = _landmark(3, kf=1, feat=4)
    landmark["obs"].append({"kf": 2, "feat": 5})

    with pytest.raises(ValueError, match="missing required key 'xy'"):
        count_valid_landmark_observations(landmark)


# Keep malformed observation records covered by invariants
def test_malformed_observation_records_still_fail_invariant_checks():
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
