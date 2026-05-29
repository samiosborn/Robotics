# tests/slam/test_map_mutation.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from geometry.pnp import PnPCorrespondences
from slam.invariants import audit_seed_invariants
from slam.keyframe_state import initialise_canonical_keyframe_state
from slam.map_mutation import (
    add_report_warning,
    count_report_changes,
    merge_map_mutation_reports,
    new_map_mutation_report,
)
from slam.map_update import (
    TriangulatedLandmarkBatch,
    append_new_landmarks_to_seed,
    append_tracked_observations_to_seed,
)


# Build a feature stub with keypoints
def _features(kps_xy):
    return SimpleNamespace(kps_xy=np.asarray(kps_xy, dtype=np.float64))


# Build a project pose tuple
def _pose(tx: float = 0.0):
    return np.eye(3, dtype=np.float64), np.asarray([tx, 0.0, 0.0], dtype=np.float64)


# Build a small canonical seed with one active landmark
def _seed():
    seed = {
        "T_WC1": _pose(1.0),
        "keyframe_kf": 1,
        "feats1": _features([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        "landmark_id_by_feat1": np.asarray([0, -1, -1], dtype=np.int64),
        "landmarks": [
            {
                "id": 0,
                "X_w": np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
                "birth_source": "bootstrap",
                "birth_kf": 1,
                "obs": [
                    {"kf": 1, "feat": 0, "xy": np.asarray([10.0, 20.0], dtype=np.float64)},
                ],
            }
        ],
    }
    return initialise_canonical_keyframe_state(seed)


# Build a synthetic pose output with PnP correspondences
def _pose_out(*, landmark_ids, cur_feat_idx, x_cur, inlier_mask):
    landmark_ids = np.asarray(landmark_ids, dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(cur_feat_idx, dtype=np.int64).reshape(-1)
    x_cur = np.asarray(x_cur, dtype=np.float64)
    n_corr = int(landmark_ids.size)
    corrs = PnPCorrespondences(
        X_w=np.zeros((3, n_corr), dtype=np.float64),
        x_cur=x_cur,
        landmark_ids=landmark_ids,
        cur_feat_idx=cur_feat_idx,
        kf_feat_idx=np.zeros((n_corr,), dtype=np.int64),
    )
    return {
        "corrs": corrs,
        "pnp_inlier_mask": np.asarray(inlier_mask, dtype=bool),
    }


# Build a triangulated map-growth batch
def _batch(kf_feat_idx, cur_feat_idx):
    kf_feat_idx = np.asarray(kf_feat_idx, dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(cur_feat_idx, dtype=np.int64).reshape(-1)
    n_item = int(kf_feat_idx.size)
    x_kf = np.vstack(
        [
            30.0 + np.arange(n_item, dtype=np.float64),
            40.0 + np.arange(n_item, dtype=np.float64),
        ]
    )
    x_cur = np.vstack(
        [
            35.0 + np.arange(n_item, dtype=np.float64),
            45.0 + np.arange(n_item, dtype=np.float64),
        ]
    )
    X_w = np.vstack(
        [
            1.0 + np.arange(n_item, dtype=np.float64),
            np.zeros((n_item,), dtype=np.float64),
            5.0 * np.ones((n_item,), dtype=np.float64),
        ]
    )
    return TriangulatedLandmarkBatch(
        track_idx=np.arange(n_item, dtype=np.int64),
        kf_feat_idx=kf_feat_idx,
        cur_feat_idx=cur_feat_idx,
        x_kf=x_kf,
        x_cur=x_cur,
        X_w=X_w,
        valid_mask=np.ones((n_item,), dtype=bool),
        stats={"n_valid": int(n_item), "reason": None},
    )


# Create a fresh report with default counters
def test_new_map_mutation_report_has_default_counts():
    report = new_map_mutation_report(context="unit")

    assert report["context"] == "unit"
    assert report["added_observations"] == 0
    assert report["skipped_duplicate_observations"] == 0
    assert report["added_landmarks"] == 0
    assert report["updated_active_lookup_entries"] == 0
    assert report["warnings"] == []


# Merge reports without mutating their warnings
def test_merge_map_mutation_reports_sums_counts_and_warnings():
    report_a = new_map_mutation_report(context="a")
    report_b = new_map_mutation_report(context="b")
    report_a["added_observations"] = 2
    report_a["updated_active_lookup_entries"] = 1
    report_b["added_landmarks"] = 3
    add_report_warning(report_a, "first")
    add_report_warning(report_b, "second")

    merged = merge_map_mutation_reports(report_a, report_b, context="merged")

    assert merged["context"] == "merged"
    assert merged["added_observations"] == 2
    assert merged["added_landmarks"] == 3
    assert merged["updated_active_lookup_entries"] == 1
    assert merged["warnings"] == ["first", "second"]
    assert count_report_changes(merged) == 6


# Report one tracked observation append
def test_tracked_observation_append_reports_added_observation():
    seed = _seed()
    pose_out = _pose_out(
        landmark_ids=[0],
        cur_feat_idx=[4],
        x_cur=np.asarray([[11.0], [21.0]], dtype=np.float64),
        inlier_mask=[True],
    )

    seed, stats, report = append_tracked_observations_to_seed(seed, pose_out, current_kf=2, return_report=True)

    assert stats["n_added"] == 1
    assert report["added_observations"] == 1
    assert report["skipped_duplicate_observations"] == 0
    assert report["removed_landmarks"] == 0
    assert report["updated_active_lookup_entries"] == 0
    assert seed["last_tracked_observation_append_report"] is report


# Report one duplicate tracked observation skip
def test_tracked_observation_append_reports_skipped_duplicate_observation():
    seed = _seed()
    pose_out = _pose_out(
        landmark_ids=[0],
        cur_feat_idx=[4],
        x_cur=np.asarray([[11.0], [21.0]], dtype=np.float64),
        inlier_mask=[True],
    )
    seed, _, _ = append_tracked_observations_to_seed(seed, pose_out, current_kf=2, return_report=True)

    seed, stats, report = append_tracked_observations_to_seed(seed, pose_out, current_kf=2, return_report=True)

    assert stats["n_added"] == 0
    assert stats["n_duplicate"] == 1
    assert report["added_observations"] == 0
    assert report["skipped_duplicate_observations"] == 1
    assert len(seed["landmarks"][0]["obs"]) == 2


# Preserve the current seed state shape after tracked append
def test_tracked_observation_append_still_preserves_seed_state_shape():
    seed = _seed()
    pose_out = _pose_out(
        landmark_ids=[0],
        cur_feat_idx=[4],
        x_cur=np.asarray([[11.0], [21.0]], dtype=np.float64),
        inlier_mask=[True],
    )

    seed, stats, report = append_tracked_observations_to_seed(seed, pose_out, current_kf=2, return_report=True)

    assert "landmarks" in seed
    assert "landmark_id_by_feat1" in seed
    assert "keyframes" in seed
    assert "last_tracked_observation_append_stats" in seed
    assert stats["mutation_report"] is report
    assert audit_seed_invariants(seed)["errors"] == []


# Report one map-growth landmark and its observations
def test_map_growth_append_reports_added_landmark_and_observations():
    seed = _seed()
    batch = _batch([1], [5])

    seed, report = append_new_landmarks_to_seed(seed, batch, current_kf=2, return_report=True)

    assert report["added_landmarks"] == 1
    assert report["added_observations"] == 2
    assert report["updated_active_lookup_entries"] == 1
    assert report["skipped_landmark_candidates"] == 0
    assert int(seed["landmark_id_by_feat1"][1]) == 1
    assert seed["last_append_mutation_report"] is report


# Make duplicate map-growth feature assignment explicit
def test_map_growth_append_duplicate_candidate_behaviour_is_explicit():
    seed = _seed()
    batch = _batch([1, 1], [5, 6])

    seed, report = append_new_landmarks_to_seed(seed, batch, current_kf=2, return_report=True)

    assert report["added_landmarks"] == 1
    assert report["added_observations"] == 2
    assert report["skipped_landmark_candidates"] == 1
    assert report["skipped_mapped_keyframe_features"] == 1
    assert report["feature_assignment_conflicts"] == 1
    assert seed["last_append_stats"]["n_added"] == 1
    assert seed["last_append_stats"]["n_skipped"] == 1


# Keep existing diagnostics beside mutation reports
def test_reports_do_not_replace_existing_diagnostics_unexpectedly():
    seed = _seed()
    pose_out = _pose_out(
        landmark_ids=[0],
        cur_feat_idx=[4],
        x_cur=np.asarray([[11.0], [21.0]], dtype=np.float64),
        inlier_mask=[True],
    )
    seed, stats, report = append_tracked_observations_to_seed(seed, pose_out, current_kf=2, return_report=True)

    assert seed["last_tracked_observation_append_stats"] is stats
    assert seed["last_tracked_observation_append_report"] is report
    assert stats["n_append_total"] == 1
    assert stats["mutation_report"] is report

    batch = _batch([1], [5])
    seed, growth_report = append_new_landmarks_to_seed(seed, batch, current_kf=3, return_report=True)

    assert "last_append_stats" in seed
    assert "last_append_mutation_report" in seed
    assert seed["last_append_stats"]["mutation_report"] is growth_report


# Audit invariants after valid synthetic mutations
def test_invariant_audit_passes_after_valid_synthetic_mutations():
    seed = _seed()
    pose_out = _pose_out(
        landmark_ids=[0],
        cur_feat_idx=[4],
        x_cur=np.asarray([[11.0], [21.0]], dtype=np.float64),
        inlier_mask=[True],
    )
    seed, _, tracked_report = append_tracked_observations_to_seed(seed, pose_out, current_kf=2, return_report=True)
    seed, growth_report = append_new_landmarks_to_seed(seed, _batch([1], [5]), current_kf=3, return_report=True)

    report = audit_seed_invariants(seed)

    assert report["errors"] == []
    assert tracked_report["added_observations"] == 1
    assert growth_report["added_landmarks"] == 1
