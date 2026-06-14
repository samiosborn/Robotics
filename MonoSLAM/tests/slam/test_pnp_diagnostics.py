from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from slam.pnp_diagnostics import (
    point_spatial_summary,
    pose_reprojection_block,
    reprojection_error_histogram,
    reprojection_error_summary,
    spatial_relationship_side,
    threshold_pair_group_masks,
)


# Summarise only finite reprojection errors
def test_reprojection_error_summary_ignores_nonfinite_values():
    summary = reprojection_error_summary(
        np.asarray([1.0, 3.0, np.inf, np.nan], dtype=np.float64)
    )

    assert summary["count"] == 2
    assert summary["min"] == 1.0
    assert summary["median"] == 2.0
    assert summary["max"] == 3.0


# Keep reprojection histogram boundaries stable
def test_reprojection_error_histogram_preserves_bins():
    histogram = reprojection_error_histogram(
        np.asarray([2.0, 2.5, 4.0, 7.0, 10.0, 15.0, 21.0], dtype=np.float64)
    )

    assert histogram == {
        "le_2_px": 1,
        "gt_2_le_3_px": 1,
        "gt_3_le_5_px": 1,
        "gt_5_le_8_px": 1,
        "gt_8_le_12_px": 1,
        "gt_12_le_20_px": 1,
        "gt_20_px": 1,
    }


# Split fixed-threshold support into stable groups
def test_threshold_pair_group_masks_separate_support():
    pose_pair = {
        "pose_8px": {
            "inlier_mask": np.asarray([True, False, False, True], dtype=bool)
        },
        "pose_12px": {
            "inlier_mask": np.asarray([True, True, False, False], dtype=bool)
        },
    }

    masks = threshold_pair_group_masks(
        pose_pair,
        4,
        include_12px_all=True,
    )

    np.testing.assert_array_equal(
        masks["pnp_8px_inliers"],
        np.asarray([True, False, False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        masks["pnp_12px_only_inliers"],
        np.asarray([False, True, False, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        masks["rejected_by_both"],
        np.asarray([False, False, True, False], dtype=bool),
    )


# Compute pose reprojection summaries from correspondence arrays
def test_pose_reprojection_block_reports_depth_and_error():
    K = np.eye(3, dtype=np.float64)
    corrs = SimpleNamespace(
        X_w=np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, -1.0],
            ],
            dtype=np.float64,
        ),
        x_cur=np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    block = pose_reprojection_block(
        corrs,
        K,
        np.eye(3, dtype=np.float64),
        np.zeros((3,), dtype=np.float64),
        np.asarray([True, True], dtype=bool),
        eps=1e-12,
    )

    assert block["n_total"] == 2
    assert block["n_positive_depth"] == 1
    assert block["n_non_positive_depth"] == 1
    assert block["error_px"]["count"] == 1
    assert block["error_px"]["max"] == 0.0


# Summarise and compare separated image-point groups
def test_spatial_relationship_side_reports_separation():
    image_size = (100, 100)
    points_a = np.asarray([[10.0, 10.0], [12.0, 12.0]], dtype=np.float64)
    points_b = np.asarray([[80.0, 80.0], [82.0, 82.0]], dtype=np.float64)
    summary_a = point_spatial_summary(points_a, image_size)
    summary_b = point_spatial_summary(points_b, image_size)

    relationship = spatial_relationship_side(
        summary_a,
        summary_b,
        points_a,
        points_b,
        image_size,
    )

    assert relationship["bbox_iou"] == 0.0
    assert relationship["centroid_distance_fraction"] > 0.5
    assert relationship["nearest_distance"]["median_fraction"] > 0.5
