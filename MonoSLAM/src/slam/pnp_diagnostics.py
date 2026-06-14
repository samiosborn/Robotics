from __future__ import annotations

import numpy as np

from core.checks import align_bool_mask_1d
from geometry.camera import reprojection_errors_sq, world_to_camera_points


# Summarise a vector of reprojection errors in pixels
def reprojection_error_summary(errors_px: np.ndarray) -> dict:
    errors_px = np.asarray(errors_px, dtype=np.float64).reshape(-1)
    errors_px = errors_px[np.isfinite(errors_px)]

    if int(errors_px.size) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }

    return {
        "count": int(errors_px.size),
        "min": float(np.min(errors_px)),
        "max": float(np.max(errors_px)),
        "mean": float(np.mean(errors_px)),
        "median": float(np.median(errors_px)),
        "p10": float(np.percentile(errors_px, 10)),
        "p25": float(np.percentile(errors_px, 25)),
        "p50": float(np.percentile(errors_px, 50)),
        "p75": float(np.percentile(errors_px, 75)),
        "p90": float(np.percentile(errors_px, 90)),
        "p95": float(np.percentile(errors_px, 95)),
    }


# Count reprojection errors in the diagnostic bins
def reprojection_error_histogram(errors_px: np.ndarray) -> dict:
    errors_px = np.asarray(errors_px, dtype=np.float64).reshape(-1)
    errors_px = errors_px[np.isfinite(errors_px)]

    return {
        "le_2_px": int(np.sum(errors_px <= 2.0)),
        "gt_2_le_3_px": int(np.sum((errors_px > 2.0) & (errors_px <= 3.0))),
        "gt_3_le_5_px": int(np.sum((errors_px > 3.0) & (errors_px <= 5.0))),
        "gt_5_le_8_px": int(np.sum((errors_px > 5.0) & (errors_px <= 8.0))),
        "gt_8_le_12_px": int(np.sum((errors_px > 8.0) & (errors_px <= 12.0))),
        "gt_12_le_20_px": int(np.sum((errors_px > 12.0) & (errors_px <= 20.0))),
        "gt_20_px": int(np.sum(errors_px > 20.0)),
    }


# Build one reprojection diagnostic block from a boolean subset
def non_pnp_reprojection_block(
    errors_px: np.ndarray,
    positive_depth: np.ndarray,
    non_positive_depth: np.ndarray,
    mask: np.ndarray,
) -> dict:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    errors_px = np.asarray(errors_px, dtype=np.float64).reshape(-1)
    positive_depth = np.asarray(positive_depth, dtype=bool).reshape(-1)
    non_positive_depth = np.asarray(non_positive_depth, dtype=bool).reshape(-1)

    finite_reprojection = np.isfinite(errors_px)
    usable = mask & positive_depth & finite_reprojection

    return {
        "n_total": int(np.sum(mask)),
        "n_positive_depth": int(np.sum(mask & positive_depth)),
        "n_non_positive_depth": int(np.sum(mask & non_positive_depth)),
        "n_nonfinite_reprojection": int(np.sum(mask & ~finite_reprojection)),
        "error_px": reprojection_error_summary(errors_px[usable]),
        "hist_px": reprojection_error_histogram(errors_px[usable]),
    }


# Compute reprojection errors for a pose on all correspondences
def reprojection_errors_for_pose(
    corrs,
    K: np.ndarray,
    R,
    t,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = int(np.asarray(corrs.X_w, dtype=np.float64).shape[1])
    errors_px = np.full((N,), np.inf, dtype=np.float64)
    positive_depth = np.zeros((N,), dtype=bool)
    non_positive_depth = np.zeros((N,), dtype=bool)

    if R is None or t is None:
        return errors_px, positive_depth, non_positive_depth

    X_w = np.asarray(corrs.X_w, dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    X_c = world_to_camera_points(R, t, X_w)
    z = np.asarray(X_c[2, :], dtype=np.float64).reshape(-1)
    finite_depth = np.isfinite(z)
    positive_depth = finite_depth & (z > float(eps))
    non_positive_depth = finite_depth & ~positive_depth

    err_sq = np.asarray(
        reprojection_errors_sq(K, R, t, X_w, x_cur),
        dtype=np.float64,
    ).reshape(-1)
    finite_err_sq = np.isfinite(err_sq) & (err_sq >= 0.0)
    errors_px[finite_err_sq] = np.sqrt(err_sq[finite_err_sq])

    return errors_px, positive_depth, non_positive_depth


# Summarise reprojection behaviour for a pose and correspondence subset
def pose_reprojection_block(
    corrs,
    K: np.ndarray,
    R,
    t,
    mask: np.ndarray,
    *,
    eps: float,
) -> dict:
    errors_px, positive_depth, non_positive_depth = reprojection_errors_for_pose(
        corrs,
        K,
        R,
        t,
        eps=eps,
    )

    return non_pnp_reprojection_block(
        errors_px,
        positive_depth,
        non_positive_depth,
        mask,
    )


# Build reusable threshold-pair group masks
def threshold_pair_group_masks(
    pose_pair: dict,
    N: int,
    *,
    include_12px_all: bool = False,
) -> dict[str, np.ndarray]:
    mask_8 = align_bool_mask_1d(
        pose_pair["pose_8px"]["inlier_mask"],
        N,
        name="pnp_8px_inlier_mask",
    )
    mask_12 = align_bool_mask_1d(
        pose_pair["pose_12px"]["inlier_mask"],
        N,
        name="pnp_12px_inlier_mask",
    )
    mask_12_only = mask_12 & ~mask_8
    mask_rejected = ~(mask_8 | mask_12)

    out = {
        "pnp_8px_inliers": mask_8,
        "pnp_12px_only_inliers": mask_12_only,
        "rejected_by_both": mask_rejected,
    }
    if bool(include_12px_all):
        out = {
            "pnp_8px_inliers": mask_8,
            "pnp_12px_all_inliers": mask_12,
            "pnp_12px_only_inliers": mask_12_only,
            "rejected_by_both": mask_rejected,
        }

    return out


# Summarise finite scores for one selected group
def score_summary(scores: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]

    if int(scores.size) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


# Build a coarse spatial summary for image points
def point_spatial_summary(
    xy: np.ndarray,
    image_size: tuple[int, int],
    *,
    grid_cols: int = 4,
    grid_rows: int = 3,
) -> dict:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]

    W = float(image_size[0])
    H = float(image_size[1])
    image_area = max(W * H, 1.0)
    grid = [[0 for _ in range(int(grid_cols))] for _ in range(int(grid_rows))]
    count = int(xy.shape[0])

    if count == 0:
        return {
            "count": 0,
            "bbox": None,
            "bbox_area_fraction": None,
            "bbox_aspect_ratio": None,
            "centroid": None,
            "grid_cols": int(grid_cols),
            "grid_rows": int(grid_rows),
            "occupancy_grid": grid,
            "occupied_cells": 0,
            "occupied_cell_fraction": 0.0,
            "max_cell_count": 0,
            "max_cell_fraction": None,
            "border_count": 0,
            "border_fraction": None,
            "thin_structure_like": False,
            "heavily_concentrated": False,
        }

    xmin = float(np.min(xy[:, 0]))
    ymin = float(np.min(xy[:, 1]))
    xmax = float(np.max(xy[:, 0]))
    ymax = float(np.max(xy[:, 1]))
    bbox_w = max(0.0, xmax - xmin)
    bbox_h = max(0.0, ymax - ymin)
    bbox_area_fraction = float((bbox_w * bbox_h) / image_area)

    if bbox_w <= 1e-12 or bbox_h <= 1e-12:
        bbox_aspect_ratio = None
    else:
        bbox_aspect_ratio = float(max(bbox_w / bbox_h, bbox_h / bbox_w))

    centroid = [float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))]

    for point in xy:
        col = int(np.floor((float(point[0]) / max(W, 1.0)) * int(grid_cols)))
        row = int(np.floor((float(point[1]) / max(H, 1.0)) * int(grid_rows)))
        col = int(np.clip(col, 0, int(grid_cols) - 1))
        row = int(np.clip(row, 0, int(grid_rows) - 1))
        grid[row][col] += 1

    occupied_cells = int(sum(1 for row in grid for value in row if int(value) > 0))
    max_cell_count = int(max(max(row) for row in grid))
    max_cell_fraction = float(max_cell_count / count)
    occupied_cell_fraction = float(
        occupied_cells / max(int(grid_cols) * int(grid_rows), 1)
    )

    border_margin = 0.08 * min(W, H)
    border_mask = (
        (xy[:, 0] <= border_margin)
        | (xy[:, 1] <= border_margin)
        | (xy[:, 0] >= (W - 1.0 - border_margin))
        | (xy[:, 1] >= (H - 1.0 - border_margin))
    )
    border_count = int(np.sum(border_mask))
    border_fraction = float(border_count / count)
    thin_structure_like = bool(
        bbox_aspect_ratio is not None
        and bbox_aspect_ratio >= 4.0
        and bbox_area_fraction <= 0.35
        and count >= 5
    )
    heavily_concentrated = bool(count >= 5 and max_cell_fraction >= 0.50)

    return {
        "count": int(count),
        "bbox": [xmin, ymin, xmax, ymax],
        "bbox_area_fraction": bbox_area_fraction,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "centroid": centroid,
        "grid_cols": int(grid_cols),
        "grid_rows": int(grid_rows),
        "occupancy_grid": grid,
        "occupied_cells": int(occupied_cells),
        "occupied_cell_fraction": occupied_cell_fraction,
        "max_cell_count": int(max_cell_count),
        "max_cell_fraction": max_cell_fraction,
        "border_count": int(border_count),
        "border_fraction": border_fraction,
        "thin_structure_like": thin_structure_like,
        "heavily_concentrated": heavily_concentrated,
    }


# Build the set of occupied coarse-grid cells
def _occupied_cells(summary: dict) -> set[tuple[int, int]]:
    grid = summary.get("occupancy_grid", []) if isinstance(summary, dict) else []
    cells: set[tuple[int, int]] = set()

    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            if int(value) > 0:
                cells.add((int(row_index), int(column_index)))

    return cells


# Compute IoU between two image-space bounding boxes
def _bbox_iou(box_a, box_b) -> float | None:
    if box_a is None or box_b is None:
        return None

    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    intersection_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return None

    return float(intersection / union)


# Compute symmetric nearest-neighbour distances between point sets
def _nearest_distance_summary(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    image_size: tuple[int, int],
) -> dict:
    xy_a = np.asarray(xy_a, dtype=np.float64).reshape(-1, 2)
    xy_b = np.asarray(xy_b, dtype=np.float64).reshape(-1, 2)
    xy_a = xy_a[np.isfinite(xy_a).all(axis=1)]
    xy_b = xy_b[np.isfinite(xy_b).all(axis=1)]

    if int(xy_a.shape[0]) == 0 or int(xy_b.shape[0]) == 0:
        return {
            "count": 0,
            "min_px": None,
            "median_px": None,
            "mean_px": None,
            "p75_px": None,
            "median_fraction": None,
        }

    delta = xy_a[:, None, :] - xy_b[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    nearest = np.concatenate(
        [np.min(distance, axis=1), np.min(distance, axis=0)]
    )
    image_diagonal = max(
        float(np.hypot(float(image_size[0]), float(image_size[1]))),
        1.0,
    )

    return {
        "count": int(nearest.size),
        "min_px": float(np.min(nearest)),
        "median_px": float(np.median(nearest)),
        "mean_px": float(np.mean(nearest)),
        "p75_px": float(np.percentile(nearest, 75)),
        "median_fraction": float(np.median(nearest) / image_diagonal),
    }


# Summarise the spatial relationship between two image-point groups
def spatial_relationship_side(
    summary_a: dict,
    summary_b: dict,
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    image_size: tuple[int, int],
) -> dict:
    cells_a = _occupied_cells(summary_a)
    cells_b = _occupied_cells(summary_b)
    cell_union = cells_a | cells_b
    cell_overlap = cells_a & cells_b

    centroid_a = summary_a.get("centroid", None)
    centroid_b = summary_b.get("centroid", None)
    centroid_distance_px = None
    centroid_distance_fraction = None
    if centroid_a is not None and centroid_b is not None:
        dx = float(centroid_a[0]) - float(centroid_b[0])
        dy = float(centroid_a[1]) - float(centroid_b[1])
        centroid_distance_px = float(np.hypot(dx, dy))
        image_diagonal = max(
            float(np.hypot(float(image_size[0]), float(image_size[1]))),
            1.0,
        )
        centroid_distance_fraction = float(centroid_distance_px / image_diagonal)

    return {
        "bbox_iou": _bbox_iou(
            summary_a.get("bbox", None),
            summary_b.get("bbox", None),
        ),
        "centroid_distance_px": centroid_distance_px,
        "centroid_distance_fraction": centroid_distance_fraction,
        "shared_occupied_cells": int(len(cell_overlap)),
        "occupied_cell_union": int(len(cell_union)),
        "grid_overlap_fraction": (
            None
            if len(cell_union) == 0
            else float(len(cell_overlap) / len(cell_union))
        ),
        "nearest_distance": _nearest_distance_summary(
            xy_a,
            xy_b,
            image_size,
        ),
    }


# Classify the 8 px versus 12 px-only spatial relationship
def classify_spatial_relationship(reference: dict, current: dict) -> str:
    def separated(side: dict) -> bool:
        bbox_iou = side.get("bbox_iou", None)
        centroid_fraction = side.get("centroid_distance_fraction", None)
        nearest = side.get("nearest_distance", {})
        nearest_fraction = (
            nearest.get("median_fraction", None)
            if isinstance(nearest, dict)
            else None
        )

        return (
            bbox_iou is not None
            and centroid_fraction is not None
            and nearest_fraction is not None
            and float(bbox_iou) <= 0.02
            and float(centroid_fraction) >= 0.10
            and float(nearest_fraction) >= 0.05
        )

    def interleaved(side: dict) -> bool:
        bbox_iou = side.get("bbox_iou", None)
        grid_overlap = side.get("grid_overlap_fraction", None)
        nearest = side.get("nearest_distance", {})
        nearest_fraction = (
            nearest.get("median_fraction", None)
            if isinstance(nearest, dict)
            else None
        )

        return (
            (bbox_iou is not None and float(bbox_iou) >= 0.20)
            or (
                grid_overlap is not None
                and nearest_fraction is not None
                and float(grid_overlap) >= 0.40
                and float(nearest_fraction) <= 0.04
            )
            or (
                nearest_fraction is not None
                and float(nearest_fraction) <= 0.03
            )
        )

    reference_separated = separated(reference)
    current_separated = separated(current)
    reference_interleaved = interleaved(reference)
    current_interleaved = interleaved(current)

    if reference_separated and current_separated:
        return "spatially_separated"
    if reference_interleaved and current_interleaved:
        return "spatially_interleaved"
    if reference_separated or current_separated:
        return "ambiguous_mixed_separation"
    if reference_interleaved or current_interleaved:
        return "ambiguous_mixed_interleaving"

    return "ambiguous"
