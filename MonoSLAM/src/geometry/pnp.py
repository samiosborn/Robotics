# src/geometry/pnp.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import align_bool_mask_1d, check_3xN_2xN_cols, check_3xN_pair, check_int_ge0, check_int_gt0, check_matrix_3x3, check_points_2xN, check_points_3xN, check_positive
from geometry.camera import camera_centre, pixel_to_normalised, reprojection_errors_sq, reprojection_rmse, world_to_camera_points
from geometry.lie import hat
from geometry.rotation import angle_between_rotmats
from geometry.pose import angle_between_translations, apply_left_pose_increment_wc

# Bundle of 2D–3D correspondences for PnP
@dataclass(frozen=True)
class PnPCorrespondences:
    # World points as (3,N)
    X_w: np.ndarray
    # Current image points as (2,N)
    x_cur: np.ndarray
    # Landmark id per correspondence as (N,)
    landmark_ids: np.ndarray
    # Current feature index per correspondence as (N,)
    cur_feat_idx: np.ndarray
    # Keyframe feature index per correspondence as (N,)
    kf_feat_idx: np.ndarray


# Build lookup from landmark id to landmark dict
def _landmark_dict_by_id(seed: dict) -> dict[int, dict]:
    # Read landmarks
    landmarks = seed.get("landmarks", [])
    if not isinstance(landmarks, list):
        raise ValueError("seed['landmarks'] must be a list")

    # Build lookup
    out: dict[int, dict] = {}
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        if "id" not in lm:
            continue
        out[int(lm["id"])] = lm

    return out


# Count valid landmark observations
def _landmark_observation_count(lm: dict) -> int:
    # Read observation list
    obs = lm.get("obs", None)
    if not isinstance(obs, list):
        return 0

    # Count valid observation records
    n_obs = 0
    for ob in obs:
        if isinstance(ob, dict):
            n_obs += 1

    return int(n_obs)


# Read the explicit birth source for a landmark
def _landmark_birth_source(lm: dict) -> str | None:
    # Read birth source metadata
    birth_source = lm.get("birth_source", None)
    if not isinstance(birth_source, str):
        return None

    return str(birth_source)


# Determine whether a landmark is explicitly bootstrap-born for pose support
def _landmark_is_bootstrap_born(lm: dict) -> bool:
    return _landmark_birth_source(lm) == "bootstrap"


# Build an empty PnP correspondence bundle
def _empty_pnp_correspondences() -> PnPCorrespondences:
    return PnPCorrespondences(
        X_w=np.zeros((3, 0), dtype=np.float64),
        x_cur=np.zeros((2, 0), dtype=np.float64),
        landmark_ids=np.zeros((0,), dtype=np.int64),
        cur_feat_idx=np.zeros((0,), dtype=np.int64),
        kf_feat_idx=np.zeros((0,), dtype=np.int64),
    )


# Slice a PnP correspondence bundle consistently across all fields
def _slice_pnp_correspondences(corrs: PnPCorrespondences, idx) -> PnPCorrespondences:
    # Convert indexer
    idx = np.asarray(idx)

    # Slice all fields consistently
    return PnPCorrespondences(
        X_w=np.asarray(corrs.X_w[:, idx], dtype=np.float64),
        x_cur=np.asarray(corrs.x_cur[:, idx], dtype=np.float64),
        landmark_ids=np.asarray(corrs.landmark_ids[idx], dtype=np.int64).reshape(-1),
        cur_feat_idx=np.asarray(corrs.cur_feat_idx[idx], dtype=np.int64).reshape(-1),
        kf_feat_idx=np.asarray(corrs.kf_feat_idx[idx], dtype=np.int64).reshape(-1),
    )


# Build DLT design matrix for linear PnP
def _build_pnp_dlt_matrix(X_w, x_hat):
    # --- Checks ---
    # Require same correspondence count
    X_w, x_hat = check_3xN_pair(X_w, x_hat, dtype=float, finite=True)

    # Read sizes
    N = int(X_w.shape[1])

    # Homogeneous world points
    X_h = np.vstack([X_w, np.ones((1, N), dtype=float)])

    # Dehomogenise normalised image coordinates
    w = x_hat[2, :]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("x_hat contains points with near-zero homogeneous scale")

    u = x_hat[0, :] / w
    v = x_hat[1, :] / w

    # Allocate design matrix
    A = np.zeros((2 * N, 12), dtype=float)

    # Fill two equations per correspondence
    for i in range(N):
        # Homogeneous world point
        Xi = X_h[:, i]

        # p1^T X - u p3^T X = 0
        A[2 * i, 0:4] = Xi
        A[2 * i, 8:12] = -float(u[i]) * Xi

        # p2^T X - v p3^T X = 0
        A[2 * i + 1, 4:8] = Xi
        A[2 * i + 1, 8:12] = -float(v[i]) * Xi

    return A


# Recover a metric pose from a projective DLT camera matrix
def _pose_from_dlt_projection(P_tilde, eps=1e-12):
    # --- Checks ---
    # Convert projective camera matrix
    P_tilde = np.asarray(P_tilde, dtype=float)
    if P_tilde.shape != (3, 4):
        raise ValueError(f"P_tilde must be (3,4); got {P_tilde.shape}")

    # Split into linear and translation parts
    M = P_tilde[:, :3]
    p4 = P_tilde[:, 3]

    # Project the linear block onto the nearest rotation
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    # Enforce a proper rotation
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt

    # Recover the common scale
    scale = float(np.trace(M @ R.T) / 3.0)
    if abs(scale) < float(eps):
        raise ValueError("Recovered DLT scale is near zero")

    # Recover translation
    t = p4 / scale

    stats = {
        "M_singular_values": np.asarray(S, dtype=float),
        "scale": float(scale),
        "det_R": float(np.linalg.det(R)),
    }

    return np.asarray(R, dtype=float), np.asarray(t, dtype=float).reshape(3), stats


# Score a candidate pose by reprojection error and positive depth
def _pnp_inlier_mask_from_pose(X_w, x_cur, K, R, t, *, threshold_px=3.0, eps=1e-12):
    # --- Checks ---
    # Check threshold
    threshold_px = check_positive(threshold_px, name="threshold_px", eps=0.0)
    # Check epsilon
    eps = check_positive(eps, name="eps", eps=0.0)

    # Project world points into the candidate camera frame
    X_c = world_to_camera_points(R, t, X_w)

    # Compute squared reprojection errors
    d_sq = reprojection_errors_sq(K, R, t, X_w, x_cur)
    d_sq = np.asarray(d_sq, dtype=float).reshape(-1)

    # Reject invalid projections
    d_sq[~np.isfinite(d_sq)] = np.inf

    # Reject points behind the camera
    d_sq[X_c[2, :] <= float(eps)] = np.inf

    # Convert threshold into an inlier mask
    inlier_mask = d_sq <= float(threshold_px ** 2)

    return inlier_mask, d_sq


# Read image height and width from an image shape tuple
def _image_shape_hw(image_shape) -> tuple[int, int]:
    shape = tuple(image_shape)
    if len(shape) < 2:
        raise ValueError(f"image_shape must have at least two entries; got {image_shape}")

    H = check_int_gt0(int(shape[0]), name="image_shape[0]")
    W = check_int_gt0(int(shape[1]), name="image_shape[1]")

    return int(H), int(W)


# Find connected components in a boolean adjacency matrix
def _connected_components_from_adjacency(adjacency: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be a square matrix; got {adjacency.shape}")

    N = int(adjacency.shape[0])
    visited = np.zeros((N,), dtype=bool)
    component_id_by_item = np.full((N,), -1, dtype=np.int64)
    components: list[list[int]] = []

    for i in range(N):
        if bool(visited[i]):
            continue

        component_id = int(len(components))
        stack = [int(i)]
        visited[i] = True
        members: list[int] = []
        while len(stack) > 0:
            j = int(stack.pop())
            members.append(j)
            component_id_by_item[j] = component_id
            neighbours = np.nonzero(adjacency[j] & ~visited)[0]
            for k in neighbours:
                visited[int(k)] = True
                stack.append(int(k))

        components.append(members)

    return components, component_id_by_item


# Score spatial coverage of PnP inlier pixels in the current image
def pnp_inlier_spatial_coverage(x_cur, inlier_mask, image_shape, *, grid_cols: int = 4, grid_rows: int = 3) -> dict:
    # Check image shape
    H, W = _image_shape_hw(image_shape)

    # Check grid controls
    grid_cols = check_int_gt0(grid_cols, name="grid_cols")
    grid_rows = check_int_gt0(grid_rows, name="grid_rows")

    # Check current image points and inlier mask
    x_cur = check_points_2xN(x_cur, name="x_cur", dtype=float, finite=True)
    N = int(x_cur.shape[1])
    inlier_mask = align_bool_mask_1d(inlier_mask, N, name="pnp_inlier_mask")

    # Start with an empty occupancy grid
    grid = [[0 for _ in range(int(grid_cols))] for _ in range(int(grid_rows))]
    xy = np.asarray(x_cur[:, inlier_mask].T, dtype=np.float64)
    xy = xy[np.isfinite(xy).all(axis=1)]
    n_inliers = int(xy.shape[0])

    if n_inliers == 0:
        return {
            "n_inliers": 0,
            "image_width": int(W),
            "image_height": int(H),
            "grid_cols": int(grid_cols),
            "grid_rows": int(grid_rows),
            "occupied_cells": 0,
            "max_cell_count": 0,
            "max_cell_fraction": None,
            "bbox": None,
            "bbox_area_fraction": None,
            "occupancy_grid": grid,
        }

    # Clip pixels to the image support for grid and area accounting
    xy_clip = xy.copy()
    xy_clip[:, 0] = np.clip(xy_clip[:, 0], 0.0, float(W - 1))
    xy_clip[:, 1] = np.clip(xy_clip[:, 1], 0.0, float(H - 1))

    # Fill the coarse occupancy grid
    for p in xy_clip:
        col = int(np.floor((float(p[0]) / max(float(W), 1.0)) * int(grid_cols)))
        row = int(np.floor((float(p[1]) / max(float(H), 1.0)) * int(grid_rows)))
        col = int(np.clip(col, 0, int(grid_cols) - 1))
        row = int(np.clip(row, 0, int(grid_rows) - 1))
        grid[row][col] += 1

    # Compute cell occupancy and image-space bounding box
    occupied_cells = int(sum(1 for row in grid for value in row if int(value) > 0))
    max_cell_count = int(max(max(row) for row in grid))
    max_cell_fraction = float(max_cell_count / max(n_inliers, 1))

    xmin = float(np.min(xy_clip[:, 0]))
    ymin = float(np.min(xy_clip[:, 1]))
    xmax = float(np.max(xy_clip[:, 0]))
    ymax = float(np.max(xy_clip[:, 1]))
    bbox_w = max(0.0, xmax - xmin)
    bbox_h = max(0.0, ymax - ymin)
    image_area = max(float(W * H), 1.0)
    bbox_area_fraction = float((bbox_w * bbox_h) / image_area)

    return {
        "n_inliers": int(n_inliers),
        "image_width": int(W),
        "image_height": int(H),
        "grid_cols": int(grid_cols),
        "grid_rows": int(grid_rows),
        "occupied_cells": int(occupied_cells),
        "max_cell_count": int(max_cell_count),
        "max_cell_fraction": max_cell_fraction,
        "bbox": [xmin, ymin, xmax, ymax],
        "bbox_area_fraction": bbox_area_fraction,
        "occupancy_grid": grid,
    }


# Score connected components of PnP inlier pixels in the current image
def pnp_inlier_component_support(x_cur, inlier_mask, image_shape, *, radius_px: float = 80.0) -> dict:
    # Check image shape
    H, W = _image_shape_hw(image_shape)

    # Check component radius
    radius_px = check_positive(radius_px, name="radius_px", eps=0.0)

    # Check current image points and inlier mask
    x_cur = check_points_2xN(x_cur, name="x_cur", dtype=float, finite=True)
    N = int(x_cur.shape[1])
    inlier_mask = align_bool_mask_1d(inlier_mask, N, name="pnp_inlier_mask")

    # Collect finite inlier pixels
    xy = np.asarray(x_cur[:, inlier_mask].T, dtype=np.float64)
    xy = xy[np.isfinite(xy).all(axis=1)]
    n_inliers = int(xy.shape[0])

    if n_inliers == 0:
        return {
            "n_inliers": 0,
            "image_width": int(W),
            "image_height": int(H),
            "radius_px": float(radius_px),
            "component_count": 0,
            "largest_component_size": 0,
            "largest_component_fraction": None,
            "largest_component_bbox": None,
            "largest_component_bbox_area_fraction": None,
            "component_sizes": [],
            "component_bboxes": [],
            "component_bbox_area_fractions": [],
            "component_id_by_inlier": [],
        }

    # Clip pixels to the image support for area accounting
    xy_clip = xy.copy()
    xy_clip[:, 0] = np.clip(xy_clip[:, 0], 0.0, float(W - 1))
    xy_clip[:, 1] = np.clip(xy_clip[:, 1], 0.0, float(H - 1))

    # Build a radius adjacency matrix
    dxy = xy_clip[:, None, :] - xy_clip[None, :, :]
    dist_sq = np.sum(dxy * dxy, axis=2)
    radius_sq = float(radius_px) * float(radius_px)
    adjacency = dist_sq <= radius_sq

    # Find connected components with a small explicit stack
    components, component_id_by_inlier = _connected_components_from_adjacency(adjacency)

    # Summarise each component bbox
    image_area = max(float(W * H), 1.0)
    component_sizes: list[int] = []
    component_bboxes: list[list[float]] = []
    component_bbox_area_fractions: list[float] = []
    for members in components:
        pts = xy_clip[np.asarray(members, dtype=np.int64)]
        xmin = float(np.min(pts[:, 0]))
        ymin = float(np.min(pts[:, 1]))
        xmax = float(np.max(pts[:, 0]))
        ymax = float(np.max(pts[:, 1]))
        bbox_w = max(0.0, xmax - xmin)
        bbox_h = max(0.0, ymax - ymin)
        bbox_area_fraction = float((bbox_w * bbox_h) / image_area)

        component_sizes.append(int(len(members)))
        component_bboxes.append([xmin, ymin, xmax, ymax])
        component_bbox_area_fractions.append(bbox_area_fraction)

    # Select the largest component
    largest_component_index = int(np.argmax(np.asarray(component_sizes, dtype=np.int64)))
    largest_component_size = int(component_sizes[largest_component_index])
    largest_component_fraction = float(largest_component_size / max(n_inliers, 1))

    return {
        "n_inliers": int(n_inliers),
        "image_width": int(W),
        "image_height": int(H),
        "radius_px": float(radius_px),
        "component_count": int(len(components)),
        "largest_component_size": int(largest_component_size),
        "largest_component_fraction": largest_component_fraction,
        "largest_component_bbox": component_bboxes[largest_component_index],
        "largest_component_bbox_area_fraction": float(component_bbox_area_fractions[largest_component_index]),
        "component_sizes": component_sizes,
        "component_bboxes": component_bboxes,
        "component_bbox_area_fractions": component_bbox_area_fractions,
        "component_id_by_inlier": [int(v) for v in component_id_by_inlier.tolist()],
    }


# Evaluate configured spatial-coverage rejection reasons
def pnp_spatial_coverage_gate_reasons(
    coverage: dict,
    *,
    min_occupied_cells: int,
    max_single_cell_fraction: float,
    min_bbox_area_fraction: float,
) -> list[str]:
    coverage = coverage if isinstance(coverage, dict) else {}
    gate_reasons: list[str] = []
    occupied_cells = int(coverage.get("occupied_cells", 0))
    max_cell_fraction = coverage.get("max_cell_fraction", None)
    bbox_area_fraction = coverage.get("bbox_area_fraction", None)

    if occupied_cells < int(min_occupied_cells):
        gate_reasons.append("occupied_cells_low")
    if max_cell_fraction is None or float(max_cell_fraction) > float(max_single_cell_fraction):
        gate_reasons.append("single_cell_fraction_high")
    if bbox_area_fraction is None or float(bbox_area_fraction) < float(min_bbox_area_fraction):
        gate_reasons.append("bbox_area_fraction_low")

    return gate_reasons


# Evaluate configured component-support rejection reasons
def pnp_component_support_gate_reasons(
    component_support: dict,
    *,
    min_component_count: int,
    max_largest_component_fraction: float,
    min_largest_component_bbox_area_fraction: float,
) -> list[str]:
    component_support = component_support if isinstance(component_support, dict) else {}
    gate_reasons: list[str] = []
    component_count = int(component_support.get("component_count", 0))
    largest_component_fraction = component_support.get("largest_component_fraction", None)
    largest_component_bbox_area_fraction = component_support.get("largest_component_bbox_area_fraction", None)

    if component_count < int(min_component_count):
        gate_reasons.append("component_count_low")
    if largest_component_fraction is None or float(largest_component_fraction) > float(max_largest_component_fraction):
        gate_reasons.append("largest_component_fraction_high")
    if largest_component_bbox_area_fraction is None or float(largest_component_bbox_area_fraction) < float(min_largest_component_bbox_area_fraction):
        gate_reasons.append("largest_component_bbox_area_fraction_low")

    return gate_reasons


# Classify threshold-stability diagnostics from pose and support comparisons
def pnp_threshold_stability_flags(
    *,
    ref_pose_ok: bool,
    compare_pose_ok: bool,
    ref_threshold_px: float,
    compare_threshold_px: float,
    support_iou,
    support_union: int,
    translation_direction_delta_deg,
    camera_centre_direction_delta_deg=None,
    min_support_iou: float = 0.25,
    max_translation_direction_deg: float = 120.0,
    max_camera_centre_direction_deg: float = 120.0,
    disjoint_support_iou: float = 0.05,
) -> dict:
    ref_is_looser = float(ref_threshold_px) > float(compare_threshold_px)
    compare_is_looser = float(compare_threshold_px) > float(ref_threshold_px)
    one_solution_only_at_looser = (bool(ref_pose_ok) != bool(compare_pose_ok)) and (
        (bool(ref_pose_ok) and bool(ref_is_looser)) or (bool(compare_pose_ok) and bool(compare_is_looser))
    )
    one_solution_only_at_stricter = (bool(ref_pose_ok) != bool(compare_pose_ok)) and (
        (bool(ref_pose_ok) and bool(compare_is_looser)) or (bool(compare_pose_ok) and bool(ref_is_looser))
    )

    supports_effectively_disjoint = bool(compare_pose_ok) and support_iou is not None and int(support_union) > 0 and float(support_iou) <= float(disjoint_support_iou)
    support_iou_low = bool(compare_pose_ok) and support_iou is not None and float(support_iou) < float(min_support_iou)
    translation_direction_disagrees = translation_direction_delta_deg is not None and float(translation_direction_delta_deg) > float(max_translation_direction_deg)
    camera_centre_direction_disagrees = camera_centre_direction_delta_deg is not None and float(camera_centre_direction_delta_deg) > float(max_camera_centre_direction_deg)

    instability_reasons: list[str] = []
    if bool(one_solution_only_at_looser):
        instability_reasons.append("looser_solution_only")
    if bool(supports_effectively_disjoint):
        instability_reasons.append("support_effectively_disjoint")
    elif bool(support_iou_low):
        instability_reasons.append("support_iou_low")
    if bool(translation_direction_disagrees):
        instability_reasons.append("translation_direction_disagreement")
    if bool(camera_centre_direction_disagrees):
        instability_reasons.append("camera_centre_direction_disagreement")

    unstable = len(instability_reasons) > 0

    return {
        "one_solution_only_at_looser_threshold": bool(one_solution_only_at_looser),
        "one_solution_only_at_stricter_threshold": bool(one_solution_only_at_stricter),
        "supports_effectively_disjoint": bool(supports_effectively_disjoint),
        "support_iou_low": bool(support_iou_low),
        "translation_direction_disagrees": bool(translation_direction_disagrees),
        "camera_centre_direction_disagrees": bool(camera_centre_direction_disagrees),
        "unstable": bool(unstable),
        "instability_reasons": instability_reasons,
        "classification": "unstable" if bool(unstable) else ("stable" if bool(compare_pose_ok) else "unavailable"),
    }


# Convert correspondence pixels to canonical (N,2) image coordinates
def _as_correspondence_xy_N2(xy, *, name: str) -> np.ndarray:
    # Convert array
    arr = np.asarray(xy, dtype=np.float64)

    # Accept the tracking convention
    if arr.ndim == 2 and arr.shape[1] == 2:
        out = arr

    # Accept the PnP convention
    elif arr.ndim == 2 and arr.shape[0] == 2:
        out = arr.T

    else:
        raise ValueError(f"{name} must be (N,2) or (2,N); got {arr.shape}")

    # Require finite pixels
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")

    return np.asarray(out, dtype=np.float64)


# Score tracked pose candidates by local displacement consistency in keyframe image space
def pnp_local_displacement_consistency_mask(
    xy_kf,
    xy_cur,
    *,
    radius_px: float = 80.0,
    min_neighbours: int = 3,
    max_median_residual_px: float = 12.0,
    min_keep: int = 0,
) -> tuple[np.ndarray, dict]:
    # Check image coordinates
    xy_kf = _as_correspondence_xy_N2(xy_kf, name="xy_kf")
    xy_cur = _as_correspondence_xy_N2(xy_cur, name="xy_cur")
    if xy_kf.shape != xy_cur.shape:
        raise ValueError(f"xy_kf and xy_cur must have the same shape; got {xy_kf.shape} and {xy_cur.shape}")

    # Check controls
    radius_px = check_positive(radius_px, name="radius_px", eps=0.0)
    min_neighbours = check_int_ge0(min_neighbours, name="min_neighbours")
    max_median_residual_px = check_positive(max_median_residual_px, name="max_median_residual_px", eps=0.0)
    min_keep = check_int_ge0(min_keep, name="min_keep")

    # Read correspondence count
    N = int(xy_kf.shape[0])

    # Start from a keep-all mask for empty inputs
    if N == 0:
        stats = {
            "n_input": 0,
            "n_keep": 0,
            "n_removed": 0,
            "radius_px": float(radius_px),
            "min_neighbours": int(min_neighbours),
            "max_median_residual_px": float(max_median_residual_px),
            "min_keep": int(min_keep),
            "min_keep_fallback": False,
            "n_too_few_neighbours": 0,
            "n_motion_inconsistent": 0,
            "neighbour_count_min": None,
            "neighbour_count_median": None,
            "neighbour_count_max": None,
            "residual_median_px": None,
            "residual_p75_px": None,
            "residual_p90_px": None,
            "residual_max_px": None,
        }
        return np.zeros((0,), dtype=bool), stats

    # Compute local neighbourhoods in the reference keyframe
    dxy_kf = xy_kf[:, None, :] - xy_kf[None, :, :]
    dist_kf = np.sqrt(np.sum(dxy_kf * dxy_kf, axis=2))
    neighbour_mask = (dist_kf <= float(radius_px)) & ~np.eye(N, dtype=bool)
    neighbour_count = np.sum(neighbour_mask, axis=1).astype(np.int64)

    # Compute current-frame displacement vectors
    displacement = xy_cur - xy_kf
    residual_px = np.full((N,), np.nan, dtype=np.float64)
    keep = np.ones((N,), dtype=bool)

    # Compare each displacement with the local median displacement
    for i in range(N):
        local = neighbour_mask[i]
        if int(neighbour_count[i]) < int(min_neighbours):
            keep[i] = False
            continue
        if int(neighbour_count[i]) == 0:
            keep[i] = True
            continue

        local_disp = displacement[local]
        median_disp = np.median(local_disp, axis=0)
        residual_px[i] = float(np.linalg.norm(displacement[i] - median_disp))
        keep[i] = bool(residual_px[i] <= float(max_median_residual_px))

    # Preserve the bundle when the configured floor would be violated
    min_keep_fallback = False
    if int(min_keep) > 0 and int(np.sum(keep)) < int(min_keep):
        min_keep_fallback = True
        keep = np.ones((N,), dtype=bool)

    # Summarise residuals where a local model was available
    finite_residual = residual_px[np.isfinite(residual_px)]
    if int(finite_residual.size) == 0:
        residual_median_px = None
        residual_p75_px = None
        residual_p90_px = None
        residual_max_px = None
    else:
        residual_median_px = float(np.median(finite_residual))
        residual_p75_px = float(np.percentile(finite_residual, 75))
        residual_p90_px = float(np.percentile(finite_residual, 90))
        residual_max_px = float(np.max(finite_residual))

    # Pack compact diagnostic stats
    too_few_neighbours = neighbour_count < int(min_neighbours)
    motion_inconsistent = np.isfinite(residual_px) & (residual_px > float(max_median_residual_px))
    stats = {
        "n_input": int(N),
        "n_keep": int(np.sum(keep)),
        "n_removed": int(N - int(np.sum(keep))),
        "radius_px": float(radius_px),
        "min_neighbours": int(min_neighbours),
        "max_median_residual_px": float(max_median_residual_px),
        "min_keep": int(min_keep),
        "min_keep_fallback": bool(min_keep_fallback),
        "n_too_few_neighbours": int(np.sum(too_few_neighbours)),
        "n_motion_inconsistent": int(np.sum(motion_inconsistent)),
        "neighbour_count_min": int(np.min(neighbour_count)),
        "neighbour_count_median": float(np.median(neighbour_count)),
        "neighbour_count_max": int(np.max(neighbour_count)),
        "residual_median_px": residual_median_px,
        "residual_p75_px": residual_p75_px,
        "residual_p90_px": residual_p90_px,
        "residual_max_px": residual_max_px,
    }

    return np.asarray(keep, dtype=bool), stats


# Summarise current-image radius density
def _spatial_radius_density_stats(xy: np.ndarray, *, radius_px: float, max_points_per_radius: int) -> dict:
    # Convert image points
    xy = _as_correspondence_xy_N2(xy, name="xy")
    N = int(xy.shape[0])

    # Return an empty summary
    if N == 0:
        return {
            "count": 0,
            "radius_px": float(radius_px),
            "max_points_per_radius": int(max_points_per_radius),
            "n_dense_points": 0,
            "density_count_min": None,
            "density_count_median": None,
            "density_count_max": None,
            "component_count": 0,
            "largest_component_count": 0,
            "largest_component_fraction": None,
        }

    # Compute pairwise radius neighbourhoods
    dxy = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(dxy * dxy, axis=2))
    neighbour_mask = dist <= float(radius_px)
    density_count = np.sum(neighbour_mask, axis=1).astype(np.int64)

    # Find connected radius components
    components, _ = _connected_components_from_adjacency(neighbour_mask)
    component_sizes = [int(len(members)) for members in components]

    # Pack density and component stats
    largest_component_count = int(max(component_sizes)) if len(component_sizes) > 0 else 0
    return {
        "count": int(N),
        "radius_px": float(radius_px),
        "max_points_per_radius": int(max_points_per_radius),
        "n_dense_points": int(np.sum(density_count > int(max_points_per_radius))),
        "density_count_min": int(np.min(density_count)),
        "density_count_median": float(np.median(density_count)),
        "density_count_max": int(np.max(density_count)),
        "component_count": int(len(component_sizes)),
        "largest_component_count": int(largest_component_count),
        "largest_component_fraction": float(largest_component_count / max(N, 1)),
    }


# Thin current-image pose-support points by a local radius cap
def pnp_current_image_spatial_thinning_mask(
    xy_cur,
    *,
    radius_px: float = 20.0,
    max_points_per_radius: int = 16,
    min_keep: int = 0,
) -> tuple[np.ndarray, dict]:
    # Check image coordinates
    xy_cur = _as_correspondence_xy_N2(xy_cur, name="xy_cur")

    # Check controls
    radius_px = check_positive(radius_px, name="radius_px", eps=0.0)
    max_points_per_radius = check_int_gt0(max_points_per_radius, name="max_points_per_radius")
    min_keep = check_int_ge0(min_keep, name="min_keep")

    # Read correspondence count
    N = int(xy_cur.shape[0])

    # Start from an empty keep mask
    if N == 0:
        stats = {
            "n_input": 0,
            "n_keep": 0,
            "n_removed": 0,
            "radius_px": float(radius_px),
            "max_points_per_radius": int(max_points_per_radius),
            "min_keep": int(min_keep),
            "min_keep_fallback": False,
            "before": _spatial_radius_density_stats(xy_cur, radius_px=float(radius_px), max_points_per_radius=int(max_points_per_radius)),
            "after": _spatial_radius_density_stats(xy_cur, radius_px=float(radius_px), max_points_per_radius=int(max_points_per_radius)),
        }
        return np.zeros((0,), dtype=bool), stats

    # Compute local density in the current image
    dxy = xy_cur[:, None, :] - xy_cur[None, :, :]
    dist = np.sqrt(np.sum(dxy * dxy, axis=2))
    radius_mask = dist <= float(radius_px)
    density_count = np.sum(radius_mask, axis=1).astype(np.int64)

    # Keep sparse candidates first to preserve broad support
    order = np.lexsort((np.arange(N, dtype=np.int64), density_count))
    keep = np.zeros((N,), dtype=bool)
    kept_count_by_candidate_radius = np.zeros((N,), dtype=np.int64)

    # Greedily accept candidates without breaking the local radius cap
    for idx_raw in order:
        idx = int(idx_raw)
        affected = radius_mask[:, idx]
        affected_kept = affected & keep

        if int(kept_count_by_candidate_radius[idx]) + 1 > int(max_points_per_radius):
            continue
        if np.any(kept_count_by_candidate_radius[affected_kept] + 1 > int(max_points_per_radius)):
            continue

        keep[idx] = True
        kept_count_by_candidate_radius[affected] += 1

    # Preserve the bundle when the configured floor would be violated
    min_keep_fallback = False
    if int(min_keep) > 0 and int(np.sum(keep)) < int(min_keep):
        min_keep_fallback = True
        keep = np.ones((N,), dtype=bool)

    # Summarise thinning behaviour
    stats = {
        "n_input": int(N),
        "n_keep": int(np.sum(keep)),
        "n_removed": int(N - int(np.sum(keep))),
        "radius_px": float(radius_px),
        "max_points_per_radius": int(max_points_per_radius),
        "min_keep": int(min_keep),
        "min_keep_fallback": bool(min_keep_fallback),
        "before": _spatial_radius_density_stats(xy_cur, radius_px=float(radius_px), max_points_per_radius=int(max_points_per_radius)),
        "after": _spatial_radius_density_stats(xy_cur[keep], radius_px=float(radius_px), max_points_per_radius=int(max_points_per_radius)),
    }

    return np.asarray(keep, dtype=bool), stats


# Build the Gauss-Newton linear system for pose-only reprojection refinement
def _linearise_pose_only_reprojection(X_w, x_cur, K, R, t, eps=1e-12):
    # --- Checks ---
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    # Check correspondences
    X_w, x_cur = check_3xN_2xN_cols(X_w, x_cur, nameX="X_w", namex="x_cur", dtype=float, finite=True)

    # Transform world points into the current camera frame
    X_c = world_to_camera_points(R, t, X_w)

    # Read intrinsics
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Keep only points in front of the camera
    z = X_c[2, :]
    valid = np.isfinite(z) & (z > float(eps))

    # Require at least one valid point
    if not np.any(valid):
        raise ValueError("No valid positive-depth points for pose refinement")

    # Slice valid correspondences
    X_c_valid = X_c[:, valid]
    x_obs_valid = x_cur[:, valid]

    # Read valid count
    N = int(X_c_valid.shape[1])

    # Allocate Jacobian and residual vector
    J = np.zeros((2 * N, 6), dtype=float)
    r = np.zeros((2 * N,), dtype=float)

    # Linearise one correspondence at a time
    for i in range(N):
        # Read camera-frame point
        X = float(X_c_valid[0, i])
        Y = float(X_c_valid[1, i])
        Z = float(X_c_valid[2, i])

        # Predicted pixel location
        u_hat = fx * (X / Z) + cx
        v_hat = fy * (Y / Z) + cy

        # Observed pixel location
        u_obs = float(x_obs_valid[0, i])
        v_obs = float(x_obs_valid[1, i])

        # Residual = observed - predicted
        r[2 * i] = u_obs - u_hat
        r[2 * i + 1] = v_obs - v_hat

        # Projection Jacobian with respect to camera-frame point
        J_proj = np.array(
            [
                [fx / Z, 0.0, -fx * X / (Z ** 2)],
                [0.0, fy / Z, -fy * Y / (Z ** 2)],
            ],
            dtype=float,
        )

        # Camera-frame point Jacobian with respect to left pose increment
        J_pose = np.hstack([np.eye(3, dtype=float), -hat(X_c_valid[:, i])])

        # Residual Jacobian
        J_i = -J_proj @ J_pose

        # Write block
        J[2 * i : 2 * i + 2, :] = J_i

    return J, r, valid


# Build 2D–3D correspondences from seed and tracking output with pose-support stats
def build_pnp_correspondences_with_stats(
    seed: dict,
    track_out: dict,
    *,
    min_landmark_observations: int = 2,
    allow_bootstrap_landmarks_for_pose: bool = True,
    min_post_bootstrap_observations_for_pose: int = 3,
    enable_local_consistency_filter: bool = False,
    local_consistency_radius_px: float = 80.0,
    local_consistency_min_neighbours: int = 3,
    local_consistency_max_median_residual_px: float = 12.0,
    local_consistency_min_keep: int = 0,
    enable_spatial_thinning_filter: bool = False,
    spatial_thinning_radius_px: float = 20.0,
    spatial_thinning_max_points_per_radius: int = 16,
    spatial_thinning_min_keep: int = 0,
) -> tuple[PnPCorrespondences, dict]:
    # --- Checks ---
    # Seed must be a dict
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    # Track output must be a dict
    if not isinstance(track_out, dict):
        raise ValueError("track_out must be a dict")

    # Check landmark-quality filter
    min_landmark_observations = check_int_gt0(
        min_landmark_observations,
        name="min_landmark_observations",
    )
    allow_bootstrap_landmarks_for_pose = bool(allow_bootstrap_landmarks_for_pose)
    min_post_bootstrap_observations_for_pose = check_int_gt0(
        min_post_bootstrap_observations_for_pose,
        name="min_post_bootstrap_observations_for_pose",
    )
    enable_local_consistency_filter = bool(enable_local_consistency_filter)
    local_consistency_radius_px = check_positive(
        local_consistency_radius_px,
        name="local_consistency_radius_px",
        eps=0.0,
    )
    local_consistency_min_neighbours = check_int_ge0(
        local_consistency_min_neighbours,
        name="local_consistency_min_neighbours",
    )
    local_consistency_max_median_residual_px = check_positive(
        local_consistency_max_median_residual_px,
        name="local_consistency_max_median_residual_px",
        eps=0.0,
    )
    local_consistency_min_keep = check_int_ge0(
        local_consistency_min_keep,
        name="local_consistency_min_keep",
    )
    enable_spatial_thinning_filter = bool(enable_spatial_thinning_filter)
    spatial_thinning_radius_px = check_positive(
        spatial_thinning_radius_px,
        name="spatial_thinning_radius_px",
        eps=0.0,
    )
    spatial_thinning_max_points_per_radius = check_int_gt0(
        spatial_thinning_max_points_per_radius,
        name="spatial_thinning_max_points_per_radius",
    )
    spatial_thinning_min_keep = check_int_ge0(
        spatial_thinning_min_keep,
        name="spatial_thinning_min_keep",
    )

    # Read landmark id map from keyframe features to landmarks
    landmark_id_by_feat1 = np.asarray(
        seed.get("landmark_id_by_feat1", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)

    # Read tracked feature indices
    kf_feat_idx = np.asarray(
        track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)

    cur_feat_idx = np.asarray(
        track_out.get("cur_feat_idx", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)

    # Read current image points
    xy_cur = np.asarray(
        track_out.get("xy_cur", np.zeros((0, 2), dtype=np.float64)),
        dtype=np.float64,
    )

    xy_kf = np.asarray(
        track_out.get("xy_kf", np.zeros((0, 2), dtype=np.float64)),
        dtype=np.float64,
    )

    # Require aligned tracking outputs
    if kf_feat_idx.ndim != 1:
        raise ValueError(f"kf_feat_idx must be 1D; got {kf_feat_idx.shape}")
    if cur_feat_idx.ndim != 1:
        raise ValueError(f"cur_feat_idx must be 1D; got {cur_feat_idx.shape}")
    if kf_feat_idx.size != cur_feat_idx.size:
        raise ValueError(
            f"kf_feat_idx and cur_feat_idx must have same size; got {kf_feat_idx.size} and {cur_feat_idx.size}"
        )
    if xy_cur.ndim != 2 or xy_cur.shape[1] != 2:
        raise ValueError(f"xy_cur must be (N,2); got {xy_cur.shape}")
    if xy_cur.shape[0] != kf_feat_idx.size:
        raise ValueError(
            f"xy_cur and tracked feature indices must have same N; got {xy_cur.shape[0]} and {kf_feat_idx.size}"
        )
    if xy_kf.ndim != 2 or xy_kf.shape[1] != 2:
        raise ValueError(f"xy_kf must be (N,2); got {xy_kf.shape}")
    if xy_kf.shape[0] != kf_feat_idx.size:
        raise ValueError(
            f"xy_kf and tracked feature indices must have same N; got {xy_kf.shape[0]} and {kf_feat_idx.size}"
        )

    # Build landmark lookup
    lm_by_id = _landmark_dict_by_id(seed)

    # Collect valid correspondences
    X_cols: list[np.ndarray] = []
    x_cols: list[np.ndarray] = []
    x_kf_rows: list[np.ndarray] = []
    landmark_ids: list[int] = []
    cur_idx_keep: list[int] = []
    kf_idx_keep: list[int] = []

    # Start pose-support stats
    stats = {
        "n_corr_raw": 0,
        "n_corr_bootstrap_born": 0,
        "n_corr_post_bootstrap_born": 0,
        "n_corr_after_pose_filter": 0,
        "n_corr_after_local_consistency_filter": 0,
        "n_corr_after_spatial_thinning_filter": 0,
        "n_corr_bootstrap_used": 0,
        "n_corr_post_bootstrap_used": 0,
        "n_corr_bootstrap_after_local_consistency": 0,
        "n_corr_post_bootstrap_after_local_consistency": 0,
        "n_corr_bootstrap_after_spatial_thinning": 0,
        "n_corr_post_bootstrap_after_spatial_thinning": 0,
        "pnp_local_consistency_filter_enabled": bool(enable_local_consistency_filter),
        "pnp_local_consistency_filter_evaluated": False,
        "pnp_local_consistency_filter_applied": False,
        "pnp_local_consistency_filter_removed": 0,
        "pnp_local_consistency_filter_reason": None,
        "pnp_local_consistency_radius_px": float(local_consistency_radius_px),
        "pnp_local_consistency_min_neighbours": int(local_consistency_min_neighbours),
        "pnp_local_consistency_max_median_residual_px": float(local_consistency_max_median_residual_px),
        "pnp_local_consistency_min_keep": int(local_consistency_min_keep),
        "pnp_local_consistency_stats": None,
        "pnp_spatial_thinning_filter_enabled": bool(enable_spatial_thinning_filter),
        "pnp_spatial_thinning_filter_evaluated": False,
        "pnp_spatial_thinning_filter_applied": False,
        "pnp_spatial_thinning_filter_removed": 0,
        "pnp_spatial_thinning_filter_reason": None,
        "pnp_spatial_thinning_radius_px": float(spatial_thinning_radius_px),
        "pnp_spatial_thinning_max_points_per_radius": int(spatial_thinning_max_points_per_radius),
        "pnp_spatial_thinning_min_keep": int(spatial_thinning_min_keep),
        "pnp_spatial_thinning_stats": None,
    }

    # Walk tracked correspondences
    for i in range(kf_feat_idx.size):
        # Keyframe feature index
        feat1 = int(kf_feat_idx[i])

        # Skip invalid keyframe feature index
        if feat1 < 0 or feat1 >= landmark_id_by_feat1.size:
            continue

        # Map keyframe feature to landmark id
        lm_id = int(landmark_id_by_feat1[feat1])

        # Skip unmatched features
        if lm_id < 0:
            continue

        # Lookup landmark
        lm = lm_by_id.get(lm_id, None)
        if lm is None:
            continue

        # Read world point
        X_w = np.asarray(lm.get("X_w", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        if X_w.size != 3:
            continue
        if not np.isfinite(X_w).all():
            continue

        # Read current image point
        x = np.asarray(xy_cur[i], dtype=np.float64).reshape(-1)
        if x.size != 2:
            continue
        if not np.isfinite(x).all():
            continue

        # Read keyframe image point
        x_kf = np.asarray(xy_kf[i], dtype=np.float64).reshape(-1)
        if x_kf.size != 2:
            continue
        if not np.isfinite(x_kf).all():
            continue

        # Classify the landmark origin and update raw counts
        n_obs = _landmark_observation_count(lm)
        bootstrap_born = _landmark_is_bootstrap_born(lm)

        stats["n_corr_raw"] += 1
        if bootstrap_born:
            stats["n_corr_bootstrap_born"] += 1
        else:
            stats["n_corr_post_bootstrap_born"] += 1

        # Apply the origin-aware pose-support gate
        if bootstrap_born:
            if not bool(allow_bootstrap_landmarks_for_pose):
                continue
            min_obs_required = int(min_landmark_observations)
        else:
            min_obs_required = max(
                int(min_landmark_observations),
                int(min_post_bootstrap_observations_for_pose),
            )

        if n_obs < int(min_obs_required):
            continue

        # Append valid correspondence
        X_cols.append(X_w.reshape(3, 1))
        x_cols.append(x.reshape(2, 1))
        x_kf_rows.append(x_kf.reshape(1, 2))
        landmark_ids.append(lm_id)
        cur_idx_keep.append(int(cur_feat_idx[i]))
        kf_idx_keep.append(feat1)

        # Update kept pose-support counts
        if bootstrap_born:
            stats["n_corr_bootstrap_used"] += 1
        else:
            stats["n_corr_post_bootstrap_used"] += 1

    # Return empty bundle if nothing survived
    if len(X_cols) == 0:
        return _empty_pnp_correspondences(), stats

    # Stack into canonical arrays
    X_w = np.hstack(X_cols)
    x_cur = np.hstack(x_cols)
    xy_kf_keep = np.vstack(x_kf_rows)
    landmark_ids_arr = np.asarray(landmark_ids, dtype=np.int64)
    cur_feat_idx_arr = np.asarray(cur_idx_keep, dtype=np.int64)
    kf_feat_idx_arr = np.asarray(kf_idx_keep, dtype=np.int64)

    # Final checks
    X_w = check_points_3xN(X_w, name="X_w", dtype=float, finite=True)
    x_cur = check_points_2xN(x_cur, name="x_cur", dtype=float, finite=True)

    # Record the final kept correspondence count
    stats["n_corr_after_pose_filter"] = int(X_w.shape[1])
    stats["n_corr_after_local_consistency_filter"] = int(X_w.shape[1])
    stats["n_corr_after_spatial_thinning_filter"] = int(X_w.shape[1])
    stats["n_corr_bootstrap_after_local_consistency"] = int(stats["n_corr_bootstrap_used"])
    stats["n_corr_post_bootstrap_after_local_consistency"] = int(stats["n_corr_post_bootstrap_used"])
    stats["n_corr_bootstrap_after_spatial_thinning"] = int(stats["n_corr_bootstrap_used"])
    stats["n_corr_post_bootstrap_after_spatial_thinning"] = int(stats["n_corr_post_bootstrap_used"])

    # Build the correspondence bundle before local consistency filtering
    corrs = PnPCorrespondences(
        X_w=X_w,
        x_cur=x_cur,
        landmark_ids=landmark_ids_arr,
        cur_feat_idx=cur_feat_idx_arr,
        kf_feat_idx=kf_feat_idx_arr,
    )

    # Optionally reject locally inconsistent pose-support tracks
    if bool(enable_local_consistency_filter):
        keep_mask, local_stats = pnp_local_displacement_consistency_mask(
            xy_kf_keep,
            x_cur.T,
            radius_px=float(local_consistency_radius_px),
            min_neighbours=int(local_consistency_min_neighbours),
            max_median_residual_px=float(local_consistency_max_median_residual_px),
            min_keep=int(local_consistency_min_keep),
        )
        keep_mask = align_bool_mask_1d(keep_mask, int(X_w.shape[1]), name="pnp_local_consistency_keep_mask")
        stats["pnp_local_consistency_filter_evaluated"] = True
        stats["pnp_local_consistency_filter_applied"] = True
        stats["pnp_local_consistency_filter_removed"] = int(np.sum(~keep_mask))
        stats["pnp_local_consistency_stats"] = local_stats

        corrs = _slice_pnp_correspondences(corrs, keep_mask)
        birth_bootstrap = np.asarray([_landmark_is_bootstrap_born(lm_by_id.get(int(lm_id), {})) for lm_id in corrs.landmark_ids], dtype=bool)
        stats["n_corr_after_local_consistency_filter"] = int(corrs.X_w.shape[1])
        stats["n_corr_bootstrap_after_local_consistency"] = int(np.sum(birth_bootstrap))
        stats["n_corr_post_bootstrap_after_local_consistency"] = int(corrs.X_w.shape[1] - int(np.sum(birth_bootstrap)))
        stats["n_corr_after_spatial_thinning_filter"] = int(corrs.X_w.shape[1])
        stats["n_corr_bootstrap_after_spatial_thinning"] = int(np.sum(birth_bootstrap))
        stats["n_corr_post_bootstrap_after_spatial_thinning"] = int(corrs.X_w.shape[1] - int(np.sum(birth_bootstrap)))

    # Optionally thin dense current-image pose-support clusters
    if bool(enable_spatial_thinning_filter):
        keep_mask, thinning_stats = pnp_current_image_spatial_thinning_mask(
            corrs.x_cur,
            radius_px=float(spatial_thinning_radius_px),
            max_points_per_radius=int(spatial_thinning_max_points_per_radius),
            min_keep=int(spatial_thinning_min_keep),
        )
        keep_mask = align_bool_mask_1d(keep_mask, int(corrs.X_w.shape[1]), name="pnp_spatial_thinning_keep_mask")
        stats["pnp_spatial_thinning_filter_evaluated"] = True
        stats["pnp_spatial_thinning_filter_applied"] = True
        stats["pnp_spatial_thinning_filter_removed"] = int(np.sum(~keep_mask))
        stats["pnp_spatial_thinning_stats"] = thinning_stats

        corrs = _slice_pnp_correspondences(corrs, keep_mask)
        birth_bootstrap = np.asarray([_landmark_is_bootstrap_born(lm_by_id.get(int(lm_id), {})) for lm_id in corrs.landmark_ids], dtype=bool)
        stats["n_corr_after_spatial_thinning_filter"] = int(corrs.X_w.shape[1])
        stats["n_corr_bootstrap_after_spatial_thinning"] = int(np.sum(birth_bootstrap))
        stats["n_corr_post_bootstrap_after_spatial_thinning"] = int(corrs.X_w.shape[1] - int(np.sum(birth_bootstrap)))

    return corrs, stats


# Estimate camera pose from 2D–3D correspondences with linear DLT PnP
def estimate_pose_pnp(
    corrs: PnPCorrespondences,
    K,
    *,
    min_points: int = 6,
    rank_tol: float = 1e-10,
    min_cheirality_ratio: float = 0.5,
    eps: float = 1e-12,
):
    # --- Checks ---
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    # Check correspondences
    X_w, x_cur = check_3xN_2xN_cols(corrs.X_w, corrs.x_cur, nameX="corrs.X_w", namex="corrs.x_cur", dtype=float, finite=True)

    # Validate scalar parameters
    min_points = max(6, int(min_points))
    rank_tol = float(rank_tol)
    min_cheirality_ratio = float(min_cheirality_ratio)
    eps = float(eps)

    # Initialise stats
    N = int(X_w.shape[1])
    stats = {
        "n_corr": N,
        "min_points": int(min_points),
        "method": "dlt",
    }

    # Require enough correspondences
    if N < min_points:
        stats.update({"reason": "too_few_correspondences"})
        return None, None, stats

    # Convert pixels to normalised homogeneous image coordinates
    x_hat = pixel_to_normalised(K, x_cur)

    # Build DLT design matrix
    try:
        A = _build_pnp_dlt_matrix(X_w, x_hat)
    except Exception as exc:
        stats.update({"reason": "dlt_matrix_failed", "error": str(exc)})
        return None, None, stats

    # Solve homogeneous least squares
    try:
        _, S_A, Vt_A = np.linalg.svd(A)
    except Exception as exc:
        stats.update({"reason": "dlt_svd_failed", "error": str(exc)})
        return None, None, stats

    # Estimate numerical rank
    if S_A.size == 0:
        stats.update({"reason": "dlt_empty_spectrum"})
        return None, None, stats

    s0 = float(S_A[0])
    tol = float(rank_tol) * max(s0, eps)
    rank = int(np.sum(S_A > tol))
    stats.update({"A_rank": rank, "A_singular_values": np.asarray(S_A, dtype=float)})

    # Require near-full rank
    if rank < 11:
        stats.update({"reason": "dlt_rank_deficient"})
        return None, None, stats

    # Recover projective camera matrix
    P_tilde = Vt_A[-1, :].reshape(3, 4)

    # Recover metric pose
    try:
        R, t, pose_stats = _pose_from_dlt_projection(P_tilde, eps=eps)
    except Exception as exc:
        stats.update({"reason": "pose_from_dlt_failed", "error": str(exc)})
        return None, None, stats

    stats.update(pose_stats)

    # Check cheirality under recovered pose
    X_c = world_to_camera_points(R, t, X_w)
    z = X_c[2, :]
    cheirality_ratio = float(np.mean(z > eps))
    stats.update({"cheirality_ratio": cheirality_ratio})

    # Reject poses with too many points behind the camera
    if cheirality_ratio < min_cheirality_ratio:
        stats.update({"reason": "cheirality_ratio_too_low"})
        return None, None, stats

    # Score by reprojection error
    try:
        rmse_px = float(reprojection_rmse(K, R, t, X_w, x_cur))
    except Exception as exc:
        stats.update({"reason": "reprojection_eval_failed", "error": str(exc)})
        return None, None, stats

    stats.update({"reprojection_rmse_px": rmse_px})

    return np.asarray(R, dtype=float), np.asarray(t, dtype=float).reshape(3), stats


# Refine a PnP pose by minimising reprojection error over fixed 2D–3D correspondences
def refine_pose_pnp(
    corrs: PnPCorrespondences,
    K,
    R_init,
    t_init,
    *,
    max_iters=15,
    min_points=6,
    damping=1e-6,
    step_tol=1e-9,
    improvement_tol=1e-9,
    eps=1e-12,
):
    # --- Checks ---
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    # Check correspondences
    X_w, x_cur = check_3xN_2xN_cols(corrs.X_w, corrs.x_cur, nameX="corrs.X_w", namex="corrs.x_cur", dtype=float, finite=True)
    # Check initial pose
    R = check_matrix_3x3(R_init, name="R_init", dtype=float, finite=False)
    t = np.asarray(t_init, dtype=float).reshape(3,)
    # Check iteration controls
    max_iters = check_int_gt0(max_iters, name="max_iters")
    min_points = max(6, int(min_points))
    damping = check_positive(damping, name="damping", eps=0.0)
    step_tol = check_positive(step_tol, name="step_tol", eps=0.0)
    improvement_tol = check_positive(improvement_tol, name="improvement_tol", eps=0.0)
    eps = check_positive(eps, name="eps", eps=0.0)

    # Read total correspondence count
    N = int(X_w.shape[1])

    # Initialise stats
    stats = {
        "n_corr": N,
        "max_iters": int(max_iters),
        "n_iters": 0,
        "method": "gauss_newton_pose_only",
        "converged": False,
        "reason": None,
    }

    # Require enough correspondences
    if N < min_points:
        stats.update({"reason": "too_few_correspondences"})
        return None, None, stats

    # Initial reprojection score
    try:
        prev_rmse = float(reprojection_rmse(K, R, t, X_w, x_cur))
    except Exception as exc:
        stats.update({"reason": "initial_reprojection_eval_failed", "error": str(exc)})
        return None, None, stats

    stats["rmse_px_init"] = prev_rmse

    # Gauss-Newton iterations
    for it in range(max_iters):
        # Build local linear system
        try:
            J, r, valid = _linearise_pose_only_reprojection(X_w, x_cur, K, R, t, eps=eps)
        except Exception as exc:
            stats.update({"reason": "linearisation_failed", "error": str(exc), "n_iters": int(it)})
            return None, None, stats

        # Require enough valid points
        n_valid = int(np.sum(valid))
        if n_valid < min_points:
            stats.update({"reason": "too_few_valid_points", "n_valid": n_valid, "n_iters": int(it)})
            return None, None, stats

        # Normal equations with light damping
        H = J.T @ J + float(damping) * np.eye(6, dtype=float)
        g = J.T @ r

        # Solve for increment
        try:
            delta = -np.linalg.solve(H, g)
        except Exception as exc:
            stats.update({"reason": "normal_equations_failed", "error": str(exc), "n_iters": int(it)})
            return None, None, stats

        # Read step size
        step_norm = float(np.linalg.norm(delta))

        # Update pose
        R_new, t_new = apply_left_pose_increment_wc(R, t, delta, eps=eps)

        # Score updated pose
        try:
            rmse_new = float(reprojection_rmse(K, R_new, t_new, X_w, x_cur))
        except Exception as exc:
            stats.update({"reason": "updated_reprojection_eval_failed", "error": str(exc), "n_iters": int(it)})
            return None, None, stats

        # Accept the step
        R = R_new
        t = t_new

        # Track progress
        stats["n_iters"] = int(it + 1)
        stats["n_valid"] = n_valid
        stats["step_norm_last"] = step_norm
        stats["rmse_px_last"] = rmse_new

        # Convergence on step size
        if step_norm <= float(step_tol):
            stats["converged"] = True
            stats["reason"] = "step_tol_reached"
            break

        # Convergence on RMSE improvement
        if abs(prev_rmse - rmse_new) <= float(improvement_tol):
            stats["converged"] = True
            stats["reason"] = "improvement_tol_reached"
            break

        # Continue from new score
        prev_rmse = rmse_new

    # Final score
    try:
        rmse_final = float(reprojection_rmse(K, R, t, X_w, x_cur))
    except Exception as exc:
        stats.update({"reason": "final_reprojection_eval_failed", "error": str(exc)})
        return None, None, stats

    stats["rmse_px_final"] = rmse_final

    # Mark non-convergence if loop ended naturally
    if stats["reason"] is None:
        stats["reason"] = "max_iters_reached"

    return np.asarray(R, dtype=float), np.asarray(t, dtype=float).reshape(3), stats


# Estimate camera pose from 2D–3D correspondences with RANSAC around linear PnP
def estimate_pose_pnp_ransac(
    corrs: PnPCorrespondences,
    K,
    *,
    num_trials=1000,
    sample_size=6,
    threshold_px=3.0,
    min_inliers=12,
    seed=0,
    min_points=6,
    rank_tol=1e-10,
    min_cheirality_ratio=0.5,
    eps=1e-12,
    refit=True,
    refine_nonlinear=True,
    refine_max_iters=15,
    refine_damping=1e-6,
    refine_step_tol=1e-9,
    refine_improvement_tol=1e-9,
):
    # --- Checks ---
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    # Check RANSAC controls
    num_trials = check_int_gt0(num_trials, name="num_trials")
    sample_size = check_int_gt0(sample_size, name="sample_size")
    min_inliers = check_int_gt0(min_inliers, name="min_inliers")
    min_points = check_int_gt0(min_points, name="min_points")
    threshold_px = check_positive(threshold_px, name="threshold_px", eps=0.0)
    eps = check_positive(eps, name="eps", eps=0.0)

    # Require a valid linear PnP sample size
    if sample_size < 6:
        raise ValueError(f"sample_size must be >= 6 for linear PnP; got {sample_size}")

    # Require the solver minimum to be valid
    if min_points < 6:
        raise ValueError(f"min_points must be >= 6 for linear PnP; got {min_points}")

    # Require the inlier minimum to be meaningful
    if min_inliers < sample_size:
        raise ValueError(f"min_inliers must be >= sample_size; got {min_inliers} < {sample_size}")

    # Check correspondence arrays
    X_w, x_cur = check_3xN_2xN_cols(corrs.X_w, corrs.x_cur, nameX="corrs.X_w", namex="corrs.x_cur", dtype=float, finite=True)

    # Read total correspondence count
    N = int(X_w.shape[1])

    # Require enough correspondences
    if N < sample_size:
        raise ValueError(f"Need at least sample_size correspondences; got N={N}, sample_size={sample_size}")

    # Random number generator
    rng = np.random.default_rng(seed)

    # Initialise best model
    best_R = None
    best_t = None
    best_mask = np.zeros((N,), dtype=bool)
    best_count = 0
    best_mean_err = np.inf
    n_model_success = 0

    # RANSAC loop
    for _ in range(num_trials):
        # Sample a minimal subset
        idx = rng.choice(N, size=sample_size, replace=False)

        # Build subset correspondences
        corrs_sub = _slice_pnp_correspondences(corrs, idx)

        # Estimate a pose from the subset
        try:
            R_t, t_t, _ = estimate_pose_pnp(
                corrs_sub,
                K,
                min_points=min_points,
                rank_tol=rank_tol,
                min_cheirality_ratio=min_cheirality_ratio,
                eps=eps,
            )
        except Exception:
            continue

        # Skip failed models
        if R_t is None or t_t is None:
            continue

        # Count successful model fits
        n_model_success += 1

        # Score on all correspondences
        mask_t, d_sq_t = _pnp_inlier_mask_from_pose(
            X_w,
            x_cur,
            K,
            R_t,
            t_t,
            threshold_px=threshold_px,
            eps=eps,
        )

        # Count inliers
        count_t = int(mask_t.sum())
        if count_t == 0:
            continue

        # Mean inlier error for tie-breaks
        mean_err_t = float(np.mean(d_sq_t[mask_t]))

        # Keep the best consensus
        if (count_t > best_count) or (count_t == best_count and mean_err_t < best_mean_err):
            best_R = np.asarray(R_t, dtype=float)
            best_t = np.asarray(t_t, dtype=float).reshape(3,)
            best_mask = np.asarray(mask_t, dtype=bool)
            best_count = count_t
            best_mean_err = mean_err_t

            # Early exit on perfect agreement
            if best_count == N:
                break

    # Default stats
    stats = {
        "N": N,
        "num_trials": int(num_trials),
        "sample_size": int(sample_size),
        "threshold_px": float(threshold_px),
        "min_inliers": int(min_inliers),
        "n_model_success": int(n_model_success),
        "n_inliers": int(best_count),
        "mean_inlier_err_sq": None if not np.isfinite(best_mean_err) else float(best_mean_err),
        "refit": False,
        "reason": None,
    }

    # Fail if no good model was found
    if best_R is None or best_t is None or best_count < min_inliers:
        stats["reason"] = "pnp_ransac_failed"
        return None, None, None, stats

    # Stop here if refit is disabled
    if not bool(refit):
        return best_R, best_t, best_mask, stats

    # Refit on the best inlier set
    try:
        # Slice inlier correspondences
        corrs_in = _slice_pnp_correspondences(corrs, best_mask)

        # Re-estimate a linear pose on all inliers
        R_refit, t_refit, _ = estimate_pose_pnp(
            corrs_in,
            K,
            min_points=min_points,
            rank_tol=rank_tol,
            min_cheirality_ratio=min_cheirality_ratio,
            eps=eps,
        )

        # Keep only valid linear refits
        if R_refit is not None and t_refit is not None:
            # Optionally refine nonlinearly on the fixed inlier set
            if bool(refine_nonlinear):
                R_nonlin, t_nonlin, refine_stats = refine_pose_pnp(
                    corrs_in,
                    K,
                    R_refit,
                    t_refit,
                    max_iters=refine_max_iters,
                    min_points=min_points,
                    damping=refine_damping,
                    step_tol=refine_step_tol,
                    improvement_tol=refine_improvement_tol,
                    eps=eps,
                )

                stats["refine_stats"] = refine_stats

                if R_nonlin is not None and t_nonlin is not None:
                    R_refit = R_nonlin
                    t_refit = t_nonlin

            # Score the refined pose on all correspondences
            mask_refit, d_sq_refit = _pnp_inlier_mask_from_pose(
                X_w,
                x_cur,
                K,
                R_refit,
                t_refit,
                threshold_px=threshold_px,
                eps=eps,
            )

            # Read refined quality
            n_refit = int(mask_refit.sum())
            mean_err_refit = float(np.mean(d_sq_refit[mask_refit])) if n_refit > 0 else np.inf

            # Keep the refined model if consensus is not worse
            if n_refit >= best_count:
                best_R = np.asarray(R_refit, dtype=float)
                best_t = np.asarray(t_refit, dtype=float).reshape(3,)
                best_mask = np.asarray(mask_refit, dtype=bool)
                best_count = n_refit
                best_mean_err = mean_err_refit
                stats["refit"] = True

    except Exception as exc:
        stats["refit_error"] = str(exc)

    # Final stats
    stats["n_inliers"] = int(best_count)
    stats["mean_inlier_err_sq"] = None if not np.isfinite(best_mean_err) else float(best_mean_err)

    return best_R, best_t, best_mask, stats


# Compare an accepted PnP pose against a nearby threshold on the same correspondences
def pnp_threshold_stability_diagnostic(
    corrs: PnPCorrespondences,
    K,
    R_ref,
    t_ref,
    ref_inlier_mask,
    *,
    ref_threshold_px: float,
    compare_threshold_px: float,
    num_trials=1000,
    sample_size=6,
    min_inliers=12,
    seed=0,
    min_points=6,
    rank_tol=1e-10,
    min_cheirality_ratio=0.5,
    eps=1e-12,
    refit=True,
    refine_nonlinear=True,
    refine_max_iters=15,
    refine_damping=1e-6,
    refine_step_tol=1e-9,
    refine_improvement_tol=1e-9,
    min_support_iou: float = 0.25,
    max_translation_direction_deg: float = 120.0,
    max_camera_centre_direction_deg: float = 120.0,
    disjoint_support_iou: float = 0.05,
) -> dict:
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)

    # Check correspondence arrays
    X_w, x_cur = check_3xN_2xN_cols(corrs.X_w, corrs.x_cur, nameX="corrs.X_w", namex="corrs.x_cur", dtype=float, finite=True)
    N = int(X_w.shape[1])

    # Check solver controls
    ref_threshold_px = check_positive(ref_threshold_px, name="ref_threshold_px", eps=0.0)
    compare_threshold_px = check_positive(compare_threshold_px, name="compare_threshold_px", eps=0.0)
    num_trials = check_int_gt0(num_trials, name="num_trials")
    sample_size = check_int_gt0(sample_size, name="sample_size")
    min_inliers = check_int_gt0(min_inliers, name="min_inliers")
    min_points = check_int_gt0(min_points, name="min_points")
    rank_tol = check_positive(rank_tol, name="rank_tol", eps=0.0)
    eps = check_positive(eps, name="eps", eps=0.0)

    # Check stability controls
    min_support_iou = float(min_support_iou)
    max_translation_direction_deg = check_positive(
        max_translation_direction_deg,
        name="max_translation_direction_deg",
        eps=0.0,
    )
    max_camera_centre_direction_deg = check_positive(
        max_camera_centre_direction_deg,
        name="max_camera_centre_direction_deg",
        eps=0.0,
    )
    disjoint_support_iou = float(disjoint_support_iou)
    if not np.isfinite(min_support_iou) or min_support_iou < 0.0 or min_support_iou > 1.0:
        raise ValueError(f"min_support_iou must be in [0,1]; got {min_support_iou}")
    if not np.isfinite(disjoint_support_iou) or disjoint_support_iou < 0.0 or disjoint_support_iou > 1.0:
        raise ValueError(f"disjoint_support_iou must be in [0,1]; got {disjoint_support_iou}")

    # Read accepted pose and support
    ref_ok = (R_ref is not None) and (t_ref is not None)
    ref_mask = align_bool_mask_1d(ref_inlier_mask, N, name="ref_inlier_mask")
    ref_inliers = int(np.sum(ref_mask))

    # Start with an unavailable comparison
    out = {
        "evaluated": False,
        "classification": "unavailable",
        "reason": None,
        "ref_threshold_px": float(ref_threshold_px),
        "compare_threshold_px": float(compare_threshold_px),
        "ref_pose_ok": bool(ref_ok),
        "compare_pose_ok": False,
        "compare_pose_reason": None,
        "ref_inliers": int(ref_inliers),
        "compare_inliers": 0,
        "support_overlap": 0,
        "support_union": int(ref_inliers),
        "support_iou": None,
        "support_overlap_over_ref": None,
        "support_overlap_over_compare": None,
        "rotation_delta_deg": None,
        "translation_direction_delta_deg": None,
        "camera_centre_direction_delta_deg": None,
        "one_solution_only_at_looser_threshold": False,
        "one_solution_only_at_stricter_threshold": False,
        "supports_effectively_disjoint": False,
        "support_iou_low": False,
        "translation_direction_disagrees": False,
        "camera_centre_direction_disagrees": False,
        "unstable": False,
        "instability_reasons": [],
        "min_support_iou": float(min_support_iou),
        "max_translation_direction_deg": float(max_translation_direction_deg),
        "max_camera_centre_direction_deg": float(max_camera_centre_direction_deg),
        "disjoint_support_iou": float(disjoint_support_iou),
    }

    # Stop if the reference pose is unavailable
    if not bool(ref_ok):
        out["reason"] = "reference_pose_missing"
        return out

    # Stop when RANSAC cannot draw a valid comparison sample
    if N < int(sample_size):
        out["reason"] = "too_few_correspondences_for_ransac"
        return out

    # Estimate the comparison pose without changing the accepted solution
    try:
        R_cmp, t_cmp, cmp_mask, cmp_stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=int(num_trials),
            sample_size=int(sample_size),
            threshold_px=float(compare_threshold_px),
            min_inliers=int(min_inliers),
            seed=int(seed),
            min_points=int(min_points),
            rank_tol=float(rank_tol),
            min_cheirality_ratio=float(min_cheirality_ratio),
            eps=float(eps),
            refit=bool(refit),
            refine_nonlinear=bool(refine_nonlinear),
            refine_max_iters=int(refine_max_iters),
            refine_damping=float(refine_damping),
            refine_step_tol=float(refine_step_tol),
            refine_improvement_tol=float(refine_improvement_tol),
        )
    except Exception as exc:
        out["evaluated"] = True
        out["reason"] = "comparison_pnp_error"
        out["compare_pose_reason"] = str(exc)
        return out

    # Read comparison pose and support
    cmp_stats = cmp_stats if isinstance(cmp_stats, dict) else {}
    compare_ok = (R_cmp is not None) and (t_cmp is not None)
    cmp_mask = align_bool_mask_1d(cmp_mask, N, name="compare_inlier_mask")
    compare_inliers = int(np.sum(cmp_mask))
    overlap = ref_mask & cmp_mask
    union = ref_mask | cmp_mask
    n_overlap = int(np.sum(overlap))
    n_union = int(np.sum(union))
    support_iou = None if n_union == 0 else float(n_overlap / n_union)

    out.update(
        {
            "evaluated": True,
            "reason": None,
            "compare_pose_ok": bool(compare_ok),
            "compare_pose_reason": cmp_stats.get("reason", None),
            "compare_inliers": int(compare_inliers),
            "support_overlap": int(n_overlap),
            "support_union": int(n_union),
            "support_iou": support_iou,
            "support_overlap_over_ref": None if ref_inliers == 0 else float(n_overlap / max(ref_inliers, 1)),
            "support_overlap_over_compare": None if compare_inliers == 0 else float(n_overlap / max(compare_inliers, 1)),
        }
    )

    # Compare pose directions when both solutions exist
    if bool(compare_ok):
        R_ref_arr = np.asarray(R_ref, dtype=np.float64)
        R_cmp_arr = np.asarray(R_cmp, dtype=np.float64)
        t_ref_arr = np.asarray(t_ref, dtype=np.float64).reshape(3)
        t_cmp_arr = np.asarray(t_cmp, dtype=np.float64).reshape(3)

        try:
            out["rotation_delta_deg"] = float(np.degrees(angle_between_rotmats(R_ref_arr, R_cmp_arr)))
        except Exception:
            out["rotation_delta_deg"] = None

        try:
            out["translation_direction_delta_deg"] = float(np.degrees(angle_between_translations(t_ref_arr, t_cmp_arr)))
        except Exception:
            out["translation_direction_delta_deg"] = None

        try:
            C_ref = camera_centre(R_ref_arr, t_ref_arr)
            C_cmp = camera_centre(R_cmp_arr, t_cmp_arr)
            out["camera_centre_direction_delta_deg"] = float(np.degrees(angle_between_translations(C_ref, C_cmp)))
        except Exception:
            out["camera_centre_direction_delta_deg"] = None

    # Flag the diagnostic instability modes
    out.update(
        pnp_threshold_stability_flags(
            ref_pose_ok=bool(ref_ok),
            compare_pose_ok=bool(compare_ok),
            ref_threshold_px=float(ref_threshold_px),
            compare_threshold_px=float(compare_threshold_px),
            support_iou=support_iou,
            support_union=int(n_union),
            translation_direction_delta_deg=out["translation_direction_delta_deg"],
            camera_centre_direction_delta_deg=out["camera_centre_direction_delta_deg"],
            min_support_iou=float(min_support_iou),
            max_translation_direction_deg=float(max_translation_direction_deg),
            max_camera_centre_direction_deg=float(max_camera_centre_direction_deg),
            disjoint_support_iou=float(disjoint_support_iou),
        )
    )

    return out
