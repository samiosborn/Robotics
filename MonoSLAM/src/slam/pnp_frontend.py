# src/slam/pnp_frontend.py

from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import align_bool_mask_1d, check_finite_scalar, check_in_01, check_int_ge0, check_int_gt0, check_matrix_3x3, check_positive
from geometry.pnp import _pnp_inlier_mask_from_pose, _slice_pnp_correspondences, build_pnp_correspondences_with_stats, estimate_pose_pnp_ransac, pnp_threshold_stability_diagnostic
from slam.pnp_stats import landmark_observation_histogram, pnp_support_diagnostic_stats, pnp_support_gate_stats, pnp_threshold_stability_summary_stats


_PNP_SUPPORT_RESCUE_STRICT_THRESHOLD_PX = 8.0
_PNP_SUPPORT_RESCUE_LOOSE_THRESHOLD_PX = 12.0


# Try the frame-local loose-pose residual subset rescue on the fixed correspondence set
def _attempt_pnp_spatial_support_rescue(
    corrs,
    K: np.ndarray,
    *,
    threshold_px: float,
    base_inlier_mask,
    base_spatial_gate_reason,
    num_trials: int,
    sample_size: int,
    min_inliers: int,
    ransac_seed: int,
    min_points: int,
    rank_tol: float,
    min_cheirality_ratio: float,
    eps: float,
    refit: bool,
    refine_nonlinear: bool,
    refine_max_iters: int,
    refine_damping: float,
    refine_step_tol: float,
    refine_improvement_tol: float,
    image_shape: tuple[int, int] | None,
    enable_pnp_spatial_gate: bool,
    pnp_spatial_grid_cols: int,
    pnp_spatial_grid_rows: int,
    min_pnp_inlier_cells: int,
    max_pnp_single_cell_fraction: float,
    min_pnp_bbox_area_fraction: float,
    enable_pnp_component_gate: bool,
    pnp_component_radius_px: float,
    min_pnp_component_count: int,
    max_pnp_largest_component_fraction: float,
    min_pnp_largest_component_bbox_area_fraction: float,
) -> dict[str, Any]:
    base_inlier_mask = align_bool_mask_1d(base_inlier_mask, int(corrs.X_w.shape[1]), name="base_pnp_inlier_mask")
    base_n_inliers = int(np.sum(base_inlier_mask))
    out: dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "reason": None,
        "base_strict_inliers": int(base_n_inliers),
        "base_spatial_gate_reason": base_spatial_gate_reason,
        "loose_threshold_px": float(_PNP_SUPPORT_RESCUE_LOOSE_THRESHOLD_PX),
        "loose_pose_ok": False,
        "loose_inliers": 0,
        "loose_reason": None,
        "subset_count": 0,
        "subset_strict_inliers": 0,
        "subset_strict_reason": None,
        "fullset_strict_inliers": 0,
        "fullset_strict_inlier_delta": 0,
        "min_inlier_gain_required": int(sample_size),
        "final_spatial_gate_rejected": False,
        "final_spatial_gate_reason": None,
        "final_component_gate_rejected": False,
        "final_component_gate_reason": None,
    }

    if image_shape is None:
        out["reason"] = "image_shape_unavailable"
        return out

    if not np.isclose(float(threshold_px), float(_PNP_SUPPORT_RESCUE_STRICT_THRESHOLD_PX)):
        out["reason"] = "threshold_not_strict_8px"
        return out

    gate_reason_str = "" if base_spatial_gate_reason is None else str(base_spatial_gate_reason)
    if "bbox_area_fraction_low" not in gate_reason_str:
        out["reason"] = "spatial_gate_reason_not_bbox_area_fraction_low"
        return out

    out["attempted"] = True

    try:
        R_loose, t_loose, loose_inlier_mask, loose_stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=int(num_trials),
            sample_size=int(sample_size),
            threshold_px=float(_PNP_SUPPORT_RESCUE_LOOSE_THRESHOLD_PX),
            min_inliers=int(min_inliers),
            seed=int(ransac_seed),
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
        out["reason"] = "loose_pnp_error"
        out["error"] = str(exc)
        return out

    loose_stats = loose_stats if isinstance(loose_stats, dict) else {}
    loose_inlier_mask = align_bool_mask_1d(loose_inlier_mask, int(corrs.X_w.shape[1]), name="loose_pnp_inlier_mask")
    out["loose_pose_ok"] = bool((R_loose is not None) and (t_loose is not None))
    out["loose_inliers"] = int(np.sum(loose_inlier_mask))
    out["loose_reason"] = loose_stats.get("reason", None)

    if not bool(out["loose_pose_ok"]):
        out["reason"] = "loose_pnp_failed"
        return out

    loose_subset_mask, _ = _pnp_inlier_mask_from_pose(
        corrs.X_w,
        corrs.x_cur,
        K,
        R_loose,
        t_loose,
        threshold_px=float(_PNP_SUPPORT_RESCUE_LOOSE_THRESHOLD_PX),
        eps=float(eps),
    )
    loose_subset_mask = align_bool_mask_1d(loose_subset_mask, int(corrs.X_w.shape[1]), name="loose_subset_mask")
    subset_count = int(np.sum(loose_subset_mask))
    out["subset_count"] = int(subset_count)

    if subset_count < int(sample_size):
        out["reason"] = "loose_subset_too_small"
        return out

    corrs_subset = _slice_pnp_correspondences(corrs, loose_subset_mask)
    try:
        R_rescue, t_rescue, subset_strict_mask, subset_strict_stats = estimate_pose_pnp_ransac(
            corrs_subset,
            K,
            num_trials=int(num_trials),
            sample_size=int(sample_size),
            threshold_px=float(threshold_px),
            min_inliers=int(min_inliers),
            seed=int(ransac_seed),
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
        out["reason"] = "subset_strict_pnp_error"
        out["error"] = str(exc)
        return out

    subset_strict_stats = subset_strict_stats if isinstance(subset_strict_stats, dict) else {}
    subset_strict_mask = align_bool_mask_1d(subset_strict_mask, int(corrs_subset.X_w.shape[1]), name="subset_strict_inlier_mask")
    out["subset_strict_inliers"] = int(np.sum(subset_strict_mask))
    out["subset_strict_reason"] = subset_strict_stats.get("reason", None)

    if R_rescue is None or t_rescue is None:
        out["reason"] = "subset_strict_pnp_failed"
        return out

    rescued_full_mask, rescued_full_d_sq = _pnp_inlier_mask_from_pose(
        corrs.X_w,
        corrs.x_cur,
        K,
        R_rescue,
        t_rescue,
        threshold_px=float(threshold_px),
        eps=float(eps),
    )
    rescued_full_mask = align_bool_mask_1d(rescued_full_mask, int(corrs.X_w.shape[1]), name="rescued_full_mask")
    rescued_full_d_sq = np.asarray(rescued_full_d_sq, dtype=np.float64).reshape(-1)
    rescued_full_inliers = int(np.sum(rescued_full_mask))
    out["fullset_strict_inliers"] = int(rescued_full_inliers)
    out["fullset_strict_inlier_delta"] = int(rescued_full_inliers - base_n_inliers)

    if int(out["fullset_strict_inlier_delta"]) < int(sample_size):
        out["reason"] = "strict_inlier_gain_too_small"
        return out

    rescue_support_stats = pnp_support_diagnostic_stats(
        corrs,
        rescued_full_mask,
        image_shape,
        pnp_spatial_grid_cols=int(pnp_spatial_grid_cols),
        pnp_spatial_grid_rows=int(pnp_spatial_grid_rows),
        pnp_component_radius_px=float(pnp_component_radius_px),
    )
    rescue_gate_stats = pnp_support_gate_stats(
        True,
        rescue_support_stats,
        enable_pnp_spatial_gate=bool(enable_pnp_spatial_gate),
        min_pnp_inlier_cells=int(min_pnp_inlier_cells),
        max_pnp_single_cell_fraction=float(max_pnp_single_cell_fraction),
        min_pnp_bbox_area_fraction=float(min_pnp_bbox_area_fraction),
        enable_pnp_component_gate=bool(enable_pnp_component_gate),
        min_pnp_component_count=int(min_pnp_component_count),
        max_pnp_largest_component_fraction=float(max_pnp_largest_component_fraction),
        min_pnp_largest_component_bbox_area_fraction=float(min_pnp_largest_component_bbox_area_fraction),
    )

    out["final_spatial_gate_rejected"] = bool(rescue_gate_stats.get("pnp_spatial_gate_rejected", False))
    out["final_spatial_gate_reason"] = rescue_gate_stats.get("pnp_spatial_gate_reason", None)
    out["final_component_gate_rejected"] = bool(rescue_gate_stats.get("pnp_component_gate_rejected", False))
    out["final_component_gate_reason"] = rescue_gate_stats.get("pnp_component_gate_reason", None)

    if bool(out["final_spatial_gate_rejected"]) or bool(out["final_component_gate_rejected"]):
        out["reason"] = "rescued_support_still_rejected"
        return out

    rescue_solver_stats = dict(subset_strict_stats)
    rescue_solver_stats.update(
        {
            "n_inliers": int(rescued_full_inliers),
            "mean_inlier_err_sq": None if rescued_full_inliers <= 0 else float(np.mean(rescued_full_d_sq[rescued_full_mask])),
            "threshold_px": float(threshold_px),
        }
    )

    out["succeeded"] = True
    out["reason"] = "rescued"
    out["R"] = np.asarray(R_rescue, dtype=np.float64)
    out["t"] = np.asarray(t_rescue, dtype=np.float64).reshape(3)
    out["pnp_inlier_mask"] = np.asarray(rescued_full_mask, dtype=bool)
    out["pnp_stats"] = rescue_solver_stats
    out["support_stats"] = rescue_support_stats
    out["gate_stats"] = rescue_gate_stats
    return out


# Estimate the current pose from a bootstrap seed and tracked observations
def estimate_pose_from_seed(
    K: np.ndarray,
    seed: dict[str, Any],
    track_out: dict[str, Any],
    *,
    num_trials: int = 1000,
    sample_size: int = 6,
    threshold_px: float = 3.0,
    min_inliers: int = 12,
    ransac_seed: int = 0,
    min_points: int = 6,
    rank_tol: float = 1e-10,
    min_cheirality_ratio: float = 0.5,
    min_landmark_observations: int = 2,
    allow_bootstrap_landmarks_for_pose: bool = True,
    min_post_bootstrap_observations_for_pose: int = 3,
    eps: float = 1e-12,
    refit: bool = True,
    refine_nonlinear: bool = True,
    refine_max_iters: int = 15,
    refine_damping: float = 1e-6,
    refine_step_tol: float = 1e-9,
    refine_improvement_tol: float = 1e-9,
    image_shape: tuple[int, int] | None = None,
    enable_pnp_spatial_gate: bool = True,
    pnp_spatial_grid_cols: int = 4,
    pnp_spatial_grid_rows: int = 3,
    min_pnp_inlier_cells: int = 1,
    max_pnp_single_cell_fraction: float = 1.0,
    min_pnp_bbox_area_fraction: float = 0.01,
    enable_pnp_component_gate: bool = False,
    pnp_component_radius_px: float = 80.0,
    max_pnp_largest_component_fraction: float = 1.0,
    min_pnp_component_count: int = 0,
    min_pnp_largest_component_bbox_area_fraction: float = 0.0,
    enable_pnp_local_consistency_filter: bool = False,
    pnp_local_consistency_radius_px: float = 80.0,
    pnp_local_consistency_min_neighbours: int = 3,
    pnp_local_consistency_max_median_residual_px: float = 12.0,
    pnp_local_consistency_min_keep: int = 0,
    enable_pnp_spatial_thinning_filter: bool = False,
    pnp_spatial_thinning_radius_px: float = 20.0,
    pnp_spatial_thinning_max_points_per_radius: int = 16,
    pnp_spatial_thinning_min_keep: int = 0,
    enable_pnp_threshold_stability_diagnostic: bool = False,
    pnp_threshold_stability_compare_px: float | None = None,
    pnp_threshold_stability_min_support_iou: float = 0.25,
    pnp_threshold_stability_max_translation_direction_deg: float = 120.0,
    pnp_threshold_stability_max_camera_centre_direction_deg: float = 120.0,
    pnp_threshold_stability_disjoint_iou: float = 0.05,
    enable_pnp_threshold_stability_gate: bool = False,
) -> dict[str, Any]:
    # Check intrinsics
    check_matrix_3x3(K, name="K", dtype=float, finite=False)

    # Check containers
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    if not isinstance(track_out, dict):
        raise ValueError("track_out must be a dict")

    # Check RANSAC controls
    num_trials = check_int_gt0(num_trials, name="num_trials")
    sample_size = check_int_gt0(sample_size, name="sample_size")
    min_inliers = check_int_gt0(min_inliers, name="min_inliers")
    min_points = check_int_gt0(min_points, name="min_points")
    ransac_seed = int(ransac_seed)

    # Check linear solver controls
    threshold_px = check_positive(threshold_px, name="threshold_px", eps=0.0)
    rank_tol = check_positive(rank_tol, name="rank_tol", eps=0.0)
    min_cheirality_ratio = check_finite_scalar(min_cheirality_ratio, name="min_cheirality_ratio")
    check_in_01(min_cheirality_ratio, name="min_cheirality_ratio", eps=0.0)
    min_landmark_observations = check_int_gt0(min_landmark_observations, name="min_landmark_observations")
    allow_bootstrap_landmarks_for_pose = bool(allow_bootstrap_landmarks_for_pose)
    min_post_bootstrap_observations_for_pose = check_int_gt0(
        min_post_bootstrap_observations_for_pose,
        name="min_post_bootstrap_observations_for_pose",
    )
    eps = check_positive(eps, name="eps", eps=0.0)

    # Check nonlinear refinement controls
    refine_max_iters = check_int_gt0(refine_max_iters, name="refine_max_iters")
    refine_damping = check_positive(refine_damping, name="refine_damping", eps=0.0)
    refine_step_tol = check_positive(refine_step_tol, name="refine_step_tol", eps=0.0)
    refine_improvement_tol = check_positive(refine_improvement_tol, name="refine_improvement_tol", eps=0.0)

    # Check spatial coverage controls
    enable_pnp_spatial_gate = bool(enable_pnp_spatial_gate)
    pnp_spatial_grid_cols = check_int_gt0(pnp_spatial_grid_cols, name="pnp_spatial_grid_cols")
    pnp_spatial_grid_rows = check_int_gt0(pnp_spatial_grid_rows, name="pnp_spatial_grid_rows")
    min_pnp_inlier_cells = check_int_ge0(min_pnp_inlier_cells, name="min_pnp_inlier_cells")
    max_pnp_single_cell_fraction = check_finite_scalar(max_pnp_single_cell_fraction, name="max_pnp_single_cell_fraction")
    min_pnp_bbox_area_fraction = check_finite_scalar(min_pnp_bbox_area_fraction, name="min_pnp_bbox_area_fraction")
    check_in_01(max_pnp_single_cell_fraction, name="max_pnp_single_cell_fraction", eps=0.0)
    check_in_01(min_pnp_bbox_area_fraction, name="min_pnp_bbox_area_fraction", eps=0.0)
    if max_pnp_single_cell_fraction <= 0.0:
        raise ValueError(f"max_pnp_single_cell_fraction must be > 0; got {max_pnp_single_cell_fraction}")

    # Check component-support controls
    enable_pnp_component_gate = bool(enable_pnp_component_gate)
    pnp_component_radius_px = check_positive(pnp_component_radius_px, name="pnp_component_radius_px", eps=0.0)
    max_pnp_largest_component_fraction = check_finite_scalar(
        max_pnp_largest_component_fraction,
        name="max_pnp_largest_component_fraction",
    )
    min_pnp_component_count = check_int_ge0(min_pnp_component_count, name="min_pnp_component_count")
    min_pnp_largest_component_bbox_area_fraction = check_finite_scalar(
        min_pnp_largest_component_bbox_area_fraction,
        name="min_pnp_largest_component_bbox_area_fraction",
    )
    check_in_01(max_pnp_largest_component_fraction, name="max_pnp_largest_component_fraction", eps=0.0)
    check_in_01(min_pnp_largest_component_bbox_area_fraction, name="min_pnp_largest_component_bbox_area_fraction", eps=0.0)
    if max_pnp_largest_component_fraction <= 0.0:
        raise ValueError(f"max_pnp_largest_component_fraction must be > 0; got {max_pnp_largest_component_fraction}")

    # Check local pose-candidate consistency controls
    enable_pnp_local_consistency_filter = bool(enable_pnp_local_consistency_filter)
    pnp_local_consistency_radius_px = check_positive(
        pnp_local_consistency_radius_px,
        name="pnp_local_consistency_radius_px",
        eps=0.0,
    )
    pnp_local_consistency_min_neighbours = check_int_ge0(
        pnp_local_consistency_min_neighbours,
        name="pnp_local_consistency_min_neighbours",
    )
    pnp_local_consistency_max_median_residual_px = check_positive(
        pnp_local_consistency_max_median_residual_px,
        name="pnp_local_consistency_max_median_residual_px",
        eps=0.0,
    )
    pnp_local_consistency_min_keep = check_int_ge0(
        pnp_local_consistency_min_keep,
        name="pnp_local_consistency_min_keep",
    )

    # Check current-image spatial thinning controls
    enable_pnp_spatial_thinning_filter = bool(enable_pnp_spatial_thinning_filter)
    pnp_spatial_thinning_radius_px = check_positive(
        pnp_spatial_thinning_radius_px,
        name="pnp_spatial_thinning_radius_px",
        eps=0.0,
    )
    pnp_spatial_thinning_max_points_per_radius = check_int_gt0(
        pnp_spatial_thinning_max_points_per_radius,
        name="pnp_spatial_thinning_max_points_per_radius",
    )
    pnp_spatial_thinning_min_keep = check_int_ge0(
        pnp_spatial_thinning_min_keep,
        name="pnp_spatial_thinning_min_keep",
    )

    # Check threshold-stability diagnostic controls
    enable_pnp_threshold_stability_diagnostic = bool(enable_pnp_threshold_stability_diagnostic)
    enable_pnp_threshold_stability_gate = bool(enable_pnp_threshold_stability_gate)
    if pnp_threshold_stability_compare_px is not None:
        pnp_threshold_stability_compare_px = check_positive(
            pnp_threshold_stability_compare_px,
            name="pnp_threshold_stability_compare_px",
            eps=0.0,
        )
    pnp_threshold_stability_min_support_iou = check_finite_scalar(
        pnp_threshold_stability_min_support_iou,
        name="pnp_threshold_stability_min_support_iou",
    )
    pnp_threshold_stability_disjoint_iou = check_finite_scalar(
        pnp_threshold_stability_disjoint_iou,
        name="pnp_threshold_stability_disjoint_iou",
    )
    check_in_01(pnp_threshold_stability_min_support_iou, name="pnp_threshold_stability_min_support_iou", eps=0.0)
    check_in_01(pnp_threshold_stability_disjoint_iou, name="pnp_threshold_stability_disjoint_iou", eps=0.0)
    pnp_threshold_stability_max_translation_direction_deg = check_positive(
        pnp_threshold_stability_max_translation_direction_deg,
        name="pnp_threshold_stability_max_translation_direction_deg",
        eps=0.0,
    )
    pnp_threshold_stability_max_camera_centre_direction_deg = check_positive(
        pnp_threshold_stability_max_camera_centre_direction_deg,
        name="pnp_threshold_stability_max_camera_centre_direction_deg",
        eps=0.0,
    )

    # Require a valid DLT sample size
    if sample_size < 6:
        raise ValueError(f"sample_size must be >= 6 for linear PnP; got {sample_size}")

    # Require a valid solver minimum
    if min_points < 6:
        raise ValueError(f"min_points must be >= 6 for linear PnP; got {min_points}")

    # Require a meaningful inlier minimum
    if min_inliers < sample_size:
        raise ValueError(f"min_inliers must be >= sample_size; got {min_inliers} < {sample_size}")

    # Build 2D–3D correspondences from the tracked frame
    corrs, corr_stats = build_pnp_correspondences_with_stats(
        seed,
        track_out,
        min_landmark_observations=min_landmark_observations,
        allow_bootstrap_landmarks_for_pose=allow_bootstrap_landmarks_for_pose,
        min_post_bootstrap_observations_for_pose=min_post_bootstrap_observations_for_pose,
        enable_local_consistency_filter=enable_pnp_local_consistency_filter,
        local_consistency_radius_px=pnp_local_consistency_radius_px,
        local_consistency_min_neighbours=pnp_local_consistency_min_neighbours,
        local_consistency_max_median_residual_px=pnp_local_consistency_max_median_residual_px,
        local_consistency_min_keep=pnp_local_consistency_min_keep,
        enable_spatial_thinning_filter=enable_pnp_spatial_thinning_filter,
        spatial_thinning_radius_px=pnp_spatial_thinning_radius_px,
        spatial_thinning_max_points_per_radius=pnp_spatial_thinning_max_points_per_radius,
        spatial_thinning_min_keep=pnp_spatial_thinning_min_keep,
    )
    n_corr = int(corrs.X_w.shape[1])
    observation_histogram = landmark_observation_histogram(seed, corrs.landmark_ids)

    # Start stats from the correspondence count
    stats: dict[str, Any] = {
        "n_corr": n_corr,
        "n_corr_raw": int(corr_stats.get("n_corr_raw", 0)),
        "n_corr_bootstrap_born": int(corr_stats.get("n_corr_bootstrap_born", 0)),
        "n_corr_post_bootstrap_born": int(corr_stats.get("n_corr_post_bootstrap_born", 0)),
        "n_corr_after_pose_filter": int(corr_stats.get("n_corr_after_pose_filter", n_corr)),
        "n_corr_after_local_consistency_filter": int(corr_stats.get("n_corr_after_local_consistency_filter", n_corr)),
        "n_corr_after_spatial_thinning_filter": int(corr_stats.get("n_corr_after_spatial_thinning_filter", n_corr)),
        "n_corr_bootstrap_used": int(corr_stats.get("n_corr_bootstrap_used", 0)),
        "n_corr_post_bootstrap_used": int(corr_stats.get("n_corr_post_bootstrap_used", 0)),
        "n_corr_bootstrap_after_local_consistency": int(corr_stats.get("n_corr_bootstrap_after_local_consistency", 0)),
        "n_corr_post_bootstrap_after_local_consistency": int(corr_stats.get("n_corr_post_bootstrap_after_local_consistency", 0)),
        "n_corr_bootstrap_after_spatial_thinning": int(corr_stats.get("n_corr_bootstrap_after_spatial_thinning", 0)),
        "n_corr_post_bootstrap_after_spatial_thinning": int(corr_stats.get("n_corr_post_bootstrap_after_spatial_thinning", 0)),
        "min_landmark_observations": int(min_landmark_observations),
        "allow_bootstrap_landmarks_for_pose": bool(allow_bootstrap_landmarks_for_pose),
        "min_post_bootstrap_observations_for_pose": int(min_post_bootstrap_observations_for_pose),
        "landmark_observation_histogram": observation_histogram,
        "pnp_local_consistency_filter_enabled": bool(corr_stats.get("pnp_local_consistency_filter_enabled", False)),
        "pnp_local_consistency_filter_evaluated": bool(corr_stats.get("pnp_local_consistency_filter_evaluated", False)),
        "pnp_local_consistency_filter_applied": bool(corr_stats.get("pnp_local_consistency_filter_applied", False)),
        "pnp_local_consistency_filter_removed": int(corr_stats.get("pnp_local_consistency_filter_removed", 0)),
        "pnp_local_consistency_filter_reason": corr_stats.get("pnp_local_consistency_filter_reason", None),
        "pnp_local_consistency_radius_px": float(corr_stats.get("pnp_local_consistency_radius_px", pnp_local_consistency_radius_px)),
        "pnp_local_consistency_min_neighbours": int(corr_stats.get("pnp_local_consistency_min_neighbours", pnp_local_consistency_min_neighbours)),
        "pnp_local_consistency_max_median_residual_px": float(
            corr_stats.get(
                "pnp_local_consistency_max_median_residual_px",
                pnp_local_consistency_max_median_residual_px,
            )
        ),
        "pnp_local_consistency_min_keep": int(corr_stats.get("pnp_local_consistency_min_keep", pnp_local_consistency_min_keep)),
        "pnp_local_consistency_stats": corr_stats.get("pnp_local_consistency_stats", None),
        "pnp_spatial_thinning_filter_enabled": bool(corr_stats.get("pnp_spatial_thinning_filter_enabled", False)),
        "pnp_spatial_thinning_filter_evaluated": bool(corr_stats.get("pnp_spatial_thinning_filter_evaluated", False)),
        "pnp_spatial_thinning_filter_applied": bool(corr_stats.get("pnp_spatial_thinning_filter_applied", False)),
        "pnp_spatial_thinning_filter_removed": int(corr_stats.get("pnp_spatial_thinning_filter_removed", 0)),
        "pnp_spatial_thinning_filter_reason": corr_stats.get("pnp_spatial_thinning_filter_reason", None),
        "pnp_spatial_thinning_radius_px": float(corr_stats.get("pnp_spatial_thinning_radius_px", pnp_spatial_thinning_radius_px)),
        "pnp_spatial_thinning_max_points_per_radius": int(
            corr_stats.get(
                "pnp_spatial_thinning_max_points_per_radius",
                pnp_spatial_thinning_max_points_per_radius,
            )
        ),
        "pnp_spatial_thinning_min_keep": int(corr_stats.get("pnp_spatial_thinning_min_keep", pnp_spatial_thinning_min_keep)),
        "pnp_spatial_thinning_stats": corr_stats.get("pnp_spatial_thinning_stats", None),
        "pnp_spatial_gate_enabled": bool(enable_pnp_spatial_gate),
        "pnp_spatial_gate_evaluated": False,
        "pnp_spatial_gate_rejected": False,
        "pnp_spatial_gate_reason": None,
        "pnp_spatial_grid_cols": int(pnp_spatial_grid_cols),
        "pnp_spatial_grid_rows": int(pnp_spatial_grid_rows),
        "min_pnp_inlier_cells": int(min_pnp_inlier_cells),
        "max_pnp_single_cell_fraction": float(max_pnp_single_cell_fraction),
        "min_pnp_bbox_area_fraction": float(min_pnp_bbox_area_fraction),
        "pnp_spatial_coverage": None,
        "pnp_inlier_occupied_cells": 0,
        "pnp_inlier_max_cell_count": 0,
        "pnp_inlier_max_cell_fraction": None,
        "pnp_inlier_bbox_area_fraction": None,
        "pnp_inlier_bbox": None,
        "pnp_component_gate_enabled": bool(enable_pnp_component_gate),
        "pnp_component_gate_evaluated": False,
        "pnp_component_gate_rejected": False,
        "pnp_component_gate_reason": None,
        "pnp_component_radius_px": float(pnp_component_radius_px),
        "max_pnp_largest_component_fraction": float(max_pnp_largest_component_fraction),
        "min_pnp_component_count": int(min_pnp_component_count),
        "min_pnp_largest_component_bbox_area_fraction": float(min_pnp_largest_component_bbox_area_fraction),
        "pnp_component_support": None,
        "pnp_inlier_component_count": 0,
        "pnp_inlier_largest_component_size": 0,
        "pnp_inlier_largest_component_fraction": None,
        "pnp_inlier_largest_component_bbox_area_fraction": None,
        "pnp_inlier_largest_component_bbox": None,
        "pnp_inlier_component_sizes": [],
        "pnp_threshold_stability_diagnostic_enabled": bool(enable_pnp_threshold_stability_diagnostic),
        "pnp_threshold_stability_gate_enabled": bool(enable_pnp_threshold_stability_gate),
        "pnp_threshold_stability_evaluated": False,
        "pnp_threshold_stability_gate_rejected": False,
        "pnp_threshold_stability_gate_reason": None,
        "pnp_threshold_stability_compare_px": None if pnp_threshold_stability_compare_px is None else float(pnp_threshold_stability_compare_px),
        "pnp_threshold_stability": None,
        "pnp_threshold_stability_classification": "unavailable",
        "pnp_threshold_stability_unstable": False,
        "pnp_threshold_stability_ref_inliers": 0,
        "pnp_threshold_stability_compare_inliers": 0,
        "pnp_threshold_stability_support_iou": None,
        "pnp_threshold_stability_rotation_delta_deg": None,
        "pnp_threshold_stability_translation_direction_delta_deg": None,
        "pnp_threshold_stability_camera_centre_direction_delta_deg": None,
        "pnp_threshold_stability_looser_solution_only": False,
        "pnp_threshold_stability_supports_disjoint": False,
        "pnp_threshold_stability_reasons": [],
        "pnp_support_rescue_attempted": False,
        "pnp_support_rescue_succeeded": False,
        "pnp_support_rescue_reason": None,
        "pnp_support_rescue_base_strict_inliers": 0,
        "pnp_support_rescue_base_spatial_gate_reason": None,
        "pnp_support_rescue_loose_threshold_px": float(_PNP_SUPPORT_RESCUE_LOOSE_THRESHOLD_PX),
        "pnp_support_rescue_loose_pose_ok": False,
        "pnp_support_rescue_loose_inliers": 0,
        "pnp_support_rescue_loose_reason": None,
        "pnp_support_rescue_subset_count": 0,
        "pnp_support_rescue_subset_strict_inliers": 0,
        "pnp_support_rescue_subset_strict_reason": None,
        "pnp_support_rescue_fullset_strict_inliers": 0,
        "pnp_support_rescue_fullset_strict_inlier_delta": 0,
        "pnp_support_rescue_min_inlier_gain": int(sample_size),
        "pnp_support_rescue_final_spatial_gate_rejected": False,
        "pnp_support_rescue_final_spatial_gate_reason": None,
        "pnp_support_rescue_final_component_gate_rejected": False,
        "pnp_support_rescue_final_component_gate_reason": None,
    }

    # Stop early if nothing survived
    if n_corr == 0:
        stats.update({"reason": "no_pnp_correspondences"})
        return {
            "ok": False,
            "R": None,
            "t": None,
            "corrs": corrs,
            "pnp_inlier_mask": np.zeros((0,), dtype=bool),
            "landmark_ids": np.zeros((0,), dtype=np.int64),
            "stats": stats,
        }

    # Run robust PnP on the correspondence bundle
    try:
        R, t, pnp_inlier_mask, pnp_stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=num_trials,
            sample_size=sample_size,
            threshold_px=threshold_px,
            min_inliers=min_inliers,
            seed=ransac_seed,
            min_points=min_points,
            rank_tol=rank_tol,
            min_cheirality_ratio=min_cheirality_ratio,
            eps=eps,
            refit=refit,
            refine_nonlinear=refine_nonlinear,
            refine_max_iters=refine_max_iters,
            refine_damping=refine_damping,
            refine_step_tol=refine_step_tol,
            refine_improvement_tol=refine_improvement_tol,
        )
    except Exception as exc:
        stats.update({"reason": "pnp_failed", "error": str(exc)})
        return {
            "ok": False,
            "R": None,
            "t": None,
            "corrs": corrs,
            "pnp_inlier_mask": np.zeros((n_corr,), dtype=bool),
            "landmark_ids": np.zeros((0,), dtype=np.int64),
            "stats": stats,
        }

    # Merge solver stats into the frontend stats
    if isinstance(pnp_stats, dict):
        stats.update(pnp_stats)

    # Build an aligned PnP inlier mask
    pnp_inlier_mask = align_bool_mask_1d(pnp_inlier_mask, n_corr)
    stats.update({"n_pnp_inliers": int(pnp_inlier_mask.sum())})

    # Score spatial and component support when the current image size is available
    stats.update(
        pnp_support_diagnostic_stats(
            corrs,
            pnp_inlier_mask,
            image_shape,
            pnp_spatial_grid_cols=int(pnp_spatial_grid_cols),
            pnp_spatial_grid_rows=int(pnp_spatial_grid_rows),
            pnp_component_radius_px=float(pnp_component_radius_px),
        )
    )

    # Read inlier landmark ids
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    landmark_ids_inlier = landmark_ids[pnp_inlier_mask]

    # Determine success from the returned pose
    ok = (R is not None) and (t is not None)
    if not ok and stats.get("reason") is None:
        stats.update({"reason": "pnp_pose_missing"})

    # Try the frame-local rescue only for the diagnosed strict 8 px spatial support failure
    if bool(ok):
        strict_support_gate_stats = pnp_support_gate_stats(
            bool(ok),
            stats,
            enable_pnp_spatial_gate=bool(enable_pnp_spatial_gate),
            min_pnp_inlier_cells=int(min_pnp_inlier_cells),
            max_pnp_single_cell_fraction=float(max_pnp_single_cell_fraction),
            min_pnp_bbox_area_fraction=float(min_pnp_bbox_area_fraction),
            enable_pnp_component_gate=bool(enable_pnp_component_gate),
            min_pnp_component_count=int(min_pnp_component_count),
            max_pnp_largest_component_fraction=float(max_pnp_largest_component_fraction),
            min_pnp_largest_component_bbox_area_fraction=float(min_pnp_largest_component_bbox_area_fraction),
        )
        if bool(strict_support_gate_stats.get("pnp_spatial_gate_rejected", False)) and not bool(
            strict_support_gate_stats.get("pnp_component_gate_rejected", False)
        ):
            rescue_out = _attempt_pnp_spatial_support_rescue(
                corrs,
                K,
                threshold_px=float(threshold_px),
                base_inlier_mask=pnp_inlier_mask,
                base_spatial_gate_reason=strict_support_gate_stats.get("pnp_spatial_gate_reason", None),
                num_trials=int(num_trials),
                sample_size=int(sample_size),
                min_inliers=int(min_inliers),
                ransac_seed=int(ransac_seed),
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
                image_shape=image_shape,
                enable_pnp_spatial_gate=bool(enable_pnp_spatial_gate),
                pnp_spatial_grid_cols=int(pnp_spatial_grid_cols),
                pnp_spatial_grid_rows=int(pnp_spatial_grid_rows),
                min_pnp_inlier_cells=int(min_pnp_inlier_cells),
                max_pnp_single_cell_fraction=float(max_pnp_single_cell_fraction),
                min_pnp_bbox_area_fraction=float(min_pnp_bbox_area_fraction),
                enable_pnp_component_gate=bool(enable_pnp_component_gate),
                pnp_component_radius_px=float(pnp_component_radius_px),
                min_pnp_component_count=int(min_pnp_component_count),
                max_pnp_largest_component_fraction=float(max_pnp_largest_component_fraction),
                min_pnp_largest_component_bbox_area_fraction=float(min_pnp_largest_component_bbox_area_fraction),
            )
            stats.update(
                {
                    "pnp_support_rescue_attempted": bool(rescue_out.get("attempted", False)),
                    "pnp_support_rescue_succeeded": bool(rescue_out.get("succeeded", False)),
                    "pnp_support_rescue_reason": rescue_out.get("reason", None),
                    "pnp_support_rescue_base_strict_inliers": int(rescue_out.get("base_strict_inliers", int(np.sum(pnp_inlier_mask)))),
                    "pnp_support_rescue_base_spatial_gate_reason": rescue_out.get("base_spatial_gate_reason", strict_support_gate_stats.get("pnp_spatial_gate_reason", None)),
                    "pnp_support_rescue_loose_threshold_px": float(rescue_out.get("loose_threshold_px", _PNP_SUPPORT_RESCUE_LOOSE_THRESHOLD_PX)),
                    "pnp_support_rescue_loose_pose_ok": bool(rescue_out.get("loose_pose_ok", False)),
                    "pnp_support_rescue_loose_inliers": int(rescue_out.get("loose_inliers", 0)),
                    "pnp_support_rescue_loose_reason": rescue_out.get("loose_reason", None),
                    "pnp_support_rescue_subset_count": int(rescue_out.get("subset_count", 0)),
                    "pnp_support_rescue_subset_strict_inliers": int(rescue_out.get("subset_strict_inliers", 0)),
                    "pnp_support_rescue_subset_strict_reason": rescue_out.get("subset_strict_reason", None),
                    "pnp_support_rescue_fullset_strict_inliers": int(rescue_out.get("fullset_strict_inliers", 0)),
                    "pnp_support_rescue_fullset_strict_inlier_delta": int(rescue_out.get("fullset_strict_inlier_delta", 0)),
                    "pnp_support_rescue_min_inlier_gain": int(rescue_out.get("min_inlier_gain_required", sample_size)),
                    "pnp_support_rescue_final_spatial_gate_rejected": bool(rescue_out.get("final_spatial_gate_rejected", False)),
                    "pnp_support_rescue_final_spatial_gate_reason": rescue_out.get("final_spatial_gate_reason", None),
                    "pnp_support_rescue_final_component_gate_rejected": bool(rescue_out.get("final_component_gate_rejected", False)),
                    "pnp_support_rescue_final_component_gate_reason": rescue_out.get("final_component_gate_reason", None),
                }
            )

            if bool(rescue_out.get("succeeded", False)):
                R = np.asarray(rescue_out["R"], dtype=np.float64)
                t = np.asarray(rescue_out["t"], dtype=np.float64).reshape(3)
                pnp_inlier_mask = align_bool_mask_1d(rescue_out["pnp_inlier_mask"], n_corr, name="rescued_pnp_inlier_mask")
                stats.update(rescue_out.get("pnp_stats", {}))
                stats.update({"n_pnp_inliers": int(np.sum(pnp_inlier_mask))})
                stats.update(rescue_out.get("support_stats", {}))

    # Run the optional threshold-stability diagnostic on the accepted support
    threshold_stability_gate_rejected = False
    if bool(ok) and bool(enable_pnp_threshold_stability_diagnostic) and pnp_threshold_stability_compare_px is not None:
        try:
            stability = pnp_threshold_stability_diagnostic(
                corrs,
                K,
                R,
                t,
                pnp_inlier_mask,
                ref_threshold_px=float(threshold_px),
                compare_threshold_px=float(pnp_threshold_stability_compare_px),
                num_trials=int(num_trials),
                sample_size=int(sample_size),
                min_inliers=int(min_inliers),
                seed=int(ransac_seed),
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
                min_support_iou=float(pnp_threshold_stability_min_support_iou),
                max_translation_direction_deg=float(pnp_threshold_stability_max_translation_direction_deg),
                max_camera_centre_direction_deg=float(pnp_threshold_stability_max_camera_centre_direction_deg),
                disjoint_support_iou=float(pnp_threshold_stability_disjoint_iou),
            )
        except Exception as exc:
            stability = {
                "evaluated": False,
                "classification": "unavailable",
                "reason": str(exc),
                "unstable": False,
                "instability_reasons": [],
            }

        stats.update(pnp_threshold_stability_summary_stats(stability))

        if bool(enable_pnp_threshold_stability_gate) and bool(stability.get("unstable", False)):
            stats.update(
                {
                    "pnp_threshold_stability_gate_rejected": True,
                    "pnp_threshold_stability_gate_reason": ",".join(str(v) for v in stability.get("instability_reasons", [])),
                }
            )
            threshold_stability_gate_rejected = True

    # Read inlier landmark ids from the final selected support
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    landmark_ids_inlier = landmark_ids[pnp_inlier_mask]

    # Evaluate configured post-PnP support gates
    support_gate_stats = pnp_support_gate_stats(
        bool(ok),
        stats,
        enable_pnp_spatial_gate=bool(enable_pnp_spatial_gate),
        min_pnp_inlier_cells=int(min_pnp_inlier_cells),
        max_pnp_single_cell_fraction=float(max_pnp_single_cell_fraction),
        min_pnp_bbox_area_fraction=float(min_pnp_bbox_area_fraction),
        enable_pnp_component_gate=bool(enable_pnp_component_gate),
        min_pnp_component_count=int(min_pnp_component_count),
        max_pnp_largest_component_fraction=float(max_pnp_largest_component_fraction),
        min_pnp_largest_component_bbox_area_fraction=float(min_pnp_largest_component_bbox_area_fraction),
    )
    stats.update(support_gate_stats)
    spatial_gate_rejected = bool(support_gate_stats["pnp_spatial_gate_rejected"])
    component_gate_rejected = bool(support_gate_stats["pnp_component_gate_rejected"])

    # Reject poses whose accepted inlier support fails a post-PnP support gate
    if bool(ok) and (bool(spatial_gate_rejected) or bool(component_gate_rejected) or bool(threshold_stability_gate_rejected)):
        if bool(threshold_stability_gate_rejected):
            stats.update({"reason": "pnp_threshold_stability_failed"})
        elif bool(component_gate_rejected):
            stats.update({"reason": "pnp_component_support_failed"})
        else:
            stats.update({"reason": "pnp_spatial_coverage_failed"})
        return {
            "ok": False,
            "R": None,
            "t": None,
            "corrs": corrs,
            "pnp_inlier_mask": pnp_inlier_mask,
            "landmark_ids": np.zeros((0,), dtype=np.int64),
            "stats": stats,
        }

    return {
        "ok": bool(ok),
        "R": None if R is None else np.asarray(R, dtype=np.float64),
        "t": None if t is None else np.asarray(t, dtype=np.float64).reshape(3),
        "corrs": corrs,
        "pnp_inlier_mask": pnp_inlier_mask,
        "landmark_ids": landmark_ids_inlier,
        "stats": stats,
    }
