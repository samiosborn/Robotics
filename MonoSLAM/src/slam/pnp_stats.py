# src/slam/pnp_stats.py

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.pnp import pnp_component_support_gate_reasons, pnp_inlier_component_support, pnp_inlier_spatial_coverage, pnp_spatial_coverage_gate_reasons


# Build a histogram of observation counts for kept landmarks
def landmark_observation_histogram(seed: dict[str, Any], landmark_ids: np.ndarray) -> dict[str, int]:
    # Read landmarks
    landmarks = seed.get("landmarks", [])
    if not isinstance(landmarks, list):
        return {}

    # Build landmark lookup
    lm_by_id: dict[int, dict[str, Any]] = {}
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        if "id" not in lm:
            continue
        lm_by_id[int(lm["id"])] = lm

    # Count kept correspondences by landmark observation count
    hist: dict[str, int] = {}
    landmark_ids = np.asarray(landmark_ids, dtype=np.int64).reshape(-1)
    for lm_id in landmark_ids:
        lm = lm_by_id.get(int(lm_id), None)
        if lm is None:
            continue

        obs = lm.get("obs", None)
        if not isinstance(obs, list):
            n_obs = 0
        else:
            n_obs = int(sum(1 for ob in obs if isinstance(ob, dict)))

        key = str(int(n_obs))
        hist[key] = int(hist.get(key, 0) + 1)

    return {str(k): int(hist[k]) for k in sorted(hist, key=lambda s: int(s))}


# Build spatial and component-support stats for an accepted PnP support
def pnp_support_diagnostic_stats(
    corrs,
    pnp_inlier_mask,
    image_shape,
    *,
    pnp_spatial_grid_cols: int = 4,
    pnp_spatial_grid_rows: int = 3,
    pnp_component_radius_px: float = 80.0,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pnp_spatial_gate_evaluated": False,
        "pnp_spatial_coverage": None,
        "pnp_inlier_occupied_cells": 0,
        "pnp_inlier_max_cell_count": 0,
        "pnp_inlier_max_cell_fraction": None,
        "pnp_inlier_bbox_area_fraction": None,
        "pnp_inlier_bbox": None,
        "pnp_component_gate_evaluated": False,
        "pnp_component_support": None,
        "pnp_inlier_component_count": 0,
        "pnp_inlier_largest_component_size": 0,
        "pnp_inlier_largest_component_fraction": None,
        "pnp_inlier_largest_component_bbox_area_fraction": None,
        "pnp_inlier_largest_component_bbox": None,
        "pnp_inlier_component_sizes": [],
    }

    if image_shape is None:
        return out

    coverage = pnp_inlier_spatial_coverage(
        corrs.x_cur,
        pnp_inlier_mask,
        image_shape,
        grid_cols=int(pnp_spatial_grid_cols),
        grid_rows=int(pnp_spatial_grid_rows),
    )
    component_support = pnp_inlier_component_support(
        corrs.x_cur,
        pnp_inlier_mask,
        image_shape,
        radius_px=float(pnp_component_radius_px),
    )

    out.update(
        {
            "pnp_spatial_gate_evaluated": True,
            "pnp_spatial_coverage": coverage,
            "pnp_inlier_occupied_cells": int(coverage.get("occupied_cells", 0)),
            "pnp_inlier_max_cell_count": int(coverage.get("max_cell_count", 0)),
            "pnp_inlier_max_cell_fraction": coverage.get("max_cell_fraction", None),
            "pnp_inlier_bbox_area_fraction": coverage.get("bbox_area_fraction", None),
            "pnp_inlier_bbox": coverage.get("bbox", None),
            "pnp_component_gate_evaluated": True,
            "pnp_component_support": component_support,
            "pnp_inlier_component_count": int(component_support.get("component_count", 0)),
            "pnp_inlier_largest_component_size": int(component_support.get("largest_component_size", 0)),
            "pnp_inlier_largest_component_fraction": component_support.get("largest_component_fraction", None),
            "pnp_inlier_largest_component_bbox_area_fraction": component_support.get("largest_component_bbox_area_fraction", None),
            "pnp_inlier_largest_component_bbox": component_support.get("largest_component_bbox", None),
            "pnp_inlier_component_sizes": component_support.get("component_sizes", []),
        }
    )

    return out


# Evaluate configured PnP support gates from precomputed support stats
def pnp_support_gate_stats(
    pose_ok: bool,
    support_stats: dict[str, Any],
    *,
    enable_pnp_spatial_gate: bool = True,
    min_pnp_inlier_cells: int = 1,
    max_pnp_single_cell_fraction: float = 1.0,
    min_pnp_bbox_area_fraction: float = 0.01,
    enable_pnp_component_gate: bool = False,
    min_pnp_component_count: int = 0,
    max_pnp_largest_component_fraction: float = 1.0,
    min_pnp_largest_component_bbox_area_fraction: float = 0.0,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pnp_spatial_gate_rejected": False,
        "pnp_spatial_gate_reason": None,
        "pnp_component_gate_rejected": False,
        "pnp_component_gate_reason": None,
    }

    if bool(pose_ok) and bool(enable_pnp_spatial_gate) and bool(support_stats.get("pnp_spatial_gate_evaluated", False)):
        gate_reasons = pnp_spatial_coverage_gate_reasons(
            support_stats.get("pnp_spatial_coverage", None),
            min_occupied_cells=int(min_pnp_inlier_cells),
            max_single_cell_fraction=float(max_pnp_single_cell_fraction),
            min_bbox_area_fraction=float(min_pnp_bbox_area_fraction),
        )
        if len(gate_reasons) > 0:
            out["pnp_spatial_gate_rejected"] = True
            out["pnp_spatial_gate_reason"] = ",".join(gate_reasons)

    if bool(pose_ok) and bool(enable_pnp_component_gate) and bool(support_stats.get("pnp_component_gate_evaluated", False)):
        gate_reasons = pnp_component_support_gate_reasons(
            support_stats.get("pnp_component_support", None),
            min_component_count=int(min_pnp_component_count),
            max_largest_component_fraction=float(max_pnp_largest_component_fraction),
            min_largest_component_bbox_area_fraction=float(min_pnp_largest_component_bbox_area_fraction),
        )
        if len(gate_reasons) > 0:
            out["pnp_component_gate_rejected"] = True
            out["pnp_component_gate_reason"] = ",".join(gate_reasons)

    return out


# Flatten threshold-stability diagnostics into frontend stats
def pnp_threshold_stability_summary_stats(stability: dict[str, Any] | None) -> dict[str, Any]:
    stability = stability if isinstance(stability, dict) else {}
    return {
        "pnp_threshold_stability": stability if len(stability) > 0 else None,
        "pnp_threshold_stability_evaluated": bool(stability.get("evaluated", False)),
        "pnp_threshold_stability_classification": stability.get("classification", "unavailable"),
        "pnp_threshold_stability_unstable": bool(stability.get("unstable", False)),
        "pnp_threshold_stability_ref_inliers": int(stability.get("ref_inliers", 0)),
        "pnp_threshold_stability_compare_inliers": int(stability.get("compare_inliers", 0)),
        "pnp_threshold_stability_support_iou": stability.get("support_iou", None),
        "pnp_threshold_stability_rotation_delta_deg": stability.get("rotation_delta_deg", None),
        "pnp_threshold_stability_translation_direction_delta_deg": stability.get("translation_direction_delta_deg", None),
        "pnp_threshold_stability_camera_centre_direction_delta_deg": stability.get("camera_centre_direction_delta_deg", None),
        "pnp_threshold_stability_looser_solution_only": bool(stability.get("one_solution_only_at_looser_threshold", False)),
        "pnp_threshold_stability_supports_disjoint": bool(stability.get("supports_effectively_disjoint", False)),
        "pnp_threshold_stability_reasons": stability.get("instability_reasons", []),
    }


# Copy PnP diagnostic stats into a frame-level summary
def pnp_diagnostic_summary_stats(pose_stats: dict[str, Any], *, pnp_component_radius_px: float = 80.0) -> dict[str, Any]:
    return {
        "n_corr_after_pose_filter": int(pose_stats.get("n_corr_after_pose_filter", pose_stats.get("n_corr", 0))),
        "n_corr_after_local_consistency_filter": int(pose_stats.get("n_corr_after_local_consistency_filter", pose_stats.get("n_corr", 0))),
        "n_corr_after_spatial_thinning_filter": int(pose_stats.get("n_corr_after_spatial_thinning_filter", pose_stats.get("n_corr", 0))),
        "pnp_local_consistency_filter_enabled": bool(pose_stats.get("pnp_local_consistency_filter_enabled", False)),
        "pnp_local_consistency_filter_evaluated": bool(pose_stats.get("pnp_local_consistency_filter_evaluated", False)),
        "pnp_local_consistency_filter_removed": int(pose_stats.get("pnp_local_consistency_filter_removed", 0)),
        "pnp_spatial_thinning_filter_enabled": bool(pose_stats.get("pnp_spatial_thinning_filter_enabled", False)),
        "pnp_spatial_thinning_filter_evaluated": bool(pose_stats.get("pnp_spatial_thinning_filter_evaluated", False)),
        "pnp_spatial_thinning_filter_removed": int(pose_stats.get("pnp_spatial_thinning_filter_removed", 0)),
        "pnp_spatial_gate_enabled": bool(pose_stats.get("pnp_spatial_gate_enabled", False)),
        "pnp_spatial_gate_evaluated": bool(pose_stats.get("pnp_spatial_gate_evaluated", False)),
        "pnp_spatial_gate_rejected": bool(pose_stats.get("pnp_spatial_gate_rejected", False)),
        "pnp_spatial_gate_reason": pose_stats.get("pnp_spatial_gate_reason", None),
        "pnp_inlier_occupied_cells": int(pose_stats.get("pnp_inlier_occupied_cells", 0)),
        "pnp_inlier_max_cell_count": int(pose_stats.get("pnp_inlier_max_cell_count", 0)),
        "pnp_inlier_max_cell_fraction": pose_stats.get("pnp_inlier_max_cell_fraction", None),
        "pnp_inlier_bbox_area_fraction": pose_stats.get("pnp_inlier_bbox_area_fraction", None),
        "pnp_inlier_bbox": pose_stats.get("pnp_inlier_bbox", None),
        "pnp_component_gate_enabled": bool(pose_stats.get("pnp_component_gate_enabled", False)),
        "pnp_component_gate_evaluated": bool(pose_stats.get("pnp_component_gate_evaluated", False)),
        "pnp_component_gate_rejected": bool(pose_stats.get("pnp_component_gate_rejected", False)),
        "pnp_component_gate_reason": pose_stats.get("pnp_component_gate_reason", None),
        "pnp_component_radius_px": float(pose_stats.get("pnp_component_radius_px", pnp_component_radius_px)),
        "pnp_inlier_component_count": int(pose_stats.get("pnp_inlier_component_count", 0)),
        "pnp_inlier_largest_component_size": int(pose_stats.get("pnp_inlier_largest_component_size", 0)),
        "pnp_inlier_largest_component_fraction": pose_stats.get("pnp_inlier_largest_component_fraction", None),
        "pnp_inlier_largest_component_bbox_area_fraction": pose_stats.get("pnp_inlier_largest_component_bbox_area_fraction", None),
        "pnp_inlier_largest_component_bbox": pose_stats.get("pnp_inlier_largest_component_bbox", None),
        "pnp_threshold_stability_diagnostic_enabled": bool(pose_stats.get("pnp_threshold_stability_diagnostic_enabled", False)),
        "pnp_threshold_stability_gate_enabled": bool(pose_stats.get("pnp_threshold_stability_gate_enabled", False)),
        "pnp_threshold_stability_evaluated": bool(pose_stats.get("pnp_threshold_stability_evaluated", False)),
        "pnp_threshold_stability_gate_rejected": bool(pose_stats.get("pnp_threshold_stability_gate_rejected", False)),
        "pnp_threshold_stability_gate_reason": pose_stats.get("pnp_threshold_stability_gate_reason", None),
        "pnp_threshold_stability": pose_stats.get("pnp_threshold_stability", None),
        "pnp_threshold_stability_classification": pose_stats.get("pnp_threshold_stability_classification", "unavailable"),
        "pnp_threshold_stability_unstable": bool(pose_stats.get("pnp_threshold_stability_unstable", False)),
        "pnp_threshold_stability_ref_inliers": int(pose_stats.get("pnp_threshold_stability_ref_inliers", 0)),
        "pnp_threshold_stability_compare_inliers": int(pose_stats.get("pnp_threshold_stability_compare_inliers", 0)),
        "pnp_threshold_stability_support_iou": pose_stats.get("pnp_threshold_stability_support_iou", None),
        "pnp_threshold_stability_rotation_delta_deg": pose_stats.get("pnp_threshold_stability_rotation_delta_deg", None),
        "pnp_threshold_stability_translation_direction_delta_deg": pose_stats.get("pnp_threshold_stability_translation_direction_delta_deg", None),
        "pnp_threshold_stability_camera_centre_direction_delta_deg": pose_stats.get("pnp_threshold_stability_camera_centre_direction_delta_deg", None),
        "pnp_threshold_stability_looser_solution_only": bool(pose_stats.get("pnp_threshold_stability_looser_solution_only", False)),
        "pnp_threshold_stability_supports_disjoint": bool(pose_stats.get("pnp_threshold_stability_supports_disjoint", False)),
        "pnp_threshold_stability_reasons": pose_stats.get("pnp_threshold_stability_reasons", []),
    }
