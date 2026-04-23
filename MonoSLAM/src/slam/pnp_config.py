# src/slam/pnp_config.py

from __future__ import annotations

from typing import Any

from core.checks import check_finite_scalar, check_in_01, check_int_ge0, check_int_gt0, check_positive


# Build default PnP frontend settings
def pnp_frontend_defaults() -> dict[str, Any]:
    return {
        "num_trials": 1000,
        "sample_size": 16,
        "threshold_px": 3.0,
        "min_inliers": 16,
        "ransac_seed": 0,
        "min_points": 6,
        "rank_tol": 1e-10,
        "min_cheirality_ratio": 0.5,
        "min_landmark_observations": 2,
        "allow_bootstrap_landmarks_for_pose": True,
        "min_post_bootstrap_observations_for_pose": 3,
        "eps": 1e-12,
        "refit": True,
        "refine_nonlinear": True,
        "refine_max_iters": 15,
        "refine_damping": 1e-6,
        "refine_step_tol": 1e-9,
        "refine_improvement_tol": 1e-9,
        "enable_pnp_spatial_gate": True,
        "pnp_spatial_grid_cols": 4,
        "pnp_spatial_grid_rows": 3,
        "min_pnp_inlier_cells": 1,
        "max_pnp_single_cell_fraction": 1.0,
        "min_pnp_bbox_area_fraction": 0.01,
        "enable_pnp_component_gate": False,
        "pnp_component_radius_px": 80.0,
        "max_pnp_largest_component_fraction": 1.0,
        "min_pnp_component_count": 0,
        "min_pnp_largest_component_bbox_area_fraction": 0.0,
        "enable_pnp_local_consistency_filter": False,
        "pnp_local_consistency_radius_px": 80.0,
        "pnp_local_consistency_min_neighbours": 3,
        "pnp_local_consistency_max_median_residual_px": 12.0,
        "pnp_local_consistency_min_keep": 0,
        "enable_pnp_spatial_thinning_filter": False,
        "pnp_spatial_thinning_radius_px": 20.0,
        "pnp_spatial_thinning_max_points_per_radius": 16,
        "pnp_spatial_thinning_min_keep": 0,
        "enable_pnp_threshold_stability_diagnostic": False,
        "enable_pnp_threshold_stability_gate": False,
        "pnp_threshold_stability_compare_px": None,
        "pnp_threshold_stability_min_support_iou": 0.25,
        "pnp_threshold_stability_max_translation_direction_deg": 120.0,
        "pnp_threshold_stability_max_camera_centre_direction_deg": 120.0,
        "pnp_threshold_stability_disjoint_iou": 0.05,
    }


# Build default threshold-stability settings
def pnp_threshold_stability_defaults() -> dict[str, Any]:
    defaults = pnp_frontend_defaults()
    return {
        key: defaults[key]
        for key in [
            "enable_pnp_threshold_stability_diagnostic",
            "enable_pnp_threshold_stability_gate",
            "pnp_threshold_stability_compare_px",
            "pnp_threshold_stability_min_support_iou",
            "pnp_threshold_stability_max_translation_direction_deg",
            "pnp_threshold_stability_max_camera_centre_direction_deg",
            "pnp_threshold_stability_disjoint_iou",
        ]
    }


# Validate threshold-stability settings
def _validate_pnp_threshold_stability_cfg(out: dict[str, Any]) -> dict[str, Any]:
    out["enable_pnp_threshold_stability_diagnostic"] = bool(out["enable_pnp_threshold_stability_diagnostic"])
    out["enable_pnp_threshold_stability_gate"] = bool(out["enable_pnp_threshold_stability_gate"])
    if out["pnp_threshold_stability_compare_px"] is not None:
        out["pnp_threshold_stability_compare_px"] = check_positive(
            out["pnp_threshold_stability_compare_px"],
            name="pnp_threshold_stability_compare_px",
            eps=0.0,
        )
    out["pnp_threshold_stability_min_support_iou"] = check_finite_scalar(
        out["pnp_threshold_stability_min_support_iou"],
        name="pnp_threshold_stability_min_support_iou",
    )
    out["pnp_threshold_stability_disjoint_iou"] = check_finite_scalar(
        out["pnp_threshold_stability_disjoint_iou"],
        name="pnp_threshold_stability_disjoint_iou",
    )
    check_in_01(out["pnp_threshold_stability_min_support_iou"], name="pnp_threshold_stability_min_support_iou", eps=0.0)
    check_in_01(out["pnp_threshold_stability_disjoint_iou"], name="pnp_threshold_stability_disjoint_iou", eps=0.0)
    out["pnp_threshold_stability_max_translation_direction_deg"] = check_positive(
        out["pnp_threshold_stability_max_translation_direction_deg"],
        name="pnp_threshold_stability_max_translation_direction_deg",
        eps=0.0,
    )
    out["pnp_threshold_stability_max_camera_centre_direction_deg"] = check_positive(
        out["pnp_threshold_stability_max_camera_centre_direction_deg"],
        name="pnp_threshold_stability_max_camera_centre_direction_deg",
        eps=0.0,
    )

    return out


# Normalise threshold-stability settings from a PnP config block
def pnp_threshold_stability_cfg_from_pnp(pnp_cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pnp_cfg, dict):
        pnp_cfg = {}

    out = pnp_threshold_stability_defaults()
    for key, default in list(out.items()):
        out[key] = pnp_cfg.get(key, default)

    return _validate_pnp_threshold_stability_cfg(out)


# Build PnP frontend kwargs from a runtime PnP config block
def pnp_frontend_kwargs_from_cfg(pnp_cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pnp_cfg, dict):
        pnp_cfg = {}

    out = pnp_frontend_defaults()
    for key, default in list(out.items()):
        out[key] = pnp_cfg.get(key, default)

    out["num_trials"] = check_int_gt0(out["num_trials"], name="pnp.num_trials")
    out["sample_size"] = check_int_gt0(out["sample_size"], name="pnp.sample_size")
    out["threshold_px"] = check_positive(out["threshold_px"], name="pnp.threshold_px", eps=0.0)
    out["min_inliers"] = check_int_gt0(out["min_inliers"], name="pnp.min_inliers")
    out["ransac_seed"] = int(out["ransac_seed"])
    out["min_points"] = check_int_gt0(out["min_points"], name="pnp.min_points")
    out["rank_tol"] = check_positive(out["rank_tol"], name="pnp.rank_tol", eps=0.0)
    out["min_cheirality_ratio"] = check_finite_scalar(out["min_cheirality_ratio"], name="pnp.min_cheirality_ratio")
    check_in_01(out["min_cheirality_ratio"], name="pnp.min_cheirality_ratio", eps=0.0)
    out["min_landmark_observations"] = check_int_gt0(out["min_landmark_observations"], name="pnp.min_landmark_observations")
    out["allow_bootstrap_landmarks_for_pose"] = bool(out["allow_bootstrap_landmarks_for_pose"])
    out["min_post_bootstrap_observations_for_pose"] = check_int_gt0(
        out["min_post_bootstrap_observations_for_pose"],
        name="pnp.min_post_bootstrap_observations_for_pose",
    )
    out["eps"] = check_positive(out["eps"], name="pnp.eps", eps=0.0)
    out["refit"] = bool(out["refit"])
    out["refine_nonlinear"] = bool(out["refine_nonlinear"])
    out["refine_max_iters"] = check_int_gt0(out["refine_max_iters"], name="pnp.refine_max_iters")
    out["refine_damping"] = check_positive(out["refine_damping"], name="pnp.refine_damping", eps=0.0)
    out["refine_step_tol"] = check_positive(out["refine_step_tol"], name="pnp.refine_step_tol", eps=0.0)
    out["refine_improvement_tol"] = check_positive(out["refine_improvement_tol"], name="pnp.refine_improvement_tol", eps=0.0)

    out["enable_pnp_spatial_gate"] = bool(out["enable_pnp_spatial_gate"])
    out["pnp_spatial_grid_cols"] = check_int_gt0(out["pnp_spatial_grid_cols"], name="pnp.pnp_spatial_grid_cols")
    out["pnp_spatial_grid_rows"] = check_int_gt0(out["pnp_spatial_grid_rows"], name="pnp.pnp_spatial_grid_rows")
    out["min_pnp_inlier_cells"] = check_int_ge0(out["min_pnp_inlier_cells"], name="pnp.min_pnp_inlier_cells")
    out["max_pnp_single_cell_fraction"] = check_finite_scalar(out["max_pnp_single_cell_fraction"], name="pnp.max_pnp_single_cell_fraction")
    out["min_pnp_bbox_area_fraction"] = check_finite_scalar(out["min_pnp_bbox_area_fraction"], name="pnp.min_pnp_bbox_area_fraction")
    check_in_01(out["max_pnp_single_cell_fraction"], name="pnp.max_pnp_single_cell_fraction", eps=0.0)
    check_in_01(out["min_pnp_bbox_area_fraction"], name="pnp.min_pnp_bbox_area_fraction", eps=0.0)
    if out["max_pnp_single_cell_fraction"] <= 0.0:
        raise ValueError(f"pnp.max_pnp_single_cell_fraction must be > 0; got {out['max_pnp_single_cell_fraction']}")

    out["enable_pnp_component_gate"] = bool(out["enable_pnp_component_gate"])
    out["pnp_component_radius_px"] = check_positive(out["pnp_component_radius_px"], name="pnp.pnp_component_radius_px", eps=0.0)
    out["max_pnp_largest_component_fraction"] = check_finite_scalar(out["max_pnp_largest_component_fraction"], name="pnp.max_pnp_largest_component_fraction")
    out["min_pnp_component_count"] = check_int_ge0(out["min_pnp_component_count"], name="pnp.min_pnp_component_count")
    out["min_pnp_largest_component_bbox_area_fraction"] = check_finite_scalar(
        out["min_pnp_largest_component_bbox_area_fraction"],
        name="pnp.min_pnp_largest_component_bbox_area_fraction",
    )
    check_in_01(out["max_pnp_largest_component_fraction"], name="pnp.max_pnp_largest_component_fraction", eps=0.0)
    check_in_01(out["min_pnp_largest_component_bbox_area_fraction"], name="pnp.min_pnp_largest_component_bbox_area_fraction", eps=0.0)
    if out["max_pnp_largest_component_fraction"] <= 0.0:
        raise ValueError(f"pnp.max_pnp_largest_component_fraction must be > 0; got {out['max_pnp_largest_component_fraction']}")

    out["enable_pnp_local_consistency_filter"] = bool(out["enable_pnp_local_consistency_filter"])
    out["pnp_local_consistency_radius_px"] = check_positive(out["pnp_local_consistency_radius_px"], name="pnp.pnp_local_consistency_radius_px", eps=0.0)
    out["pnp_local_consistency_min_neighbours"] = check_int_ge0(out["pnp_local_consistency_min_neighbours"], name="pnp.pnp_local_consistency_min_neighbours")
    out["pnp_local_consistency_max_median_residual_px"] = check_positive(
        out["pnp_local_consistency_max_median_residual_px"],
        name="pnp.pnp_local_consistency_max_median_residual_px",
        eps=0.0,
    )
    out["pnp_local_consistency_min_keep"] = check_int_ge0(out["pnp_local_consistency_min_keep"], name="pnp.pnp_local_consistency_min_keep")

    out["enable_pnp_spatial_thinning_filter"] = bool(out["enable_pnp_spatial_thinning_filter"])
    out["pnp_spatial_thinning_radius_px"] = check_positive(out["pnp_spatial_thinning_radius_px"], name="pnp.pnp_spatial_thinning_radius_px", eps=0.0)
    out["pnp_spatial_thinning_max_points_per_radius"] = check_int_gt0(out["pnp_spatial_thinning_max_points_per_radius"], name="pnp.pnp_spatial_thinning_max_points_per_radius")
    out["pnp_spatial_thinning_min_keep"] = check_int_ge0(out["pnp_spatial_thinning_min_keep"], name="pnp.pnp_spatial_thinning_min_keep")

    out.update(_validate_pnp_threshold_stability_cfg({key: out[key] for key in pnp_threshold_stability_defaults()}))

    return out
