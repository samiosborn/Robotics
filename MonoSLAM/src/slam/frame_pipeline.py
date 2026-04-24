# src/slam/frame_pipeline.py

from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import check_int_ge0, check_matrix_3x3, check_positive
from features.pipeline import FrameFeatures
from slam.keyframe import consider_promote_keyframe
from slam.map_update import append_tracked_observations_to_seed, grow_map_from_tracking_result
from slam.pnp_frontend import estimate_pose_from_seed
from slam.pnp_stats import pnp_diagnostic_summary_stats
from slam.seed import seed_keyframe_pose
from slam.tracking import track_against_keyframe


# Process one new frame against the current seed map
def process_frame_against_seed(
    K: np.ndarray,
    seed: dict[str, Any],
    keyframe_feats: FrameFeatures,
    cur_im: np.ndarray,
    *,
    feature_cfg: dict[str, Any],
    match_mode: str | None = None,
    ncc_min_score: float = 0.7,
    brief_mode: str = "nn",
    brief_max_dist: int | None = 80,
    brief_ratio: float = 0.8,
    mutual: bool = True,
    max_matches: int | None = None,
    scale_gate: int = 1,
    F_cfg: dict[str, Any],
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
    keyframe_kf: int = 1,
    current_kf: int = -1,
    grow_map: bool = True,
    min_parallax_deg: float = 1.0,
    max_depth_ratio: float = 200.0,
    max_reproj_error_px: float | None = 3.0,
    max_append_reproj_error_px_existing: float = 2.0,
    consider_keyframe: bool = True,
    keyframe_min_track_inliers: int = 80,
    keyframe_min_pnp_inliers: int = 40,
    keyframe_min_landmark_growth: int = 20,
    keyframe_min_linked_landmarks_for_promotion: int = 100,
    keyframe_min_translation_m: float = 0.10,
    keyframe_min_rotation_deg: float = 5.0,
    keyframe_require_pose: bool = True,
) -> dict[str, Any]:
    # --- Checks ---
    # Check intrinsics
    check_matrix_3x3(K, name="K", dtype=float, finite=False)

    # Check containers
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    if not isinstance(feature_cfg, dict):
        raise ValueError("feature_cfg must be a dict")
    if not isinstance(F_cfg, dict):
        raise ValueError("F_cfg must be a dict")

    # Read current image shape for PnP spatial coverage checks
    if image_shape is None:
        cur_im_arr = np.asarray(cur_im)
        if cur_im_arr.ndim < 2:
            raise ValueError(f"cur_im must have at least two dimensions; got {cur_im_arr.shape}")
        image_shape = (int(cur_im_arr.shape[0]), int(cur_im_arr.shape[1]))

    # Check frame indices
    keyframe_kf = check_int_ge0(keyframe_kf, name="keyframe_kf")
    current_kf = int(current_kf)
    if current_kf < -1:
        raise ValueError(f"current_kf must be >= -1; got {current_kf}")

    # Check map-growth controls
    min_parallax_deg = check_positive(min_parallax_deg, name="min_parallax_deg", eps=0.0)
    max_depth_ratio = check_positive(max_depth_ratio, name="max_depth_ratio", eps=0.0)
    if max_reproj_error_px is not None:
        max_reproj_error_px = check_positive(max_reproj_error_px, name="max_reproj_error_px", eps=0.0)
    max_append_reproj_error_px_existing = check_positive(
        max_append_reproj_error_px_existing,
        name="max_append_reproj_error_px_existing",
        eps=0.0,
    )

    # Check keyframe-promotion controls
    keyframe_min_track_inliers = check_int_ge0(keyframe_min_track_inliers, name="keyframe_min_track_inliers")
    keyframe_min_pnp_inliers = check_int_ge0(keyframe_min_pnp_inliers, name="keyframe_min_pnp_inliers")
    keyframe_min_landmark_growth = check_int_ge0(keyframe_min_landmark_growth, name="keyframe_min_landmark_growth")
    keyframe_min_linked_landmarks_for_promotion = check_int_ge0(
        keyframe_min_linked_landmarks_for_promotion,
        name="keyframe_min_linked_landmarks_for_promotion",
    )
    keyframe_min_translation_m = check_positive(keyframe_min_translation_m, name="keyframe_min_translation_m", eps=0.0)
    keyframe_min_rotation_deg = check_positive(keyframe_min_rotation_deg, name="keyframe_min_rotation_deg", eps=0.0)

    # Require a valid current keyframe index when promotion is enabled
    if bool(consider_keyframe) and current_kf < 0:
        raise ValueError("current_kf must be >= 0 when consider_keyframe is True")

    # Track current frame against the reference keyframe
    track_out = track_against_keyframe(
        K,
        keyframe_feats,
        cur_im,
        feature_cfg=feature_cfg,
        match_mode=match_mode,
        ncc_min_score=ncc_min_score,
        brief_mode=brief_mode,
        brief_max_dist=brief_max_dist,
        brief_ratio=brief_ratio,
        mutual=mutual,
        max_matches=max_matches,
        scale_gate=scale_gate,
        F_cfg=F_cfg,
    )

    # Read tracking stats
    track_stats = track_out.get("stats", {}) if isinstance(track_out, dict) else {}
    n_track_inliers = int(track_stats.get("n_inliers", 0))

    # Early exit if tracking produced no geometric inliers
    if n_track_inliers <= 0:
        stats = {
            "ok": False,
            "reason": track_stats.get("reason", "tracking_failed"),
            "n_track_inliers": 0,
            "n_pnp_corr": 0,
            "n_pnp_inliers": 0,
            "n_new_candidates": 0,
            "n_new_triangulated": 0,
            "n_new_added": 0,
            "keyframe_make": False,
            "keyframe_promoted": False,
            "keyframe_reason": None,
        }
        return {
            "ok": False,
            "seed": seed,
            "track_out": track_out,
            "pose_out": None,
            "map_growth_out": None,
            "keyframe_out": None,
            "R": None,
            "t": None,
            "stats": stats,
        }

    # Estimate current pose from the seed map
    pose_out = estimate_pose_from_seed(
        K,
        seed,
        track_out,
        num_trials=num_trials,
        sample_size=sample_size,
        threshold_px=threshold_px,
        min_inliers=min_inliers,
        ransac_seed=ransac_seed,
        min_points=min_points,
        rank_tol=rank_tol,
        min_cheirality_ratio=min_cheirality_ratio,
        min_landmark_observations=min_landmark_observations,
        allow_bootstrap_landmarks_for_pose=allow_bootstrap_landmarks_for_pose,
        min_post_bootstrap_observations_for_pose=min_post_bootstrap_observations_for_pose,
        eps=eps,
        refit=refit,
        refine_nonlinear=refine_nonlinear,
        refine_max_iters=refine_max_iters,
        refine_damping=refine_damping,
        refine_step_tol=refine_step_tol,
        refine_improvement_tol=refine_improvement_tol,
        image_shape=image_shape,
        enable_pnp_spatial_gate=enable_pnp_spatial_gate,
        pnp_spatial_grid_cols=pnp_spatial_grid_cols,
        pnp_spatial_grid_rows=pnp_spatial_grid_rows,
        min_pnp_inlier_cells=min_pnp_inlier_cells,
        max_pnp_single_cell_fraction=max_pnp_single_cell_fraction,
        min_pnp_bbox_area_fraction=min_pnp_bbox_area_fraction,
        enable_pnp_component_gate=enable_pnp_component_gate,
        pnp_component_radius_px=pnp_component_radius_px,
        max_pnp_largest_component_fraction=max_pnp_largest_component_fraction,
        min_pnp_component_count=min_pnp_component_count,
        min_pnp_largest_component_bbox_area_fraction=min_pnp_largest_component_bbox_area_fraction,
        enable_pnp_local_consistency_filter=enable_pnp_local_consistency_filter,
        pnp_local_consistency_radius_px=pnp_local_consistency_radius_px,
        pnp_local_consistency_min_neighbours=pnp_local_consistency_min_neighbours,
        pnp_local_consistency_max_median_residual_px=pnp_local_consistency_max_median_residual_px,
        pnp_local_consistency_min_keep=pnp_local_consistency_min_keep,
        enable_pnp_spatial_thinning_filter=enable_pnp_spatial_thinning_filter,
        pnp_spatial_thinning_radius_px=pnp_spatial_thinning_radius_px,
        pnp_spatial_thinning_max_points_per_radius=pnp_spatial_thinning_max_points_per_radius,
        pnp_spatial_thinning_min_keep=pnp_spatial_thinning_min_keep,
        enable_pnp_threshold_stability_diagnostic=enable_pnp_threshold_stability_diagnostic,
        pnp_threshold_stability_compare_px=pnp_threshold_stability_compare_px,
        pnp_threshold_stability_min_support_iou=pnp_threshold_stability_min_support_iou,
        pnp_threshold_stability_max_translation_direction_deg=pnp_threshold_stability_max_translation_direction_deg,
        pnp_threshold_stability_max_camera_centre_direction_deg=pnp_threshold_stability_max_camera_centre_direction_deg,
        pnp_threshold_stability_disjoint_iou=pnp_threshold_stability_disjoint_iou,
        enable_pnp_threshold_stability_gate=enable_pnp_threshold_stability_gate,
    )

    # Read pose stats
    pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
    ok = bool(pose_out.get("ok", False)) if isinstance(pose_out, dict) else False

    # Stop if pose estimation failed
    if not ok:
        stats = {
            "ok": False,
            "reason": pose_stats.get("reason", "pnp_failed"),
            "n_track_matches": int(track_stats.get("n_matches", 0)),
            "n_track_inliers": int(track_stats.get("n_inliers", 0)),
            "n_pnp_corr": int(pose_stats.get("n_corr", 0)),
            "n_pnp_inliers": int(pose_stats.get("n_pnp_inliers", 0)),
            **pnp_diagnostic_summary_stats(pose_stats, pnp_component_radius_px=pnp_component_radius_px),
            "n_new_candidates": 0,
            "n_new_triangulated": 0,
            "n_new_added": 0,
            "keyframe_make": False,
            "keyframe_promoted": False,
            "keyframe_reason": None,
        }
        return {
            "ok": False,
            "seed": seed,
            "track_out": track_out,
            "pose_out": pose_out,
            "map_growth_out": None,
            "keyframe_out": None,
            "R": None,
            "t": None,
            "stats": stats,
        }

    localisation_only_rescue_frame = bool(pose_stats.get("pnp_support_rescue_succeeded", False))

    # Default map-growth output
    seed_out = seed
    tracked_obs_stats: dict[str, Any] = {}
    map_growth_out = None

    if not localisation_only_rescue_frame:
        seed_out, tracked_obs_stats = append_tracked_observations_to_seed(
            seed,
            pose_out,
            keyframe_kf=keyframe_kf,
            current_kf=current_kf,
            K=K,
            track_out=track_out,
            max_append_reproj_error_px_existing=max_append_reproj_error_px_existing,
            eps=eps,
        )

        # Grow the map only after a valid pose has been recovered
        if bool(grow_map):
            # Read the frozen keyframe pose from the seed
            R_kf, t_kf = seed_keyframe_pose(seed_out)

            # Read the current pose
            R_cur = np.asarray(pose_out["R"], dtype=np.float64)
            t_cur = np.asarray(pose_out["t"], dtype=np.float64).reshape(3)

            # Run one map-growth step from the tracked frame
            map_growth_out = grow_map_from_tracking_result(
                seed_out,
                track_out,
                K,
                K,
                R_kf,
                t_kf,
                R_cur,
                t_cur,
                keyframe_kf=keyframe_kf,
                current_kf=current_kf,
                descriptor_source=track_out.get("cur_feats", None),
                min_parallax_deg=min_parallax_deg,
                max_depth_ratio=max_depth_ratio,
                max_reproj_error_px=max_reproj_error_px,
                eps=eps,
            )

            # Read the updated seed
            seed_out = map_growth_out.seed

    # Default keyframe-consideration output
    keyframe_out = None

    # Consider promoting the current frame to a new keyframe
    if bool(consider_keyframe) and not localisation_only_rescue_frame:
        keyframe_out = consider_promote_keyframe(
            seed_out,
            pose_out,
            track_out,
            map_growth_out=map_growth_out,
            current_kf=current_kf,
            image_shape=image_shape,
            min_track_inliers=keyframe_min_track_inliers,
            min_pnp_inliers=keyframe_min_pnp_inliers,
            min_landmark_growth=keyframe_min_landmark_growth,
            min_linked_landmarks_for_promotion=keyframe_min_linked_landmarks_for_promotion,
            min_translation_m=keyframe_min_translation_m,
            min_rotation_deg=keyframe_min_rotation_deg,
            require_pose=keyframe_require_pose,
        )

        # Read the updated seed after any promotion
        seed_out = keyframe_out.seed

    # Read map-growth stats
    map_stats = map_growth_out.stats if map_growth_out is not None else {}

    # Read keyframe stats
    keyframe_stats = keyframe_out.stats if keyframe_out is not None else {}

    # Pack a single frontend result
    stats = {
        "ok": True,
        "reason": None,
        "n_track_matches": int(track_stats.get("n_matches", 0)),
        "n_track_inliers": int(track_stats.get("n_inliers", 0)),
        "n_pnp_corr": int(pose_stats.get("n_corr", 0)),
        "n_pnp_inliers": int(pose_stats.get("n_pnp_inliers", 0)),
        **pnp_diagnostic_summary_stats(pose_stats, pnp_component_radius_px=pnp_component_radius_px),
        "n_tracked_obs_added": int(tracked_obs_stats.get("n_added", 0)),
        "n_append_candidates_existing": int(tracked_obs_stats.get("n_append_candidates_existing", 0)),
        "n_append_pnp_inliers": int(tracked_obs_stats.get("n_append_pnp_inliers", 0)),
        "n_append_extra_reproj_pass": int(tracked_obs_stats.get("n_append_extra_reproj_pass", 0)),
        "n_append_total": int(tracked_obs_stats.get("n_append_total", 0)),
        "n_append_duplicates": int(tracked_obs_stats.get("n_append_duplicates", 0)),
        "n_landmarks_with_obs_current_kf_after_append": int(
            tracked_obs_stats.get("n_landmarks_with_obs_current_kf_after_append", 0)
        ),
        "max_append_reproj_error_px_existing": float(max_append_reproj_error_px_existing),
        "n_new_candidates": int(map_stats.get("n_candidates", 0)),
        "n_new_triangulated": int(map_stats.get("n_triangulated_valid", 0)),
        "n_new_added": int(map_stats.get("n_added", 0)),
        "seed_landmarks_after": int(len(seed_out.get("landmarks", []))),
        "localisation_only_rescue_frame": bool(localisation_only_rescue_frame),
        "n_linked_landmarks_candidate": int(keyframe_stats.get("n_linked_landmarks_candidate", 0)),
        "keyframe_make": bool(keyframe_stats.get("make_keyframe", False)),
        "keyframe_promoted": bool(keyframe_stats.get("promoted", False)),
        "keyframe_reason": keyframe_stats.get("reason", None),
    }

    return {
        "ok": True,
        "seed": seed_out,
        "track_out": track_out,
        "pose_out": pose_out,
        "map_growth_out": map_growth_out,
        "keyframe_out": keyframe_out,
        "R": np.asarray(pose_out["R"], dtype=np.float64),
        "t": np.asarray(pose_out["t"], dtype=np.float64).reshape(3),
        "stats": stats,
    }
