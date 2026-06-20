# src/slam/frame_pipeline.py

from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import check_int_ge0, check_mask_bool_N, check_matrix_3x3, check_positive
from geometry.camera import reprojection_errors_sq
from geometry.pnp import _pnp_inlier_mask_from_pose, estimate_pose_pnp_ransac
from slam.bundle_adjustment import local_bundle_adjustment_not_run_stats, run_local_bundle_adjustment
from slam.keyframe import consider_promote_keyframe
from slam.keyframe_state import get_active_keyframe_features, get_pose_for_kf, has_active_keyframe_state, set_active_keyframe_record, store_current_pose
from slam.map_update import append_tracked_observations_to_seed, grow_map_from_tracking_result
from slam.map_mutation import merge_map_mutation_reports
from slam.pnp_frontend import estimate_pose_from_seed
from slam.pnp_stats import pnp_diagnostic_summary_stats
from slam.seed import seed_keyframe_pose
from slam.tracking import track_against_keyframe


_CANONICAL_POSE_PROXY_GATE_MEDIAN_PX = 8.0
_CANONICAL_POSE_PROXY_STRICT_EVAL_PX = 8.0
_CANONICAL_POSE_PROXY_RESCUE_THRESHOLD_PX = 40.0
_CANONICAL_POSE_PROXY_NUM_TRIALS = 5000
_CANONICAL_POSE_PROXY_SEEDS = (0, 1, 2, 3, 7)


# Copy a pose block for accepted-pose storage
def _copy_pose_blocks(R, t) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(R, dtype=np.float64).copy(),
        np.asarray(t, dtype=np.float64).reshape(3).copy(),
    )


def _copy_seed_for_active_record(seed: dict[str, Any]) -> dict[str, Any]:
    seed_out = dict(seed)
    if "poses" in seed and isinstance(seed["poses"], dict):
        seed_out["poses"] = dict(seed["poses"])
    if "keyframes" in seed and isinstance(seed["keyframes"], dict):
        seed_out["keyframes"] = dict(seed["keyframes"])
    return seed_out


def _canonical_pose_proxy_default_stats() -> dict[str, Any]:
    return {
        "canonical_pose_proxy_residual_shape_trigger_fired": False,
        "canonical_pose_proxy_trigger_residual_median_px": None,
        "canonical_pose_proxy_num_seeds_tried": 0,
        "canonical_pose_proxy_selected": False,
        "canonical_pose_proxy_selected_strict_8px_inliers": 0,
        "canonical_pose_proxy_selected_residual_median_px": None,
        "canonical_pose_proxy_storage_replaced": False,
        "canonical_pose_proxy_failed_fallback": False,
        "canonical_pose_proxy_reason": "not_evaluated",
    }


def _canonical_pose_proxy_residual_median_px(
    K: np.ndarray,
    R,
    t,
    corrs,
    *,
    eps: float,
    mask=None,
) -> float | None:
    X_w = np.asarray(corrs.X_w, dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    n_corr = int(X_w.shape[1])
    if mask is not None:
        support_mask = check_mask_bool_N(mask, n_corr, name="canonical_pose_proxy_mask")
        if support_mask is None:
            support_mask = np.zeros((n_corr,), dtype=bool)
        X_w = X_w[:, support_mask]
        x_cur = x_cur[:, support_mask]
    if int(X_w.shape[1]) == 0:
        return None

    R_arr = np.asarray(R, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64).reshape(3)
    X_c = R_arr @ X_w + t_arr.reshape(3, 1)
    err_sq = np.asarray(reprojection_errors_sq(K, R_arr, t_arr, X_w, x_cur), dtype=np.float64).reshape(-1)
    depth = np.asarray(X_c[2, :], dtype=np.float64).reshape(-1)
    valid = np.isfinite(depth) & (depth > float(eps)) & np.isfinite(err_sq) & (err_sq >= 0.0)
    if not bool(np.any(valid)):
        return None

    errors_px = np.sqrt(err_sq[valid])
    return float(np.median(errors_px))


def _canonical_pose_proxy_strict_inliers(
    K: np.ndarray,
    R,
    t,
    corrs,
    *,
    eps: float,
) -> int:
    mask, _ = _pnp_inlier_mask_from_pose(
        corrs.X_w,
        corrs.x_cur,
        K,
        np.asarray(R, dtype=np.float64),
        np.asarray(t, dtype=np.float64).reshape(3),
        threshold_px=float(_CANONICAL_POSE_PROXY_STRICT_EVAL_PX),
        eps=float(eps),
    )
    return int(np.sum(np.asarray(mask, dtype=bool).reshape(-1)))


def _select_canonical_pose_proxy(
    K: np.ndarray,
    pose_out: dict[str, Any],
    pose_stats: dict[str, Any],
    *,
    sample_size: int,
    min_inliers: int,
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
) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict[str, Any]]:
    stats = _canonical_pose_proxy_default_stats()
    if not bool(pose_stats.get("pnp_support_rescue_loose_localisation_fallback_succeeded", False)):
        stats["canonical_pose_proxy_reason"] = "not_loose_rescue"
        return None, stats
    if not isinstance(pose_out, dict) or pose_out.get("corrs", None) is None:
        stats["canonical_pose_proxy_reason"] = "correspondences_unavailable"
        return None, stats
    if pose_out.get("R", None) is None or pose_out.get("t", None) is None:
        stats["canonical_pose_proxy_reason"] = "accepted_pose_unavailable"
        return None, stats

    corrs = pose_out["corrs"]
    n_corr = int(corrs.X_w.shape[1])
    accepted_mask = check_mask_bool_N(
        pose_out.get("pnp_inlier_mask", np.zeros((n_corr,), dtype=bool)),
        n_corr,
        name="canonical_pose_proxy_accepted_mask",
    )
    if accepted_mask is None:
        accepted_mask = np.zeros((n_corr,), dtype=bool)

    accepted_median_px = _canonical_pose_proxy_residual_median_px(
        K,
        pose_out["R"],
        pose_out["t"],
        corrs,
        eps=float(eps),
        mask=accepted_mask,
    )
    trigger_fired = bool(
        accepted_median_px is not None
        and float(accepted_median_px) > float(_CANONICAL_POSE_PROXY_GATE_MEDIAN_PX)
    )
    stats.update(
        {
            "canonical_pose_proxy_residual_shape_trigger_fired": bool(trigger_fired),
            "canonical_pose_proxy_trigger_residual_median_px": accepted_median_px,
        }
    )
    if not bool(trigger_fired):
        stats["canonical_pose_proxy_reason"] = "trigger_not_fired"
        return None, stats

    if n_corr < int(sample_size):
        stats["canonical_pose_proxy_failed_fallback"] = True
        stats["canonical_pose_proxy_reason"] = "too_few_correspondences"
        return None, stats

    stats["canonical_pose_proxy_num_seeds_tried"] = int(len(_CANONICAL_POSE_PROXY_SEEDS))
    best_pose: tuple[np.ndarray, np.ndarray] | None = None
    best_strict_inliers = 0
    best_median_px: float | None = None

    for seed_value in _CANONICAL_POSE_PROXY_SEEDS:
        try:
            R_proxy, t_proxy, _, _ = estimate_pose_pnp_ransac(
                corrs,
                K,
                num_trials=int(_CANONICAL_POSE_PROXY_NUM_TRIALS),
                sample_size=int(sample_size),
                threshold_px=float(_CANONICAL_POSE_PROXY_RESCUE_THRESHOLD_PX),
                min_inliers=int(min_inliers),
                seed=int(seed_value),
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
        except Exception:
            continue
        if R_proxy is None or t_proxy is None:
            continue

        strict_inliers = _canonical_pose_proxy_strict_inliers(
            K,
            R_proxy,
            t_proxy,
            corrs,
            eps=float(eps),
        )
        median_px = _canonical_pose_proxy_residual_median_px(
            K,
            R_proxy,
            t_proxy,
            corrs,
            eps=float(eps),
        )
        candidate_median = float("inf") if median_px is None else float(median_px)
        best_candidate_median = float("inf") if best_median_px is None else float(best_median_px)
        if best_pose is None or int(strict_inliers) > int(best_strict_inliers):
            keep_candidate = True
        elif int(strict_inliers) == int(best_strict_inliers) and candidate_median < best_candidate_median:
            keep_candidate = True
        else:
            keep_candidate = False

        if bool(keep_candidate):
            best_pose = (
                np.asarray(R_proxy, dtype=np.float64).copy(),
                np.asarray(t_proxy, dtype=np.float64).reshape(3).copy(),
            )
            best_strict_inliers = int(strict_inliers)
            best_median_px = None if median_px is None else float(median_px)

    if best_pose is None:
        stats["canonical_pose_proxy_failed_fallback"] = True
        stats["canonical_pose_proxy_reason"] = "all_proxy_seeds_failed"
        return None, stats

    stats.update(
        {
            "canonical_pose_proxy_selected": True,
            "canonical_pose_proxy_selected_strict_8px_inliers": int(best_strict_inliers),
            "canonical_pose_proxy_selected_residual_median_px": best_median_px,
            "canonical_pose_proxy_reason": "proxy_selected",
        }
    )
    return best_pose, stats


# Flatten local BA diagnostics into frame stats
def _local_ba_summary_stats(local_ba_stats: dict[str, Any]) -> dict[str, Any]:
    stats = local_ba_stats if isinstance(local_ba_stats, dict) else local_bundle_adjustment_not_run_stats("stats_unavailable")
    return {
        "local_ba_attempted": bool(stats.get("attempted", False)),
        "local_ba_skipped": bool(stats.get("skipped", False)),
        "local_ba_succeeded": bool(stats.get("succeeded", False)),
        "local_ba_skip_reason": stats.get("skip_reason", None),
        "local_ba_reason": stats.get("reason", None),
        "local_ba_acceptance_reason": stats.get("acceptance_reason", None),
        "local_ba_rejection_reason": stats.get("rejection_reason", None),
        "local_ba_n_local_keyframes": int(stats.get("n_local_keyframes", 0)),
        "local_ba_local_keyframes": list(stats.get("local_keyframes", [])),
        "local_ba_anchor_kf": stats.get("anchor_kf", None),
        "local_ba_optimised_keyframes": list(stats.get("optimised_keyframes", [])),
        "local_ba_n_local_landmarks": int(stats.get("n_local_landmarks", 0)),
        "local_ba_n_observations": int(stats.get("n_observations", 0)),
        "local_ba_initial_mean_reproj_error_px": stats.get("initial_mean_reproj_error_px", None),
        "local_ba_initial_median_reproj_error_px": stats.get("initial_median_reproj_error_px", None),
        "local_ba_final_mean_reproj_error_px": stats.get("final_mean_reproj_error_px", None),
        "local_ba_final_median_reproj_error_px": stats.get("final_median_reproj_error_px", None),
        "local_ba_iterations": int(stats.get("iterations", 0)),
        "local_ba_accepted_iterations": int(stats.get("accepted_iterations", 0)),
        "local_ba_initial_damping": stats.get("initial_damping", None),
        "local_ba_final_damping": stats.get("final_damping", None),
        "local_ba_stats": stats,
    }


def _support_basis_from_rescued_pose(
    track_out: dict[str, Any],
    pose_out: dict[str, Any],
    current_kf: int,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    cur_feats = track_out.get("cur_feats", None)
    if cur_feats is None or not hasattr(cur_feats, "kps_xy"):
        return None, {
            "n_feat": 0,
            "n_corr": 0,
            "n_accepted_support": 0,
            "n_lookup_mapped": 0,
            "n_conflicts": 0,
            "n_out_of_range": 0,
        }

    corrs = pose_out["corrs"]
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(corrs.cur_feat_idx, dtype=np.int64).reshape(-1)
    if int(cur_feat_idx.size) != int(landmark_ids.size):
        raise ValueError(
            f"pose_out['corrs'].cur_feat_idx must have {landmark_ids.size} entries to match landmark_ids; got {cur_feat_idx.size}"
        )
    support_mask = check_mask_bool_N(
        pose_out.get("pnp_inlier_mask", np.zeros((landmark_ids.size,), dtype=bool)),
        int(landmark_ids.size),
        name="pose_out['pnp_inlier_mask']",
    )
    if support_mask is None:
        support_mask = np.zeros((int(landmark_ids.size),), dtype=bool)
    n_feat = int(np.asarray(cur_feats.kps_xy).shape[0])
    n_corr = int(landmark_ids.size)
    lookup = np.full((n_feat,), -1, dtype=np.int64)

    n_mapped = 0
    n_conflicts = 0
    n_out_of_range = 0
    for i in np.flatnonzero(support_mask[:n_corr]):
        lm_id = int(landmark_ids[int(i)])
        feat_idx = int(cur_feat_idx[int(i)])
        if feat_idx < 0 or feat_idx >= int(n_feat):
            n_out_of_range += 1
            continue
        prev = int(lookup[feat_idx])
        if prev >= 0 and prev != lm_id:
            n_conflicts += 1
            continue
        if prev < 0:
            n_mapped += 1
        lookup[feat_idx] = lm_id

    stats = {
        "n_feat": int(n_feat),
        "n_corr": int(n_corr),
        "n_accepted_support": int(np.sum(support_mask[:n_corr])),
        "n_lookup_mapped": int(n_mapped),
        "n_conflicts": int(n_conflicts),
        "n_out_of_range": int(n_out_of_range),
    }
    if n_mapped <= 0 or n_conflicts > 0 or n_out_of_range > 0:
        return None, stats

    return {
        "kf": int(current_kf),
        "R": np.asarray(pose_out["R"], dtype=np.float64).copy(),
        "t": np.asarray(pose_out["t"], dtype=np.float64).reshape(3).copy(),
        "feats": cur_feats,
        "landmark_id_by_feat": lookup,
        "n_lookup_mapped": int(n_mapped),
        "n_accepted_support": int(np.sum(support_mask[:n_corr])),
    }, stats


def _seed_with_support_basis(seed: dict[str, Any], basis: dict[str, Any]) -> dict[str, Any]:
    seed_out = _copy_seed_for_active_record(seed)
    pose = (
        np.asarray(basis["R"], dtype=np.float64).copy(),
        np.asarray(basis["t"], dtype=np.float64).reshape(3).copy(),
    )
    lookup_raw = basis.get("landmark_id_by_feat", None)
    lookup = np.asarray(lookup_raw, dtype=np.int64).reshape(-1).copy()
    set_active_keyframe_record(seed_out, int(basis["kf"]), pose, basis["feats"], lookup)
    return seed_out


def _refresh_active_lookup_basis_from_rescued_support(
    seed: dict[str, Any],
    track_out: dict[str, Any],
    pose_out: dict[str, Any],
    current_kf: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    seed_out = _copy_seed_for_active_record(seed)
    basis, stats = _support_basis_from_rescued_pose(track_out, pose_out, current_kf)
    if basis is None:
        return seed_out, stats
    return _seed_with_support_basis(seed_out, basis), stats


def _rescued_support_history_stats(
    seed: dict[str, Any],
    pose_out: dict[str, Any],
    K: np.ndarray,
    current_kf: int,
) -> dict[str, Any]:
    corrs = pose_out["corrs"]
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    support_mask = check_mask_bool_N(
        pose_out.get("pnp_inlier_mask", np.zeros((landmark_ids.size,), dtype=bool)),
        int(landmark_ids.size),
        name="pose_out['pnp_inlier_mask']",
    )
    if support_mask is None:
        support_mask = np.zeros((int(landmark_ids.size),), dtype=bool)

    support_landmark_ids = sorted(set(int(v) for v in landmark_ids[support_mask]))
    landmark_by_id = {
        int(lm["id"]): lm
        for lm in seed.get("landmarks", [])
        if isinstance(lm, dict) and "id" in lm
    }
    n_evaluated = 0
    n_inconsistent = 0

    for lm_id in support_landmark_ids:
        lm = landmark_by_id.get(int(lm_id), None)
        if not isinstance(lm, dict):
            continue
        X_w = np.asarray(lm.get("X_w", None), dtype=np.float64).reshape(-1)
        observations = lm.get("obs", [])
        if X_w.size != 3 or not np.isfinite(X_w).all() or not isinstance(observations, list):
            continue

        errors_px: list[float] = []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            obs_kf = int(observation.get("kf", -1))
            xy = np.asarray(observation.get("xy", None), dtype=np.float64).reshape(-1)
            if obs_kf < 0 or obs_kf >= int(current_kf) or xy.size != 2 or not np.isfinite(xy).all():
                continue
            try:
                R_obs, t_obs = get_pose_for_kf(seed, obs_kf, context="rescued support history")
            except ValueError:
                continue
            R_obs = np.asarray(R_obs, dtype=np.float64)
            t_obs = np.asarray(t_obs, dtype=np.float64).reshape(3)
            X_c = R_obs @ X_w + t_obs
            if not np.isfinite(X_c).all() or float(X_c[2]) <= 0.0:
                continue
            err_sq = np.asarray(
                reprojection_errors_sq(
                    K,
                    R_obs,
                    t_obs,
                    X_w.reshape(3, 1),
                    xy.reshape(2, 1),
                ),
                dtype=np.float64,
            ).reshape(-1)
            if err_sq.size == 1 and np.isfinite(err_sq[0]) and float(err_sq[0]) >= 0.0:
                errors_px.append(float(np.sqrt(err_sq[0])))

        if len(errors_px) < 2:
            continue
        errors = np.asarray(errors_px, dtype=np.float64)
        n_evaluated += 1
        if bool(
            float(np.median(errors)) > 3.0
            or float(np.percentile(errors, 90.0)) > 8.0
            or float(np.max(errors)) > 12.0
        ):
            n_inconsistent += 1

    return {
        "n_support": int(len(support_landmark_ids)),
        "n_evaluated": int(n_evaluated),
        "n_inconsistent": int(n_inconsistent),
        "inconsistent_fraction": float(n_inconsistent / max(len(support_landmark_ids), 1)),
    }


def _attempt_incoherent_support_recovery(
    K: np.ndarray,
    seed: dict[str, Any],
    cur_im: np.ndarray,
    *,
    feature_cfg: dict[str, Any],
    match_mode: str | None,
    ncc_min_score: float,
    brief_mode: str,
    brief_max_dist: int | None,
    brief_ratio: float,
    mutual: bool,
    max_matches: int | None,
    scale_gate: int,
    F_cfg: dict[str, Any],
    base_n_corr: int,
    num_trials: int,
    sample_size: int,
    threshold_px: float,
    min_inliers: int,
    ransac_seed: int,
    min_points: int,
    rank_tol: float,
    min_cheirality_ratio: float,
    min_landmark_observations: int,
    allow_bootstrap_landmarks_for_pose: bool,
    min_post_bootstrap_observations_for_pose: int,
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
    max_pnp_largest_component_fraction: float,
    min_pnp_component_count: int,
    min_pnp_largest_component_bbox_area_fraction: float,
    enable_pnp_local_consistency_filter: bool,
    pnp_local_consistency_radius_px: float,
    pnp_local_consistency_min_neighbours: int,
    pnp_local_consistency_max_median_residual_px: float,
    pnp_local_consistency_min_keep: int,
    enable_pnp_spatial_thinning_filter: bool,
    pnp_spatial_thinning_radius_px: float,
    pnp_spatial_thinning_max_points_per_radius: int,
    pnp_spatial_thinning_min_keep: int,
    enable_pnp_threshold_stability_diagnostic: bool,
    pnp_threshold_stability_compare_px: float | None,
    pnp_threshold_stability_min_support_iou: float,
    pnp_threshold_stability_max_translation_direction_deg: float,
    pnp_threshold_stability_max_camera_centre_direction_deg: float,
    pnp_threshold_stability_disjoint_iou: float,
    enable_pnp_threshold_stability_gate: bool,
    temporal_reference_R,
    temporal_reference_t,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    basis = seed.get("_incoherent_recovery_support_basis", None)
    if not isinstance(basis, dict):
        return None, None
    if basis.get("feats", None) is None:
        return None, None
    if basis.get("landmark_id_by_feat", None) is None:
        return None, None

    anchor_seed = _seed_with_support_basis(seed, basis)
    try:
        recovery_track_out = track_against_keyframe(
            K,
            basis["feats"],
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
    except Exception:
        return None, None

    recovery_pose_out = estimate_pose_from_seed(
        K,
        anchor_seed,
        recovery_track_out,
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
        temporal_reference_R=temporal_reference_R,
        temporal_reference_t=temporal_reference_t,
    )

    recovery_stats = recovery_pose_out.get("stats", {}) if isinstance(recovery_pose_out, dict) else {}
    recovery_stats = recovery_stats if isinstance(recovery_stats, dict) else {}
    candidate_n_corr = int(recovery_stats.get("n_corr", 0))
    candidate_n_inliers = int(recovery_stats.get("n_pnp_inliers", recovery_stats.get("n_inliers", 0)))
    support_better = candidate_n_corr > int(base_n_corr)
    pose_ok = bool(recovery_pose_out.get("ok", False)) if isinstance(recovery_pose_out, dict) else False
    if not bool(pose_ok and support_better and candidate_n_inliers >= int(min_inliers)):
        return None, None

    recovery_stats = dict(recovery_stats)
    recovery_stats.update(
        {
            "pnp_incoherent_recovery_attempted": True,
            "pnp_incoherent_recovery_succeeded": True,
            "pnp_incoherent_recovery_reason": "previous_rescued_support_basis",
            "pnp_incoherent_recovery_source_kf": int(basis.get("kf", -1)),
            "pnp_incoherent_recovery_base_n_corr": int(base_n_corr),
            "pnp_incoherent_recovery_candidate_n_corr": int(candidate_n_corr),
            "pnp_incoherent_recovery_candidate_n_inliers": int(candidate_n_inliers),
        }
    )
    recovery_pose_out = dict(recovery_pose_out)
    recovery_pose_out["stats"] = recovery_stats
    return recovery_track_out, recovery_pose_out


# Process one new frame against the current seed map
def process_frame_against_seed(
    K: np.ndarray,
    seed: dict[str, Any],
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
    enable_local_ba: bool = True,
    local_ba_max_keyframes: int = 3,
    local_ba_min_keyframes: int = 2,
    local_ba_min_landmarks: int = 6,
    local_ba_min_observations: int = 12,
    local_ba_max_iters: int = 5,
    local_ba_initial_damping: float = 1e-3,
    local_ba_max_damping: float = 1e9,
    local_ba_step_tol: float = 1e-7,
    local_ba_improvement_tol: float = 1e-6,
) -> dict[str, Any]:
    # Validate inputs
    check_matrix_3x3(K, name="K", dtype=float, finite=False)

    # Check containers
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    if not isinstance(feature_cfg, dict):
        raise ValueError("feature_cfg must be a dict")
    if not isinstance(F_cfg, dict):
        raise ValueError("F_cfg must be a dict")
    if not has_active_keyframe_state(seed):
        raise ValueError("seed must contain canonical active keyframe state")
    keyframe_feats = get_active_keyframe_features(seed)

    # Read current image shape for PnP spatial coverage checks
    if image_shape is None:
        cur_im_arr = np.asarray(cur_im)
        if cur_im_arr.ndim < 2:
            raise ValueError(f"cur_im must have at least two dimensions; got {cur_im_arr.shape}")
        image_shape = (int(cur_im_arr.shape[0]), int(cur_im_arr.shape[1]))

    # Check frame indices
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
            **_local_ba_summary_stats(local_bundle_adjustment_not_run_stats("tracking_failed")),
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
    temporal_reference_pose = seed.get("last_accepted_pose", None)
    if isinstance(temporal_reference_pose, dict):
        temporal_reference_R = temporal_reference_pose.get("R", None)
        temporal_reference_t = temporal_reference_pose.get("t", None)
    else:
        temporal_reference_R, temporal_reference_t = seed_keyframe_pose(seed)

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
        temporal_reference_R=temporal_reference_R,
        temporal_reference_t=temporal_reference_t,
    )

    # Read pose stats
    pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
    ok = bool(pose_out.get("ok", False)) if isinstance(pose_out, dict) else False

    # Try one map-safe rebasing attempt for incoherent support
    if not bool(ok) and str(pose_stats.get("reason", "")) == "pnp_support_incoherent":
        recovery_track_out, recovery_pose_out = _attempt_incoherent_support_recovery(
            K,
            seed,
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
            base_n_corr=int(pose_stats.get("n_corr", 0)),
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
            temporal_reference_R=temporal_reference_R,
            temporal_reference_t=temporal_reference_t,
        )
        if recovery_pose_out is not None and recovery_track_out is not None:
            track_out = recovery_track_out
            track_stats = track_out.get("stats", {}) if isinstance(track_out, dict) else {}
            pose_out = recovery_pose_out
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
            **_local_ba_summary_stats(local_bundle_adjustment_not_run_stats("pnp_failed")),
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

    incoherent_recovery_frame = bool(pose_stats.get("pnp_incoherent_recovery_succeeded", False))
    localisation_only_rescue_frame = bool(pose_stats.get("pnp_support_rescue_succeeded", False) or incoherent_recovery_frame)

    # Localisation-only rescue frames may append existing-landmark observations only
    seed_out = seed
    tracked_obs_stats: dict[str, Any] = {}
    tracked_obs_report: dict[str, Any] | None = None
    map_growth_out = None
    guarded_support_refresh_stats: dict[str, Any] = {
        "triggered": False,
        "reason": None,
        "n_lookup_mapped": 0,
        "n_accepted_support": 0,
        "history_evaluated": 0,
        "history_inconsistent": 0,
        "history_inconsistent_fraction": 0.0,
    }

    if not localisation_only_rescue_frame:
        seed_out, tracked_obs_stats, tracked_obs_report = append_tracked_observations_to_seed(
            seed,
            pose_out,
            current_kf=current_kf,
            K=K,
            track_out=track_out,
            max_append_reproj_error_px_existing=max_append_reproj_error_px_existing,
            prune_stale_map_growth=True,
            eps=eps,
            return_report=True,
        )

        # Grow the map only from non-rescue poses
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
                current_kf=current_kf,
                descriptor_source=track_out.get("cur_feats", None),
                min_parallax_deg=min_parallax_deg,
                max_depth_ratio=max_depth_ratio,
                max_reproj_error_px=max_reproj_error_px,
                eps=eps,
            )

            # Read the updated seed
            seed_out = map_growth_out.seed
    else:
        seed_out, tracked_obs_stats, tracked_obs_report = append_tracked_observations_to_seed(
            seed,
            pose_out,
            current_kf=current_kf,
            K=K,
            track_out=track_out,
            max_append_reproj_error_px_existing=max_append_reproj_error_px_existing,
            prune_stale_map_growth=False,
            eps=eps,
            return_report=True,
        )

        n_pnp_inliers = int(pose_stats.get("n_pnp_inliers", 0))
        pnp_occupied_cells = int(pose_stats.get("pnp_inlier_occupied_cells", 0))
        bbox_area = pose_stats.get("pnp_inlier_bbox_area_fraction", None)
        pnp_bbox_area_fraction = 0.0 if bbox_area is None else float(bbox_area)
        support_strong_enough = n_pnp_inliers >= max(int(min_inliers) + 4, 20) and (
            pnp_occupied_cells >= 2 or pnp_bbox_area_fraction >= 0.05
        )
        max_cell_fraction = pose_stats.get("pnp_inlier_max_cell_fraction", None)
        largest_component_fraction = pose_stats.get("pnp_inlier_largest_component_fraction", None)
        spatially_concentrated = bool(
            max_cell_fraction is not None
            and largest_component_fraction is not None
            and float(max_cell_fraction) >= 0.90
            and float(largest_component_fraction) >= 0.90
        )
        history_stats = {
            "n_evaluated": 0,
            "n_inconsistent": 0,
            "inconsistent_fraction": 0.0,
        }
        if bool(support_strong_enough and spatially_concentrated):
            history_stats = _rescued_support_history_stats(seed, pose_out, K, int(current_kf))
        guarded_support_refresh_stats.update(
            {
                "history_evaluated": int(history_stats.get("n_evaluated", 0)),
                "history_inconsistent": int(history_stats.get("n_inconsistent", 0)),
                "history_inconsistent_fraction": float(history_stats.get("inconsistent_fraction", 0.0)),
            }
        )
        history_inconsistent = float(history_stats.get("inconsistent_fraction", 0.0)) >= 0.50

        if bool(incoherent_recovery_frame):
            guarded_support_refresh_stats["reason"] = "incoherent_recovery_localisation_only"
        elif not bool(support_strong_enough):
            guarded_support_refresh_stats["reason"] = "rescued_support_too_weak"
        elif bool(spatially_concentrated and history_inconsistent):
            guarded_support_refresh_stats["reason"] = "rescued_support_concentrated_history_inconsistent"
        else:
            seed_out, refresh_stats = _refresh_active_lookup_basis_from_rescued_support(
                seed_out,
                track_out,
                pose_out,
                int(current_kf),
            )
            guarded_support_refresh_stats.update(
                {
                    "triggered": True,
                    "reason": "rescued_support_refresh",
                    "n_lookup_mapped": int(refresh_stats.get("n_lookup_mapped", 0)),
                    "n_accepted_support": int(refresh_stats.get("n_accepted_support", 0)),
                    "n_conflicts": int(refresh_stats.get("n_conflicts", 0)),
                    "n_out_of_range": int(refresh_stats.get("n_out_of_range", 0)),
                }
            )

    # Keep one previous rescued basis for explicit incoherent recovery
    if bool(pose_stats.get("pnp_support_rescue_succeeded", False)):
        last_rescued_basis = seed.get("_last_rescued_support_basis", None)
        rescued_basis, _ = _support_basis_from_rescued_pose(
            track_out,
            pose_out,
            int(current_kf),
        )
        if rescued_basis is not None:
            seed_out = dict(seed_out)
            if isinstance(last_rescued_basis, dict):
                seed_out["_incoherent_recovery_support_basis"] = last_rescued_basis
            seed_out["_last_rescued_support_basis"] = rescued_basis

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
            min_landmark_observations=min_landmark_observations,
            allow_bootstrap_landmarks_for_pose=allow_bootstrap_landmarks_for_pose,
            min_post_bootstrap_observations_for_pose=min_post_bootstrap_observations_for_pose,
            min_translation_m=keyframe_min_translation_m,
            min_rotation_deg=keyframe_min_rotation_deg,
            require_pose=keyframe_require_pose,
        )

        # Read the updated seed after any promotion
        seed_out = keyframe_out.seed

    # Read map-growth stats
    map_stats = map_growth_out.stats if map_growth_out is not None else {}
    map_growth_report = map_growth_out.mutation_report if map_growth_out is not None else None
    mutation_reports = [report for report in (tracked_obs_report, map_growth_report) if isinstance(report, dict)]
    map_mutation_report = None
    if len(mutation_reports) > 0:
        map_mutation_report = merge_map_mutation_reports(*mutation_reports, context="frame_pipeline")

    # Read keyframe stats
    keyframe_stats = keyframe_out.stats if keyframe_out is not None else {}

    seed_out = dict(seed_out)
    accepted_R, accepted_t = _copy_pose_blocks(pose_out["R"], pose_out["t"])
    canonical_R, canonical_t = accepted_R.copy(), accepted_t.copy()
    canonical_pose_proxy, canonical_pose_proxy_stats = _select_canonical_pose_proxy(
        K,
        pose_out,
        pose_stats,
        sample_size=int(sample_size),
        min_inliers=int(min_inliers),
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
    if canonical_pose_proxy is not None:
        canonical_R, canonical_t = _copy_pose_blocks(canonical_pose_proxy[0], canonical_pose_proxy[1])
    seed_out["last_accepted_pose"] = {
        "kf": int(current_kf),
        "R": accepted_R,
        "t": accepted_t,
        "localisation_only": bool(localisation_only_rescue_frame),
    }
    if int(current_kf) >= 0:
        store_current_pose(seed_out, int(current_kf), canonical_R, canonical_t)
        if canonical_pose_proxy is not None:
            canonical_pose_proxy_stats["canonical_pose_proxy_storage_replaced"] = True

    if not bool(enable_local_ba):
        local_ba_stats = local_bundle_adjustment_not_run_stats("disabled")
    elif bool(localisation_only_rescue_frame):
        local_ba_stats = local_bundle_adjustment_not_run_stats("localisation_only_rescue_frame")
    elif not bool(keyframe_stats.get("promoted", False)):
        local_ba_stats = local_bundle_adjustment_not_run_stats("no_new_keyframe")
    else:
        local_ba_stats = run_local_bundle_adjustment(
            K,
            seed_out,
            max_keyframes=local_ba_max_keyframes,
            min_keyframes=local_ba_min_keyframes,
            min_landmarks=local_ba_min_landmarks,
            min_observations=local_ba_min_observations,
            max_iters=local_ba_max_iters,
            initial_damping=local_ba_initial_damping,
            max_damping=local_ba_max_damping,
            step_tol=local_ba_step_tol,
            improvement_tol=local_ba_improvement_tol,
            eps=eps,
        )

        if bool(local_ba_stats.get("succeeded", False)) and int(current_kf) in set(
            int(kf) for kf in local_ba_stats.get("optimised_keyframes", [])
        ):
            current_pose = get_pose_for_kf(seed_out, int(current_kf), context="local BA accepted current pose")
            accepted_R, accepted_t = _copy_pose_blocks(current_pose[0], current_pose[1])
            seed_out["last_accepted_pose"] = {
                "kf": int(current_kf),
                "R": accepted_R.copy(),
                "t": accepted_t.copy(),
                "localisation_only": False,
            }

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
        **canonical_pose_proxy_stats,
        **_local_ba_summary_stats(local_ba_stats),
        "guarded_support_refresh_triggered": bool(guarded_support_refresh_stats.get("triggered", False)),
        "guarded_support_refresh_reason": guarded_support_refresh_stats.get("reason", None),
        "guarded_support_refresh_n_lookup_mapped": int(guarded_support_refresh_stats.get("n_lookup_mapped", 0)),
        "guarded_support_refresh_n_accepted_support": int(guarded_support_refresh_stats.get("n_accepted_support", 0)),
        "guarded_support_refresh_n_conflicts": int(guarded_support_refresh_stats.get("n_conflicts", 0)),
        "guarded_support_refresh_n_out_of_range": int(guarded_support_refresh_stats.get("n_out_of_range", 0)),
        "guarded_support_refresh_history_evaluated": int(guarded_support_refresh_stats.get("history_evaluated", 0)),
        "guarded_support_refresh_history_inconsistent": int(guarded_support_refresh_stats.get("history_inconsistent", 0)),
        "guarded_support_refresh_history_inconsistent_fraction": float(
            guarded_support_refresh_stats.get("history_inconsistent_fraction", 0.0)
        ),
        "n_linked_landmarks_candidate": int(keyframe_stats.get("n_linked_landmarks_candidate", 0)),
        "n_pose_eligible_linked_landmarks_candidate": int(
            keyframe_stats.get("n_pose_eligible_linked_landmarks_candidate", 0)
        ),
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
        "tracked_observation_report": tracked_obs_report,
        "map_mutation_report": map_mutation_report,
        "keyframe_out": keyframe_out,
        "R": accepted_R.copy(),
        "t": accepted_t.copy(),
        "stats": stats,
    }
