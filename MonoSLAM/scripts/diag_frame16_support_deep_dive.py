# scripts/diag_frame16_support_deep_dive.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_runtime_cfg as _load_runtime_cfg, seed_landmark_count as _seed_landmark_count, standard_frame_stats as _standard_frame_stats

from datasets.eth3d import load_eth3d_sequence
from geometry.camera import reprojection_residuals, world_to_camera_points
from geometry.pnp import _pnp_inlier_mask_from_pose, _slice_pnp_correspondences, build_pnp_correspondences_with_stats, estimate_pose_pnp_ransac, pnp_local_displacement_consistency_mask
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.pnp_stats import pnp_support_diagnostic_stats, pnp_support_gate_stats
from slam.seed import seed_keyframe_pose


# Convert numpy values for JSON output
def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


# Count valid observation records
def _obs_count(lm: dict) -> int:
    obs = lm.get("obs", None)
    if not isinstance(obs, list):
        return 0
    return int(sum(1 for ob in obs if isinstance(ob, dict)))


# Build landmark lookup by id
def _landmarks_by_id(seed: dict) -> dict[int, dict]:
    return {
        int(lm["id"]): lm
        for lm in seed.get("landmarks", [])
        if isinstance(lm, dict) and "id" in lm
    }


# Count categorical values
def _value_counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = int(counts.get(key, 0) + 1)
    return {key: int(counts[key]) for key in sorted(counts)}


# Summarise a numeric vector
def _numeric_summary(values) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


# Summarise image-space spread
def _spatial_summary(xy: np.ndarray, image_shape: tuple[int, int]) -> dict:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    H = int(image_shape[0])
    W = int(image_shape[1])
    grid = [[0 for _ in range(4)] for _ in range(3)]
    if int(xy.shape[0]) == 0:
        return {
            "count": 0,
            "bbox": None,
            "bbox_area_fraction": None,
            "centroid": None,
            "occupied_cells": 0,
            "occupancy_grid": grid,
        }

    x = xy[:, 0]
    y = xy[:, 1]
    bbox = [float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))]
    bbox_area_fraction = float(max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0) / max(float(W * H), 1.0))
    for p in xy:
        col = int(np.clip(np.floor((float(p[0]) / max(float(W), 1.0)) * 4), 0, 3))
        row = int(np.clip(np.floor((float(p[1]) / max(float(H), 1.0)) * 3), 0, 2))
        grid[row][col] += 1

    return {
        "count": int(xy.shape[0]),
        "bbox": bbox,
        "bbox_area_fraction": bbox_area_fraction,
        "centroid": [float(np.mean(x)), float(np.mean(y))],
        "occupied_cells": int(sum(1 for row in grid for value in row if int(value) > 0)),
        "occupancy_grid": grid,
    }


# Copy a pose into plain arrays
def _copy_pose(pose: dict | None) -> dict | None:
    if not isinstance(pose, dict):
        return None
    if pose.get("R", None) is None or pose.get("t", None) is None:
        return None
    return {
        "kf": None if pose.get("kf", None) is None else int(pose.get("kf")),
        "R": np.asarray(pose["R"], dtype=np.float64).copy(),
        "t": np.asarray(pose["t"], dtype=np.float64).reshape(3).copy(),
        "localisation_only": bool(pose.get("localisation_only", False)),
    }


# Build the support funnel from one tracked frame
def _support_funnel(seed: dict, track_out: dict, pnp_cfg: dict, image_shape: tuple[int, int]) -> dict:
    landmark_id_by_feat1 = np.asarray(seed.get("landmark_id_by_feat1", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    lm_by_id = _landmarks_by_id(seed)
    kf_feat_idx = np.asarray(track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    xy_kf = np.asarray(track_out.get("xy_kf", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    xy_cur = np.asarray(track_out.get("xy_cur", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    n_tracks = int(kf_feat_idx.size)

    in_range = (kf_feat_idx >= 0) & (kf_feat_idx < int(landmark_id_by_feat1.size))
    mapped = np.zeros((n_tracks,), dtype=bool)
    valid_landmark = np.zeros((n_tracks,), dtype=bool)
    valid_xw = np.zeros((n_tracks,), dtype=bool)
    obs_gate = np.zeros((n_tracks,), dtype=bool)
    mapped_lm_ids = np.full((n_tracks,), -1, dtype=np.int64)

    for i in range(n_tracks):
        if not bool(in_range[i]):
            continue
        lm_id = int(landmark_id_by_feat1[int(kf_feat_idx[i])])
        if lm_id < 0:
            continue
        mapped[i] = True
        mapped_lm_ids[i] = int(lm_id)
        lm = lm_by_id.get(lm_id, None)
        if not isinstance(lm, dict):
            continue
        valid_landmark[i] = True
        X_w = np.asarray(lm.get("X_w", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        if X_w.size == 3 and np.isfinite(X_w).all():
            valid_xw[i] = True
        is_bootstrap = str(lm.get("birth_source", "")) == "bootstrap"
        if is_bootstrap:
            min_obs_required = int(pnp_cfg["min_landmark_observations"])
        else:
            min_obs_required = max(
                int(pnp_cfg["min_landmark_observations"]),
                int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
            )
        if not bool(pnp_cfg["allow_bootstrap_landmarks_for_pose"]) and is_bootstrap:
            obs_gate[i] = False
        else:
            obs_gate[i] = bool(valid_xw[i] and _obs_count(lm) >= int(min_obs_required))

    corrs, corr_stats = build_pnp_correspondences_with_stats(
        seed,
        track_out,
        min_landmark_observations=int(pnp_cfg["min_landmark_observations"]),
        allow_bootstrap_landmarks_for_pose=bool(pnp_cfg["allow_bootstrap_landmarks_for_pose"]),
        min_post_bootstrap_observations_for_pose=int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
        enable_local_consistency_filter=bool(pnp_cfg["enable_pnp_local_consistency_filter"]),
        local_consistency_radius_px=float(pnp_cfg["pnp_local_consistency_radius_px"]),
        local_consistency_min_neighbours=int(pnp_cfg["pnp_local_consistency_min_neighbours"]),
        local_consistency_max_median_residual_px=float(pnp_cfg["pnp_local_consistency_max_median_residual_px"]),
        local_consistency_min_keep=int(pnp_cfg["pnp_local_consistency_min_keep"]),
        enable_spatial_thinning_filter=bool(pnp_cfg["enable_pnp_spatial_thinning_filter"]),
        spatial_thinning_radius_px=float(pnp_cfg["pnp_spatial_thinning_radius_px"]),
        spatial_thinning_max_points_per_radius=int(pnp_cfg["pnp_spatial_thinning_max_points_per_radius"]),
        spatial_thinning_min_keep=int(pnp_cfg["pnp_spatial_thinning_min_keep"]),
    )

    return {
        "raw_tracked_pairs": int(n_tracks),
        "track_matches_reported": int(track_out.get("stats", {}).get("n_matches", 0)),
        "track_inliers_reported": int(track_out.get("stats", {}).get("n_inliers", 0)),
        "kf_feat_idx_in_range": int(np.sum(in_range)),
        "mapped_by_keyframe_lookup": int(np.sum(mapped)),
        "valid_landmarks": int(np.sum(valid_landmark)),
        "valid_X_w": int(np.sum(valid_xw)),
        "observation_gated_pass": int(np.sum(obs_gate)),
        "final_pnp_correspondences": int(corrs.X_w.shape[1]),
        "corr_stats": corr_stats,
        "spatial": {
            "mapped_current": _spatial_summary(xy_cur[mapped], image_shape),
            "eligible_current": _spatial_summary(xy_cur[obs_gate], image_shape),
            "mapped_keyframe": _spatial_summary(xy_kf[mapped], image_shape),
            "eligible_keyframe": _spatial_summary(xy_kf[obs_gate], image_shape),
        },
        "eligible_landmark_ids": [int(v) for v in np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1).tolist()],
    }


# Run one threshold on a fixed eligible set
def _run_threshold(corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], threshold_px: float) -> dict:
    n_corr = int(corrs.X_w.shape[1])
    out = {
        "threshold_px": float(threshold_px),
        "n_corr": int(n_corr),
        "ok": False,
        "reason": None,
        "n_inliers": 0,
        "n_model_success": 0,
        "spatial_gate_rejected": False,
        "spatial_gate_reason": None,
        "component_gate_rejected": False,
        "component_gate_reason": None,
        "occupied_cells": 0,
        "bbox_area_fraction": None,
        "max_cell_fraction": None,
        "component_count": 0,
        "largest_component_size": 0,
        "largest_component_fraction": None,
        "R": None,
        "t": None,
        "inlier_mask": np.zeros((n_corr,), dtype=bool),
    }
    if n_corr < int(pnp_cfg["sample_size"]):
        out["reason"] = "too_few_correspondences_for_ransac"
        return out

    num_trials = 5000 if float(threshold_px) >= 20.0 else int(pnp_cfg["num_trials"])
    try:
        R, t, inlier_mask, stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=int(num_trials),
            sample_size=int(pnp_cfg["sample_size"]),
            threshold_px=float(threshold_px),
            min_inliers=int(pnp_cfg["min_inliers"]),
            seed=int(pnp_cfg["ransac_seed"]),
            min_points=int(pnp_cfg["min_points"]),
            rank_tol=float(pnp_cfg["rank_tol"]),
            min_cheirality_ratio=float(pnp_cfg["min_cheirality_ratio"]),
            eps=float(pnp_cfg["eps"]),
            refit=bool(pnp_cfg["refit"]),
            refine_nonlinear=bool(pnp_cfg["refine_nonlinear"]),
            refine_max_iters=int(pnp_cfg["refine_max_iters"]),
            refine_damping=float(pnp_cfg["refine_damping"]),
            refine_step_tol=float(pnp_cfg["refine_step_tol"]),
            refine_improvement_tol=float(pnp_cfg["refine_improvement_tol"]),
        )
    except Exception as exc:
        out["reason"] = "pnp_ransac_error"
        out["error"] = str(exc)
        return out

    stats = stats if isinstance(stats, dict) else {}
    mask = np.zeros((n_corr,), dtype=bool) if inlier_mask is None else np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if int(mask.size) != n_corr:
        mask = np.zeros((n_corr,), dtype=bool)

    ok = bool(R is not None and t is not None)
    support_stats = pnp_support_diagnostic_stats(
        corrs,
        mask,
        image_shape,
        pnp_spatial_grid_cols=int(pnp_cfg["pnp_spatial_grid_cols"]),
        pnp_spatial_grid_rows=int(pnp_cfg["pnp_spatial_grid_rows"]),
        pnp_component_radius_px=float(pnp_cfg["pnp_component_radius_px"]),
    )
    gate_stats = pnp_support_gate_stats(
        ok,
        support_stats,
        enable_pnp_spatial_gate=bool(pnp_cfg["enable_pnp_spatial_gate"]),
        min_pnp_inlier_cells=int(pnp_cfg["min_pnp_inlier_cells"]),
        max_pnp_single_cell_fraction=float(pnp_cfg["max_pnp_single_cell_fraction"]),
        min_pnp_bbox_area_fraction=float(pnp_cfg["min_pnp_bbox_area_fraction"]),
        enable_pnp_component_gate=bool(pnp_cfg["enable_pnp_component_gate"]),
        min_pnp_component_count=int(pnp_cfg["min_pnp_component_count"]),
        max_pnp_largest_component_fraction=float(pnp_cfg["max_pnp_largest_component_fraction"]),
        min_pnp_largest_component_bbox_area_fraction=float(pnp_cfg["min_pnp_largest_component_bbox_area_fraction"]),
    )

    out.update(
        {
            "ok": bool(ok),
            "reason": stats.get("reason", None),
            "n_inliers": int(np.sum(mask)),
            "n_model_success": int(stats.get("n_model_success", 0)),
            "spatial_gate_rejected": bool(gate_stats.get("pnp_spatial_gate_rejected", False)),
            "spatial_gate_reason": gate_stats.get("pnp_spatial_gate_reason", None),
            "component_gate_rejected": bool(gate_stats.get("pnp_component_gate_rejected", False)),
            "component_gate_reason": gate_stats.get("pnp_component_gate_reason", None),
            "occupied_cells": int(support_stats.get("pnp_inlier_occupied_cells", 0)),
            "bbox_area_fraction": support_stats.get("pnp_inlier_bbox_area_fraction", None),
            "max_cell_fraction": support_stats.get("pnp_inlier_max_cell_fraction", None),
            "component_count": int(support_stats.get("pnp_inlier_component_count", 0)),
            "largest_component_size": int(support_stats.get("pnp_inlier_largest_component_size", 0)),
            "largest_component_fraction": support_stats.get("pnp_inlier_largest_component_fraction", None),
            "R": None if R is None else np.asarray(R, dtype=np.float64),
            "t": None if t is None else np.asarray(t, dtype=np.float64).reshape(3),
            "inlier_mask": mask,
        }
    )
    return out


# Strip pose arrays from threshold rows
def _public_threshold_row(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"R", "t", "inlier_mask"}
    }


# Score a reference pose on fixed correspondences
def _residual_summary(corrs, K: np.ndarray, R, t, *, eps: float) -> dict:
    n_corr = int(corrs.X_w.shape[1])
    out = {
        "available": False,
        "n_corr": int(n_corr),
        "n_positive_depth": 0,
        "n_non_positive_depth": 0,
        "n_finite_residual": 0,
        "within": {"8": 0, "12": 0, "20": 0, "40": 0},
        "median_px": None,
        "p90_px": None,
        "mean_signed_residual_px": None,
        "error_px": {},
    }
    if R is None or t is None or n_corr == 0:
        return out

    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    X_c = world_to_camera_points(R, t, corrs.X_w)
    z = np.asarray(X_c[2, :], dtype=np.float64).reshape(-1)
    residuals = np.asarray(reprojection_residuals(K, R, t, corrs.X_w, corrs.x_cur), dtype=np.float64)
    err = np.sqrt(np.sum(residuals * residuals, axis=0))
    finite = np.isfinite(err) & np.isfinite(residuals).all(axis=0)
    positive = np.isfinite(z) & (z > float(eps))
    usable = finite & positive
    err_use = err[usable]
    res_use = residuals[:, usable]

    within = {
        "8": int(np.sum(err_use <= 8.0)),
        "12": int(np.sum(err_use <= 12.0)),
        "20": int(np.sum(err_use <= 20.0)),
        "40": int(np.sum(err_use <= 40.0)),
    }
    out.update(
        {
            "available": True,
            "n_positive_depth": int(np.sum(positive)),
            "n_non_positive_depth": int(np.sum(np.isfinite(z) & ~positive)),
            "n_finite_residual": int(np.sum(finite)),
            "within": within,
            "median_px": None if int(err_use.size) == 0 else float(np.median(err_use)),
            "p90_px": None if int(err_use.size) == 0 else float(np.percentile(err_use, 90)),
            "mean_signed_residual_px": None if int(err_use.size) == 0 else [float(v) for v in np.mean(res_use, axis=1)],
            "error_px": _numeric_summary(err_use),
        }
    )
    return out


# Score one landmark group under a reference pose
def _group_residual_summary(corrs, K: np.ndarray, R, t, group_ids: set[int], *, eps: float) -> dict:
    ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    mask = np.asarray([int(v) in group_ids for v in ids], dtype=bool)
    if R is None or t is None or not np.any(mask):
        return {
            "count": int(np.sum(mask)),
            "within_40": 0,
            "median_px": None,
            "p90_px": None,
        }
    sub_err = _residual_errors(corrs, K, R, t, eps=eps)
    err = sub_err[mask]
    err = err[np.isfinite(err)]
    if int(err.size) == 0:
        return {
            "count": int(np.sum(mask)),
            "within_40": 0,
            "median_px": None,
            "p90_px": None,
        }
    return {
        "count": int(np.sum(mask)),
        "within_40": int(np.sum(err <= 40.0)),
        "median_px": float(np.median(err)),
        "p90_px": float(np.percentile(err, 90)),
    }


# Compute positive-depth residual errors
def _residual_errors(corrs, K: np.ndarray, R, t, *, eps: float) -> np.ndarray:
    n_corr = int(corrs.X_w.shape[1])
    out = np.full((n_corr,), np.inf, dtype=np.float64)
    if R is None or t is None or n_corr == 0:
        return out
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    X_c = world_to_camera_points(R, t, corrs.X_w)
    z = np.asarray(X_c[2, :], dtype=np.float64).reshape(-1)
    residuals = np.asarray(reprojection_residuals(K, R, t, corrs.X_w, corrs.x_cur), dtype=np.float64)
    err = np.sqrt(np.sum(residuals * residuals, axis=0))
    valid = np.isfinite(err) & np.isfinite(residuals).all(axis=0) & np.isfinite(z) & (z > float(eps))
    out[valid] = err[valid]
    return out


# Build landmark metadata summaries
def _landmark_set_summary(seed: dict, ids: set[int]) -> dict:
    lm_by_id = _landmarks_by_id(seed)
    records = []
    for lm_id in sorted(ids):
        lm = lm_by_id.get(int(lm_id), {})
        records.append(
            {
                "id": int(lm_id),
                "birth_source": str(lm.get("birth_source", "missing")),
                "birth_kf": int(lm.get("birth_kf", -1)) if isinstance(lm, dict) else -1,
                "obs_count": _obs_count(lm) if isinstance(lm, dict) else 0,
            }
        )
    return {
        "count": int(len(records)),
        "ids": [int(record["id"]) for record in records],
        "source_split": _value_counts([record["birth_source"] for record in records]),
        "obs_count_distribution": _value_counts([record["obs_count"] for record in records]),
        "birth_kf_distribution": _value_counts([record["birth_kf"] for record in records]),
        "records": records,
    }


# Build frame-15 to frame-16 landmark continuity
def _landmark_continuity(seed: dict, frame15: dict, frame16: dict, corrs16, K: np.ndarray, last_pose: dict | None, *, eps: float) -> dict:
    ids15 = set(int(v) for v in frame15["funnel"]["eligible_landmark_ids"])
    ids16 = set(int(v) for v in frame16["funnel"]["eligible_landmark_ids"])
    intersection = ids15 & ids16
    dropped = ids15 - ids16
    new = ids16 - ids15
    R = None if last_pose is None else last_pose.get("R", None)
    t = None if last_pose is None else last_pose.get("t", None)
    return {
        "frame15": _landmark_set_summary(seed, ids15),
        "frame16": _landmark_set_summary(seed, ids16),
        "intersection": _landmark_set_summary(seed, intersection),
        "dropped_from_frame15": _landmark_set_summary(seed, dropped),
        "new_in_frame16": _landmark_set_summary(seed, new),
        "frame16_residual_groups_under_last_accepted_pose": {
            "surviving_intersection": _group_residual_summary(corrs16, K, R, t, intersection, eps=eps),
            "new_in_frame16": _group_residual_summary(corrs16, K, R, t, new, eps=eps),
        },
    }


# Summarise a processed frame without changing production state
def _analyse_processed_frame(seed_before: dict, keyframe_index: int, out: dict, seq, K: np.ndarray, pnp_cfg: dict, frame_index: int) -> dict:
    cur_im, cur_ts, cur_id = seq.get(frame_index)
    image_shape = (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1]))
    track_out = out.get("track_out", {})
    pose_out = out.get("pose_out", {})
    corrs = pose_out.get("corrs", None) if isinstance(pose_out, dict) else None
    if corrs is None:
        corrs, _ = build_pnp_correspondences_with_stats(seed_before, track_out)

    last_pose = _copy_pose(seed_before.get("last_accepted_pose", None))
    thresholds = [_run_threshold(corrs, K, pnp_cfg, image_shape, threshold_px=v) for v in [8.0, 12.0, 20.0, 40.0]]
    stats = out.get("stats", {}) if isinstance(out, dict) else {}
    pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
    pose_stats = pose_stats if isinstance(pose_stats, dict) else {}
    standard = _standard_frame_stats(
        frame_index=frame_index,
        reference_keyframe_index=keyframe_index,
        frontend_out=out,
        seed_landmarks_before=_seed_landmark_count(seed_before),
        seed_landmarks_after=_seed_landmark_count(out.get("seed", {})),
    )

    return {
        "frame_index": int(frame_index),
        "frame_id": str(cur_id),
        "timestamp": float(cur_ts),
        "active_keyframe": int(keyframe_index),
        "seed_landmarks": int(len(seed_before.get("landmarks", []))),
        "last_accepted_pose": None if last_pose is None else {
            "kf": last_pose["kf"],
            "localisation_only": bool(last_pose["localisation_only"]),
        },
        "pipeline": {
            **standard,
            "support_quality_veto_attempted": bool(
                pose_stats.get("pnp_support_quality_veto_attempted", stats.get("pnp_support_quality_veto_attempted", False))
            ),
            "support_quality_veto_evaluated": bool(
                pose_stats.get("pnp_support_quality_veto_evaluated", stats.get("pnp_support_quality_veto_evaluated", False))
            ),
            "support_quality_veto_triggered": bool(
                pose_stats.get("pnp_support_quality_veto_triggered", stats.get("pnp_support_quality_veto_triggered", False))
            ),
            "support_quality_veto_reason": pose_stats.get(
                "pnp_support_quality_veto_reason",
                stats.get("pnp_support_quality_veto_reason", None),
            ),
            "support_quality_veto_candidate_count": int(
                pose_stats.get("pnp_support_quality_veto_candidate_count", stats.get("pnp_support_quality_veto_candidate_count", 0))
            ),
            "support_quality_veto_local_consistency_retained": int(
                pose_stats.get(
                    "pnp_support_quality_veto_local_consistency_retained",
                    stats.get("pnp_support_quality_veto_local_consistency_retained", 0),
                )
            ),
            "support_quality_veto_local_consistency_retention": pose_stats.get(
                "pnp_support_quality_veto_local_consistency_retention",
                stats.get("pnp_support_quality_veto_local_consistency_retention", None),
            ),
            "support_quality_veto_signal": pose_stats.get(
                "pnp_support_quality_veto_signal",
                stats.get("pnp_support_quality_veto_signal", None),
            ),
            "rescue_reason": pose_stats.get("pnp_support_rescue_reason", stats.get("pnp_support_rescue_reason", None)),
            "rescue_trigger_reason": pose_stats.get("pnp_support_rescue_trigger_reason", None),
            "rescue_loose_threshold_px": pose_stats.get("pnp_support_rescue_loose_threshold_px", stats.get("pnp_support_rescue_loose_threshold_px", None)),
            "rescue_loose_pose_ok": bool(pose_stats.get("pnp_support_rescue_loose_pose_ok", False)),
            "rescue_loose_inliers": int(pose_stats.get("pnp_support_rescue_loose_inliers", stats.get("pnp_support_rescue_loose_inliers", 0))),
            "second_stage_attempted": bool(pose_stats.get("pnp_support_rescue_second_stage_attempted", False)),
            "second_stage_succeeded": bool(pose_stats.get("pnp_support_rescue_second_stage_succeeded", False)),
            "seeded_20px_fallback_attempted": bool(pose_stats.get("pnp_support_rescue_second_stage_seeded_20px_fallback_attempted", False)),
            "seeded_20px_fallback_triggered": bool(pose_stats.get("pnp_support_rescue_second_stage_seeded_20px_fallback_triggered", False)),
            "seeded_20px_fallback_inliers": int(pose_stats.get("pnp_support_rescue_second_stage_seeded_20px_fallback_inliers", 0)),
            "seeded_20px_fallback_reason": pose_stats.get("pnp_support_rescue_second_stage_seeded_20px_fallback_reason", None),
            "loose_localisation_fallback_succeeded": bool(
                pose_stats.get(
                    "pnp_support_rescue_loose_localisation_fallback_succeeded",
                    stats.get("pnp_support_rescue_loose_localisation_fallback_succeeded", False),
                )
            ),
            "loose_localisation_fallback_reason": pose_stats.get(
                "pnp_support_rescue_loose_localisation_fallback_reason",
                stats.get("pnp_support_rescue_loose_localisation_fallback_reason", None),
            ),
        },
        "funnel": _support_funnel(seed_before, track_out, pnp_cfg, image_shape),
        "thresholds_on_fixed_eligible_set": [_public_threshold_row(row) for row in thresholds],
        "_corrs": corrs,
        "_thresholds": thresholds,
    }


# Build recent reference-pose residual summaries for frame 16
def _reference_pose_residuals(seed_before: dict, accepted_history: list[dict], corrs, K: np.ndarray, *, eps: float) -> list[dict]:
    refs: list[dict] = []
    last_pose = _copy_pose(seed_before.get("last_accepted_pose", None))
    if last_pose is not None:
        refs.append({"label": f"last_accepted_kf_{last_pose['kf']}", **last_pose})

    R_kf, t_kf = seed_keyframe_pose(seed_before)
    refs.append(
        {
            "label": f"active_keyframe_kf_{int(seed_before.get('keyframe_kf', -1))}",
            "kf": int(seed_before.get("keyframe_kf", -1)),
            "R": np.asarray(R_kf, dtype=np.float64),
            "t": np.asarray(t_kf, dtype=np.float64).reshape(3),
            "localisation_only": False,
        }
    )

    last_kf = None if last_pose is None else last_pose.get("kf", None)
    for pose in reversed(accepted_history):
        if last_kf is not None and int(pose.get("kf", -1)) == int(last_kf):
            continue
        refs.append({"label": f"previous_accepted_kf_{pose['kf']}", **pose})
        break

    out = []
    seen_labels: set[str] = set()
    for ref in refs:
        label = str(ref["label"])
        if label in seen_labels:
            continue
        seen_labels.add(label)
        summary = _residual_summary(corrs, K, ref.get("R", None), ref.get("t", None), eps=eps)
        out.append(
            {
                "label": label,
                "kf": None if ref.get("kf", None) is None else int(ref.get("kf")),
                "localisation_only": bool(ref.get("localisation_only", False)),
                **summary,
            }
        )
    return out


# Infer whether loose support exists
def _loose_pose_summary(thresholds: list[dict]) -> dict:
    rows = [row for row in thresholds if float(row["threshold_px"]) in (20.0, 40.0)]
    coherent = [
        row
        for row in rows
        if bool(row.get("ok", False))
        and not bool(row.get("spatial_gate_rejected", False))
        and not bool(row.get("component_gate_rejected", False))
    ]
    return {
        "coherent_loose_pose_exists": bool(len(coherent) > 0),
        "coherent_thresholds_px": [float(row["threshold_px"]) for row in coherent],
    }


# Summarise compact threshold outcomes
def _compact_thresholds(rows: list[dict]) -> dict:
    out = {}
    for row in rows:
        threshold_key = f"{float(row['threshold_px']):.0f}px"
        out[threshold_key] = {
            "ok": bool(row.get("ok", False)),
            "reason": row.get("reason", None),
            "n_inliers": int(row.get("n_inliers", 0)),
            "n_model_success": int(row.get("n_model_success", 0)),
            "spatial_gate_rejected": bool(row.get("spatial_gate_rejected", False)),
            "spatial_gate_reason": row.get("spatial_gate_reason", None),
        }
    return out


# Run one diagnostic support-selection experiment
def _support_filter_experiment(corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], *, name: str, mask: np.ndarray, residuals_px: np.ndarray | None = None, extra: dict | None = None) -> dict:
    N = int(corrs.X_w.shape[1])
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if int(mask.size) != N:
        raise ValueError(f"Filter mask must have size {N}; got {mask.size}")

    corrs_filtered = _slice_pnp_correspondences(corrs, mask)
    rows = [_run_threshold(corrs_filtered, K, pnp_cfg, image_shape, threshold_px=v) for v in [8.0, 12.0, 20.0, 40.0]]
    retained_residuals = None
    if residuals_px is not None:
        residuals_px = np.asarray(residuals_px, dtype=np.float64).reshape(-1)
        retained_residuals = _numeric_summary(residuals_px[mask])

    return {
        "name": str(name),
        "retained_count": int(np.sum(mask)),
        "dropped_count": int(N - int(np.sum(mask))),
        "retained_landmark_ids": [int(v) for v in np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)[mask].tolist()],
        "retained_residual_px": retained_residuals,
        "extra": {} if extra is None else extra,
        "thresholds": [_public_threshold_row(row) for row in rows],
        "threshold_summary": _compact_thresholds(rows),
        "loose_pose_summary": _loose_pose_summary(rows),
    }


# Build diagnostic support-selection experiments for frame 16
def _frame16_support_filter_experiments(seed_before: dict, corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int]) -> dict:
    N = int(corrs.X_w.shape[1])
    last_pose = _copy_pose(seed_before.get("last_accepted_pose", None))
    if last_pose is None:
        raise RuntimeError("Frame 16 support filtering requires a last accepted pose")

    residuals_px = _residual_errors(
        corrs,
        K,
        last_pose["R"],
        last_pose["t"],
        eps=float(pnp_cfg["eps"]),
    )
    finite_order = np.argsort(residuals_px, kind="stable")
    experiments = []

    for cap_px in [40.0, 30.0, 25.0]:
        mask = np.isfinite(residuals_px) & (residuals_px <= float(cap_px))
        experiments.append(
            _support_filter_experiment(
                corrs,
                K,
                pnp_cfg,
                image_shape,
                name=f"last_pose_residual_le_{cap_px:.0f}px",
                mask=mask,
                residuals_px=residuals_px,
                extra={"cap_px": float(cap_px), "reference_pose_kf": int(last_pose["kf"])},
            )
        )

    for k in [10, 12, 14, 16]:
        keep_count = min(int(k), N)
        mask = np.zeros((N,), dtype=bool)
        mask[finite_order[:keep_count]] = True
        experiments.append(
            _support_filter_experiment(
                corrs,
                K,
                pnp_cfg,
                image_shape,
                name=f"last_pose_residual_top_{int(k)}",
                mask=mask,
                residuals_px=residuals_px,
                extra={"requested_k": int(k), "reference_pose_kf": int(last_pose["kf"])},
            )
        )

    kf_feats = seed_before.get("feats1", None)
    if kf_feats is not None and hasattr(kf_feats, "kps_xy"):
        kps_xy = np.asarray(kf_feats.kps_xy, dtype=np.float64)
        kf_feat_idx = np.asarray(corrs.kf_feat_idx, dtype=np.int64).reshape(-1)
        valid = (kf_feat_idx >= 0) & (kf_feat_idx < int(kps_xy.shape[0]))
        if bool(np.all(valid)):
            xy_kf = np.asarray(kps_xy[kf_feat_idx, :2], dtype=np.float64)
            xy_cur = np.asarray(corrs.x_cur, dtype=np.float64).T
            local_mask, local_stats = pnp_local_displacement_consistency_mask(
                xy_kf,
                xy_cur,
                radius_px=float(pnp_cfg["pnp_local_consistency_radius_px"]),
                min_neighbours=int(pnp_cfg["pnp_local_consistency_min_neighbours"]),
                max_median_residual_px=float(pnp_cfg["pnp_local_consistency_max_median_residual_px"]),
                min_keep=int(pnp_cfg["pnp_local_consistency_min_keep"]),
            )
            experiments.append(
                _support_filter_experiment(
                    corrs,
                    K,
                    pnp_cfg,
                    image_shape,
                    name="existing_local_displacement_consistency",
                    mask=local_mask,
                    residuals_px=residuals_px,
                    extra={"local_consistency_stats": local_stats},
                )
            )
        else:
            experiments.append(
                {
                    "name": "existing_local_displacement_consistency",
                    "available": False,
                    "reason": "invalid_keyframe_feature_indices",
                }
            )
    else:
        experiments.append(
            {
                "name": "existing_local_displacement_consistency",
                "available": False,
                "reason": "missing_keyframe_features",
            }
        )

    any_loose_pose = any(bool(row.get("loose_pose_summary", {}).get("coherent_loose_pose_exists", False)) for row in experiments if isinstance(row, dict))
    return {
        "reference_pose": {
            "label": f"last_accepted_kf_{last_pose['kf']}",
            "kf": int(last_pose["kf"]),
            "localisation_only": bool(last_pose["localisation_only"]),
        },
        "residual_px": _numeric_summary(residuals_px),
        "residual_order_landmark_ids": [int(v) for v in np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)[finite_order].tolist()],
        "experiments": experiments,
        "any_coherent_loose_pose": bool(any_loose_pose),
    }


# Run the frame-16 deep dive
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--target_frame", type=int, default=16)
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    target_frame = int(args.target_frame)
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)
    pnp_cfg = frontend_kwargs["pnp_frontend_kwargs"]
    dataset_cfg = cfg["dataset"]
    dataset_root = (ROOT / dataset_cfg["root"]).resolve()
    seq = load_eth3d_sequence(
        dataset_root,
        str(dataset_cfg["seq"]),
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    max_frames = dataset_cfg.get("max_frames", None)
    requested_len = int(target_frame) + 1
    if max_frames is None:
        n_effective = len(seq)
    else:
        n_effective = min(len(seq), max(int(max_frames), int(requested_len)))
    if target_frame >= n_effective:
        raise IndexError(f"target_frame={target_frame} outside effective sequence length {n_effective}")

    im0, _, _ = seq.get(0)
    im1, _, _ = seq.get(1)
    boot = bootstrap_from_two_frames(
        K,
        K,
        im0,
        im1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )
    if not bool(boot.get("ok", False)):
        raise RuntimeError(f"Bootstrap failed: {boot.get('stats', {}).get('reason', None)}")

    seed = boot["seed"]
    keyframe_feats = seed["feats1"]
    keyframe_index = 1
    accepted_history: list[dict] = []
    analysed: dict[int, dict] = {}
    frame16_seed_before = None
    frame16_accepted_history: list[dict] = []

    for frame_index in range(2, int(target_frame) + 1):
        cur_im, _, _ = seq.get(frame_index)
        seed_before = seed
        keyframe_feats_before = keyframe_feats
        keyframe_index_before = int(keyframe_index)
        accepted_history_before = [dict(pose) for pose in accepted_history]

        out = process_frame_against_seed(
            K,
            seed,
            keyframe_feats,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            keyframe_kf=keyframe_index,
            current_kf=frame_index,
            **pnp_cfg,
        )

        if frame_index in (15, 16):
            analysed[frame_index] = _analyse_processed_frame(
                seed_before,
                keyframe_index_before,
                out,
                seq,
                K,
                pnp_cfg,
                frame_index,
            )
        if frame_index == 16:
            frame16_seed_before = seed_before
            frame16_accepted_history = accepted_history_before

        seed = out["seed"]
        if bool(out.get("ok", False)):
            pose = _copy_pose(seed.get("last_accepted_pose", None))
            if pose is not None:
                accepted_history.append(pose)
        if bool(out.get("stats", {}).get("keyframe_promoted", False)):
            keyframe_feats = out["track_out"]["cur_feats"]
            keyframe_index = int(frame_index)

    if 15 not in analysed or 16 not in analysed or frame16_seed_before is None:
        raise RuntimeError("Expected frame 15 and frame 16 analyses to be available")

    corrs16 = analysed[16]["_corrs"]
    thresholds16 = analysed[16]["_thresholds"]
    cur_im16, _, _ = seq.get(16)
    image_shape16 = (int(np.asarray(cur_im16).shape[0]), int(np.asarray(cur_im16).shape[1]))
    residuals16 = _reference_pose_residuals(
        frame16_seed_before,
        frame16_accepted_history,
        corrs16,
        K,
        eps=float(pnp_cfg["eps"]),
    )
    last_pose16 = _copy_pose(frame16_seed_before.get("last_accepted_pose", None))
    continuity = _landmark_continuity(
        frame16_seed_before,
        analysed[15],
        analysed[16],
        corrs16,
        K,
        last_pose16,
        eps=float(pnp_cfg["eps"]),
    )
    support_filter_experiments = _frame16_support_filter_experiments(
        frame16_seed_before,
        corrs16,
        K,
        pnp_cfg,
        image_shape16,
    )

    for frame in analysed.values():
        frame.pop("_corrs", None)
        frame.pop("_thresholds", None)

    result = {
        "event": "frame16_support_deep_dive",
        "profile": str(profile_path),
        "dataset_root": str(dataset_root),
        "sequence": str(dataset_cfg["seq"]),
        "pnp_config": {
            "sample_size": int(pnp_cfg["sample_size"]),
            "min_inliers": int(pnp_cfg["min_inliers"]),
            "threshold_px": float(pnp_cfg["threshold_px"]),
            "num_trials": int(pnp_cfg["num_trials"]),
        },
        "frames": {
            "15": analysed[15],
            "16": analysed[16],
        },
        "frame16_fixed_set": {
            "n_corr": int(corrs16.X_w.shape[1]),
            "loose_pose_summary": _loose_pose_summary(thresholds16),
            "thresholds": [_public_threshold_row(row) for row in thresholds16],
        },
        "frame16_reference_pose_residuals": residuals16,
        "frame15_frame16_landmark_continuity": continuity,
        "frame16_support_filter_experiments": support_filter_experiments,
    }

    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
