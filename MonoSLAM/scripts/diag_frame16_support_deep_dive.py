# scripts/diag_frame16_support_deep_dive.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_runtime_cfg as _load_runtime_cfg, seed_landmark_count as _seed_landmark_count, standard_frame_stats as _standard_frame_stats

from datasets.eth3d import load_eth3d_sequence
from geometry.camera import reprojection_residuals, world_to_camera_points
from geometry.pnp import PnPCorrespondences, _pnp_inlier_mask_from_pose, _slice_pnp_correspondences, build_pnp_correspondences_with_stats, estimate_pose_pnp_ransac, pnp_local_displacement_consistency_mask
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.pnp_stats import pnp_support_diagnostic_stats, pnp_support_gate_stats
from slam.seed import seed_keyframe_pose
from slam.tracking import track_against_keyframe


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


# Collect landmark ids for one birth source
def _landmark_ids_by_source(seed: dict, birth_source: str) -> set[int]:
    return {
        int(lm["id"])
        for lm in seed.get("landmarks", [])
        if isinstance(lm, dict)
        and "id" in lm
        and str(lm.get("birth_source", "")) == str(birth_source)
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


# Summarise tracked 2D motion without using landmarks
def _tracking_2d_summary(track_out: dict, image_shape: tuple[int, int], pnp_cfg: dict, mask: np.ndarray | None = None) -> dict:
    xy_kf = np.asarray(track_out.get("xy_kf", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    xy_cur = np.asarray(track_out.get("xy_cur", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    n_raw = min(int(xy_kf.shape[0]) if xy_kf.ndim == 2 else 0, int(xy_cur.shape[0]) if xy_cur.ndim == 2 else 0)
    xy_kf = xy_kf[:n_raw, :2]
    xy_cur = xy_cur[:n_raw, :2]

    if mask is None:
        keep = np.ones((n_raw,), dtype=bool)
    else:
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        if int(keep.size) != n_raw:
            keep_aligned = np.zeros((n_raw,), dtype=bool)
            n_copy = min(int(keep.size), n_raw)
            keep_aligned[:n_copy] = keep[:n_copy]
            keep = keep_aligned

    xy_kf_keep = xy_kf[keep]
    xy_cur_keep = xy_cur[keep]
    n_keep = int(xy_kf_keep.shape[0])

    out = {
        "count": int(n_keep),
        "keyframe_spread": _spatial_summary(xy_kf_keep, image_shape),
        "current_spread": _spatial_summary(xy_cur_keep, image_shape),
        "displacement_px": _numeric_summary(np.linalg.norm(xy_cur_keep - xy_kf_keep, axis=1) if n_keep > 0 else []),
        "displacement_xy_mean": None,
        "displacement_xy_median": None,
        "local_displacement_consistency": None,
    }
    if n_keep > 0:
        displacement = xy_cur_keep - xy_kf_keep
        out["displacement_xy_mean"] = [float(v) for v in np.mean(displacement, axis=0)]
        out["displacement_xy_median"] = [float(v) for v in np.median(displacement, axis=0)]

    if n_keep > 0:
        local_mask, local_stats = pnp_local_displacement_consistency_mask(
            xy_kf_keep,
            xy_cur_keep,
            radius_px=float(pnp_cfg["pnp_local_consistency_radius_px"]),
            min_neighbours=int(pnp_cfg["pnp_local_consistency_min_neighbours"]),
            max_median_residual_px=float(pnp_cfg["pnp_local_consistency_max_median_residual_px"]),
            min_keep=int(pnp_cfg["pnp_local_consistency_min_keep"]),
        )
        local_stats = dict(local_stats)
        local_stats["retention"] = float(np.sum(local_mask) / max(n_keep, 1))
        out["local_displacement_consistency"] = local_stats

    return out


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


# Snapshot mutable seed fields for diagnostics
def _diagnostic_seed_snapshot(seed: dict) -> dict:
    out = dict(seed)

    landmarks = []
    for lm in seed.get("landmarks", []):
        if not isinstance(lm, dict):
            landmarks.append(lm)
            continue

        lm_copy = dict(lm)
        if "X_w" in lm_copy:
            lm_copy["X_w"] = np.asarray(lm_copy["X_w"], dtype=np.float64).copy()
        if isinstance(lm_copy.get("obs", None), list):
            obs_copy = []
            for ob in lm_copy["obs"]:
                if not isinstance(ob, dict):
                    obs_copy.append(ob)
                    continue
                ob_copy = dict(ob)
                if "xy" in ob_copy:
                    ob_copy["xy"] = np.asarray(ob_copy["xy"], dtype=np.float64).copy()
                obs_copy.append(ob_copy)
            lm_copy["obs"] = obs_copy
        if isinstance(lm_copy.get("quality", None), dict):
            lm_copy["quality"] = dict(lm_copy["quality"])
        landmarks.append(lm_copy)

    out["landmarks"] = landmarks
    if "landmark_id_by_feat1" in seed:
        out["landmark_id_by_feat1"] = np.asarray(seed["landmark_id_by_feat1"], dtype=np.int64).copy()

    pose = _copy_pose(seed.get("last_accepted_pose", None))
    if pose is not None:
        out["last_accepted_pose"] = pose

    return out


# Copy the active lookup basis into a reusable anchor
def _copy_basis_snapshot(
    seed: dict,
    keyframe_index: int,
    *,
    label: str,
    kind: str,
    source_frame: int | None,
    localisation_only: bool,
) -> dict | None:
    if not isinstance(seed, dict):
        return None
    feats = seed.get("feats1", None)
    if feats is None or not hasattr(feats, "kps_xy"):
        return None
    lookup_raw = seed.get("landmark_id_by_feat1", None)
    if lookup_raw is None:
        return None
    try:
        R, t = seed_keyframe_pose(seed)
    except Exception:
        return None

    lookup = np.asarray(lookup_raw, dtype=np.int64).reshape(-1).copy()
    kps_xy = np.asarray(feats.kps_xy)
    n_feat = int(kps_xy.shape[0]) if kps_xy.ndim >= 1 else 0
    return {
        "label": str(label),
        "kind": str(kind),
        "source_frame": None if source_frame is None else int(source_frame),
        "kf": int(seed.get("keyframe_kf", keyframe_index)),
        "R": np.asarray(R, dtype=np.float64).copy(),
        "t": np.asarray(t, dtype=np.float64).reshape(3).copy(),
        "feats": feats,
        "landmark_id_by_feat1": lookup,
        "localisation_only": bool(localisation_only),
        "n_feat": int(n_feat),
        "n_lookup_mapped": int(np.sum(lookup >= 0)),
    }


# Rebuild a rescued frame support basis for diagnostics
def _reconstruct_rescued_support_basis(track_out: dict, pose_out: dict, current_kf: int) -> dict | None:
    if not isinstance(track_out, dict) or not isinstance(pose_out, dict):
        return None
    cur_feats = track_out.get("cur_feats", None)
    if cur_feats is None or not hasattr(cur_feats, "kps_xy"):
        return None
    corrs = pose_out.get("corrs", None)
    if corrs is None:
        return None
    if pose_out.get("R", None) is None or pose_out.get("t", None) is None:
        return None

    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(corrs.cur_feat_idx, dtype=np.int64).reshape(-1)
    support_mask = np.asarray(pose_out.get("pnp_inlier_mask", np.zeros((landmark_ids.size,), dtype=bool)), dtype=bool).reshape(-1)
    n_feat = int(np.asarray(cur_feats.kps_xy).shape[0])
    n_corr = min(int(landmark_ids.size), int(cur_feat_idx.size), int(support_mask.size))
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

    reconstruction = {
        "n_feat": int(n_feat),
        "n_corr": int(n_corr),
        "n_accepted_support": int(np.sum(support_mask[:n_corr])),
        "n_lookup_mapped": int(n_mapped),
        "n_conflicts": int(n_conflicts),
        "n_out_of_range": int(n_out_of_range),
        "clean": bool(n_mapped > 0 and n_conflicts == 0 and n_out_of_range == 0),
    }
    if not bool(reconstruction["clean"]):
        return {
            "label": f"rescued_support_basis_kf_{int(current_kf)}",
            "kind": "rescued_support_basis",
            "source_frame": int(current_kf),
            "kf": int(current_kf),
            "available": False,
            "reconstruction": reconstruction,
        }

    return {
        "label": f"rescued_support_basis_kf_{int(current_kf)}",
        "kind": "rescued_support_basis",
        "source_frame": int(current_kf),
        "kf": int(current_kf),
        "R": np.asarray(pose_out["R"], dtype=np.float64).copy(),
        "t": np.asarray(pose_out["t"], dtype=np.float64).reshape(3).copy(),
        "feats": cur_feats,
        "landmark_id_by_feat1": lookup,
        "localisation_only": True,
        "n_feat": int(n_feat),
        "n_lookup_mapped": int(n_mapped),
        "available": True,
        "reconstruction": reconstruction,
    }


# Build a seed with a diagnostic active basis
def _seed_with_basis(seed: dict, basis: dict) -> dict:
    out = dict(seed)
    out["feats1"] = basis["feats"]
    out["keyframe_kf"] = int(basis["kf"])
    out["T_WC1"] = (
        np.asarray(basis["R"], dtype=np.float64).copy(),
        np.asarray(basis["t"], dtype=np.float64).reshape(3).copy(),
    )
    out["landmark_id_by_feat1"] = np.asarray(basis["landmark_id_by_feat1"], dtype=np.int64).reshape(-1).copy()
    return out


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
        "track_2d": {
            "raw": _tracking_2d_summary(track_out, image_shape, pnp_cfg),
            "mapped": _tracking_2d_summary(track_out, image_shape, pnp_cfg, mapped),
            "eligible": _tracking_2d_summary(track_out, image_shape, pnp_cfg, obs_gate),
        },
        "mapped_landmark_ids": [int(v) for v in mapped_lm_ids[mapped].tolist()],
        "eligible_landmark_ids": [int(v) for v in np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1).tolist()],
    }


# Build mapped correspondences before observation eligibility gates
def _build_mapped_correspondences(seed: dict, track_out: dict) -> PnPCorrespondences:
    landmark_id_by_feat1 = np.asarray(seed.get("landmark_id_by_feat1", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    lm_by_id = _landmarks_by_id(seed)
    kf_feat_idx = np.asarray(track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(track_out.get("cur_feat_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    xy_cur = np.asarray(track_out.get("xy_cur", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    xy_kf = np.asarray(track_out.get("xy_kf", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)

    n_tracks = min(
        int(kf_feat_idx.size),
        int(cur_feat_idx.size),
        int(xy_cur.shape[0]) if xy_cur.ndim == 2 else 0,
        int(xy_kf.shape[0]) if xy_kf.ndim == 2 else 0,
    )

    X_cols: list[np.ndarray] = []
    x_cols: list[np.ndarray] = []
    landmark_ids: list[int] = []
    cur_idx_keep: list[int] = []
    kf_idx_keep: list[int] = []

    for i in range(n_tracks):
        feat1 = int(kf_feat_idx[i])
        if feat1 < 0 or feat1 >= int(landmark_id_by_feat1.size):
            continue

        lm_id = int(landmark_id_by_feat1[feat1])
        if lm_id < 0:
            continue

        lm = lm_by_id.get(lm_id, None)
        if not isinstance(lm, dict):
            continue

        X_w = np.asarray(lm.get("X_w", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue

        x = np.asarray(xy_cur[i, :2], dtype=np.float64).reshape(-1)
        if x.size != 2 or not np.isfinite(x).all():
            continue

        X_cols.append(X_w.reshape(3, 1))
        x_cols.append(x.reshape(2, 1))
        landmark_ids.append(int(lm_id))
        cur_idx_keep.append(int(cur_feat_idx[i]))
        kf_idx_keep.append(int(feat1))

    if len(X_cols) == 0:
        return PnPCorrespondences(
            X_w=np.zeros((3, 0), dtype=np.float64),
            x_cur=np.zeros((2, 0), dtype=np.float64),
            landmark_ids=np.zeros((0,), dtype=np.int64),
            cur_feat_idx=np.zeros((0,), dtype=np.int64),
            kf_feat_idx=np.zeros((0,), dtype=np.int64),
        )

    return PnPCorrespondences(
        X_w=np.hstack(X_cols),
        x_cur=np.hstack(x_cols),
        landmark_ids=np.asarray(landmark_ids, dtype=np.int64),
        cur_feat_idx=np.asarray(cur_idx_keep, dtype=np.int64),
        kf_feat_idx=np.asarray(kf_idx_keep, dtype=np.int64),
    )


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


# Score one fixed reference pose at one threshold
def _score_fixed_pose_threshold(corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], R, t, threshold_px: float) -> dict:
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
    }
    if R is None or t is None:
        out["reason"] = "pose_unavailable"
        return out
    if n_corr == 0:
        out["reason"] = "no_correspondences"
        return out

    mask, _ = _pnp_inlier_mask_from_pose(
        corrs.X_w,
        corrs.x_cur,
        K,
        R,
        t,
        threshold_px=float(threshold_px),
        eps=float(pnp_cfg["eps"]),
    )
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if int(mask.size) != n_corr:
        mask = np.zeros((n_corr,), dtype=bool)

    n_inliers = int(np.sum(mask))
    pose_support_ok = bool(n_inliers >= int(pnp_cfg["min_inliers"]))
    support_stats = pnp_support_diagnostic_stats(
        corrs,
        mask,
        image_shape,
        pnp_spatial_grid_cols=int(pnp_cfg["pnp_spatial_grid_cols"]),
        pnp_spatial_grid_rows=int(pnp_cfg["pnp_spatial_grid_rows"]),
        pnp_component_radius_px=float(pnp_cfg["pnp_component_radius_px"]),
    )
    gate_stats = pnp_support_gate_stats(
        pose_support_ok,
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

    spatial_rejected = bool(gate_stats.get("pnp_spatial_gate_rejected", False))
    component_rejected = bool(gate_stats.get("pnp_component_gate_rejected", False))
    ok = bool(pose_support_ok and not spatial_rejected and not component_rejected)
    reason = None
    if not bool(pose_support_ok):
        reason = "too_few_pose_inliers"
    elif bool(component_rejected):
        reason = "component_support_failed"
    elif bool(spatial_rejected):
        reason = "spatial_coverage_failed"

    out.update(
        {
            "ok": bool(ok),
            "reason": reason,
            "n_inliers": int(n_inliers),
            "spatial_gate_rejected": spatial_rejected,
            "spatial_gate_reason": gate_stats.get("pnp_spatial_gate_reason", None),
            "component_gate_rejected": component_rejected,
            "component_gate_reason": gate_stats.get("pnp_component_gate_reason", None),
            "occupied_cells": int(support_stats.get("pnp_inlier_occupied_cells", 0)),
            "bbox_area_fraction": support_stats.get("pnp_inlier_bbox_area_fraction", None),
            "max_cell_fraction": support_stats.get("pnp_inlier_max_cell_fraction", None),
            "component_count": int(support_stats.get("pnp_inlier_component_count", 0)),
            "largest_component_size": int(support_stats.get("pnp_inlier_largest_component_size", 0)),
            "largest_component_fraction": support_stats.get("pnp_inlier_largest_component_fraction", None),
        }
    )
    return out


# Score one fixed reference pose across diagnostic thresholds
def _score_fixed_pose_thresholds(corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], R, t) -> list[dict]:
    return [_score_fixed_pose_threshold(corrs, K, pnp_cfg, image_shape, R, t, threshold_px=v) for v in [8.0, 12.0, 20.0, 40.0]]


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
        "within": {"8": 0, "12": 0, "20": 0, "30": 0, "40": 0, "60": 0},
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
        "30": int(np.sum(err_use <= 30.0)),
        "40": int(np.sum(err_use <= 40.0)),
        "60": int(np.sum(err_use <= 60.0)),
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


# Build metadata records aligned to one correspondence bundle
def _correspondence_metadata_records(seed: dict, corrs, *, frame_index: int) -> list[dict]:
    lm_by_id = _landmarks_by_id(seed)
    records = []
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    kf_feat_idx = np.asarray(corrs.kf_feat_idx, dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(corrs.cur_feat_idx, dtype=np.int64).reshape(-1)

    for i, lm_id_raw in enumerate(landmark_ids.tolist()):
        lm_id = int(lm_id_raw)
        lm = lm_by_id.get(lm_id, None)
        if not isinstance(lm, dict):
            records.append(
                {
                    "corr_index": int(i),
                    "id": int(lm_id),
                    "birth_source": "missing",
                    "birth_kf": -1,
                    "age": None,
                    "obs_count": 0,
                    "obs_frames": [],
                    "kf_feat_idx": int(kf_feat_idx[i]) if i < int(kf_feat_idx.size) else -1,
                    "cur_feat_idx": int(cur_feat_idx[i]) if i < int(cur_feat_idx.size) else -1,
                }
            )
            continue

        birth_kf = int(lm.get("birth_kf", -1))
        age = None if birth_kf < 0 else int(frame_index) - int(birth_kf)
        records.append(
            {
                "corr_index": int(i),
                "id": int(lm_id),
                "birth_source": str(lm.get("birth_source", "missing")),
                "birth_kf": int(birth_kf),
                "age": age,
                "obs_count": _obs_count(lm),
                "obs_frames": _obs_frames(lm),
                "kf_feat_idx": int(kf_feat_idx[i]) if i < int(kf_feat_idx.size) else -1,
                "cur_feat_idx": int(cur_feat_idx[i]) if i < int(cur_feat_idx.size) else -1,
            }
        )

    return records


# Convert metadata records into a boolean correspondence mask
def _metadata_mask(records: list[dict], predicate) -> np.ndarray:
    return np.asarray([bool(predicate(record)) for record in records], dtype=bool)


# Score local displacement consistency for a correspondence subgroup
def _subgroup_local_displacement_summary(seed: dict, corrs, pnp_cfg: dict, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    N = int(corrs.X_w.shape[1])
    if int(mask.size) != N:
        return {
            "available": False,
            "reason": "mask_size_mismatch",
            "count": int(np.sum(mask)),
        }

    feats = seed.get("feats1", None)
    if feats is None or not hasattr(feats, "kps_xy"):
        return {
            "available": False,
            "reason": "missing_keyframe_features",
            "count": int(np.sum(mask)),
        }

    kps_xy = np.asarray(feats.kps_xy, dtype=np.float64)
    kf_feat_idx = np.asarray(corrs.kf_feat_idx, dtype=np.int64).reshape(-1)
    if int(kf_feat_idx.size) != N or kps_xy.ndim != 2 or int(kps_xy.shape[1]) < 2:
        return {
            "available": False,
            "reason": "keyframe_features_unusable",
            "count": int(np.sum(mask)),
        }

    valid = mask & (kf_feat_idx >= 0) & (kf_feat_idx < int(kps_xy.shape[0]))
    count = int(np.sum(mask))
    valid_count = int(np.sum(valid))
    if valid_count != count:
        return {
            "available": False,
            "reason": "invalid_keyframe_feature_indices",
            "count": int(count),
            "valid_count": int(valid_count),
        }

    if count == 0:
        return {
            "available": True,
            "count": 0,
            "stats": None,
            "retention": None,
        }

    xy_kf = np.asarray(kps_xy[kf_feat_idx[mask], :2], dtype=np.float64)
    xy_cur = np.asarray(corrs.x_cur, dtype=np.float64).T[mask]
    local_mask, local_stats = pnp_local_displacement_consistency_mask(
        xy_kf,
        xy_cur,
        radius_px=float(pnp_cfg["pnp_local_consistency_radius_px"]),
        min_neighbours=int(pnp_cfg["pnp_local_consistency_min_neighbours"]),
        max_median_residual_px=float(pnp_cfg["pnp_local_consistency_max_median_residual_px"]),
        min_keep=0,
    )
    local_stats = dict(local_stats)
    retention = float(np.sum(local_mask) / max(count, 1))
    local_stats["retention"] = float(retention)

    return {
        "available": True,
        "count": int(count),
        "keep_count": int(np.sum(local_mask)),
        "reject_count": int(count - int(np.sum(local_mask))),
        "retention": float(retention),
        "stats": local_stats,
    }


# Collect compact pathology flags for one subgroup
def _subgroup_pathologies(count: int, recent_residual: dict, local_summary: dict) -> list[str]:
    pathologies: list[str] = []
    if int(count) == 0:
        return pathologies

    n_non_positive_depth = int(recent_residual.get("n_non_positive_depth", 0))
    within = recent_residual.get("within", {})
    within_20 = int(within.get("20", 0)) if isinstance(within, dict) else 0
    within_40 = int(within.get("40", 0)) if isinstance(within, dict) else 0
    median_px = recent_residual.get("median_px", None)
    p90_px = recent_residual.get("p90_px", None)

    if n_non_positive_depth > 0:
        pathologies.append("non_positive_depth_under_recent_pose")
    if median_px is not None and float(median_px) > 40.0:
        pathologies.append("high_recent_pose_median_residual")
    if p90_px is not None and float(p90_px) > 80.0:
        pathologies.append("very_high_recent_pose_p90_residual")
    if within_20 == 0 and int(count) >= 4:
        pathologies.append("no_recent_pose_support_within_20px")
    if within_40 < min(int(count), 4):
        pathologies.append("weak_recent_pose_support_within_40px")

    if bool(local_summary.get("available", False)) and local_summary.get("retention", None) is not None:
        retention = float(local_summary.get("retention", 0.0))
        if retention <= 0.20 and int(count) >= 4:
            pathologies.append("low_local_displacement_retention")
        stats = local_summary.get("stats", {})
        if isinstance(stats, dict) and int(stats.get("n_too_few_neighbours", 0)) == int(count) and int(count) >= 4:
            pathologies.append("locally_sparse_support")

    return pathologies


# Summarise one fixed correspondence subgroup
def _frame16_subgroup_summary(
    *,
    seed: dict,
    corrs,
    K: np.ndarray,
    pnp_cfg: dict,
    image_shape: tuple[int, int],
    reference_poses: list[dict],
    records: list[dict],
    label: str,
    group_type: str,
    mask: np.ndarray,
    residuals_px: np.ndarray | None = None,
) -> dict:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    sub_corrs = _slice_pnp_correspondences(corrs, mask)
    count = int(sub_corrs.X_w.shape[1])
    ref_rows = []
    for ref in reference_poses:
        residuals = _residual_summary(
            sub_corrs,
            K,
            ref.get("R", None),
            ref.get("t", None),
            eps=float(pnp_cfg["eps"]),
        )
        ref_rows.append(
            {
                "label": str(ref.get("label", "")),
                "kf": None if ref.get("kf", None) is None else int(ref.get("kf")),
                "localisation_only": bool(ref.get("localisation_only", False)),
                **residuals,
            }
        )

    recent_residual = ref_rows[0] if len(ref_rows) > 0 else _residual_summary(sub_corrs, K, None, None, eps=float(pnp_cfg["eps"]))
    local_summary = _subgroup_local_displacement_summary(seed, corrs, pnp_cfg, mask)
    ids = [int(v) for v in np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)[mask].tolist()]
    group_records = [record for record, keep in zip(records, mask.tolist()) if bool(keep)]
    residual_summary = None
    if residuals_px is not None:
        residuals_px = np.asarray(residuals_px, dtype=np.float64).reshape(-1)
        residual_summary = _numeric_summary(residuals_px[mask])

    return {
        "label": str(label),
        "group_type": str(group_type),
        "count": int(count),
        "landmark_ids": ids,
        "source_split": _value_counts([record["birth_source"] for record in group_records]),
        "birth_kf_distribution": _value_counts([record["birth_kf"] for record in group_records]),
        "obs_count_distribution": _value_counts([record["obs_count"] for record in group_records]),
        "current_spread": _spatial_summary(np.asarray(sub_corrs.x_cur, dtype=np.float64).T, image_shape),
        "recent_pose_residual": recent_residual,
        "residual_px_under_recent_pose": residual_summary,
        "local_displacement_consistency": local_summary,
        "pathologies": _subgroup_pathologies(count, recent_residual, local_summary),
    }


# Build frame-16 subgroup composition diagnostics
def _frame16_subgroup_composition(seed_before: dict, corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], reference_poses: list[dict], *, frame_index: int) -> dict:
    records = _correspondence_metadata_records(seed_before, corrs, frame_index=int(frame_index))
    N = int(corrs.X_w.shape[1])
    last_pose = _copy_pose(seed_before.get("last_accepted_pose", None))
    residuals_px = None
    if last_pose is not None:
        residuals_px = _residual_errors(
            corrs,
            K,
            last_pose["R"],
            last_pose["t"],
            eps=float(pnp_cfg["eps"]),
        )

    groups = []

    def _add_group(label: str, group_type: str, mask: np.ndarray) -> None:
        if int(np.sum(mask)) == 0:
            return
        groups.append(
            _frame16_subgroup_summary(
                seed=seed_before,
                corrs=corrs,
                K=K,
                pnp_cfg=pnp_cfg,
                image_shape=image_shape,
                reference_poses=reference_poses,
                records=records,
                label=label,
                group_type=group_type,
                mask=mask,
                residuals_px=residuals_px,
            )
        )

    for source in sorted(set(str(record["birth_source"]) for record in records)):
        _add_group(
            f"source:{source}",
            "birth_source",
            _metadata_mask(records, lambda record, source=source: str(record["birth_source"]) == str(source)),
        )

    for birth_kf in sorted(set(int(record["birth_kf"]) for record in records)):
        _add_group(
            f"birth_kf:{int(birth_kf)}",
            "birth_kf",
            _metadata_mask(records, lambda record, birth_kf=birth_kf: int(record["birth_kf"]) == int(birth_kf)),
        )

    for obs_count in sorted(set(int(record["obs_count"]) for record in records)):
        _add_group(
            f"obs_count:{int(obs_count)}",
            "obs_count",
            _metadata_mask(records, lambda record, obs_count=obs_count: int(record["obs_count"]) == int(obs_count)),
        )

    residual_bin_counts = {}
    if residuals_px is not None and int(residuals_px.size) == N:
        bins = [
            ("recent_pose_residual:le20px", np.isfinite(residuals_px) & (residuals_px <= 20.0)),
            ("recent_pose_residual:20_40px", np.isfinite(residuals_px) & (residuals_px > 20.0) & (residuals_px <= 40.0)),
            ("recent_pose_residual:gt40px", np.isfinite(residuals_px) & (residuals_px > 40.0)),
            ("recent_pose_residual:invalid", ~np.isfinite(residuals_px)),
        ]
        for label, mask in bins:
            residual_bin_counts[label] = int(np.sum(mask))
            _add_group(label, "recent_pose_residual_bin", mask)

    return {
        "eligible_count": int(N),
        "source_split": _value_counts([record["birth_source"] for record in records]),
        "birth_kf_distribution": _value_counts([record["birth_kf"] for record in records]),
        "obs_count_distribution": _value_counts([record["obs_count"] for record in records]),
        "residual_bin_counts": residual_bin_counts,
        "records": records,
        "groups": groups,
    }


# Label the outcome of one diagnostic exclusion
def _diagnostic_exclusion_assessment(experiment: dict, pnp_cfg: dict) -> str:
    retained = int(experiment.get("retained_count", 0))
    if retained < int(pnp_cfg["sample_size"]):
        return "too_few_after_exclusion"
    if bool(experiment.get("loose_pose_summary", {}).get("coherent_loose_pose_exists", False)):
        return "coherent_loose_pose_appears"
    return "no_coherent_loose_pose"


# Add a diagnostic exclusion experiment
def _append_exclusion_experiment(
    experiments: list[dict],
    corrs,
    K: np.ndarray,
    pnp_cfg: dict,
    image_shape: tuple[int, int],
    *,
    name: str,
    drop_mask: np.ndarray,
    residuals_px: np.ndarray | None,
    extra: dict | None = None,
) -> None:
    N = int(corrs.X_w.shape[1])
    drop_mask = np.asarray(drop_mask, dtype=bool).reshape(-1)
    if int(drop_mask.size) != N or int(np.sum(drop_mask)) == 0:
        return
    keep_mask = ~drop_mask
    row = _support_filter_experiment(
        corrs,
        K,
        pnp_cfg,
        image_shape,
        name=str(name),
        mask=keep_mask,
        residuals_px=residuals_px,
        extra={} if extra is None else extra,
    )
    row["dropped_landmark_ids"] = [int(v) for v in np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)[drop_mask].tolist()]
    row["assessment"] = _diagnostic_exclusion_assessment(row, pnp_cfg)
    experiments.append(row)


# Build narrow frame-16 subgroup exclusion tests
def _frame16_subgroup_exclusion_experiments(seed_before: dict, corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], *, frame_index: int) -> dict:
    records = _correspondence_metadata_records(seed_before, corrs, frame_index=int(frame_index))
    N = int(corrs.X_w.shape[1])
    ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    last_pose = _copy_pose(seed_before.get("last_accepted_pose", None))
    residuals_px = None
    if last_pose is not None:
        residuals_px = _residual_errors(
            corrs,
            K,
            last_pose["R"],
            last_pose["t"],
            eps=float(pnp_cfg["eps"]),
        )

    experiments: list[dict] = []
    source_map_growth = _metadata_mask(records, lambda record: str(record["birth_source"]) == "map_growth")
    _append_exclusion_experiment(
        experiments,
        corrs,
        K,
        pnp_cfg,
        image_shape,
        name="exclude_all_map_growth",
        drop_mask=source_map_growth,
        residuals_px=residuals_px,
        extra={"drop_rule": "birth_source == map_growth"},
    )

    bootstrap_birth_kfs = sorted(
        set(
            int(record["birth_kf"])
            for record in records
            if str(record["birth_source"]) == "bootstrap" and int(record["birth_kf"]) >= 0
        )
    )
    if len(bootstrap_birth_kfs) > 0:
        oldest_bootstrap_birth_kf = int(bootstrap_birth_kfs[0])
        _append_exclusion_experiment(
            experiments,
            corrs,
            K,
            pnp_cfg,
            image_shape,
            name=f"exclude_bootstrap_birth_kf_{oldest_bootstrap_birth_kf}",
            drop_mask=_metadata_mask(
                records,
                lambda record, oldest_bootstrap_birth_kf=oldest_bootstrap_birth_kf: str(record["birth_source"]) == "bootstrap"
                and int(record["birth_kf"]) == int(oldest_bootstrap_birth_kf),
            ),
            residuals_px=residuals_px,
            extra={"drop_rule": "oldest bootstrap birth_kf", "birth_kf": int(oldest_bootstrap_birth_kf)},
        )

    map_growth_birth_kfs = sorted(
        set(
            int(record["birth_kf"])
            for record in records
            if str(record["birth_source"]) == "map_growth" and int(record["birth_kf"]) >= 0
        )
    )
    if len(map_growth_birth_kfs) > 0:
        counts = {
            int(birth_kf): int(
                sum(
                    1
                    for record in records
                    if str(record["birth_source"]) == "map_growth" and int(record["birth_kf"]) == int(birth_kf)
                )
            )
            for birth_kf in map_growth_birth_kfs
        }
        largest_birth_kf = sorted(counts, key=lambda birth_kf: (-int(counts[int(birth_kf)]), int(birth_kf)))[0]
        _append_exclusion_experiment(
            experiments,
            corrs,
            K,
            pnp_cfg,
            image_shape,
            name=f"exclude_map_growth_birth_kf_{int(largest_birth_kf)}",
            drop_mask=_metadata_mask(
                records,
                lambda record, largest_birth_kf=largest_birth_kf: str(record["birth_source"]) == "map_growth"
                and int(record["birth_kf"]) == int(largest_birth_kf),
            ),
            residuals_px=residuals_px,
            extra={
                "drop_rule": "largest map_growth birth_kf subgroup",
                "birth_kf": int(largest_birth_kf),
                "subgroup_count": int(counts[int(largest_birth_kf)]),
            },
        )

    if residuals_px is not None and int(residuals_px.size) == N:
        invalid = ~np.isfinite(residuals_px)
        gt40 = np.isfinite(residuals_px) & (residuals_px > 40.0)
        gt20 = np.isfinite(residuals_px) & (residuals_px > 20.0)
        if np.any(invalid):
            worst_mask = invalid
            worst_label = "exclude_recent_pose_residual_invalid"
        elif np.any(gt40):
            worst_mask = gt40
            worst_label = "exclude_recent_pose_residual_gt40px"
        else:
            worst_mask = gt20
            worst_label = "exclude_recent_pose_residual_gt20px"
        _append_exclusion_experiment(
            experiments,
            corrs,
            K,
            pnp_cfg,
            image_shape,
            name=worst_label,
            drop_mask=worst_mask,
            residuals_px=residuals_px,
            extra={"drop_rule": "worst recent-pose residual subgroup"},
        )

        tail_count = min(4, max(1, int(np.ceil(0.20 * max(N, 1)))))
        sort_key = np.where(np.isfinite(residuals_px), residuals_px, np.inf)
        worst_order = np.argsort(-sort_key, kind="stable")
        tail_idx = worst_order[:tail_count]
        tail_mask = np.zeros((N,), dtype=bool)
        tail_mask[tail_idx] = True
        _append_exclusion_experiment(
            experiments,
            corrs,
            K,
            pnp_cfg,
            image_shape,
            name=f"exclude_highest_recent_pose_residual_tail_{tail_count}",
            drop_mask=tail_mask,
            residuals_px=residuals_px,
            extra={
                "drop_rule": "highest recent-pose residual tail",
                "tail_count": int(tail_count),
                "dropped_residual_px": [None if not np.isfinite(float(residuals_px[i])) else float(residuals_px[i]) for i in tail_idx.tolist()],
                "dropped_landmark_ids_ordered": [int(v) for v in ids[tail_idx].tolist()],
            },
        )

    return {
        "eligible_count": int(N),
        "reference_pose": None if last_pose is None else {
            "label": f"last_accepted_kf_{last_pose['kf']}",
            "kf": int(last_pose["kf"]),
            "localisation_only": bool(last_pose["localisation_only"]),
        },
        "experiments": experiments,
    }


# Summarise stored landmark quality fields
def _landmark_quality_summary(records: list[dict]) -> dict:
    keys = sorted({str(key) for record in records for key in record.get("quality", {}).keys()})
    out = {}
    for key in keys:
        values = [record.get("quality", {}).get(key, None) for record in records]
        numeric_values = [
            float(value)
            for value in values
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))
        ]
        out[key] = {
            "present": int(sum(value is not None for value in values)),
            "none": int(sum(value is None for value in values)),
            "numeric": _numeric_summary(numeric_values),
        }
    return out


# Build recent poses for fixed frame-16 landmark checks
def _frame16_reference_poses(seed_before: dict, accepted_history: list[dict]) -> list[dict]:
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

    out: list[dict] = []
    seen_labels: set[str] = set()
    for ref in refs:
        label = str(ref["label"])
        if label in seen_labels:
            continue
        seen_labels.add(label)
        out.append(ref)
    return out


# Summarise 3D landmark fields and depths under reference poses
def _landmark_geometry_summary(seed: dict, ids: set[int], reference_poses: list[dict], *, eps: float, frame_index: int | None = None) -> dict:
    lm_by_id = _landmarks_by_id(seed)
    records = []
    X_cols = []
    missing_ids = []
    invalid_xw_ids = []

    for lm_id in sorted(ids):
        lm = lm_by_id.get(int(lm_id), None)
        if not isinstance(lm, dict):
            missing_ids.append(int(lm_id))
            continue

        X_w = np.asarray(lm.get("X_w", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        xw_valid = bool(X_w.size == 3 and np.isfinite(X_w).all())
        if bool(xw_valid):
            X_cols.append(X_w.reshape(3, 1))
        else:
            invalid_xw_ids.append(int(lm_id))

        quality = lm.get("quality", {})
        birth_kf = int(lm.get("birth_kf", -1))
        age = None
        if frame_index is not None and birth_kf >= 0:
            age = int(frame_index) - int(birth_kf)

        records.append(
            {
                "id": int(lm_id),
                "birth_source": str(lm.get("birth_source", "missing")),
                "birth_kf": int(birth_kf),
                "age": age,
                "obs_count": _obs_count(lm),
                "xw_valid": bool(xw_valid),
                "quality": quality if isinstance(quality, dict) else {},
            }
        )

    X_w_all = np.hstack(X_cols) if len(X_cols) > 0 else np.zeros((3, 0), dtype=np.float64)
    depth_by_reference_pose = []
    for ref in reference_poses:
        if X_w_all.shape[1] == 0 or ref.get("R", None) is None or ref.get("t", None) is None:
            depth_by_reference_pose.append(
                {
                    "label": str(ref.get("label", "")),
                    "kf": None if ref.get("kf", None) is None else int(ref.get("kf")),
                    "count": int(X_w_all.shape[1]),
                    "n_positive_depth": 0,
                    "n_non_positive_depth": 0,
                    "depth": _numeric_summary([]),
                }
            )
            continue

        X_c = world_to_camera_points(ref["R"], ref["t"], X_w_all)
        z = np.asarray(X_c[2, :], dtype=np.float64).reshape(-1)
        finite = np.isfinite(z)
        positive = finite & (z > float(eps))
        depth_by_reference_pose.append(
            {
                "label": str(ref.get("label", "")),
                "kf": None if ref.get("kf", None) is None else int(ref.get("kf")),
                "localisation_only": bool(ref.get("localisation_only", False)),
                "count": int(z.size),
                "n_positive_depth": int(np.sum(positive)),
                "n_non_positive_depth": int(np.sum(finite & ~positive)),
                "depth": _numeric_summary(z[finite]),
            }
        )

    return {
        "count": int(len(ids)),
        "missing_ids": missing_ids,
        "invalid_xw_ids": invalid_xw_ids,
        "source_split": _value_counts([record["birth_source"] for record in records]),
        "birth_kf_distribution": _value_counts([record["birth_kf"] for record in records]),
        "age_distribution": _value_counts([record["age"] for record in records if record["age"] is not None]),
        "obs_count_distribution": _value_counts([record["obs_count"] for record in records]),
        "quality_fields": _landmark_quality_summary(records),
        "world_norm": _numeric_summary(np.linalg.norm(X_w_all, axis=0) if X_w_all.shape[1] > 0 else []),
        "depth_by_reference_pose": depth_by_reference_pose,
        "records": records,
    }


# Mask correspondences by landmark birth source
def _correspondence_birth_source_mask(seed: dict, corrs, birth_source: str) -> np.ndarray:
    lm_by_id = _landmarks_by_id(seed)
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    return np.asarray(
        [
            str(lm_by_id.get(int(lm_id), {}).get("birth_source", "missing")) == str(birth_source)
            for lm_id in landmark_ids
        ],
        dtype=bool,
    )


# Summarise one source class in a correspondence set
def _source_correspondence_summary(
    seed: dict,
    corrs,
    K: np.ndarray,
    reference_poses: list[dict],
    image_shape: tuple[int, int],
    *,
    birth_source: str,
    eps: float,
    frame_index: int,
) -> dict:
    mask = _correspondence_birth_source_mask(seed, corrs, birth_source)
    sub_corrs = _slice_pnp_correspondences(corrs, mask)
    ids = set(int(v) for v in np.asarray(sub_corrs.landmark_ids, dtype=np.int64).reshape(-1).tolist())
    residual_rows = []
    for ref in reference_poses:
        residuals = _residual_summary(
            sub_corrs,
            K,
            ref.get("R", None),
            ref.get("t", None),
            eps=float(eps),
        )
        residual_rows.append(
            {
                "label": str(ref.get("label", "")),
                "kf": None if ref.get("kf", None) is None else int(ref.get("kf")),
                "localisation_only": bool(ref.get("localisation_only", False)),
                **residuals,
            }
        )

    return {
        "birth_source": str(birth_source),
        "correspondence_count": int(np.sum(mask)),
        "unique_landmark_count": int(len(ids)),
        "unique_landmark_ids": [int(v) for v in sorted(ids)],
        "current_spread": _spatial_summary(np.asarray(sub_corrs.x_cur, dtype=np.float64).T, image_shape),
        "landmark_geometry": _landmark_geometry_summary(
            seed,
            ids,
            reference_poses,
            eps=float(eps),
            frame_index=int(frame_index),
        ),
        "residuals_by_reference_pose": residual_rows,
    }


# Compare bootstrap and map-growth support at frame 16
def _frame16_support_source_comparison(
    seed_before: dict,
    mapped_corrs,
    eligible_corrs,
    K: np.ndarray,
    reference_poses: list[dict],
    image_shape: tuple[int, int],
    *,
    eps: float,
    frame_index: int,
) -> dict:
    return {
        "mapped": {
            "bootstrap": _source_correspondence_summary(
                seed_before,
                mapped_corrs,
                K,
                reference_poses,
                image_shape,
                birth_source="bootstrap",
                eps=float(eps),
                frame_index=int(frame_index),
            ),
            "map_growth": _source_correspondence_summary(
                seed_before,
                mapped_corrs,
                K,
                reference_poses,
                image_shape,
                birth_source="map_growth",
                eps=float(eps),
                frame_index=int(frame_index),
            ),
        },
        "eligible": {
            "bootstrap": _source_correspondence_summary(
                seed_before,
                eligible_corrs,
                K,
                reference_poses,
                image_shape,
                birth_source="bootstrap",
                eps=float(eps),
                frame_index=int(frame_index),
            ),
            "map_growth": _source_correspondence_summary(
                seed_before,
                eligible_corrs,
                K,
                reference_poses,
                image_shape,
                birth_source="map_growth",
                eps=float(eps),
                frame_index=int(frame_index),
            ),
        },
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


# Pack compact records for one source
def _compact_source_records(seed: dict, birth_source: str, frame_index: int) -> list[dict]:
    records = []
    for lm in seed.get("landmarks", []):
        if not isinstance(lm, dict) or "id" not in lm:
            continue
        if str(lm.get("birth_source", "")) != str(birth_source):
            continue
        birth_kf = int(lm.get("birth_kf", -1))
        age = None if birth_kf < 0 else int(frame_index) - int(birth_kf)
        records.append(
            {
                "id": int(lm["id"]),
                "birth_kf": int(birth_kf),
                "age": age,
                "obs_count": _obs_count(lm),
            }
        )
    return records


# Capture lifecycle state around one processed frame
def _lifecycle_frame_snapshot(seed_before: dict, seed_after: dict, out: dict, seq, pnp_cfg: dict, frame_index: int) -> dict:
    cur_im, _, _ = seq.get(frame_index)
    image_shape = (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1]))
    track_out = out.get("track_out", {}) if isinstance(out, dict) else {}
    funnel = _support_funnel(seed_before, track_out, pnp_cfg, image_shape)

    before_ids = set(_landmarks_by_id(seed_before).keys())
    after_ids = set(_landmarks_by_id(seed_after).keys())
    before_bootstrap = _landmark_ids_by_source(seed_before, "bootstrap")
    before_map_growth = _landmark_ids_by_source(seed_before, "map_growth")
    after_map_growth = _landmark_ids_by_source(seed_after, "map_growth")
    mapped_ids = set(int(v) for v in funnel.get("mapped_landmark_ids", []))
    eligible_ids = set(int(v) for v in funnel.get("eligible_landmark_ids", []))

    return {
        "frame_index": int(frame_index),
        "seed_landmarks_before": int(len(before_ids)),
        "seed_landmarks_after": int(len(after_ids)),
        "bootstrap_present_before": int(len(before_bootstrap)),
        "map_growth_present_before": int(len(before_map_growth)),
        "map_growth_present_after": int(len(after_map_growth)),
        "map_growth_added_ids": [int(v) for v in sorted(after_map_growth - before_map_growth)],
        "map_growth_removed_ids": [int(v) for v in sorted(before_map_growth - after_map_growth)],
        "mapped_ids": [int(v) for v in sorted(mapped_ids)],
        "eligible_ids": [int(v) for v in sorted(eligible_ids)],
        "mapped_bootstrap_ids": [int(v) for v in sorted(mapped_ids & before_bootstrap)],
        "eligible_bootstrap_ids": [int(v) for v in sorted(eligible_ids & before_bootstrap)],
        "mapped_map_growth_ids": [int(v) for v in sorted(mapped_ids & before_map_growth)],
        "eligible_map_growth_ids": [int(v) for v in sorted(eligible_ids & before_map_growth)],
        "map_growth_records_before": _compact_source_records(seed_before, "map_growth", int(frame_index)),
        "map_growth_records_after": _compact_source_records(seed_after, "map_growth", int(frame_index)),
        "funnel_counts": {
            "raw_tracked_pairs": int(funnel.get("raw_tracked_pairs", 0)),
            "mapped_by_keyframe_lookup": int(funnel.get("mapped_by_keyframe_lookup", 0)),
            "observation_gated_pass": int(funnel.get("observation_gated_pass", 0)),
            "final_pnp_correspondences": int(funnel.get("final_pnp_correspondences", 0)),
        },
    }


# Summarise map-growth lifecycle across a frame range
def _landmark_lifecycle_summary(
    seed_at_frame16: dict,
    frame_rows: list[dict],
    mapped16_ids: set[int],
    eligible16_ids: set[int],
    *,
    start_frame: int,
    end_frame: int,
) -> dict:
    meta_by_id: dict[int, dict] = {}
    eligible_frames_by_id: dict[int, list[int]] = {}
    mapped_frames_by_id: dict[int, list[int]] = {}

    for row in frame_rows:
        frame_index = int(row.get("frame_index", -1))
        for record in row.get("map_growth_records_before", []) + row.get("map_growth_records_after", []):
            lm_id = int(record["id"])
            meta_by_id.setdefault(lm_id, dict(record))
        for lm_id in row.get("eligible_map_growth_ids", []):
            eligible_frames_by_id.setdefault(int(lm_id), []).append(int(frame_index))
        for lm_id in row.get("mapped_map_growth_ids", []):
            mapped_frames_by_id.setdefault(int(lm_id), []).append(int(frame_index))

    lm16_by_id = _landmarks_by_id(seed_at_frame16)
    present16_map_growth_ids = _landmark_ids_by_source(seed_at_frame16, "map_growth")
    for lm_id in present16_map_growth_ids:
        lm = lm16_by_id.get(int(lm_id), {})
        birth_kf = int(lm.get("birth_kf", -1)) if isinstance(lm, dict) else -1
        meta_by_id[int(lm_id)] = {
            "id": int(lm_id),
            "birth_kf": int(birth_kf),
            "age": None if birth_kf < 0 else int(end_frame) - int(birth_kf),
            "obs_count": _obs_count(lm) if isinstance(lm, dict) else 0,
        }

    all_ids = set(meta_by_id.keys())
    for row in frame_rows:
        all_ids.update(int(v) for v in row.get("map_growth_added_ids", []))
        all_ids.update(int(v) for v in row.get("map_growth_removed_ids", []))

    records = []
    for lm_id in sorted(all_ids):
        meta = dict(meta_by_id.get(int(lm_id), {"id": int(lm_id), "birth_kf": None, "age": None, "obs_count": None}))
        eligible_frames = sorted(set(int(v) for v in eligible_frames_by_id.get(int(lm_id), [])))
        mapped_frames = sorted(set(int(v) for v in mapped_frames_by_id.get(int(lm_id), [])))
        meta.update(
            {
                "present_at_frame16": bool(int(lm_id) in present16_map_growth_ids),
                "mapped_frames": mapped_frames,
                "eligible_frames": eligible_frames,
                "ever_mapped": bool(len(mapped_frames) > 0),
                "ever_pose_eligible": bool(len(eligible_frames) > 0),
                "frame16_mapped": bool(int(lm_id) in mapped16_ids),
                "frame16_eligible": bool(int(lm_id) in eligible16_ids),
            }
        )
        records.append(meta)

    present_records = [record for record in records if bool(record.get("present_at_frame16", False))]
    return {
        "frame_range": [int(start_frame), int(end_frame)],
        "map_growth_added_by_frame": {
            str(int(row["frame_index"])): int(len(row.get("map_growth_added_ids", [])))
            for row in frame_rows
        },
        "map_growth_removed_by_frame": {
            str(int(row["frame_index"])): int(len(row.get("map_growth_removed_ids", [])))
            for row in frame_rows
        },
        "map_growth_seen_count": int(len(records)),
        "map_growth_present_at_frame16_count": int(len(present_records)),
        "map_growth_ever_mapped_count": int(sum(1 for record in records if bool(record.get("ever_mapped", False)))),
        "map_growth_ever_pose_eligible_count": int(sum(1 for record in records if bool(record.get("ever_pose_eligible", False)))),
        "map_growth_frame16_mapped_count": int(sum(1 for record in records if bool(record.get("frame16_mapped", False)))),
        "map_growth_frame16_eligible_count": int(sum(1 for record in records if bool(record.get("frame16_eligible", False)))),
        "present_frame16_birth_kf_distribution": _value_counts([record.get("birth_kf", None) for record in present_records]),
        "present_frame16_obs_count_distribution": _value_counts([record.get("obs_count", None) for record in present_records]),
        "present_frame16_age_distribution": _value_counts([record.get("age", None) for record in present_records if record.get("age", None) is not None]),
        "records": records,
    }


# Read one landmark from a seed snapshot
def _landmark_by_id_or_none(seed: dict, lm_id: int) -> dict | None:
    return _landmarks_by_id(seed).get(int(lm_id), None)


# Collect observation frame indices for one landmark
def _obs_frames(lm: dict | None) -> list[int]:
    if not isinstance(lm, dict):
        return []
    obs = lm.get("obs", None)
    if not isinstance(obs, list):
        return []
    frames = []
    for ob in obs:
        if not isinstance(ob, dict):
            continue
        frames.append(int(ob.get("kf", -1)))
    return sorted(frames)


# Find feature indices that link a landmark to the active lookup basis
def _active_basis_features_for_landmark(seed: dict, lm_id: int, keyframe_index: int) -> list[int]:
    lm = _landmark_by_id_or_none(seed, int(lm_id))
    if not isinstance(lm, dict):
        return []

    obs = lm.get("obs", None)
    if not isinstance(obs, list):
        return []

    lookup = np.asarray(seed.get("landmark_id_by_feat1", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    feats = []
    for ob in obs:
        if not isinstance(ob, dict):
            continue
        if int(ob.get("kf", -1)) != int(keyframe_index):
            continue
        feat = int(ob.get("feat", -1))
        if feat < 0 or feat >= int(lookup.size):
            continue
        if int(lookup[feat]) != int(lm_id):
            continue
        feats.append(int(feat))
    return sorted(set(feats))


# Test whether a landmark can pass the observation gate
def _pose_obs_gate_ready(lm: dict | None, pnp_cfg: dict) -> bool:
    if not isinstance(lm, dict):
        return False
    is_bootstrap = str(lm.get("birth_source", "")) == "bootstrap"
    if is_bootstrap:
        min_obs_required = int(pnp_cfg["min_landmark_observations"])
        if not bool(pnp_cfg["allow_bootstrap_landmarks_for_pose"]):
            return False
    else:
        min_obs_required = max(
            int(pnp_cfg["min_landmark_observations"]),
            int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
        )
    return bool(_obs_count(lm) >= int(min_obs_required))


# Pick a stable bootstrap comparison cohort
def _bootstrap_comparison_ids(seed: dict, frame_inputs: list[dict], count: int) -> list[int]:
    bootstrap_ids = _landmark_ids_by_source(seed, "bootstrap")
    mapped_counts = {int(lm_id): 0 for lm_id in bootstrap_ids}
    eligible_counts = {int(lm_id): 0 for lm_id in bootstrap_ids}

    for row in frame_inputs:
        funnel = _support_funnel(
            row["seed_before"],
            row["track_out"],
            row["pnp_cfg"],
            row["image_shape"],
        )
        mapped = set(int(v) for v in funnel.get("mapped_landmark_ids", []))
        eligible = set(int(v) for v in funnel.get("eligible_landmark_ids", []))
        for lm_id in bootstrap_ids:
            if int(lm_id) in mapped:
                mapped_counts[int(lm_id)] += 1
            if int(lm_id) in eligible:
                eligible_counts[int(lm_id)] += 1

    ranked = sorted(
        bootstrap_ids,
        key=lambda lm_id: (
            -int(eligible_counts.get(int(lm_id), 0)),
            -int(mapped_counts.get(int(lm_id), 0)),
            int(lm_id),
        ),
    )
    return [int(v) for v in ranked[: int(count)]]


# Classify one landmark-frame maturation step
def _cohort_failure_step(record: dict) -> str:
    if not bool(record["present_before"]):
        return "not_present_before_frame"
    if int(record["active_basis_feature_count"]) == 0:
        return "lookup_basis_missing_link"
    if not bool(record["tracked_from_active_basis"]):
        return "not_tracked_from_active_basis"
    if not bool(record["mapped_back_to_landmark"]):
        return "tracked_but_not_mapped_back"
    if bool(record["appended_current_observation"]):
        return "observation_appended"
    if bool(record["localisation_only_or_failed"]):
        return "append_skipped_for_localisation_or_failed_pose"
    return "mapped_but_observation_not_appended"


# Trace one source cohort across processed frames
def _trace_landmark_cohort(source: str, cohort_ids: list[int], frame_inputs: list[dict], pnp_cfg: dict) -> dict:
    rows = []
    per_landmark: dict[int, list[dict]] = {int(lm_id): [] for lm_id in cohort_ids}

    for row in frame_inputs:
        frame_index = int(row["frame_index"])
        seed_before = row["seed_before"]
        seed_after = row["seed_after"]
        track_out = row["track_out"]
        pose_out = row["pose_out"]
        keyframe_index = int(row["keyframe_index_before"])
        funnel = _support_funnel(seed_before, track_out, pnp_cfg, row["image_shape"])
        mapped_ids = set(int(v) for v in funnel.get("mapped_landmark_ids", []))
        eligible_ids = set(int(v) for v in funnel.get("eligible_landmark_ids", []))
        kf_feat_idx = np.asarray(track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
        pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
        pose_stats = pose_stats if isinstance(pose_stats, dict) else {}
        pose_ok = bool(isinstance(pose_out, dict) and bool(pose_out.get("ok", False)))
        support_rescue = bool(pose_stats.get("pnp_support_rescue_succeeded", False))
        incoherent_recovery = bool(pose_stats.get("pnp_incoherent_recovery_succeeded", False))
        localisation_or_failed = bool(not pose_ok or support_rescue or incoherent_recovery)
        if not bool(pose_ok):
            append_block_reason = str(pose_stats.get("reason", "pose_failed"))
        elif bool(support_rescue):
            append_block_reason = "support_rescue_localisation_only"
        elif bool(incoherent_recovery):
            append_block_reason = "incoherent_recovery_localisation_only"
        else:
            append_block_reason = None

        frame_records = []
        for lm_id in cohort_ids:
            lm_before = _landmark_by_id_or_none(seed_before, int(lm_id))
            lm_after = _landmark_by_id_or_none(seed_after, int(lm_id))
            active_features = _active_basis_features_for_landmark(seed_before, int(lm_id), keyframe_index)
            active_feature_set = set(int(v) for v in active_features)
            tracked = any(int(v) in active_feature_set for v in kf_feat_idx.tolist())
            present_before = bool(isinstance(lm_before, dict))
            present_after = bool(isinstance(lm_after, dict))
            obs_before = _obs_count(lm_before) if isinstance(lm_before, dict) else 0
            obs_after = _obs_count(lm_after) if isinstance(lm_after, dict) else 0
            obs_frames_before = _obs_frames(lm_before)
            obs_frames_after = _obs_frames(lm_after)
            appended = bool(present_before and obs_after > obs_before and frame_index in obs_frames_after and frame_index not in obs_frames_before)
            record = {
                "id": int(lm_id),
                "frame_index": int(frame_index),
                "present_before": bool(present_before),
                "present_after": bool(present_after),
                "created_this_frame": bool(not present_before and present_after),
                "active_keyframe": int(keyframe_index),
                "active_basis_feature_count": int(len(active_features)),
                "active_basis_features": [int(v) for v in active_features],
                "tracked_from_active_basis": bool(tracked),
                "mapped_back_to_landmark": bool(int(lm_id) in mapped_ids),
                "pose_eligible_this_frame": bool(int(lm_id) in eligible_ids),
                "obs_gate_ready_before": _pose_obs_gate_ready(lm_before, pnp_cfg),
                "obs_count_before": int(obs_before),
                "obs_count_after": int(obs_after),
                "obs_frames_before": obs_frames_before,
                "obs_frames_after": obs_frames_after,
                "appended_current_observation": bool(appended),
                "localisation_only_or_failed": bool(localisation_or_failed),
                "append_block_reason": append_block_reason,
            }
            record["failure_step"] = _cohort_failure_step(record)
            frame_records.append(record)
            per_landmark[int(lm_id)].append(record)

        rows.append(
            {
                "frame_index": int(frame_index),
                "active_keyframe": int(keyframe_index),
                "pose_ok": bool(pose_ok),
                "append_path_open": bool(not localisation_or_failed),
                "append_block_reason": append_block_reason,
                "cohort_present_before": int(sum(1 for record in frame_records if bool(record["present_before"]))),
                "cohort_present_after": int(sum(1 for record in frame_records if bool(record["present_after"]))),
                "cohort_linked_to_active_basis": int(sum(1 for record in frame_records if int(record["active_basis_feature_count"]) > 0)),
                "cohort_tracked_from_active_basis": int(sum(1 for record in frame_records if bool(record["tracked_from_active_basis"]))),
                "cohort_mapped_back": int(sum(1 for record in frame_records if bool(record["mapped_back_to_landmark"]))),
                "cohort_appended": int(sum(1 for record in frame_records if bool(record["appended_current_observation"]))),
                "cohort_obs_ge_3_after": int(sum(1 for record in frame_records if int(record["obs_count_after"]) >= 3)),
                "cohort_pose_eligible": int(sum(1 for record in frame_records if bool(record["pose_eligible_this_frame"]))),
                "failure_steps": _value_counts([record["failure_step"] for record in frame_records]),
            }
        )

    final_records = []
    for lm_id in cohort_ids:
        landmark_rows = per_landmark[int(lm_id)]
        final_row = landmark_rows[-1] if len(landmark_rows) > 0 else {}
        final_records.append(
            {
                "id": int(lm_id),
                "present_at_end": bool(final_row.get("present_after", False)),
                "final_obs_count": int(final_row.get("obs_count_after", 0)),
                "ever_tracked_from_active_basis": bool(any(bool(row["tracked_from_active_basis"]) for row in landmark_rows)),
                "ever_mapped_back": bool(any(bool(row["mapped_back_to_landmark"]) for row in landmark_rows)),
                "ever_appended": bool(any(bool(row["appended_current_observation"]) for row in landmark_rows)),
                "ever_pose_eligible": bool(any(bool(row["pose_eligible_this_frame"]) for row in landmark_rows)),
                "failure_steps": _value_counts([row["failure_step"] for row in landmark_rows]),
                "records": landmark_rows,
            }
        )

    return {
        "source": str(source),
        "cohort_ids": [int(v) for v in cohort_ids],
        "frame_rows": rows,
        "final": {
            "count": int(len(cohort_ids)),
            "present_at_end": int(sum(1 for record in final_records if bool(record["present_at_end"]))),
            "ever_tracked_from_active_basis": int(sum(1 for record in final_records if bool(record["ever_tracked_from_active_basis"]))),
            "ever_mapped_back": int(sum(1 for record in final_records if bool(record["ever_mapped_back"]))),
            "ever_appended": int(sum(1 for record in final_records if bool(record["ever_appended"]))),
            "obs_ge_3_at_end": int(sum(1 for record in final_records if int(record["final_obs_count"]) >= 3)),
            "ever_pose_eligible": int(sum(1 for record in final_records if bool(record["ever_pose_eligible"]))),
            "final_obs_count_distribution": _value_counts([record["final_obs_count"] for record in final_records]),
            "dominant_failure_steps": _value_counts(
                [
                    sorted(record["failure_steps"].items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]
                    for record in final_records
                    if len(record["failure_steps"]) > 0
                ]
            ),
        },
        "sample_records": final_records[: min(8, len(final_records))],
    }


# Build cohort accrual diagnostics from stored frame snapshots
def _cohort_accrual_diagnostic(seed_at_end: dict, frame_inputs: list[dict], pnp_cfg: dict, *, cohort_birth_kf: int = 8) -> dict:
    birth_rows = [row for row in frame_inputs if int(row["frame_index"]) == int(cohort_birth_kf)]
    birth_seed = birth_rows[0]["seed_after"] if len(birth_rows) > 0 else seed_at_end
    map_growth_ids = [
        int(lm["id"])
        for lm in birth_seed.get("landmarks", [])
        if isinstance(lm, dict)
        and str(lm.get("birth_source", "")) == "map_growth"
        and int(lm.get("birth_kf", -1)) == int(cohort_birth_kf)
    ]
    map_growth_ids = sorted(map_growth_ids)
    bootstrap_ids = _bootstrap_comparison_ids(seed_at_end, frame_inputs, min(max(len(map_growth_ids), 1), 24))
    comparison_frames = [row for row in frame_inputs if int(row["frame_index"]) >= int(cohort_birth_kf)]

    return {
        "cohort_birth_kf": int(cohort_birth_kf),
        "map_growth": _trace_landmark_cohort("map_growth", map_growth_ids, comparison_frames, pnp_cfg),
        "bootstrap_comparison": _trace_landmark_cohort("bootstrap", bootstrap_ids, comparison_frames, pnp_cfg),
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


# Parse a compact comma-separated frame list
def _parse_frame_indices(raw: str) -> list[int]:
    frames: list[int] = []
    for part in str(raw).split(","):
        value = str(part).strip()
        if value == "":
            continue
        frame_index = int(value)
        if frame_index < 2:
            raise ValueError(f"comparison frame must be >= 2; got {frame_index}")
        if frame_index not in frames:
            frames.append(int(frame_index))
    return frames


# Select a compact state label for comparison rows
def _comparison_state_label(frame_index: int, standard: dict, pose_stats: dict, frame16_index: int) -> str:
    if int(frame_index) == int(frame16_index):
        return "incoherent"
    if bool(pose_stats.get("pnp_support_rescue_succeeded", False)) or bool(standard.get("localisation_only_rescue_frame", False)):
        return "rescue-worthy"
    if bool(standard.get("pipeline_ok", False)):
        return "healthy"
    return "failed"


# Build a compact geometry-consistency row for one processed frame
def _geometry_consistency_comparison_row(
    seed_before: dict,
    keyframe_index: int,
    out: dict,
    seq,
    K: np.ndarray,
    pnp_cfg: dict,
    frame_index: int,
    *,
    incoherent_frame_index: int,
) -> dict:
    cur_im, cur_ts, cur_id = seq.get(frame_index)
    image_shape = (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1]))
    track_out = out.get("track_out", {})
    pose_out = out.get("pose_out", {})
    pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
    pose_stats = pose_stats if isinstance(pose_stats, dict) else {}
    standard = _standard_frame_stats(
        frame_index=frame_index,
        reference_keyframe_index=keyframe_index,
        frontend_out=out,
        seed_landmarks_before=_seed_landmark_count(seed_before),
        seed_landmarks_after=_seed_landmark_count(out.get("seed", {})),
    )

    funnel = _support_funnel(seed_before, track_out, pnp_cfg, image_shape)
    corrs, _ = build_pnp_correspondences_with_stats(
        seed_before,
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

    last_pose = _copy_pose(seed_before.get("last_accepted_pose", None))
    R_ref = None if last_pose is None else last_pose.get("R", None)
    t_ref = None if last_pose is None else last_pose.get("t", None)
    residuals = _residual_summary(corrs, K, R_ref, t_ref, eps=float(pnp_cfg["eps"]))
    within = residuals.get("within", {})
    local_stats = funnel.get("track_2d", {}).get("eligible", {}).get("local_displacement_consistency", None)
    local_stats = local_stats if isinstance(local_stats, dict) else {}

    return {
        "frame_index": int(frame_index),
        "frame_id": str(cur_id),
        "timestamp": float(cur_ts),
        "state_label": _comparison_state_label(frame_index, standard, pose_stats, int(incoherent_frame_index)),
        "pipeline_ok": bool(standard.get("pipeline_ok", False)),
        "pipeline_reason": standard.get("pipeline_reason", None),
        "rescue_succeeded": bool(pose_stats.get("pnp_support_rescue_succeeded", False)),
        "localisation_only_rescue_frame": bool(standard.get("localisation_only_rescue_frame", False)),
        "last_accepted_pose": None if last_pose is None else {
            "kf": last_pose["kf"],
            "localisation_only": bool(last_pose["localisation_only"]),
        },
        "mapped": int(funnel.get("mapped_by_keyframe_lookup", 0)),
        "eligible": int(corrs.X_w.shape[1]),
        "eligible_landmark_ids": [int(v) for v in np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1).tolist()],
        "local_displacement_retained": int(local_stats.get("n_keep", 0)),
        "local_displacement_retention": local_stats.get("retention", None),
        "recent_pose_positive_depth": int(residuals.get("n_positive_depth", 0)),
        "recent_pose_within_20_px": int(within.get("20", 0)),
        "recent_pose_within_30_px": int(within.get("30", 0)),
        "recent_pose_within_40_px": int(within.get("40", 0)),
        "recent_pose_within_60_px": int(within.get("60", 0)),
        "recent_pose_median_px": residuals.get("median_px", None),
        "recent_pose_p90_px": residuals.get("p90_px", None),
        "recent_pose_residual_px": residuals.get("error_px", {}),
    }


# Build recent reference-pose residual summaries for frame 16
def _reference_pose_residuals(seed_before: dict, accepted_history: list[dict], corrs, K: np.ndarray, *, eps: float) -> list[dict]:
    out = []
    for ref in _frame16_reference_poses(seed_before, accepted_history):
        label = str(ref["label"])
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


# Count threshold support compactly
def _threshold_counts(rows: list[dict]) -> dict[str, int]:
    return {f"{float(row['threshold_px']):.0f}px": int(row.get("n_inliers", 0)) for row in rows}


# Score whether a threshold set has a loose coherent pose
def _coherent_loose_pose(rows: list[dict]) -> bool:
    return bool(_loose_pose_summary(rows).get("coherent_loose_pose_exists", False))


# Score a pose-only anchor on the current frame-16 support
def _evaluate_pose_anchor(
    *,
    label: str,
    pose: dict,
    corrs,
    K: np.ndarray,
    pnp_cfg: dict,
    image_shape: tuple[int, int],
    current_funnel: dict,
) -> dict:
    rows = _score_fixed_pose_thresholds(corrs, K, pnp_cfg, image_shape, pose.get("R", None), pose.get("t", None))
    residuals = _residual_summary(corrs, K, pose.get("R", None), pose.get("t", None), eps=float(pnp_cfg["eps"]))
    return {
        "label": str(label),
        "anchor_kind": "pose_rescore",
        "analysis_method": "rescore_current_fixed_support",
        "source_frame": None if pose.get("kf", None) is None else int(pose.get("kf")),
        "kf": None if pose.get("kf", None) is None else int(pose.get("kf")),
        "localisation_only": bool(pose.get("localisation_only", False)),
        "tracked": int(current_funnel.get("raw_tracked_pairs", 0)),
        "mapped": int(current_funnel.get("mapped_by_keyframe_lookup", 0)),
        "eligible": int(corrs.X_w.shape[1]),
        "threshold_counts": _threshold_counts(rows),
        "threshold_summary": _compact_thresholds(rows),
        "coherent_loose_pose_exists": _coherent_loose_pose(rows),
        "median_residual_px": residuals.get("median_px", None),
        "p90_residual_px": residuals.get("p90_px", None),
    }


# Score a lookup-basis anchor by re-tracking frame 16
def _evaluate_basis_anchor(
    *,
    label: str,
    basis: dict,
    frame16_seed_before: dict,
    seq,
    K: np.ndarray,
    frontend_kwargs: dict,
    pnp_cfg: dict,
    frame_index: int,
) -> dict:
    cur_im, _, _ = seq.get(frame_index)
    image_shape = (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1]))
    anchor_seed = _seed_with_basis(frame16_seed_before, basis)

    try:
        track_out = track_against_keyframe(
            K,
            basis["feats"],
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
        )
    except Exception as exc:
        return {
            "label": str(label),
            "anchor_kind": str(basis.get("kind", "basis")),
            "analysis_method": "retrack_against_basis",
            "source_frame": basis.get("source_frame", None),
            "kf": int(basis.get("kf", -1)),
            "localisation_only": bool(basis.get("localisation_only", False)),
            "available": False,
            "reason": "tracking_error",
            "error": str(exc),
        }

    funnel = _support_funnel(anchor_seed, track_out, pnp_cfg, image_shape)
    corrs, _ = build_pnp_correspondences_with_stats(
        anchor_seed,
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
    rows = [_run_threshold(corrs, K, pnp_cfg, image_shape, threshold_px=v) for v in [8.0, 12.0, 20.0, 40.0]]
    return {
        "label": str(label),
        "anchor_kind": str(basis.get("kind", "basis")),
        "analysis_method": "retrack_against_basis",
        "source_frame": basis.get("source_frame", None),
        "kf": int(basis.get("kf", -1)),
        "localisation_only": bool(basis.get("localisation_only", False)),
        "available": True,
        "basis_lookup_mapped": int(basis.get("n_lookup_mapped", 0)),
        "tracked": int(funnel.get("raw_tracked_pairs", 0)),
        "mapped": int(funnel.get("mapped_by_keyframe_lookup", 0)),
        "eligible": int(funnel.get("final_pnp_correspondences", 0)),
        "track_matches_reported": int(funnel.get("track_matches_reported", 0)),
        "track_inliers_reported": int(funnel.get("track_inliers_reported", 0)),
        "track_2d": funnel.get("track_2d", {}),
        "threshold_counts": _threshold_counts(rows),
        "threshold_summary": _compact_thresholds(rows),
        "coherent_loose_pose_exists": _coherent_loose_pose(rows),
        "loose_pose_summary": _loose_pose_summary(rows),
        "reconstruction": basis.get("reconstruction", None),
    }


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


# Compare 2D-only coherence with 3D-backed reprojection on the same landmarks
def _frame16_2d_3d_separation_test(seed_before: dict, corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], reference_poses: list[dict]) -> dict:
    N = int(corrs.X_w.shape[1])
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    kf_feat_idx = np.asarray(corrs.kf_feat_idx, dtype=np.int64).reshape(-1)
    xy_cur_all = np.asarray(corrs.x_cur, dtype=np.float64).T

    feats = seed_before.get("feats1", None)
    if feats is None or not hasattr(feats, "kps_xy"):
        return {
            "available": False,
            "reason": "missing_keyframe_features",
            "eligible_count": int(N),
        }

    kps_xy = np.asarray(feats.kps_xy, dtype=np.float64)
    valid = (kf_feat_idx >= 0) & (kf_feat_idx < int(kps_xy.shape[0]))
    if int(valid.size) != N:
        valid = np.zeros((N,), dtype=bool)

    xy_kf_valid = np.asarray(kps_xy[kf_feat_idx[valid], :2], dtype=np.float64)
    xy_cur_valid = np.asarray(xy_cur_all[valid, :2], dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    local_keep = np.zeros((N,), dtype=bool)

    if int(xy_kf_valid.shape[0]) > 0:
        local_keep_valid, local_stats = pnp_local_displacement_consistency_mask(
            xy_kf_valid,
            xy_cur_valid,
            radius_px=float(pnp_cfg["pnp_local_consistency_radius_px"]),
            min_neighbours=int(pnp_cfg["pnp_local_consistency_min_neighbours"]),
            max_median_residual_px=float(pnp_cfg["pnp_local_consistency_max_median_residual_px"]),
            min_keep=int(pnp_cfg["pnp_local_consistency_min_keep"]),
        )
        local_keep[valid_indices] = np.asarray(local_keep_valid, dtype=bool).reshape(-1)
        local_stats = dict(local_stats)
    else:
        local_stats = {
            "n_input": 0,
            "n_keep": 0,
            "n_removed": 0,
            "reason": "no_valid_keyframe_feature_indices",
        }

    local_stats["retention"] = float(np.sum(local_keep) / max(N, 1))
    pose_rows = []
    for ref in reference_poses:
        errors_px = _residual_errors(
            corrs,
            K,
            ref.get("R", None),
            ref.get("t", None),
            eps=float(pnp_cfg["eps"]),
        )
        within_20 = np.isfinite(errors_px) & (errors_px <= 20.0)
        within_40 = np.isfinite(errors_px) & (errors_px <= 40.0)
        pose_rows.append(
            {
                "label": str(ref.get("label", "")),
                "kf": None if ref.get("kf", None) is None else int(ref.get("kf")),
                "localisation_only": bool(ref.get("localisation_only", False)),
                "within_20_count": int(np.sum(within_20)),
                "within_40_count": int(np.sum(within_40)),
                "two_d_keep_and_within_20_count": int(np.sum(local_keep & within_20)),
                "two_d_keep_and_within_40_count": int(np.sum(local_keep & within_40)),
                "two_d_reject_but_within_40_count": int(np.sum(~local_keep & within_40)),
                "within_40_landmark_ids": [int(v) for v in landmark_ids[within_40].tolist()],
                "two_d_keep_and_within_40_landmark_ids": [int(v) for v in landmark_ids[local_keep & within_40].tolist()],
                "residual_px": _numeric_summary(errors_px[np.isfinite(errors_px)]),
            }
        )

    return {
        "available": True,
        "name": "eligible_2d_local_consistency_vs_recent_pose_reprojection",
        "eligible_count": int(N),
        "valid_keyframe_feature_indices": int(np.sum(valid)),
        "two_d_local_keep_count": int(np.sum(local_keep)),
        "two_d_local_reject_count": int(N - int(np.sum(local_keep))),
        "two_d_local_keep_landmark_ids": [int(v) for v in landmark_ids[local_keep].tolist()],
        "two_d_local_reject_landmark_ids": [int(v) for v in landmark_ids[~local_keep].tolist()],
        "two_d_local_stats": local_stats,
        "two_d_spread": {
            "all_keyframe": _spatial_summary(xy_kf_valid, image_shape),
            "all_current": _spatial_summary(xy_cur_valid, image_shape),
            "kept_keyframe": _spatial_summary(xy_kf_valid[np.asarray(local_keep[valid], dtype=bool)], image_shape),
            "kept_current": _spatial_summary(xy_cur_valid[np.asarray(local_keep[valid], dtype=bool)], image_shape),
        },
        "pose_reprojection": pose_rows,
    }


# Build the requested recovery-anchor comparison
def _frame16_candidate_recovery_anchors(
    *,
    frame16_seed_before: dict,
    frame16_keyframe_index_before: int,
    frame16_accepted_history: list[dict],
    promoted_basis_history: list[dict],
    rescued_support_basis_history: list[dict],
    analysed16: dict,
    corrs16,
    seq,
    K: np.ndarray,
    frontend_kwargs: dict,
    pnp_cfg: dict,
    frame_index: int,
) -> dict:
    cur_im, _, _ = seq.get(frame_index)
    image_shape = (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1]))
    current_funnel = analysed16["funnel"]
    rows: list[dict] = []

    current_basis = _copy_basis_snapshot(
        frame16_seed_before,
        int(frame16_keyframe_index_before),
        label="current_active_keyframe_basis",
        kind="active_keyframe_basis",
        source_frame=int(frame16_keyframe_index_before),
        localisation_only=False,
    )
    if current_basis is not None:
        rows.append(
            _evaluate_basis_anchor(
                label="current_active_keyframe_basis",
                basis=current_basis,
                frame16_seed_before=frame16_seed_before,
                seq=seq,
                K=K,
                frontend_kwargs=frontend_kwargs,
                pnp_cfg=pnp_cfg,
                frame_index=frame_index,
            )
        )

    last_pose = _copy_pose(frame16_seed_before.get("last_accepted_pose", None))
    if last_pose is not None:
        rows.append(
            _evaluate_pose_anchor(
                label="last_accepted_pose",
                pose=last_pose,
                corrs=corrs16,
                K=K,
                pnp_cfg=pnp_cfg,
                image_shape=image_shape,
                current_funnel=current_funnel,
            )
        )

    previous_pose = None
    last_kf = None if last_pose is None else last_pose.get("kf", None)
    for pose in reversed(frame16_accepted_history):
        if last_kf is not None and int(pose.get("kf", -1)) == int(last_kf):
            continue
        previous_pose = pose
        break
    if previous_pose is not None:
        rows.append(
            _evaluate_pose_anchor(
                label="previous_accepted_pose",
                pose=previous_pose,
                corrs=corrs16,
                K=K,
                pnp_cfg=pnp_cfg,
                image_shape=image_shape,
                current_funnel=current_funnel,
            )
        )

    last_promoted_basis = promoted_basis_history[-1] if len(promoted_basis_history) > 0 else None
    if last_promoted_basis is not None:
        rows.append(
            _evaluate_basis_anchor(
                label="last_non_localisation_promoted_keyframe_basis",
                basis=last_promoted_basis,
                frame16_seed_before=frame16_seed_before,
                seq=seq,
                K=K,
                frontend_kwargs=frontend_kwargs,
                pnp_cfg=pnp_cfg,
                frame_index=frame_index,
            )
        )

    clean_rescued_bases = [basis for basis in rescued_support_basis_history if bool(basis.get("available", True))]
    most_recent_rescued_basis = clean_rescued_bases[-1] if len(clean_rescued_bases) > 0 else None
    if most_recent_rescued_basis is not None:
        rows.append(
            _evaluate_basis_anchor(
                label="most_recent_rescued_support_basis",
                basis=most_recent_rescued_basis,
                frame16_seed_before=frame16_seed_before,
                seq=seq,
                K=K,
                frontend_kwargs=frontend_kwargs,
                pnp_cfg=pnp_cfg,
                frame_index=frame_index,
            )
        )

    recent_rescued_candidates = clean_rescued_bases[-3:]
    recent_basis_rows = []
    for basis in recent_rescued_candidates:
        recent_basis_rows.append(
            _evaluate_basis_anchor(
                label=f"recent_rescued_support_basis_kf_{int(basis.get('kf', -1))}",
                basis=basis,
                frame16_seed_before=frame16_seed_before,
                seq=seq,
                K=K,
                frontend_kwargs=frontend_kwargs,
                pnp_cfg=pnp_cfg,
                frame_index=frame_index,
            )
        )

    def _row_score(row: dict) -> tuple[int, int, int, int]:
        counts = row.get("threshold_counts", {})
        return (
            int(bool(row.get("coherent_loose_pose_exists", False))),
            int(counts.get("40px", 0)),
            int(counts.get("20px", 0)),
            int(row.get("eligible", 0)),
        )

    strongest_recent_basis = None
    if len(recent_basis_rows) > 0:
        strongest_recent_basis = max(recent_basis_rows, key=_row_score)
        already_present = any(
            str(row.get("anchor_kind", "")) == str(strongest_recent_basis.get("anchor_kind", ""))
            and int(row.get("kf", -999)) == int(strongest_recent_basis.get("kf", -998))
            and str(row.get("label", "")) in {"most_recent_rescued_support_basis", "current_active_keyframe_basis", "last_non_localisation_promoted_keyframe_basis"}
            for row in rows
        )
        if not bool(already_present):
            strongest_recent_basis = dict(strongest_recent_basis)
            strongest_recent_basis["label"] = "strongest_recent_basis"
            rows.append(strongest_recent_basis)

    current_row = rows[0] if len(rows) > 0 else None
    current_counts = current_row.get("threshold_counts", {}) if current_row is not None else {}
    current_pose_score = (
        int(bool(current_row.get("coherent_loose_pose_exists", False))) if current_row is not None else 0,
        int(current_counts.get("40px", 0)),
        int(current_counts.get("20px", 0)),
    )
    for row in rows:
        counts = row.get("threshold_counts", {})
        pose_score = (
            int(bool(row.get("coherent_loose_pose_exists", False))),
            int(counts.get("40px", 0)),
            int(counts.get("20px", 0)),
        )
        row["meaningfully_better_than_current"] = bool(pose_score > current_pose_score)

    return {
        "available_promoted_bases": [
            {
                "label": str(basis.get("label", "")),
                "kf": int(basis.get("kf", -1)),
                "source_frame": basis.get("source_frame", None),
                "n_lookup_mapped": int(basis.get("n_lookup_mapped", 0)),
            }
            for basis in promoted_basis_history
        ],
        "available_rescued_support_bases": [
            {
                "label": str(basis.get("label", "")),
                "kf": int(basis.get("kf", -1)),
                "source_frame": basis.get("source_frame", None),
                "available": bool(basis.get("available", True)),
                "n_lookup_mapped": int(basis.get("n_lookup_mapped", 0)),
                "reconstruction": basis.get("reconstruction", None),
            }
            for basis in rescued_support_basis_history
        ],
        "rows": rows,
        "strongest_recent_basis": strongest_recent_basis,
    }


# Run the frame-16 deep dive
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--target_frame", type=int, default=16)
    parser.add_argument("--comparison_frames", type=str, default="7,12,16")
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    target_frame = int(args.target_frame)
    comparison_frames = _parse_frame_indices(args.comparison_frames)
    if len(comparison_frames) > 0:
        target_frame = max(int(target_frame), max(comparison_frames))
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
    comparison_rows: dict[int, dict] = {}
    frame16_seed_before = None
    frame16_keyframe_index_before = None
    frame16_track_out = None
    frame16_accepted_history: list[dict] = []
    promoted_basis_history: list[dict] = []
    rescued_support_basis_history: list[dict] = []
    lifecycle_start_frame = 8
    lifecycle_end_frame = 16
    lifecycle_rows: list[dict] = []
    cohort_frame_inputs: list[dict] = []

    initial_basis = _copy_basis_snapshot(
        seed,
        keyframe_index,
        label="initial_bootstrap_keyframe_basis",
        kind="promoted_keyframe_basis",
        source_frame=1,
        localisation_only=False,
    )
    if initial_basis is not None:
        promoted_basis_history.append(initial_basis)

    for frame_index in range(2, int(target_frame) + 1):
        cur_im, _, _ = seq.get(frame_index)
        seed_before = seed
        seed_before_snapshot = _diagnostic_seed_snapshot(seed)
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

        if int(lifecycle_start_frame) <= int(frame_index) <= int(lifecycle_end_frame):
            cohort_frame_inputs.append(
                {
                    "frame_index": int(frame_index),
                    "keyframe_index_before": int(keyframe_index_before),
                    "seed_before": seed_before_snapshot,
                    "seed_after": _diagnostic_seed_snapshot(out.get("seed", seed_before)),
                    "track_out": out.get("track_out", {}),
                    "pose_out": out.get("pose_out", {}),
                    "pnp_cfg": pnp_cfg,
                    "image_shape": (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1])),
                }
            )
        if int(lifecycle_start_frame) <= int(frame_index) <= int(lifecycle_end_frame):
            lifecycle_rows.append(
                _lifecycle_frame_snapshot(
                    seed_before_snapshot,
                    out.get("seed", seed_before),
                    out,
                    seq,
                    pnp_cfg,
                    int(frame_index),
                )
            )
        if frame_index in (15, 16):
            analysed[frame_index] = _analyse_processed_frame(
                seed_before_snapshot,
                keyframe_index_before,
                out,
                seq,
                K,
                pnp_cfg,
                frame_index,
            )
        if frame_index in comparison_frames:
            comparison_rows[frame_index] = _geometry_consistency_comparison_row(
                seed_before_snapshot,
                keyframe_index_before,
                out,
                seq,
                K,
                pnp_cfg,
                frame_index,
                incoherent_frame_index=16,
            )
        if frame_index == 16:
            frame16_seed_before = seed_before_snapshot
            frame16_keyframe_index_before = int(keyframe_index_before)
            frame16_track_out = out.get("track_out", {})
            frame16_accepted_history = accepted_history_before

        seed = out["seed"]
        if bool(out.get("ok", False)):
            pose = _copy_pose(seed.get("last_accepted_pose", None))
            if pose is not None:
                accepted_history.append(pose)
            pose_out = out.get("pose_out", {})
            pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
            pose_stats = pose_stats if isinstance(pose_stats, dict) else {}
            if bool(pose_stats.get("pnp_support_rescue_succeeded", False)):
                rescued_basis = _reconstruct_rescued_support_basis(
                    out.get("track_out", {}),
                    pose_out,
                    int(frame_index),
                )
                if rescued_basis is not None:
                    rescued_support_basis_history.append(rescued_basis)
        if bool(out.get("stats", {}).get("keyframe_promoted", False)):
            keyframe_feats = out["track_out"]["cur_feats"]
            keyframe_index = int(frame_index)
            promoted_basis = _copy_basis_snapshot(
                seed,
                keyframe_index,
                label=f"promoted_keyframe_basis_kf_{int(frame_index)}",
                kind="promoted_keyframe_basis",
                source_frame=int(frame_index),
                localisation_only=False,
            )
            if promoted_basis is not None:
                promoted_basis_history.append(promoted_basis)

    if 15 not in analysed or 16 not in analysed or frame16_seed_before is None or frame16_keyframe_index_before is None or frame16_track_out is None:
        raise RuntimeError("Expected frame 15 and frame 16 analyses to be available")

    corrs16 = analysed[16]["_corrs"]
    thresholds16 = analysed[16]["_thresholds"]
    cur_im16, _, _ = seq.get(16)
    image_shape16 = (int(np.asarray(cur_im16).shape[0]), int(np.asarray(cur_im16).shape[1]))
    reference_poses16 = _frame16_reference_poses(frame16_seed_before, frame16_accepted_history)
    residuals16 = _reference_pose_residuals(
        frame16_seed_before,
        frame16_accepted_history,
        corrs16,
        K,
        eps=float(pnp_cfg["eps"]),
    )
    mapped16_ids = set(int(v) for v in analysed[16]["funnel"].get("mapped_landmark_ids", []))
    eligible16_ids = set(int(v) for v in analysed[16]["funnel"].get("eligible_landmark_ids", []))
    mapped16_corrs = _build_mapped_correspondences(frame16_seed_before, frame16_track_out)
    landmark_geometry16 = {
        "mapped": _landmark_geometry_summary(
            frame16_seed_before,
            mapped16_ids,
            reference_poses16,
            eps=float(pnp_cfg["eps"]),
            frame_index=16,
        ),
        "eligible": _landmark_geometry_summary(
            frame16_seed_before,
            eligible16_ids,
            reference_poses16,
            eps=float(pnp_cfg["eps"]),
            frame_index=16,
        ),
    }
    landmark_lifecycle = _landmark_lifecycle_summary(
        frame16_seed_before,
        lifecycle_rows,
        mapped16_ids,
        eligible16_ids,
        start_frame=int(lifecycle_start_frame),
        end_frame=int(lifecycle_end_frame),
    )
    cohort_accrual = _cohort_accrual_diagnostic(
        seed,
        cohort_frame_inputs,
        pnp_cfg,
        cohort_birth_kf=8,
    )
    support_source_comparison = _frame16_support_source_comparison(
        frame16_seed_before,
        mapped16_corrs,
        corrs16,
        K,
        reference_poses16,
        image_shape16,
        eps=float(pnp_cfg["eps"]),
        frame_index=16,
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
    subgroup_composition = _frame16_subgroup_composition(
        frame16_seed_before,
        corrs16,
        K,
        pnp_cfg,
        image_shape16,
        reference_poses16,
        frame_index=16,
    )
    subgroup_exclusion_experiments = _frame16_subgroup_exclusion_experiments(
        frame16_seed_before,
        corrs16,
        K,
        pnp_cfg,
        image_shape16,
        frame_index=16,
    )
    separation_test = _frame16_2d_3d_separation_test(
        frame16_seed_before,
        corrs16,
        K,
        pnp_cfg,
        image_shape16,
        reference_poses16,
    )
    candidate_recovery_anchors = _frame16_candidate_recovery_anchors(
        frame16_seed_before=frame16_seed_before,
        frame16_keyframe_index_before=int(frame16_keyframe_index_before),
        frame16_accepted_history=frame16_accepted_history,
        promoted_basis_history=promoted_basis_history,
        rescued_support_basis_history=rescued_support_basis_history,
        analysed16=analysed[16],
        corrs16=corrs16,
        seq=seq,
        K=K,
        frontend_kwargs=frontend_kwargs,
        pnp_cfg=pnp_cfg,
        frame_index=16,
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
        "frame16_landmark_geometry": landmark_geometry16,
        "frame16_support_source_comparison": support_source_comparison,
        "frame8_frame16_landmark_lifecycle": {
            "frames": lifecycle_rows,
            "summary": landmark_lifecycle,
        },
        "frame8_cohort_observation_accrual": cohort_accrual,
        "frame15_frame16_landmark_continuity": continuity,
        "frame16_support_filter_experiments": support_filter_experiments,
        "frame16_subgroup_composition": subgroup_composition,
        "frame16_subgroup_exclusion_experiments": subgroup_exclusion_experiments,
        "frame16_2d_3d_separation_test": separation_test,
        "frame16_candidate_recovery_anchors": candidate_recovery_anchors,
        "recent_pose_geometry_consistency_comparison": {
            "comparison_frames": [int(v) for v in comparison_frames],
            "reference": "eligible PnP correspondences scored under the last accepted pose before each frame",
            "rows": [comparison_rows[int(frame_index)] for frame_index in comparison_frames if int(frame_index) in comparison_rows],
        },
    }

    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
