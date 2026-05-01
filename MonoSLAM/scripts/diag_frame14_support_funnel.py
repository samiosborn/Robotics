from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_runtime_cfg as _load_runtime_cfg, seed_landmark_count as _seed_landmark_count, standard_frame_stats as _standard_frame_stats

from datasets.eth3d import load_eth3d_sequence
from geometry.camera import camera_centre, reprojection_errors_sq, world_to_camera_points
from geometry.pnp import _pnp_inlier_mask_from_pose, _slice_pnp_correspondences, build_pnp_correspondences_with_stats, estimate_pose_pnp, estimate_pose_pnp_ransac, pnp_inlier_spatial_coverage
from geometry.pose import angle_between_translations
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe import _build_landmark_id_by_feat_for_kf
from slam.tracking import track_against_keyframe


# Count valid observation records
def _obs_count(lm: dict) -> int:
    obs = lm.get("obs", [])
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


# Summarise numeric values
def _numeric_summary(values) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p75": None,
            "max": None,
        }

    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


# Count categorical values
def _value_counts(values) -> dict:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if int(arr.size) == 0:
        return {}
    labels, counts = np.unique(arr, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


# Summarise quality dictionaries
def _quality_summary(records: list[dict]) -> dict:
    keys: set[str] = set()
    for record in records:
        quality = record.get("quality", {})
        if not isinstance(quality, dict):
            continue
        for key in quality.keys():
            keys.add(str(key))

    out = {
        "keys": sorted(keys),
        "numeric": {},
    }
    for key in sorted(keys):
        vals = []
        for record in records:
            quality = record.get("quality", {})
            if not isinstance(quality, dict):
                continue
            value = quality.get(key, None)
            if value is None:
                continue
            try:
                vals.append(float(value))
            except (TypeError, ValueError):
                continue
        out["numeric"][key] = _numeric_summary(vals)

    return out


# Summarise one landmark subgroup
def _summarise_landmark_records(records: list[dict], image_shape: tuple[int, int]) -> dict:
    xy = np.asarray([record["xy_cur"] for record in records], dtype=np.float64).reshape(-1, 2) if len(records) else np.zeros((0, 2), dtype=np.float64)
    mask = np.ones((xy.shape[0],), dtype=bool)
    return {
        "count": int(len(records)),
        "birth_source": _value_counts([record.get("birth_source", "unknown") for record in records]),
        "birth_kf": _value_counts([record.get("birth_kf", -1) for record in records]),
        "age": _numeric_summary([record.get("age", np.nan) for record in records]),
        "obs_count": _numeric_summary([record.get("obs_count", np.nan) for record in records]),
        "depth_m": _numeric_summary([record.get("depth_m", np.nan) for record in records]),
        "error_px_under_40_pose": _numeric_summary([record.get("error_px_under_40_pose", np.nan) for record in records]),
        "xw_norm": _numeric_summary([record.get("xw_norm", np.nan) for record in records]),
        "quality": _quality_summary(records),
        "image_region": _spatial_summary(xy, mask, image_shape),
    }


# Build a diagnostic keyframe lookup from current-frame tracks
def _diagnostic_refresh_keyframe(seed: dict, track_out: dict, pose_out: dict, current_kf: int) -> tuple[dict, object, dict]:
    seed_out = copy.copy(seed)
    cur_feats = track_out.get("cur_feats", None)
    if cur_feats is None or not hasattr(cur_feats, "kps_xy"):
        raise RuntimeError("Cannot refresh diagnostic keyframe without current features")

    old_lookup = np.asarray(seed.get("landmark_id_by_feat1", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    kf_feat_idx = np.asarray(track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(track_out.get("cur_feat_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    n_feat = int(np.asarray(cur_feats.kps_xy).shape[0])
    new_lookup = np.full((n_feat,), -1, dtype=np.int64)
    n_candidates = min(int(kf_feat_idx.size), int(cur_feat_idx.size))
    n_mapped = 0
    n_conflicts = 0

    for i in range(n_candidates):
        old_feat = int(kf_feat_idx[i])
        cur_feat = int(cur_feat_idx[i])
        if old_feat < 0 or old_feat >= int(old_lookup.size):
            continue
        if cur_feat < 0 or cur_feat >= int(new_lookup.size):
            continue
        lm_id = int(old_lookup[old_feat])
        if lm_id < 0:
            continue
        prev = int(new_lookup[cur_feat])
        if prev >= 0 and prev != lm_id:
            n_conflicts += 1
            continue
        if prev < 0:
            n_mapped += 1
        new_lookup[cur_feat] = lm_id

    seed_out["feats1"] = cur_feats
    seed_out["landmark_id_by_feat1"] = new_lookup
    seed_out["keyframe_kf"] = int(current_kf)
    seed_out["T_WC1"] = (
        np.asarray(pose_out["R"], dtype=np.float64),
        np.asarray(pose_out["t"], dtype=np.float64).reshape(3),
    )
    seed_out["diagnostic_keyframe_refresh"] = {
        "current_kf": int(current_kf),
        "n_feat": int(n_feat),
        "n_candidates": int(n_candidates),
        "n_mapped": int(n_mapped),
        "n_conflicts": int(n_conflicts),
    }

    return seed_out, cur_feats, dict(seed_out["diagnostic_keyframe_refresh"])


# Promote a rescued frame into a diagnostic lookup basis from visible landmarks
def _diagnostic_promote_rescued_lookup_basis(seed: dict, track_out: dict, pose_out: dict, current_kf: int) -> tuple[dict, object, dict]:
    seed_out = copy.copy(seed)
    cur_feats = track_out.get("cur_feats", None)
    if cur_feats is None or not hasattr(cur_feats, "kps_xy"):
        raise RuntimeError("Cannot promote diagnostic lookup basis without current features")

    corrs = pose_out["corrs"]
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(corrs.cur_feat_idx, dtype=np.int64).reshape(-1)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    n_feat = int(np.asarray(cur_feats.kps_xy).shape[0])
    N = min(int(landmark_ids.size), int(cur_feat_idx.size), int(x_cur.shape[1]))

    landmarks = list(seed.get("landmarks", []))
    landmark_pos_by_id = {
        int(lm["id"]): i
        for i, lm in enumerate(landmarks)
        if isinstance(lm, dict) and "id" in lm
    }

    n_added = 0
    n_duplicate = 0
    n_missing = 0
    for i in range(N):
        lm_id = int(landmark_ids[i])
        feat_idx = int(cur_feat_idx[i])
        if feat_idx < 0 or feat_idx >= int(n_feat):
            continue
        lm_pos = landmark_pos_by_id.get(lm_id, None)
        if lm_pos is None:
            n_missing += 1
            continue

        lm = landmarks[lm_pos]
        obs = lm.get("obs", None)
        if not isinstance(obs, list):
            obs = []

        duplicate = False
        for ob in obs:
            if not isinstance(ob, dict):
                continue
            if int(ob.get("kf", -1)) == int(current_kf) and int(ob.get("feat", -1)) == int(feat_idx):
                duplicate = True
                break
        if bool(duplicate):
            n_duplicate += 1
            lm["obs"] = obs
            continue

        obs.append(
            {
                "kf": int(current_kf),
                "feat": int(feat_idx),
                "xy": np.asarray(x_cur[:, i], dtype=np.float64).reshape(2),
            }
        )
        lm["obs"] = obs
        n_added += 1

    seed_out["landmarks"] = landmarks
    seed_out["feats1"] = cur_feats
    seed_out["T_WC1"] = (
        np.asarray(pose_out["R"], dtype=np.float64),
        np.asarray(pose_out["t"], dtype=np.float64).reshape(3),
    )
    seed_out["keyframe_kf"] = int(current_kf)
    seed_out["landmark_id_by_feat1"] = _build_landmark_id_by_feat_for_kf(seed_out, n_feat, int(current_kf))
    stats = {
        "current_kf": int(current_kf),
        "n_feat": int(n_feat),
        "n_visible_corr": int(N),
        "n_observations_added": int(n_added),
        "n_duplicate_observations": int(n_duplicate),
        "n_missing_landmarks": int(n_missing),
        "n_lookup_mapped": int(np.sum(np.asarray(seed_out["landmark_id_by_feat1"], dtype=np.int64) >= 0)),
    }
    seed_out["diagnostic_rescued_lookup_promotion"] = stats

    return seed_out, cur_feats, stats


# Append existing-landmark PnP inlier observations without pruning or growth
def _diagnostic_append_existing_observations_no_prune(seed: dict, pose_out: dict, current_kf: int) -> tuple[dict, dict]:
    seed_out = copy.copy(seed)
    landmarks = list(seed.get("landmarks", []))
    corrs = pose_out["corrs"]
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(corrs.cur_feat_idx, dtype=np.int64).reshape(-1)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    inlier_mask = np.asarray(pose_out.get("pnp_inlier_mask", np.zeros((landmark_ids.size,), dtype=bool)), dtype=bool).reshape(-1)

    N = min(int(landmark_ids.size), int(cur_feat_idx.size), int(x_cur.shape[1]), int(inlier_mask.size))
    landmark_pos_by_id = {
        int(lm["id"]): i
        for i, lm in enumerate(landmarks)
        if isinstance(lm, dict) and "id" in lm
    }

    stats = {
        "current_kf": int(current_kf),
        "n_corr": int(N),
        "n_inlier_corr": int(np.sum(inlier_mask[:N])),
        "n_added": 0,
        "n_duplicate": 0,
        "n_missing_landmark": 0,
        "n_landmarks_with_obs_current_kf_after_append": 0,
    }

    for i in np.flatnonzero(inlier_mask[:N]):
        lm_id = int(landmark_ids[int(i)])
        feat_idx = int(cur_feat_idx[int(i)])
        lm_pos = landmark_pos_by_id.get(lm_id, None)
        if lm_pos is None:
            stats["n_missing_landmark"] += 1
            continue

        lm = landmarks[lm_pos]
        obs = lm.get("obs", None)
        if not isinstance(obs, list):
            obs = []

        duplicate = False
        for ob in obs:
            if not isinstance(ob, dict):
                continue
            if int(ob.get("kf", -1)) == int(current_kf) and int(ob.get("feat", -1)) == int(feat_idx):
                duplicate = True
                break

        if bool(duplicate):
            stats["n_duplicate"] += 1
            lm["obs"] = obs
            continue

        obs.append(
            {
                "kf": int(current_kf),
                "feat": int(feat_idx),
                "xy": np.asarray(x_cur[:, int(i)], dtype=np.float64).reshape(2),
            }
        )
        lm["obs"] = obs
        stats["n_added"] += 1

    n_linked = 0
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        obs = lm.get("obs", None)
        if not isinstance(obs, list):
            continue
        if any(isinstance(ob, dict) and int(ob.get("kf", -1)) == int(current_kf) for ob in obs):
            n_linked += 1

    stats["n_landmarks_with_obs_current_kf_after_append"] = int(n_linked)
    seed_out["landmarks"] = landmarks
    seed_out["last_diagnostic_append_existing_no_prune_stats"] = stats

    return seed_out, stats


# Summarise image-space spread for a boolean subset
def _spatial_summary(xy: np.ndarray, mask: np.ndarray, image_shape: tuple[int, int]) -> dict:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if xy.shape[0] != mask.size:
        return {"count": 0, "reason": "unaligned"}

    pts = xy[mask]
    pts = pts[np.isfinite(pts).all(axis=1)]
    H = int(image_shape[0])
    W = int(image_shape[1])
    if pts.shape[0] == 0:
        return {
            "count": 0,
            "bbox": None,
            "bbox_area_fraction": None,
            "centroid": None,
            "occupied_cells": 0,
            "occupancy_grid": [[0 for _ in range(4)] for _ in range(3)],
        }

    x = pts[:, 0]
    y = pts[:, 1]
    bbox = [float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))]
    bbox_area_fraction = float(max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0) / max(float(W * H), 1.0))
    grid = [[0 for _ in range(4)] for _ in range(3)]
    for p in pts:
        col = int(np.clip(np.floor((float(p[0]) / max(float(W), 1.0)) * 4), 0, 3))
        row = int(np.clip(np.floor((float(p[1]) / max(float(H), 1.0)) * 3), 0, 2))
        grid[row][col] += 1

    return {
        "count": int(pts.shape[0]),
        "bbox": bbox,
        "bbox_area_fraction": bbox_area_fraction,
        "centroid": [float(np.mean(x)), float(np.mean(y))],
        "occupied_cells": int(sum(1 for row in grid for value in row if int(value) > 0)),
        "occupancy_grid": grid,
    }


# Run one PnP threshold on the fixed eligible set
def _run_threshold(corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], threshold_px: float) -> dict:
    n_corr = int(corrs.X_w.shape[1])
    if n_corr < int(pnp_cfg["sample_size"]):
        return {
            "threshold_px": float(threshold_px),
            "ok": False,
            "reason": "too_few_correspondences_for_ransac",
            "n_inliers": 0,
            "coverage": None,
        }

    R, t, inlier_mask, stats = estimate_pose_pnp_ransac(
        corrs,
        K,
        num_trials=5000 if float(threshold_px) >= 20.0 else int(pnp_cfg["num_trials"]),
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
    stats = stats if isinstance(stats, dict) else {}
    mask = np.zeros((n_corr,), dtype=bool) if inlier_mask is None else np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.size != n_corr:
        mask = np.zeros((n_corr,), dtype=bool)
    coverage = pnp_inlier_spatial_coverage(corrs.x_cur, mask, image_shape, grid_cols=4, grid_rows=3)
    n_inliers = int(np.sum(mask))
    if int(stats.get("n_inliers", 0)) > int(n_inliers):
        n_inliers = int(stats.get("n_inliers", 0))

    return {
        "threshold_px": float(threshold_px),
        "ok": bool(R is not None and t is not None),
        "reason": stats.get("reason", None),
        "n_inliers": int(n_inliers),
        "n_model_success": int(stats.get("n_model_success", 0)),
        "coverage": {
            "occupied_cells": int(coverage.get("occupied_cells", 0)),
            "bbox_area_fraction": coverage.get("bbox_area_fraction", None),
            "max_cell_fraction": coverage.get("max_cell_fraction", None),
        },
        "R": None if R is None else np.asarray(R, dtype=np.float64),
        "t": None if t is None else np.asarray(t, dtype=np.float64).reshape(3),
    }


# Replay RANSAC sampling and retain rejected candidate diagnostics
def _replay_ransac_candidate_path(corrs, K: np.ndarray, pnp_cfg: dict, image_shape: tuple[int, int], threshold_px: float) -> dict:
    X_w = np.asarray(corrs.X_w, dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    N = int(X_w.shape[1])
    sample_size = int(pnp_cfg["sample_size"])
    min_inliers = int(pnp_cfg["min_inliers"])
    num_trials = 5000 if float(threshold_px) >= 20.0 else int(pnp_cfg["num_trials"])

    if N < sample_size:
        return {
            "available": False,
            "reason": "too_few_correspondences_for_ransac",
        }

    x_min = float(np.min(x_cur[0, :]))
    y_min = float(np.min(x_cur[1, :]))
    x_span = max(float(np.max(x_cur[0, :]) - x_min), 1.0)
    y_span = max(float(np.max(x_cur[1, :]) - y_min), 1.0)
    x_cur_rank = np.asarray(x_cur, dtype=np.float64).copy()
    x_cur_rank[0, :] -= x_min
    x_cur_rank[1, :] -= y_min
    rank_image_shape = (int(np.ceil(y_span)) + 1, int(np.ceil(x_span)) + 1)

    def _support_rank(inlier_mask):
        coverage = pnp_inlier_spatial_coverage(
            x_cur_rank,
            inlier_mask,
            rank_image_shape,
            grid_cols=4,
            grid_rows=3,
        )
        bbox_area_fraction = coverage.get("bbox_area_fraction", None)
        max_cell_fraction = coverage.get("max_cell_fraction", None)
        return (
            int(coverage.get("occupied_cells", 0)),
            -np.inf if bbox_area_fraction is None else float(bbox_area_fraction),
            -np.inf if max_cell_fraction is None else -float(max_cell_fraction),
        )

    rng = np.random.default_rng(int(pnp_cfg["ransac_seed"]))
    best_R = None
    best_t = None
    best_mask = np.zeros((N,), dtype=bool)
    best_count = 0
    best_mean_err = np.inf
    best_support_rank = (0, -np.inf, -np.inf)
    best_trial = None
    n_model_success = 0
    n_viable_seen = 0
    n_displaced_by_lower_count_spatial_tie = 0
    best_count_seen = 0
    best_count_seen_trial = None
    spatial_tie_inlier_gap = 2

    for trial in range(int(num_trials)):
        idx = rng.choice(N, size=sample_size, replace=False)
        corrs_sub = _slice_pnp_correspondences(corrs, idx)
        try:
            R_t, t_t, _ = estimate_pose_pnp(
                corrs_sub,
                K,
                min_points=int(pnp_cfg["min_points"]),
                rank_tol=float(pnp_cfg["rank_tol"]),
                min_cheirality_ratio=float(pnp_cfg["min_cheirality_ratio"]),
                eps=float(pnp_cfg["eps"]),
            )
        except Exception:
            continue

        if R_t is None or t_t is None:
            continue

        n_model_success += 1
        mask_t, d_sq_t = _pnp_inlier_mask_from_pose(
            X_w,
            x_cur,
            K,
            R_t,
            t_t,
            threshold_px=float(threshold_px),
            eps=float(pnp_cfg["eps"]),
        )
        count_t = int(mask_t.sum())
        if count_t == 0:
            continue

        if count_t > best_count_seen:
            best_count_seen = int(count_t)
            best_count_seen_trial = int(trial)
        if count_t >= min_inliers:
            n_viable_seen += 1

        mean_err_t = float(np.mean(d_sq_t[mask_t]))
        support_rank_t = _support_rank(mask_t)
        inlier_gap_t = int(best_count - count_t)
        better_support_t = support_rank_t > best_support_rank
        same_support_t = support_rank_t == best_support_rank
        keep_t = False
        displacement_reason = None
        if count_t > best_count:
            keep_t = True
            displacement_reason = "higher_count"
        elif best_count >= min_inliers and count_t < best_count:
            keep_t = False
        elif 0 <= inlier_gap_t <= int(spatial_tie_inlier_gap):
            if better_support_t:
                keep_t = True
                displacement_reason = "spatial_tie"
            elif same_support_t and mean_err_t < best_mean_err:
                keep_t = True
                displacement_reason = "mean_error_tie"

        if keep_t:
            if displacement_reason == "spatial_tie" and count_t < best_count:
                n_displaced_by_lower_count_spatial_tie += 1
            best_R = np.asarray(R_t, dtype=np.float64)
            best_t = np.asarray(t_t, dtype=np.float64).reshape(3)
            best_mask = np.asarray(mask_t, dtype=bool)
            best_count = int(count_t)
            best_mean_err = float(mean_err_t)
            best_support_rank = support_rank_t
            best_trial = int(trial)
            if best_count == N:
                break

    raw_coverage = pnp_inlier_spatial_coverage(corrs.x_cur, best_mask, image_shape, grid_cols=4, grid_rows=3)
    out = {
        "available": True,
        "threshold_px": float(threshold_px),
        "N": int(N),
        "num_trials": int(num_trials),
        "sample_size": int(sample_size),
        "min_inliers": int(min_inliers),
        "n_model_success": int(n_model_success),
        "best_raw_trial": best_trial,
        "best_raw_inliers": int(best_count),
        "best_raw_mean_err_sq": None if not np.isfinite(best_mean_err) else float(best_mean_err),
        "best_raw_support_rank": [int(best_support_rank[0]), float(best_support_rank[1]), float(best_support_rank[2])],
        "best_raw_coverage": {
            "occupied_cells": int(raw_coverage.get("occupied_cells", 0)),
            "bbox_area_fraction": raw_coverage.get("bbox_area_fraction", None),
            "max_cell_fraction": raw_coverage.get("max_cell_fraction", None),
        },
        "best_count_seen": int(best_count_seen),
        "best_count_seen_trial": best_count_seen_trial,
        "n_viable_candidates_seen": int(n_viable_seen),
        "n_displaced_by_lower_count_spatial_tie": int(n_displaced_by_lower_count_spatial_tie),
        "raw_viable": bool(best_R is not None and best_t is not None and best_count >= min_inliers),
        "refit_attempted": False,
        "refit_inliers": None,
        "refit_kept": False,
    }

    if best_R is None or best_t is None or best_count < min_inliers:
        return out

    if not bool(pnp_cfg["refit"]):
        return out

    out["refit_attempted"] = True
    try:
        corrs_in = _slice_pnp_correspondences(corrs, best_mask)
        R_refit, t_refit, _ = estimate_pose_pnp(
            corrs_in,
            K,
            min_points=int(pnp_cfg["min_points"]),
            rank_tol=float(pnp_cfg["rank_tol"]),
            min_cheirality_ratio=float(pnp_cfg["min_cheirality_ratio"]),
            eps=float(pnp_cfg["eps"]),
        )
        if R_refit is not None and t_refit is not None:
            mask_refit, _ = _pnp_inlier_mask_from_pose(
                X_w,
                x_cur,
                K,
                R_refit,
                t_refit,
                threshold_px=float(threshold_px),
                eps=float(pnp_cfg["eps"]),
            )
            n_refit = int(np.sum(mask_refit))
            out["refit_inliers"] = int(n_refit)
            out["refit_kept"] = bool(n_refit >= best_count)
    except Exception as exc:
        out["refit_error"] = str(exc)

    return out


# Rescore a pose at a diagnostic threshold
def _rescore_pose(corrs, K: np.ndarray, image_shape: tuple[int, int], R, t, threshold_px: float, eps: float) -> dict:
    mask, d_sq = _pnp_inlier_mask_from_pose(
        corrs.X_w,
        corrs.x_cur,
        K,
        np.asarray(R, dtype=np.float64),
        np.asarray(t, dtype=np.float64).reshape(3),
        threshold_px=float(threshold_px),
        eps=float(eps),
    )
    coverage = pnp_inlier_spatial_coverage(corrs.x_cur, mask, image_shape, grid_cols=4, grid_rows=3)
    n_inliers = int(np.sum(mask))
    return {
        "threshold_px": float(threshold_px),
        "n_inliers": int(n_inliers),
        "mean_err_sq": None if n_inliers == 0 else float(np.mean(np.asarray(d_sq, dtype=np.float64)[mask])),
        "coverage": {
            "occupied_cells": int(coverage.get("occupied_cells", 0)),
            "bbox_area_fraction": coverage.get("bbox_area_fraction", None),
            "max_cell_fraction": coverage.get("max_cell_fraction", None),
        },
    }


# Summarise pose difference from the previous accepted pose
def _pose_delta(reference_pose: dict | None, pose_row: dict) -> dict:
    if not isinstance(reference_pose, dict) or pose_row.get("R", None) is None or pose_row.get("t", None) is None:
        return {
            "available": False,
            "rotation_delta_deg": None,
            "translation_direction_delta_deg": None,
            "camera_centre_direction_delta_deg": None,
        }

    R_ref = np.asarray(reference_pose["R"], dtype=np.float64)
    t_ref = np.asarray(reference_pose["t"], dtype=np.float64).reshape(3)
    R = np.asarray(pose_row["R"], dtype=np.float64)
    t = np.asarray(pose_row["t"], dtype=np.float64).reshape(3)
    return {
        "available": True,
        "rotation_delta_deg": float(np.degrees(angle_between_rotmats(R_ref, R))),
        "translation_direction_delta_deg": float(np.degrees(angle_between_translations(t_ref, t))),
        "camera_centre_direction_delta_deg": float(np.degrees(angle_between_translations(camera_centre(R_ref, t_ref), camera_centre(R, t)))),
    }


# Analyse one frame before advancing the frontend state
def _analyse_frame(seed: dict, keyframe_feats, keyframe_index: int, seq, K: np.ndarray, frontend_kwargs: dict, frame_index: int, *, experiment: str) -> dict:
    cur_im, cur_ts, cur_id = seq.get(frame_index)
    image_shape = (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1]))
    pnp_cfg = frontend_kwargs["pnp_frontend_kwargs"]
    track_out = track_against_keyframe(
        K,
        keyframe_feats,
        cur_im,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
    )

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
    mapped_xw = np.full((n_tracks, 3), np.nan, dtype=np.float64)
    bootstrap_born = np.zeros((n_tracks,), dtype=bool)
    post_bootstrap_born = np.zeros((n_tracks,), dtype=bool)
    obs_counts = np.full((n_tracks,), -1, dtype=np.int64)

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
            mapped_xw[i, :] = X_w[:3]
        n_obs = _obs_count(lm)
        obs_counts[i] = int(n_obs)
        is_bootstrap = str(lm.get("birth_source", "")) == "bootstrap"
        bootstrap_born[i] = bool(is_bootstrap)
        post_bootstrap_born[i] = not bool(is_bootstrap)
        min_obs_required = int(pnp_cfg["min_landmark_observations"]) if is_bootstrap else max(
            int(pnp_cfg["min_landmark_observations"]),
            int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
        )
        obs_gate[i] = bool(valid_xw[i] and n_obs >= int(min_obs_required))

    corrs, corr_stats = build_pnp_correspondences_with_stats(
        seed,
        track_out,
        min_landmark_observations=int(pnp_cfg["min_landmark_observations"]),
        allow_bootstrap_landmarks_for_pose=bool(pnp_cfg["allow_bootstrap_landmarks_for_pose"]),
        min_post_bootstrap_observations_for_pose=int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
        enable_local_consistency_filter=False,
        enable_spatial_thinning_filter=False,
    )

    thresholds = [_run_threshold(corrs, K, pnp_cfg, image_shape, threshold_px=v) for v in [8.0, 12.0, 20.0, 40.0]]
    reference_pose = seed.get("last_accepted_pose", None)
    threshold_rows = []
    for row in thresholds:
        out = {k: v for k, v in row.items() if k not in {"R", "t"}}
        out["delta_from_last_accepted_pose"] = _pose_delta(reference_pose, row)
        threshold_rows.append(out)

    landmark_subgroups = None
    ransac_candidate_path = None
    pose_40 = next((row for row in thresholds if float(row["threshold_px"]) == 40.0 and bool(row["ok"])), None)
    if int(frame_index) == 14 and pose_40 is not None:
        R40 = np.asarray(pose_40["R"], dtype=np.float64)
        t40 = np.asarray(pose_40["t"], dtype=np.float64).reshape(3)
        ransac_candidate_path = {
            "twenty_px_replay": _replay_ransac_candidate_path(corrs, K, pnp_cfg, image_shape, threshold_px=20.0),
            "forty_px_pose_rescored_at_20px": _rescore_pose(
                corrs,
                K,
                image_shape,
                R40,
                t40,
                threshold_px=20.0,
                eps=float(pnp_cfg["eps"]),
            ),
            "forty_px_pose_at_40px": _rescore_pose(
                corrs,
                K,
                image_shape,
                R40,
                t40,
                threshold_px=40.0,
                eps=float(pnp_cfg["eps"]),
            ),
        }

        mapped_records = []
        for i in np.flatnonzero(mapped & valid_landmark & valid_xw):
            lm_id = int(mapped_lm_ids[int(i)])
            lm = lm_by_id.get(lm_id, {})
            X_i = np.asarray(mapped_xw[int(i), :], dtype=np.float64).reshape(3, 1)
            x_i = np.asarray(xy_cur[int(i), :2], dtype=np.float64).reshape(2, 1)
            X_c_i = world_to_camera_points(R40, t40, X_i)
            d_sq_i = np.asarray(reprojection_errors_sq(K, R40, t40, X_i, x_i), dtype=np.float64).reshape(-1)
            depth_m = float(X_c_i[2, 0]) if X_c_i.shape[1] == 1 and np.isfinite(X_c_i[2, 0]) else np.nan
            err_px = float(np.sqrt(d_sq_i[0])) if d_sq_i.size == 1 and np.isfinite(d_sq_i[0]) and depth_m > 0.0 else np.inf
            record = {
                "lm_id": lm_id,
                "birth_source": str(lm.get("birth_source", "unknown")) if isinstance(lm, dict) else "unknown",
                "birth_kf": int(lm.get("birth_kf", -1)) if isinstance(lm, dict) else -1,
                "age": int(frame_index) - int(lm.get("birth_kf", -1)) if isinstance(lm, dict) else -1,
                "obs_count": _obs_count(lm) if isinstance(lm, dict) else 0,
                "depth_m": depth_m,
                "error_px_under_40_pose": err_px,
                "xw_norm": float(np.linalg.norm(np.asarray(mapped_xw[int(i), :], dtype=np.float64))),
                "quality": lm.get("quality", {}) if isinstance(lm, dict) else {},
                "xy_cur": np.asarray(xy_cur[int(i), :2], dtype=np.float64),
                "eligible": bool(obs_gate[int(i)]),
            }
            mapped_records.append(record)

        eligible_records = [record for record in mapped_records if bool(record["eligible"])]
        support40_records = [
            record
            for record in eligible_records
            if np.isfinite(record["error_px_under_40_pose"]) and float(record["error_px_under_40_pose"]) <= 40.0
        ]
        support20_under40_records = [
            record
            for record in support40_records
            if float(record["error_px_under_40_pose"]) <= 20.0
        ]
        fail20_under40_records = [
            record
            for record in support40_records
            if float(record["error_px_under_40_pose"]) > 20.0
        ]
        gated_records = [record for record in mapped_records if not bool(record["eligible"])]
        landmark_subgroups = {
            "pose_40_inliers": _summarise_landmark_records(support40_records, image_shape),
            "pose_40_inliers_also_within_20px": _summarise_landmark_records(support20_under40_records, image_shape),
            "pose_40_inliers_fail_20px": _summarise_landmark_records(fail20_under40_records, image_shape),
            "mapped_removed_by_observation_gate": _summarise_landmark_records(gated_records, image_shape),
        }

    return {
        "event": "frame_support_funnel",
        "experiment": str(experiment),
        "frame_index": int(frame_index),
        "frame_id": str(cur_id),
        "timestamp": float(cur_ts),
        "active_keyframe_index": int(keyframe_index),
        "reference_keyframe_index": int(keyframe_index),
        "seed_feats1_size": int(getattr(seed.get("feats1", None), "kps_xy", np.zeros((0, 2))).shape[0]),
        "landmark_id_by_feat1_size": int(landmark_id_by_feat1.size),
        "landmark_id_by_feat1_mapped_count": int(np.sum(landmark_id_by_feat1 >= 0)),
        "seed_landmarks": int(len(seed.get("landmarks", []))),
        "last_accepted_pose_kf": None if not isinstance(reference_pose, dict) else reference_pose.get("kf", None),
        "last_accepted_pose_localisation_only": None if not isinstance(reference_pose, dict) else reference_pose.get("localisation_only", None),
        "track_stats": track_out.get("stats", {}),
        "funnel": {
            "raw_tracked_pairs": int(n_tracks),
            "track_inliers_reported": int(track_out.get("stats", {}).get("n_inliers", 0)),
            "kf_feat_idx_in_range": int(np.sum(in_range)),
            "mapped_by_landmark_id_by_feat1": int(np.sum(mapped)),
            "unmapped_in_range": int(np.sum(in_range & ~mapped)),
            "valid_landmark": int(np.sum(valid_landmark)),
            "valid_X_w": int(np.sum(valid_xw)),
            "passes_observation_count_rules": int(np.sum(obs_gate)),
            "final_pnp_eligible_correspondences": int(corrs.X_w.shape[1]),
            "corr_stats": corr_stats,
            "obs_gate_rejected_bootstrap_born": int(np.sum(valid_xw & bootstrap_born & ~obs_gate)),
            "obs_gate_rejected_post_bootstrap_born": int(np.sum(valid_xw & post_bootstrap_born & ~obs_gate)),
        },
        "spatial": {
            "mapped_current": _spatial_summary(xy_cur, mapped, image_shape),
            "unmapped_current": _spatial_summary(xy_cur, in_range & ~mapped, image_shape),
            "eligible_current": _spatial_summary(xy_cur, obs_gate, image_shape),
            "mapped_keyframe": _spatial_summary(xy_kf, mapped, image_shape),
            "unmapped_keyframe": _spatial_summary(xy_kf, in_range & ~mapped, image_shape),
            "eligible_keyframe": _spatial_summary(xy_kf, obs_gate, image_shape),
        },
        "thresholds_on_fixed_eligible_set": threshold_rows,
        "landmark_subgroups": landmark_subgroups,
        "ransac_candidate_path": ransac_candidate_path,
    }


# Run the frontend up to the requested target frames
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--refresh_frame", type=int, default=None)
    parser.add_argument("--append_existing_on_rescue", action="store_true")
    parser.add_argument("--append_existing_on_rescue_from", type=int, default=0)
    parser.add_argument("--rescued_lookup_promotion_frame", type=int, default=None)
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)
    dataset_cfg = cfg["dataset"]
    dataset_root = (ROOT / dataset_cfg["root"]).resolve()
    seq = load_eth3d_sequence(
        dataset_root,
        str(dataset_cfg["seq"]),
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

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
    refresh_frame = None if args.refresh_frame is None else int(args.refresh_frame)
    rescued_lookup_promotion_frame = (
        None if args.rescued_lookup_promotion_frame is None else int(args.rescued_lookup_promotion_frame)
    )
    experiment = "baseline"
    if refresh_frame is not None:
        experiment = f"diagnostic_refresh_after_frame_{refresh_frame}"
    if rescued_lookup_promotion_frame is not None:
        experiment = f"rescued_lookup_promotion_after_frame_{rescued_lookup_promotion_frame}"
    append_existing_on_rescue_from = int(args.append_existing_on_rescue_from)
    if bool(args.append_existing_on_rescue):
        suffix = f"append_existing_on_rescue_from_{append_existing_on_rescue_from}"
        experiment = suffix if experiment == "baseline" else f"{experiment}_{suffix}"

    for frame_index in range(2, 15):
        if frame_index in (13, 14):
            print(
                json.dumps(
                    _analyse_frame(
                        seed,
                        keyframe_feats,
                        keyframe_index,
                        seq,
                        K,
                        frontend_kwargs,
                        frame_index,
                        experiment=experiment,
                    ),
                    sort_keys=True,
                )
            )

        cur_im, _, _ = seq.get(frame_index)
        seed_before = seed
        seed_landmarks_before = _seed_landmark_count(seed)
        out = process_frame_against_seed(
            K,
            seed,
            keyframe_feats,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            keyframe_kf=keyframe_index,
            current_kf=frame_index,
            **frontend_kwargs["pnp_frontend_kwargs"],
        )
        pipeline_standard = _standard_frame_stats(
            frame_index=frame_index,
            reference_keyframe_index=keyframe_index,
            frontend_out=out,
            seed_after=out.get("seed", {}),
            seed_landmarks_before=seed_landmarks_before,
        )
        print(
            json.dumps(
                {
                    "event": "pipeline_step",
                    "experiment": str(experiment),
                    **pipeline_standard,
                    "ok": bool(pipeline_standard["pipeline_ok"]),
                    "reason": pipeline_standard["pipeline_reason"],
                    "active_keyframe_before": int(keyframe_index),
                    "keyframe_promoted": bool(pipeline_standard["pipeline_keyframe_promoted"]),
                },
                sort_keys=True,
            )
        )
        seed = out["seed"]
        if (
            bool(args.append_existing_on_rescue)
            and int(frame_index) >= int(append_existing_on_rescue_from)
            and bool(out.get("ok", False))
            and bool(out.get("stats", {}).get("localisation_only_rescue_frame", False))
        ):
            seed, append_stats = _diagnostic_append_existing_observations_no_prune(
                seed,
                out["pose_out"],
                current_kf=frame_index,
            )
            print(
                json.dumps(
                    {
                        "event": "diagnostic_append_existing_on_rescue",
                        "experiment": str(experiment),
                        "frame_index": int(frame_index),
                        "active_keyframe": int(keyframe_index),
                        "n_append_total": int(append_stats.get("n_added", 0)),
                        "n_append_pnp_inliers_added": int(append_stats.get("n_added", 0)),
                        "n_append_extra_reproj_added": 0,
                        "n_landmarks_with_obs_current_kf_after_append": int(
                            append_stats.get("n_landmarks_with_obs_current_kf_after_append", 0)
                        ),
                        "n_stale_map_growth_removed": 0,
                        "seed_landmarks_after_append": int(len(seed.get("landmarks", []))),
                    },
                    sort_keys=True,
                )
            )
        if bool(out.get("stats", {}).get("keyframe_promoted", False)):
            keyframe_feats = out["track_out"]["cur_feats"]
            keyframe_index = int(frame_index)
        elif (
            rescued_lookup_promotion_frame is not None
            and int(frame_index) == int(rescued_lookup_promotion_frame)
            and bool(out.get("ok", False))
            and bool(out.get("stats", {}).get("localisation_only_rescue_frame", False))
        ):
            seed, keyframe_feats, promotion_stats = _diagnostic_promote_rescued_lookup_basis(
                seed,
                out["track_out"],
                out["pose_out"],
                int(frame_index),
            )
            keyframe_index = int(frame_index)
            print(
                json.dumps(
                    {
                        "event": "diagnostic_rescued_lookup_promotion",
                        "experiment": str(experiment),
                        "frame_index": int(frame_index),
                        "old_active_keyframe": int(seed_before.get("keyframe_kf", keyframe_index)),
                        "new_active_keyframe": int(keyframe_index),
                        "stats": promotion_stats,
                    },
                    sort_keys=True,
                )
            )
        elif refresh_frame is not None and int(frame_index) == int(refresh_frame) and bool(out.get("ok", False)):
            seed, keyframe_feats, refresh_stats = _diagnostic_refresh_keyframe(
                seed,
                out["track_out"],
                out["pose_out"],
                int(frame_index),
            )
            keyframe_index = int(frame_index)
            print(
                json.dumps(
                    {
                        "event": "diagnostic_keyframe_refresh",
                        "experiment": str(experiment),
                        "frame_index": int(frame_index),
                        "old_active_keyframe": int(seed_before.get("keyframe_kf", keyframe_index)),
                        "new_active_keyframe": int(keyframe_index),
                        "stats": refresh_stats,
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
