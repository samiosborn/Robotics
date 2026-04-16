# scripts/diag_pnp_eth3d.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from frontend_eth3d_common import ROOT, add_pnp_threshold_stability_args as _add_pnp_threshold_stability_args, append_jsonl as _append_jsonl, apply_pnp_threshold_stability_cli_overrides as _apply_pnp_threshold_stability_cli_overrides, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_pil_greyscale as _load_pil_greyscale, load_runtime_cfg as _load_runtime_cfg

from core.checks import align_bool_mask_1d, check_dir, check_finite_scalar, check_in_01, check_int_ge0, check_int_gt0, check_positive
from datasets.eth3d import load_eth3d_sequence
from geometry.camera import camera_centre, reprojection_errors_sq, reprojection_rmse, world_to_camera_points
from geometry.pose import angle_between_translations
from geometry.pnp import estimate_pose_pnp_ransac, pnp_current_image_spatial_thinning_mask, pnp_local_displacement_consistency_mask, pnp_threshold_stability_flags
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.pnp_frontend import estimate_pose_from_seed, pnp_frontend_kwargs_from_cfg, pnp_support_diagnostic_stats, pnp_support_gate_stats
from slam.tracking import track_against_keyframe


# Colours for the diagnostic-only frame-4 spatial groups
_FRAME4_SPATIAL_GROUPS = {
    "pnp_8px_inliers": {
        "label": "8 px PnP inliers",
        "colour": (30, 220, 80),
        "alpha": 235,
        "width": 2,
    },
    "pnp_12px_only_inliers": {
        "label": "12 px-only PnP inliers",
        "colour": (255, 50, 190),
        "alpha": 235,
        "width": 2,
    },
    "rejected_by_both": {
        "label": "Rejected by both",
        "colour": (0, 170, 255),
        "alpha": 105,
        "width": 1,
    },
}


# Build the diagnostic PnP solver settings from the shared frontend config parser
def _pnp_solver_cfg(profile_pnp_cfg: dict | None = None) -> dict:
    source = {
        "sample_size": 8,
        "pnp_threshold_stability_compare_px": 12.0,
    }
    if isinstance(profile_pnp_cfg, dict):
        source.update(profile_pnp_cfg)

    out = pnp_frontend_kwargs_from_cfg(source)
    out["apply_pnp_local_consistency_filter_to_pipeline"] = False
    out["apply_pnp_spatial_thinning_filter_to_pipeline"] = False
    if isinstance(profile_pnp_cfg, dict):
        out["apply_pnp_local_consistency_filter_to_pipeline"] = bool(
            profile_pnp_cfg.get("apply_pnp_local_consistency_filter_to_pipeline", False)
        )
        out["apply_pnp_spatial_thinning_filter_to_pipeline"] = bool(
            profile_pnp_cfg.get("apply_pnp_spatial_thinning_filter_to_pipeline", False)
        )

    return out


# Validate the threshold sweep list
def _parse_thresholds(values: list[float]) -> list[float]:
    if len(values) == 0:
        raise ValueError("Expected at least one threshold")
    return [check_positive(v, name="threshold_px", eps=0.0) for v in values]


# Measure final pose quality for logging
def _pose_metrics(corrs, K: np.ndarray, R, t, *, eps: float) -> dict:
    if R is None or t is None:
        return {
            "reprojection_rmse_px": None,
            "cheirality_ratio": None,
        }

    rmse_px = None
    try:
        rmse_px = float(reprojection_rmse(K, R, t, corrs.X_w, corrs.x_cur))
    except Exception:
        rmse_px = None

    cheirality_ratio = None
    try:
        X_c = world_to_camera_points(R, t, corrs.X_w)
        if int(X_c.shape[1]) > 0:
            cheirality_ratio = float(np.mean(np.asarray(X_c[2, :], dtype=np.float64) > float(eps)))
    except Exception:
        cheirality_ratio = None

    return {
        "reprojection_rmse_px": rmse_px,
        "cheirality_ratio": cheirality_ratio,
    }


# Run one threshold sweep item on a fixed correspondence set
def _run_threshold_diag(corrs, K: np.ndarray, *, threshold_px: float, pnp_cfg: dict, image_shape: tuple[int, int] | None = None) -> dict:
    n_pnp_corr = int(np.asarray(corrs.X_w, dtype=np.float64).shape[1])

    row = {
        "threshold_px": float(threshold_px),
        "n_pnp_corr": int(n_pnp_corr),
        "n_inliers": 0,
        "ok": False,
        "reason": None,
        "reprojection_rmse_px": None,
        "cheirality_ratio": None,
        "n_model_success": 0,
        "refit_requested": bool(pnp_cfg["refit"]),
        "refit_used": False,
        "refine_nonlinear_requested": bool(pnp_cfg["refine_nonlinear"]),
        "refine_converged": None,
        "refine_reason": None,
        "n_inliers_before_refit": 0,
        "refit_changed_inlier_count": False,
        "refit_inlier_delta": 0,
        "pnp_spatial_gate_enabled": bool(pnp_cfg.get("enable_pnp_spatial_gate", True)),
        "pnp_spatial_gate_evaluated": False,
        "pnp_spatial_gate_rejected": False,
        "pnp_spatial_gate_reason": None,
        "pnp_inlier_occupied_cells": 0,
        "pnp_inlier_max_cell_count": 0,
        "pnp_inlier_max_cell_fraction": None,
        "pnp_inlier_bbox_area_fraction": None,
        "pnp_inlier_bbox": None,
        "pnp_component_gate_enabled": bool(pnp_cfg.get("enable_pnp_component_gate", False)),
        "pnp_component_gate_evaluated": False,
        "pnp_component_gate_rejected": False,
        "pnp_component_gate_reason": None,
        "pnp_component_radius_px": float(pnp_cfg.get("pnp_component_radius_px", 80.0)),
        "pnp_inlier_component_count": 0,
        "pnp_inlier_largest_component_size": 0,
        "pnp_inlier_largest_component_fraction": None,
        "pnp_inlier_largest_component_bbox_area_fraction": None,
        "pnp_inlier_largest_component_bbox": None,
        "pnp_inlier_component_sizes": [],
    }

    # Stop early when no correspondence bundle is available
    if n_pnp_corr == 0:
        row["reason"] = "no_pnp_correspondences"
        return row

    # Stop early when RANSAC cannot draw a valid minimal sample
    if n_pnp_corr < int(pnp_cfg["sample_size"]):
        row["reason"] = "too_few_correspondences_for_ransac"
        return row

    try:
        _, _, _, stats_raw = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=int(pnp_cfg["num_trials"]),
            sample_size=int(pnp_cfg["sample_size"]),
            threshold_px=float(threshold_px),
            min_inliers=int(pnp_cfg["min_inliers"]),
            seed=int(pnp_cfg["ransac_seed"]),
            min_points=int(pnp_cfg["min_points"]),
            rank_tol=float(pnp_cfg["rank_tol"]),
            min_cheirality_ratio=float(pnp_cfg["min_cheirality_ratio"]),
            eps=float(pnp_cfg["eps"]),
            refit=False,
            refine_nonlinear=False,
            refine_max_iters=int(pnp_cfg["refine_max_iters"]),
            refine_damping=float(pnp_cfg["refine_damping"]),
            refine_step_tol=float(pnp_cfg["refine_step_tol"]),
            refine_improvement_tol=float(pnp_cfg["refine_improvement_tol"]),
        )
        row["n_inliers_before_refit"] = int(stats_raw.get("n_inliers", 0))
    except Exception as exc:
        row["reason"] = "pnp_ransac_error"
        row["error"] = str(exc)
        return row

    try:
        R, t, pnp_inlier_mask, stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=int(pnp_cfg["num_trials"]),
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
        row["reason"] = "pnp_ransac_error"
        row["error"] = str(exc)
        return row

    refine_stats = stats.get("refine_stats", {}) if isinstance(stats, dict) else {}
    metrics = _pose_metrics(corrs, K, R, t, eps=float(pnp_cfg["eps"]))
    n_inliers = int(stats.get("n_inliers", 0)) if isinstance(stats, dict) else 0
    ok = (R is not None) and (t is not None)
    pnp_inlier_mask = align_bool_mask_1d(pnp_inlier_mask, n_pnp_corr, name="pnp_inlier_mask")

    row.update(
        {
            "n_inliers": int(n_inliers),
            "ok": bool(ok),
            "reason": stats.get("reason", None) if isinstance(stats, dict) else None,
            "reprojection_rmse_px": metrics["reprojection_rmse_px"],
            "cheirality_ratio": metrics["cheirality_ratio"],
            "n_model_success": int(stats.get("n_model_success", 0)) if isinstance(stats, dict) else 0,
            "refit_used": bool(stats.get("refit", False)) if isinstance(stats, dict) else False,
            "refine_converged": None if not isinstance(refine_stats, dict) else refine_stats.get("converged", None),
            "refine_reason": None if not isinstance(refine_stats, dict) else refine_stats.get("reason", None),
            "refit_changed_inlier_count": int(n_inliers) != int(row["n_inliers_before_refit"]),
            "refit_inlier_delta": int(n_inliers) - int(row["n_inliers_before_refit"]),
        }
    )

    # Apply the same post-solver support gates used by the frontend
    if image_shape is not None:
        support_stats = pnp_support_diagnostic_stats(
            corrs,
            pnp_inlier_mask,
            image_shape,
            pnp_spatial_grid_cols=int(pnp_cfg["pnp_spatial_grid_cols"]),
            pnp_spatial_grid_rows=int(pnp_cfg["pnp_spatial_grid_rows"]),
            pnp_component_radius_px=float(pnp_cfg["pnp_component_radius_px"]),
        )
        row.update(support_stats)
        gate_stats = pnp_support_gate_stats(
            bool(ok),
            row,
            enable_pnp_spatial_gate=bool(pnp_cfg.get("enable_pnp_spatial_gate", True)),
            min_pnp_inlier_cells=int(pnp_cfg["min_pnp_inlier_cells"]),
            max_pnp_single_cell_fraction=float(pnp_cfg["max_pnp_single_cell_fraction"]),
            min_pnp_bbox_area_fraction=float(pnp_cfg["min_pnp_bbox_area_fraction"]),
            enable_pnp_component_gate=bool(pnp_cfg.get("enable_pnp_component_gate", False)),
            min_pnp_component_count=int(pnp_cfg["min_pnp_component_count"]),
            max_pnp_largest_component_fraction=float(pnp_cfg["max_pnp_largest_component_fraction"]),
            min_pnp_largest_component_bbox_area_fraction=float(pnp_cfg["min_pnp_largest_component_bbox_area_fraction"]),
        )
        row.update(gate_stats)

        if bool(ok) and bool(row["pnp_component_gate_rejected"]):
            row["ok"] = False
            row["reason"] = "pnp_component_support_failed"
        elif bool(ok) and bool(row["pnp_spatial_gate_rejected"]):
            row["ok"] = False
            row["reason"] = "pnp_spatial_coverage_failed"

    if not bool(ok) and row["reason"] is None:
        row["reason"] = "pnp_pose_missing"

    return row


# Format one concise threshold summary line
def _format_threshold_summary(rows: list[dict]) -> str:
    parts: list[str] = []
    for row in rows:
        status = "ok" if bool(row["ok"]) else "fail"
        parts.append(f"{row['threshold_px']:.0f}px:{int(row['n_inliers'])}/{status}")
    return " ".join(parts)


# Format an optional pixel error for concise terminal output
def _format_optional_px(value) -> str:
    if value is None:
        return "None"

    return f"{float(value):.2f}"


# Summarise a vector of reprojection errors in pixels
def _reprojection_error_summary(errors_px: np.ndarray) -> dict:
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
def _reprojection_error_histogram(errors_px: np.ndarray) -> dict:
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
def _non_pnp_reprojection_block(errors_px: np.ndarray, positive_depth: np.ndarray, non_positive_depth: np.ndarray, mask: np.ndarray) -> dict:
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
        "error_px": _reprojection_error_summary(errors_px[usable]),
        "hist_px": _reprojection_error_histogram(errors_px[usable]),
    }


# Run PnP once for the frame-4 pose comparison
def _run_pnp_pose_for_comparison(corrs, K: np.ndarray, *, threshold_px: float, pnp_cfg: dict, image_shape: tuple[int, int] | None = None) -> dict:
    N = int(np.asarray(corrs.X_w, dtype=np.float64).shape[1])
    out = {
        "threshold_px": float(threshold_px),
        "ok": False,
        "reason": None,
        "R": None,
        "t": None,
        "inlier_mask": np.zeros((N,), dtype=bool),
        "n_inliers": 0,
        "stats": {},
        "pnp_component_gate_enabled": bool(pnp_cfg.get("enable_pnp_component_gate", False)),
        "pnp_component_gate_evaluated": False,
        "pnp_component_gate_rejected": False,
        "pnp_component_gate_reason": None,
        "pnp_component_support": None,
        "pnp_inlier_component_count": 0,
        "pnp_inlier_largest_component_size": 0,
        "pnp_inlier_largest_component_fraction": None,
        "pnp_inlier_largest_component_bbox_area_fraction": None,
        "pnp_inlier_largest_component_bbox": None,
        "pnp_inlier_component_sizes": [],
    }

    # Stop when RANSAC cannot draw a minimal sample
    if N < int(pnp_cfg["sample_size"]):
        out["reason"] = "too_few_correspondences_for_ransac"
        return out

    try:
        R, t, inlier_mask, stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=int(pnp_cfg["num_trials"]),
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

    # Record the fitted pose and inlier mask
    stats = stats if isinstance(stats, dict) else {}
    inlier_mask = align_bool_mask_1d(inlier_mask, N, name="pnp_inlier_mask")
    ok = (R is not None) and (t is not None)

    out.update(
        {
            "ok": bool(ok),
            "reason": stats.get("reason", None),
            "R": None if R is None else np.asarray(R, dtype=np.float64),
            "t": None if t is None else np.asarray(t, dtype=np.float64).reshape(3),
            "inlier_mask": inlier_mask,
            "n_inliers": int(np.sum(inlier_mask)),
            "stats": stats,
        }
    )

    if not bool(ok) and out["reason"] is None:
        out["reason"] = "pnp_pose_missing"

    # Score support without changing the raw pose-comparison status
    if image_shape is not None:
        support_stats = pnp_support_diagnostic_stats(
            corrs,
            inlier_mask,
            image_shape,
            pnp_spatial_grid_cols=int(pnp_cfg["pnp_spatial_grid_cols"]),
            pnp_spatial_grid_rows=int(pnp_cfg["pnp_spatial_grid_rows"]),
            pnp_component_radius_px=float(pnp_cfg["pnp_component_radius_px"]),
        )
        out.update(support_stats)
        gate_stats = pnp_support_gate_stats(
            bool(ok),
            out,
            enable_pnp_spatial_gate=False,
            enable_pnp_component_gate=bool(pnp_cfg.get("enable_pnp_component_gate", False)),
            min_pnp_component_count=int(pnp_cfg["min_pnp_component_count"]),
            max_pnp_largest_component_fraction=float(pnp_cfg["max_pnp_largest_component_fraction"]),
            min_pnp_largest_component_bbox_area_fraction=float(pnp_cfg["min_pnp_largest_component_bbox_area_fraction"]),
        )
        out.update(gate_stats)

    return out


# Run the fixed 8 px and 12 px PnP pair once for frame-4 diagnostics
def _frame4_pnp_8px_12px_pose_pair(corrs, K: np.ndarray, *, pnp_cfg: dict, image_shape: tuple[int, int] | None = None) -> dict:
    return {
        "pose_8px": _run_pnp_pose_for_comparison(corrs, K, threshold_px=8.0, pnp_cfg=pnp_cfg, image_shape=image_shape),
        "pose_12px": _run_pnp_pose_for_comparison(corrs, K, threshold_px=12.0, pnp_cfg=pnp_cfg, image_shape=image_shape),
    }


# Build reusable frame-4 threshold group masks
def _frame4_threshold_group_masks(pose_pair: dict, N: int, *, include_12px_all: bool = False) -> dict[str, np.ndarray]:
    mask_8 = align_bool_mask_1d(pose_pair["pose_8px"]["inlier_mask"], N, name="pnp_8px_inlier_mask")
    mask_12 = align_bool_mask_1d(pose_pair["pose_12px"]["inlier_mask"], N, name="pnp_12px_inlier_mask")
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


# Summarise descriptor match scores for a selected correspondence group
def _score_summary(scores: np.ndarray) -> dict:
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


# Recover the original tracking match score for each PnP correspondence
def _match_scores_for_pnp_correspondences(track_out: dict, kf_feat_idx: np.ndarray, cur_feat_idx: np.ndarray) -> tuple[np.ndarray, str]:
    kf_feat_idx = np.asarray(kf_feat_idx, dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(cur_feat_idx, dtype=np.int64).reshape(-1)
    N = int(kf_feat_idx.size)
    scores = np.full((N,), np.nan, dtype=np.float64)
    semantics = "higher_is_better"

    if kf_feat_idx.size != cur_feat_idx.size:
        return scores, semantics

    matches = track_out.get("matches", None) if isinstance(track_out, dict) else None
    if matches is None:
        return scores, semantics

    ia = np.asarray(getattr(matches, "ia", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    ib = np.asarray(getattr(matches, "ib", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    score = np.asarray(getattr(matches, "score", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    M = min(int(ia.size), int(ib.size), int(score.size))

    if M == 0:
        return scores, semantics

    if np.nanmax(score[:M]) <= 0.0:
        semantics = "higher_is_better_negative_hamming_distance"

    score_by_pair: dict[tuple[int, int], float] = {}
    for j in range(M):
        score_by_pair[(int(ia[j]), int(ib[j]))] = float(score[j])

    for j in range(N):
        scores[j] = score_by_pair.get((int(kf_feat_idx[j]), int(cur_feat_idx[j])), np.nan)

    return scores, semantics


# Build a coarse spatial summary for one set of image points
def _point_spatial_summary(xy: np.ndarray, image_size: tuple[int, int], *, grid_cols: int = 4, grid_rows: int = 3) -> dict:
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

    for p in xy:
        col = int(np.floor((float(p[0]) / max(W, 1.0)) * int(grid_cols)))
        row = int(np.floor((float(p[1]) / max(H, 1.0)) * int(grid_rows)))
        col = int(np.clip(col, 0, int(grid_cols) - 1))
        row = int(np.clip(row, 0, int(grid_rows) - 1))
        grid[row][col] += 1

    occupied_cells = int(sum(1 for row in grid for v in row if int(v) > 0))
    max_cell_count = int(max(max(row) for row in grid))
    max_cell_fraction = float(max_cell_count / count)
    occupied_cell_fraction = float(occupied_cells / max(int(grid_cols) * int(grid_rows), 1))

    border_margin = 0.08 * min(W, H)
    border_mask = (
        (xy[:, 0] <= border_margin)
        | (xy[:, 1] <= border_margin)
        | (xy[:, 0] >= (W - 1.0 - border_margin))
        | (xy[:, 1] >= (H - 1.0 - border_margin))
    )
    border_count = int(np.sum(border_mask))
    border_fraction = float(border_count / count)
    thin_structure_like = bool(bbox_aspect_ratio is not None and bbox_aspect_ratio >= 4.0 and bbox_area_fraction <= 0.35 and count >= 5)
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

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if int(value) > 0:
                cells.add((int(r), int(c)))

    return cells


# Compute IoU between two image-space bounding boxes
def _bbox_iou(box_a, box_b) -> float | None:
    if box_a is None or box_b is None:
        return None

    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter

    if union <= 0.0:
        return None

    return float(inter / union)


# Compute symmetric nearest-neighbour distances between two point sets
def _nearest_distance_summary(xy_a: np.ndarray, xy_b: np.ndarray, image_size: tuple[int, int]) -> dict:
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

    d = xy_a[:, None, :] - xy_b[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    nearest = np.concatenate([np.min(dist, axis=1), np.min(dist, axis=0)])
    diag = float(np.hypot(float(image_size[0]), float(image_size[1])))
    diag = max(diag, 1.0)

    return {
        "count": int(nearest.size),
        "min_px": float(np.min(nearest)),
        "median_px": float(np.median(nearest)),
        "mean_px": float(np.mean(nearest)),
        "p75_px": float(np.percentile(nearest, 75)),
        "median_fraction": float(np.median(nearest) / diag),
    }


# Summarise spatial relationship between two groups on one image side
def _spatial_relationship_side(summary_a: dict, summary_b: dict, xy_a: np.ndarray, xy_b: np.ndarray, image_size: tuple[int, int]) -> dict:
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
        centroid_distance_fraction = float(centroid_distance_px / max(float(np.hypot(float(image_size[0]), float(image_size[1]))), 1.0))

    return {
        "bbox_iou": _bbox_iou(summary_a.get("bbox", None), summary_b.get("bbox", None)),
        "centroid_distance_px": centroid_distance_px,
        "centroid_distance_fraction": centroid_distance_fraction,
        "shared_occupied_cells": int(len(cell_overlap)),
        "occupied_cell_union": int(len(cell_union)),
        "grid_overlap_fraction": None if len(cell_union) == 0 else float(len(cell_overlap) / len(cell_union)),
        "nearest_distance": _nearest_distance_summary(xy_a, xy_b, image_size),
    }


# Classify the 8 px versus 12 px-only spatial relationship
def _classify_spatial_relationship(reference: dict, current: dict) -> str:
    def separated(side: dict) -> bool:
        bbox_iou = side.get("bbox_iou", None)
        centroid_fraction = side.get("centroid_distance_fraction", None)
        nearest = side.get("nearest_distance", {})
        nearest_fraction = nearest.get("median_fraction", None) if isinstance(nearest, dict) else None

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
        nearest_fraction = nearest.get("median_fraction", None) if isinstance(nearest, dict) else None

        return (
            (bbox_iou is not None and float(bbox_iou) >= 0.20)
            or (grid_overlap is not None and nearest_fraction is not None and float(grid_overlap) >= 0.40 and float(nearest_fraction) <= 0.04)
            or (nearest_fraction is not None and float(nearest_fraction) <= 0.03)
        )

    ref_separated = separated(reference)
    cur_separated = separated(current)
    ref_interleaved = interleaved(reference)
    cur_interleaved = interleaved(current)

    if ref_separated and cur_separated:
        return "spatially_separated"
    if ref_interleaved and cur_interleaved:
        return "spatially_interleaved"
    if ref_separated or cur_separated:
        return "ambiguous_mixed_separation"
    if ref_interleaved or cur_interleaved:
        return "ambiguous_mixed_interleaving"

    return "ambiguous"


# Draw diagnostic correspondences in colour on the image pair
def _draw_frame4_spatial_groups(img_ref, img_cur, kps_ref: np.ndarray, kps_cur: np.ndarray, ia: np.ndarray, ib: np.ndarray, groups: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    out_path = Path(out_path)
    A = img_ref.convert("RGB")
    B = img_cur.convert("RGB")
    WA, HA = A.size
    WB, HB = B.size
    canvas = Image.new("RGB", (WA + WB, max(HA, HB)))
    canvas.paste(A, (0, 0))
    canvas.paste(B, (WA, 0))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    kps_ref = np.asarray(kps_ref, dtype=np.float64)
    kps_cur = np.asarray(kps_cur, dtype=np.float64)
    ia = np.asarray(ia, dtype=np.int64).reshape(-1)
    ib = np.asarray(ib, dtype=np.int64).reshape(-1)
    N = min(int(ia.size), int(ib.size))

    for group_name, mask in groups:
        spec = _FRAME4_SPATIAL_GROUPS[group_name]
        mask = align_bool_mask_1d(mask, N, name=f"{group_name}_mask")
        colour = tuple(int(v) for v in spec["colour"])
        alpha = int(spec["alpha"])
        line_colour = (colour[0], colour[1], colour[2], alpha)
        point_colour = (colour[0], colour[1], colour[2], min(255, alpha + 20))
        width = int(spec["width"])
        rr = 4 if width >= 2 else 3

        for idx in np.nonzero(mask)[0]:
            i = int(ia[idx])
            j = int(ib[idx])
            if i < 0 or i >= int(kps_ref.shape[0]) or j < 0 or j >= int(kps_cur.shape[0]):
                continue

            x0 = float(kps_ref[i, 0])
            y0 = float(kps_ref[i, 1])
            x1 = float(kps_cur[j, 0]) + float(WA)
            y1 = float(kps_cur[j, 1])
            draw.line((x0, y0, x1, y1), fill=line_colour, width=width)
            draw.ellipse((x0 - rr, y0 - rr, x0 + rr, y0 + rr), fill=point_colour)
            draw.ellipse((x1 - rr, y1 - rr, x1 + rr, y1 + rr), fill=point_colour)

    composed = Image.alpha_composite(canvas.convert("RGBA"), overlay)
    legend = ImageDraw.Draw(composed)
    legend_h = 10 + 20 * len(groups)
    legend.rectangle((6, 6, 360, legend_h), fill=(0, 0, 0, 155))

    y = 12
    for group_name, mask in groups:
        spec = _FRAME4_SPATIAL_GROUPS[group_name]
        colour = tuple(int(v) for v in spec["colour"])
        n_group = int(np.sum(align_bool_mask_1d(mask, N, name=f"{group_name}_legend_mask")))
        legend.rectangle((12, y + 2, 24, y + 14), fill=(colour[0], colour[1], colour[2], 255))
        legend.text((30, y), f"{spec['label']}  n={n_group}", fill=(255, 255, 255, 255))
        y += 20

    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(str(out_path))


# Format a nullable floating-point value
def _format_optional_float(value, *, digits: int = 2) -> str:
    if value is None:
        return "None"

    return f"{float(value):.{int(digits)}f}"


# Format a nullable image-space point list
def _format_optional_point(value) -> str:
    if value is None:
        return "None"

    return "[" + ",".join(f"{float(v):.1f}" for v in value) + "]"


# Format the threshold-stability diagnostic line
def _format_pnp_threshold_stability_diag(stability: dict | None) -> str:
    if not isinstance(stability, dict):
        return "  pnp_threshold_stability: unavailable reason=missing_diag"

    reasons = stability.get("instability_reasons", [])
    if isinstance(reasons, list):
        reason_text = ",".join(str(v) for v in reasons)
    else:
        reason_text = str(reasons)
    if reason_text == "":
        reason_text = "None"

    return (
        f"  pnp_threshold_stability: class={stability.get('classification', 'unavailable')} "
        f"thresholds={_format_optional_float(stability.get('ref_threshold_px', None), digits=0)}"
        f"->{_format_optional_float(stability.get('compare_threshold_px', None), digits=0)} "
        f"ref_ok={bool(stability.get('ref_pose_ok', False))} "
        f"cmp_ok={bool(stability.get('compare_pose_ok', False))} "
        f"n_ref={int(stability.get('ref_inliers', 0))} "
        f"n_cmp={int(stability.get('compare_inliers', 0))} "
        f"iou={_format_optional_float(stability.get('support_iou', None), digits=3)} "
        f"rot_deg={_format_optional_float(stability.get('rotation_delta_deg', None), digits=2)} "
        f"t_dir_deg={_format_optional_float(stability.get('translation_direction_delta_deg', None), digits=2)} "
        f"C_dir_deg={_format_optional_float(stability.get('camera_centre_direction_delta_deg', None), digits=2)} "
        f"looser_only={bool(stability.get('one_solution_only_at_looser_threshold', False))} "
        f"disjoint={bool(stability.get('supports_effectively_disjoint', False))} "
        f"reasons={reason_text}"
    )


# Format one frame-4 spatial diagnostic group line
def _format_frame4_spatial_group(group_name: str, group: dict) -> str:
    current = group["current"]
    reference = group["reference"]
    score = group["match_score"]

    return (
        f"  frame4_spatial_group: group={group_name} "
        f"count={int(group['count'])} "
        f"cur_bbox={_format_optional_point(current['bbox'])} "
        f"cur_area_frac={_format_optional_float(current['bbox_area_fraction'], digits=3)} "
        f"cur_centroid={_format_optional_point(current['centroid'])} "
        f"cur_grid={current['occupancy_grid']} "
        f"cur_max_cell_frac={_format_optional_float(current['max_cell_fraction'], digits=2)} "
        f"cur_concentrated={bool(current['heavily_concentrated'])} "
        f"cur_border_frac={_format_optional_float(current['border_fraction'], digits=2)} "
        f"ref_bbox={_format_optional_point(reference['bbox'])} "
        f"ref_area_frac={_format_optional_float(reference['bbox_area_fraction'], digits=3)} "
        f"ref_centroid={_format_optional_point(reference['centroid'])} "
        f"ref_grid={reference['occupancy_grid']} "
        f"ref_max_cell_frac={_format_optional_float(reference['max_cell_fraction'], digits=2)} "
        f"ref_concentrated={bool(reference['heavily_concentrated'])} "
        f"score_mean={_format_optional_float(score['mean'], digits=2)} "
        f"score_median={_format_optional_float(score['median'], digits=2)}"
    )


# Format frame-4 spatial diagnostic terminal lines
def _format_frame4_spatial_diag(diag: dict) -> list[str]:
    if not isinstance(diag, dict) or not bool(diag.get("ok", False)):
        return [f"  frame4_spatial_diag: unavailable reason={diag.get('reason', None) if isinstance(diag, dict) else None}"]

    paths = diag["visualisation_paths"]
    relationship = diag["relationship_8px_vs_12px_only"]
    cur = relationship["current"]
    ref = relationship["reference"]

    lines = [
        (
            f"  frame4_spatial_paths: combined={paths['combined']} "
            f"group8={paths['pnp_8px_inliers']} "
            f"group12only={paths['pnp_12px_only_inliers']} "
            f"rejected={paths['rejected_by_both']}"
        )
    ]

    for group_name in ["pnp_8px_inliers", "pnp_12px_only_inliers", "rejected_by_both"]:
        lines.append(_format_frame4_spatial_group(group_name, diag["groups"][group_name]))

    lines.append(
        f"  frame4_spatial_relationship: classification={relationship['classification']} "
        f"cur_bbox_iou={_format_optional_float(cur['bbox_iou'], digits=3)} "
        f"cur_centroid_dist_frac={_format_optional_float(cur['centroid_distance_fraction'], digits=3)} "
        f"cur_nn_median_px={_format_optional_float(cur['nearest_distance']['median_px'], digits=2)} "
        f"cur_grid_overlap={_format_optional_float(cur['grid_overlap_fraction'], digits=2)} "
        f"ref_bbox_iou={_format_optional_float(ref['bbox_iou'], digits=3)} "
        f"ref_centroid_dist_frac={_format_optional_float(ref['centroid_distance_fraction'], digits=3)} "
        f"ref_nn_median_px={_format_optional_float(ref['nearest_distance']['median_px'], digits=2)} "
        f"ref_grid_overlap={_format_optional_float(ref['grid_overlap_fraction'], digits=2)} "
        f"score_semantics={diag['match_score_semantics']}"
    )

    return lines


# Build and export the frame-4 spatial diagnostic
def _build_frame4_pnp_spatial_diag(seq, ref_keyframe_index: int, frame_index: int, ref_keyframe_feats, track_out: dict, corrs, out_dir: Path, *, pose_pair: dict, name_suffix: str = "") -> dict:
    N = int(np.asarray(corrs.X_w, dtype=np.float64).shape[1])

    out = {
        "ok": False,
        "reason": None,
        "frame_index": int(frame_index),
        "reference_keyframe_index": int(ref_keyframe_index),
        "n_pose_eligible": int(N),
        "visualisation_paths": {},
        "groups": {},
        "relationship_8px_vs_12px_only": {},
        "match_score_semantics": "higher_is_better",
    }

    if N == 0:
        out["reason"] = "no_pnp_correspondences"
        return out

    img_ref = _load_pil_greyscale(seq.frame_info(ref_keyframe_index).path)
    img_cur = _load_pil_greyscale(seq.frame_info(frame_index).path)

    cur_feats = track_out.get("cur_feats", None) if isinstance(track_out, dict) else None
    if cur_feats is None:
        out["reason"] = "missing_current_features"
        return out

    kps_ref = np.asarray(ref_keyframe_feats.kps_xy, dtype=np.float64)
    kps_cur = np.asarray(cur_feats.kps_xy, dtype=np.float64)
    ia = np.asarray(corrs.kf_feat_idx, dtype=np.int64).reshape(-1)
    ib = np.asarray(corrs.cur_feat_idx, dtype=np.int64).reshape(-1)

    if int(ia.size) != N or int(ib.size) != N:
        out["reason"] = "correspondence_feature_index_length_mismatch"
        return out

    group_masks = _frame4_threshold_group_masks(pose_pair, N)
    mask_8 = group_masks["pnp_8px_inliers"]
    mask_12_only = group_masks["pnp_12px_only_inliers"]

    suffix = "" if str(name_suffix).strip() == "" else f"_{str(name_suffix).strip()}"
    combined_path = out_dir / f"frame4_pnp_spatial{suffix}_groups_combined.png"
    group_paths = {
        "combined": str(combined_path),
        "pnp_8px_inliers": str(out_dir / f"frame4_pnp_spatial{suffix}_8px_inliers.png"),
        "pnp_12px_only_inliers": str(out_dir / f"frame4_pnp_spatial{suffix}_12px_only_inliers.png"),
        "rejected_by_both": str(out_dir / f"frame4_pnp_spatial{suffix}_rejected_by_both.png"),
    }

    draw_order = [
        ("rejected_by_both", group_masks["rejected_by_both"]),
        ("pnp_12px_only_inliers", group_masks["pnp_12px_only_inliers"]),
        ("pnp_8px_inliers", group_masks["pnp_8px_inliers"]),
    ]
    _draw_frame4_spatial_groups(img_ref, img_cur, kps_ref, kps_cur, ia, ib, draw_order, combined_path)

    for group_name in ["pnp_8px_inliers", "pnp_12px_only_inliers", "rejected_by_both"]:
        _draw_frame4_spatial_groups(
            img_ref,
            img_cur,
            kps_ref,
            kps_cur,
            ia,
            ib,
            [(group_name, group_masks[group_name])],
            Path(group_paths[group_name]),
        )

    xy_ref = np.full((N, 2), np.nan, dtype=np.float64)
    xy_cur = np.full((N, 2), np.nan, dtype=np.float64)
    valid = (ia >= 0) & (ia < int(kps_ref.shape[0])) & (ib >= 0) & (ib < int(kps_cur.shape[0]))
    xy_ref[valid] = kps_ref[ia[valid], :2]
    xy_cur[valid] = kps_cur[ib[valid], :2]

    match_scores, score_semantics = _match_scores_for_pnp_correspondences(track_out, ia, ib)

    groups_out = {}
    for group_name, mask in group_masks.items():
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        groups_out[group_name] = {
            "label": str(_FRAME4_SPATIAL_GROUPS[group_name]["label"]),
            "count": int(np.sum(mask)),
            "reference": _point_spatial_summary(xy_ref[mask], img_ref.size),
            "current": _point_spatial_summary(xy_cur[mask], img_cur.size),
            "match_score": _score_summary(match_scores[mask]),
        }

    ref_relationship = _spatial_relationship_side(
        groups_out["pnp_8px_inliers"]["reference"],
        groups_out["pnp_12px_only_inliers"]["reference"],
        xy_ref[mask_8],
        xy_ref[mask_12_only],
        img_ref.size,
    )
    cur_relationship = _spatial_relationship_side(
        groups_out["pnp_8px_inliers"]["current"],
        groups_out["pnp_12px_only_inliers"]["current"],
        xy_cur[mask_8],
        xy_cur[mask_12_only],
        img_cur.size,
    )

    relationship = {
        "classification": _classify_spatial_relationship(ref_relationship, cur_relationship),
        "reference": ref_relationship,
        "current": cur_relationship,
    }

    out.update(
        {
            "ok": True,
            "reason": None,
            "visualisation_paths": group_paths,
            "groups": groups_out,
            "relationship_8px_vs_12px_only": relationship,
            "match_score_semantics": score_semantics,
        }
    )

    return out


# Build a frame-4 diagnostic for the local displacement-consistency filter
def _build_frame4_local_consistency_diag(seq, ref_keyframe_index: int, frame_index: int, ref_keyframe_feats, corrs, *, pose_pair: dict, pnp_cfg: dict) -> dict:
    N = int(np.asarray(corrs.X_w, dtype=np.float64).shape[1])

    out = {
        "ok": False,
        "reason": None,
        "frame_index": int(frame_index),
        "reference_keyframe_index": int(ref_keyframe_index),
        "n_pose_eligible_before": int(N),
        "n_pose_eligible_after": 0,
        "n_removed": 0,
        "filter_stats": {},
        "removed_by_group": {},
        "removed_spatial": {},
        "kept_spatial": {},
    }

    if N == 0:
        out["reason"] = "no_pnp_correspondences"
        return out

    img_ref = _load_pil_greyscale(seq.frame_info(ref_keyframe_index).path)
    img_cur = _load_pil_greyscale(seq.frame_info(frame_index).path)

    kps_ref = np.asarray(ref_keyframe_feats.kps_xy, dtype=np.float64)
    ia = np.asarray(corrs.kf_feat_idx, dtype=np.int64).reshape(-1)
    xy_cur = np.asarray(corrs.x_cur, dtype=np.float64).T

    if int(ia.size) != N or int(xy_cur.shape[0]) != N:
        out["reason"] = "correspondence_length_mismatch"
        return out

    valid = (ia >= 0) & (ia < int(kps_ref.shape[0]))
    if not np.all(valid):
        out["reason"] = "invalid_keyframe_feature_index"
        return out

    xy_ref = np.asarray(kps_ref[ia, :2], dtype=np.float64)

    keep_mask, filter_stats = pnp_local_displacement_consistency_mask(
        xy_ref,
        xy_cur,
        radius_px=float(pnp_cfg["pnp_local_consistency_radius_px"]),
        min_neighbours=int(pnp_cfg["pnp_local_consistency_min_neighbours"]),
        max_median_residual_px=float(pnp_cfg["pnp_local_consistency_max_median_residual_px"]),
        min_keep=int(pnp_cfg["pnp_local_consistency_min_keep"]),
    )
    keep_mask = align_bool_mask_1d(keep_mask, N, name="local_consistency_keep_mask")
    removed_mask = ~keep_mask

    group_masks = _frame4_threshold_group_masks(pose_pair, N)

    removed_by_group = {}
    for group_name, group_mask in group_masks.items():
        group_count = int(np.sum(group_mask))
        removed_count = int(np.sum(removed_mask & group_mask))
        removed_by_group[group_name] = {
            "count": int(group_count),
            "removed": int(removed_count),
            "removed_fraction": None if group_count == 0 else float(removed_count / group_count),
        }

    out.update(
        {
            "ok": True,
            "reason": None,
            "n_pose_eligible_after": int(np.sum(keep_mask)),
            "n_removed": int(np.sum(removed_mask)),
            "filter_stats": filter_stats,
            "removed_by_group": removed_by_group,
            "removed_spatial": {
                "reference": _point_spatial_summary(xy_ref[removed_mask], img_ref.size),
                "current": _point_spatial_summary(xy_cur[removed_mask], img_cur.size),
            },
            "kept_spatial": {
                "reference": _point_spatial_summary(xy_ref[keep_mask], img_ref.size),
                "current": _point_spatial_summary(xy_cur[keep_mask], img_cur.size),
            },
        }
    )

    return out


# Format the frame-4 local consistency diagnostic
def _format_frame4_local_consistency_diag(diag: dict) -> list[str]:
    if not isinstance(diag, dict) or not bool(diag.get("ok", False)):
        return [f"  frame4_local_consistency: unavailable reason={diag.get('reason', None) if isinstance(diag, dict) else None}"]

    stats = diag["filter_stats"]
    removed = diag["removed_by_group"]
    rem_cur = diag["removed_spatial"]["current"]

    line_filter = (
        f"  frame4_local_consistency: before={int(diag['n_pose_eligible_before'])} "
        f"after={int(diag['n_pose_eligible_after'])} "
        f"removed={int(diag['n_removed'])} "
        f"too_few_neighbours={int(stats.get('n_too_few_neighbours', 0))} "
        f"motion_inconsistent={int(stats.get('n_motion_inconsistent', 0))} "
        f"resid_med={_format_optional_float(stats.get('residual_median_px', None), digits=2)} "
        f"resid_p90={_format_optional_float(stats.get('residual_p90_px', None), digits=2)}"
    )

    line_groups = (
        f"  frame4_local_consistency_removed_groups: "
        f"pnp8={int(removed['pnp_8px_inliers']['removed'])}/{int(removed['pnp_8px_inliers']['count'])} "
        f"pnp12only={int(removed['pnp_12px_only_inliers']['removed'])}/{int(removed['pnp_12px_only_inliers']['count'])} "
        f"rejected={int(removed['rejected_by_both']['removed'])}/{int(removed['rejected_by_both']['count'])}"
    )

    line_spatial = (
        f"  frame4_local_consistency_removed_spatial: "
        f"cur_bbox={_format_optional_point(rem_cur['bbox'])} "
        f"cur_area_frac={_format_optional_float(rem_cur['bbox_area_fraction'], digits=3)} "
        f"cur_grid={rem_cur['occupancy_grid']} "
        f"cur_max_cell_frac={_format_optional_float(rem_cur['max_cell_fraction'], digits=2)}"
    )

    return [line_filter, line_groups, line_spatial]


# Build a frame-4 diagnostic for current-image spatial thinning
def _build_frame4_spatial_thinning_diag(seq, ref_keyframe_index: int, frame_index: int, ref_keyframe_feats, corrs, *, pose_pair: dict, pnp_cfg: dict) -> dict:
    N = int(np.asarray(corrs.X_w, dtype=np.float64).shape[1])

    out = {
        "ok": False,
        "reason": None,
        "frame_index": int(frame_index),
        "reference_keyframe_index": int(ref_keyframe_index),
        "n_pose_eligible_before": int(N),
        "n_pose_eligible_after": 0,
        "n_removed": 0,
        "filter_stats": {},
        "removed_by_group": {},
        "removed_spatial": {},
        "kept_spatial": {},
    }

    if N == 0:
        out["reason"] = "no_pnp_correspondences"
        return out

    img_ref = _load_pil_greyscale(seq.frame_info(ref_keyframe_index).path)
    img_cur = _load_pil_greyscale(seq.frame_info(frame_index).path)

    kps_ref = np.asarray(ref_keyframe_feats.kps_xy, dtype=np.float64)
    ia = np.asarray(corrs.kf_feat_idx, dtype=np.int64).reshape(-1)
    xy_cur = np.asarray(corrs.x_cur, dtype=np.float64).T

    if int(ia.size) != N or int(xy_cur.shape[0]) != N:
        out["reason"] = "correspondence_length_mismatch"
        return out

    valid = (ia >= 0) & (ia < int(kps_ref.shape[0]))
    if not np.all(valid):
        out["reason"] = "invalid_keyframe_feature_index"
        return out

    xy_ref = np.asarray(kps_ref[ia, :2], dtype=np.float64)

    keep_mask, filter_stats = pnp_current_image_spatial_thinning_mask(
        xy_cur,
        radius_px=float(pnp_cfg["pnp_spatial_thinning_radius_px"]),
        max_points_per_radius=int(pnp_cfg["pnp_spatial_thinning_max_points_per_radius"]),
        min_keep=int(pnp_cfg["pnp_spatial_thinning_min_keep"]),
    )
    keep_mask = align_bool_mask_1d(keep_mask, N, name="spatial_thinning_keep_mask")
    removed_mask = ~keep_mask

    group_masks = _frame4_threshold_group_masks(pose_pair, N, include_12px_all=True)

    removed_by_group = {}
    for group_name, group_mask in group_masks.items():
        group_count = int(np.sum(group_mask))
        removed_count = int(np.sum(removed_mask & group_mask))
        removed_by_group[group_name] = {
            "count": int(group_count),
            "removed": int(removed_count),
            "removed_fraction": None if group_count == 0 else float(removed_count / group_count),
        }

    out.update(
        {
            "ok": True,
            "reason": None,
            "n_pose_eligible_after": int(np.sum(keep_mask)),
            "n_removed": int(np.sum(removed_mask)),
            "filter_stats": filter_stats,
            "removed_by_group": removed_by_group,
            "removed_spatial": {
                "reference": _point_spatial_summary(xy_ref[removed_mask], img_ref.size),
                "current": _point_spatial_summary(xy_cur[removed_mask], img_cur.size),
            },
            "kept_spatial": {
                "reference": _point_spatial_summary(xy_ref[keep_mask], img_ref.size),
                "current": _point_spatial_summary(xy_cur[keep_mask], img_cur.size),
            },
        }
    )

    return out


# Format the frame-4 current-image spatial thinning diagnostic
def _format_frame4_spatial_thinning_diag(diag: dict) -> list[str]:
    if not isinstance(diag, dict) or not bool(diag.get("ok", False)):
        return [f"  frame4_spatial_thinning: unavailable reason={diag.get('reason', None) if isinstance(diag, dict) else None}"]

    stats = diag["filter_stats"]
    before = stats.get("before", {})
    after = stats.get("after", {})
    removed = diag["removed_by_group"]
    rem_cur = diag["removed_spatial"]["current"]
    kept_cur = diag["kept_spatial"]["current"]

    line_filter = (
        f"  frame4_spatial_thinning: before={int(diag['n_pose_eligible_before'])} "
        f"after={int(diag['n_pose_eligible_after'])} "
        f"removed={int(diag['n_removed'])} "
        f"radius={float(stats.get('radius_px', 0.0)):.1f} "
        f"cap={int(stats.get('max_points_per_radius', 0))} "
        f"dense_before={int(before.get('n_dense_points', 0))} "
        f"dense_after={int(after.get('n_dense_points', 0))} "
        f"density_max_before={int(before.get('density_count_max', 0) or 0)} "
        f"density_max_after={int(after.get('density_count_max', 0) or 0)} "
        f"largest_comp_before={int(before.get('largest_component_count', 0))} "
        f"largest_comp_after={int(after.get('largest_component_count', 0))}"
    )

    line_groups = (
        f"  frame4_spatial_thinning_removed_groups: "
        f"pnp8={int(removed['pnp_8px_inliers']['removed'])}/{int(removed['pnp_8px_inliers']['count'])} "
        f"pnp12all={int(removed['pnp_12px_all_inliers']['removed'])}/{int(removed['pnp_12px_all_inliers']['count'])} "
        f"pnp12only={int(removed['pnp_12px_only_inliers']['removed'])}/{int(removed['pnp_12px_only_inliers']['count'])} "
        f"rejected={int(removed['rejected_by_both']['removed'])}/{int(removed['rejected_by_both']['count'])}"
    )

    line_removed_spatial = (
        f"  frame4_spatial_thinning_removed_spatial: "
        f"cur_bbox={_format_optional_point(rem_cur['bbox'])} "
        f"cur_area_frac={_format_optional_float(rem_cur['bbox_area_fraction'], digits=3)} "
        f"cur_grid={rem_cur['occupancy_grid']} "
        f"cur_max_cell_frac={_format_optional_float(rem_cur['max_cell_fraction'], digits=2)}"
    )

    line_kept_spatial = (
        f"  frame4_spatial_thinning_kept_spatial: "
        f"cur_bbox={_format_optional_point(kept_cur['bbox'])} "
        f"cur_area_frac={_format_optional_float(kept_cur['bbox_area_fraction'], digits=3)} "
        f"cur_cells={int(kept_cur['occupied_cells'])} "
        f"cur_max_cell_frac={_format_optional_float(kept_cur['max_cell_fraction'], digits=2)}"
    )

    return [line_filter, line_groups, line_removed_spatial, line_kept_spatial]


# Compute reprojection errors for a pose on all correspondences
def _reprojection_errors_for_pose(corrs, K: np.ndarray, R, t, *, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = int(np.asarray(corrs.X_w, dtype=np.float64).shape[1])
    errors_px = np.full((N,), np.inf, dtype=np.float64)
    positive_depth = np.zeros((N,), dtype=bool)
    non_positive_depth = np.zeros((N,), dtype=bool)

    # Stop when no valid pose is available
    if R is None or t is None:
        return errors_px, positive_depth, non_positive_depth

    # Compute positive depth and reprojection error
    X_w = np.asarray(corrs.X_w, dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    X_c = world_to_camera_points(R, t, X_w)
    z = np.asarray(X_c[2, :], dtype=np.float64).reshape(-1)
    finite_depth = np.isfinite(z)
    positive_depth = finite_depth & (z > float(eps))
    non_positive_depth = finite_depth & ~positive_depth

    err_sq = np.asarray(reprojection_errors_sq(K, R, t, X_w, x_cur), dtype=np.float64).reshape(-1)
    finite_err_sq = np.isfinite(err_sq) & (err_sq >= 0.0)
    errors_px[finite_err_sq] = np.sqrt(err_sq[finite_err_sq])

    return errors_px, positive_depth, non_positive_depth


# Summarise reprojection behaviour for a pose and correspondence subset
def _pose_reprojection_block(corrs, K: np.ndarray, R, t, mask: np.ndarray, *, eps: float) -> dict:
    errors_px, positive_depth, non_positive_depth = _reprojection_errors_for_pose(corrs, K, R, t, eps=eps)

    return _non_pnp_reprojection_block(errors_px, positive_depth, non_positive_depth, mask)


# Build landmark birth-source masks for a correspondence bundle
def _birth_source_masks(seed: dict, landmark_ids: np.ndarray) -> dict:
    landmark_ids = np.asarray(landmark_ids, dtype=np.int64).reshape(-1)
    birth_sources = np.full((int(landmark_ids.size),), "unknown", dtype=object)

    # Build landmark lookup
    lm_by_id: dict[int, dict] = {}
    landmarks = seed.get("landmarks", []) if isinstance(seed, dict) else []
    if isinstance(landmarks, list):
        for lm in landmarks:
            if not isinstance(lm, dict):
                continue
            if "id" not in lm:
                continue
            lm_by_id[int(lm["id"])] = lm

    # Read birth source for each correspondence
    for j, lm_id in enumerate(landmark_ids):
        lm = lm_by_id.get(int(lm_id), None)
        if not isinstance(lm, dict):
            continue
        birth_source = lm.get("birth_source", None)
        if isinstance(birth_source, str):
            birth_sources[j] = str(birth_source)

    return {
        "bootstrap": birth_sources == "bootstrap",
        "map_growth": birth_sources == "map_growth",
        "unknown": birth_sources == "unknown",
    }


# Compare frame-4 PnP poses at 8 px and 12 px
def _compare_frame4_pnp_8px_12px(seed: dict, corrs, K: np.ndarray, *, pnp_cfg: dict, pose_pair: dict | None = None) -> dict:
    N = int(np.asarray(corrs.X_w, dtype=np.float64).shape[1])
    all_mask = np.ones((N,), dtype=bool)

    out = {
        "ok": False,
        "reason": None,
        "n_pose_eligible": int(N),
        "pose_8px_ok": False,
        "pose_12px_ok": False,
        "rotation_diff_deg": None,
        "translation_direction_angle_deg": None,
        "camera_centre_direction_angle_deg": None,
        "inliers_8px": 0,
        "inliers_12px": 0,
        "inlier_overlap": 0,
        "inlier_iou": None,
        "inlier_overlap_over_8px": None,
        "inlier_overlap_over_12px": None,
        "inliers_unique_to_12px": 0,
        "threshold_stability_classification": "unavailable",
        "threshold_stability_unstable": False,
        "threshold_stability_reasons": [],
        "threshold_stability_supports_disjoint": False,
        "threshold_stability_support_iou_low": False,
        "threshold_stability_translation_direction_disagrees": False,
        "threshold_stability_camera_centre_direction_disagrees": False,
        "component_8px": {},
        "component_12px": {},
        "component_gate_8px_rejected": False,
        "component_gate_12px_rejected": False,
        "component_gate_8px_reason": None,
        "component_gate_12px_reason": None,
        "reprojection_all_8px": _pose_reprojection_block(corrs, K, None, None, all_mask, eps=float(pnp_cfg["eps"])),
        "reprojection_all_12px": _pose_reprojection_block(corrs, K, None, None, all_mask, eps=float(pnp_cfg["eps"])),
        "unique_12px": {
            "count": 0,
            "birth_source": {"bootstrap": 0, "map_growth": 0, "unknown": 0},
            "reprojection_under_8px": _pose_reprojection_block(corrs, K, None, None, np.zeros((N,), dtype=bool), eps=float(pnp_cfg["eps"])),
            "reprojection_under_12px": _pose_reprojection_block(corrs, K, None, None, np.zeros((N,), dtype=bool), eps=float(pnp_cfg["eps"])),
        },
    }

    # Run both PnP thresholds on the same correspondence bundle
    if pose_pair is None:
        pose_pair = _frame4_pnp_8px_12px_pose_pair(corrs, K, pnp_cfg=pnp_cfg)
    pose_8 = pose_pair["pose_8px"]
    pose_12 = pose_pair["pose_12px"]

    group_masks = _frame4_threshold_group_masks(pose_pair, N, include_12px_all=True)
    mask_8 = group_masks["pnp_8px_inliers"]
    mask_12 = group_masks["pnp_12px_all_inliers"]
    overlap = mask_8 & mask_12
    union = mask_8 | mask_12
    unique_12 = mask_12 & ~mask_8

    n_8 = int(np.sum(mask_8))
    n_12 = int(np.sum(mask_12))
    n_overlap = int(np.sum(overlap))
    n_union = int(np.sum(union))
    n_unique_12 = int(np.sum(unique_12))

    out.update(
        {
            "pose_8px_ok": bool(pose_8["ok"]),
            "pose_12px_ok": bool(pose_12["ok"]),
            "pose_8px_reason": pose_8.get("reason", None),
            "pose_12px_reason": pose_12.get("reason", None),
            "inliers_8px": int(n_8),
            "inliers_12px": int(n_12),
            "inlier_overlap": int(n_overlap),
            "inlier_iou": None if n_union == 0 else float(n_overlap / n_union),
            "inlier_overlap_over_8px": None if n_8 == 0 else float(n_overlap / n_8),
            "inlier_overlap_over_12px": None if n_12 == 0 else float(n_overlap / n_12),
            "inliers_unique_to_12px": int(n_unique_12),
            "component_8px": pose_8.get("pnp_component_support", None),
            "component_12px": pose_12.get("pnp_component_support", None),
            "component_gate_8px_rejected": bool(pose_8.get("pnp_component_gate_rejected", False)),
            "component_gate_12px_rejected": bool(pose_12.get("pnp_component_gate_rejected", False)),
            "component_gate_8px_reason": pose_8.get("pnp_component_gate_reason", None),
            "component_gate_12px_reason": pose_12.get("pnp_component_gate_reason", None),
        }
    )

    # Compare recovered poses when both are available
    if bool(pose_8["ok"]) and bool(pose_12["ok"]):
        R_8 = np.asarray(pose_8["R"], dtype=np.float64)
        R_12 = np.asarray(pose_12["R"], dtype=np.float64)
        t_8 = np.asarray(pose_8["t"], dtype=np.float64).reshape(3)
        t_12 = np.asarray(pose_12["t"], dtype=np.float64).reshape(3)

        try:
            translation_direction_angle_deg = float(np.degrees(angle_between_translations(t_8, t_12)))
        except Exception:
            translation_direction_angle_deg = None

        C_8 = camera_centre(R_8, t_8)
        C_12 = camera_centre(R_12, t_12)

        try:
            camera_centre_direction_angle_deg = float(np.degrees(angle_between_translations(C_8, C_12)))
        except Exception:
            camera_centre_direction_angle_deg = None

        out["rotation_diff_deg"] = float(np.degrees(angle_between_rotmats(R_8, R_12)))
        out["translation_direction_angle_deg"] = translation_direction_angle_deg
        out["camera_centre_direction_angle_deg"] = camera_centre_direction_angle_deg
        out["reprojection_all_8px"] = _pose_reprojection_block(corrs, K, R_8, t_8, all_mask, eps=float(pnp_cfg["eps"]))
        out["reprojection_all_12px"] = _pose_reprojection_block(corrs, K, R_12, t_12, all_mask, eps=float(pnp_cfg["eps"]))
        out["unique_12px"]["reprojection_under_8px"] = _pose_reprojection_block(corrs, K, R_8, t_8, unique_12, eps=float(pnp_cfg["eps"]))
        out["unique_12px"]["reprojection_under_12px"] = _pose_reprojection_block(corrs, K, R_12, t_12, unique_12, eps=float(pnp_cfg["eps"]))

    # Classify the fixed 8 px versus 12 px threshold stability
    stability_flags = pnp_threshold_stability_flags(
        ref_pose_ok=bool(pose_8["ok"]),
        compare_pose_ok=bool(pose_12["ok"]),
        ref_threshold_px=8.0,
        compare_threshold_px=12.0,
        support_iou=out.get("inlier_iou", None),
        support_union=int(n_union),
        translation_direction_delta_deg=out["translation_direction_angle_deg"],
        camera_centre_direction_delta_deg=out["camera_centre_direction_angle_deg"],
        min_support_iou=float(pnp_cfg.get("pnp_threshold_stability_min_support_iou", 0.25)),
        max_translation_direction_angle_deg=float(pnp_cfg.get("pnp_threshold_stability_max_translation_direction_deg", 120.0)),
        max_camera_centre_direction_angle_deg=float(pnp_cfg.get("pnp_threshold_stability_max_camera_centre_direction_deg", 120.0)),
        disjoint_support_iou=float(pnp_cfg.get("pnp_threshold_stability_disjoint_iou", 0.05)),
    )
    out.update(
        {
            "threshold_stability_classification": stability_flags["classification"],
            "threshold_stability_unstable": bool(stability_flags["unstable"]),
            "threshold_stability_reasons": stability_flags["instability_reasons"],
            "threshold_stability_supports_disjoint": bool(stability_flags["supports_effectively_disjoint"]),
            "threshold_stability_support_iou_low": bool(stability_flags["support_iou_low"]),
            "threshold_stability_translation_direction_disagrees": bool(stability_flags["translation_direction_disagrees"]),
            "threshold_stability_camera_centre_direction_disagrees": bool(stability_flags["camera_centre_direction_disagrees"]),
        }
    )

    # Split 12-only inliers by landmark birth source
    birth_masks = _birth_source_masks(seed, np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1))
    out["unique_12px"]["count"] = int(n_unique_12)
    out["unique_12px"]["birth_source"] = {
        "bootstrap": int(np.sum(unique_12 & birth_masks["bootstrap"])),
        "map_growth": int(np.sum(unique_12 & birth_masks["map_growth"])),
        "unknown": int(np.sum(unique_12 & birth_masks["unknown"])),
    }

    out["ok"] = bool(pose_8["ok"]) and bool(pose_12["ok"])
    if not bool(out["ok"]):
        out["reason"] = "one_or_both_poses_failed"

    return out


# Analyse pose-eligible non-PnP correspondences under the recovered pose
def _analyse_non_pnp_pose_eligible_reprojection(seed: dict, pose_out: dict, K: np.ndarray, *, eps: float) -> dict:
    out = {
        "ok": False,
        "reason": None,
        "n_pose_eligible": 0,
        "n_pnp_inliers": 0,
        "n_non_pnp": 0,
        "all": _non_pnp_reprojection_block(np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)),
        "by_birth_source": {
            "bootstrap": _non_pnp_reprojection_block(np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)),
            "map_growth": _non_pnp_reprojection_block(np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)),
        },
    }

    # Stop when no recovered pose is available
    if not isinstance(pose_out, dict) or not bool(pose_out.get("ok", False)):
        out["reason"] = "pose_not_recovered"
        return out

    # Read PnP correspondence bundle
    corrs = pose_out.get("corrs", None)
    if corrs is None or not hasattr(corrs, "X_w") or not hasattr(corrs, "x_cur") or not hasattr(corrs, "landmark_ids"):
        out["reason"] = "missing_correspondences"
        return out

    # Read correspondence arrays
    X_w = np.asarray(corrs.X_w, dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)

    if X_w.ndim != 2 or X_w.shape[0] != 3:
        out["reason"] = "bad_world_points"
        return out
    if x_cur.ndim != 2 or x_cur.shape[0] != 2:
        out["reason"] = "bad_image_points"
        return out

    N = int(X_w.shape[1])
    if int(x_cur.shape[1]) != N or int(landmark_ids.size) != N:
        out["reason"] = "correspondence_length_mismatch"
        return out

    pnp_inlier_mask = align_bool_mask_1d(pose_out.get("pnp_inlier_mask", np.zeros((N,), dtype=bool)), N, name="pose_out['pnp_inlier_mask']")
    non_pnp_mask = ~pnp_inlier_mask

    out["n_pose_eligible"] = int(N)
    out["n_pnp_inliers"] = int(np.sum(pnp_inlier_mask))
    out["n_non_pnp"] = int(np.sum(non_pnp_mask))

    if N == 0 or not np.any(non_pnp_mask):
        out["ok"] = True
        out["reason"] = None
        return out

    # Build landmark birth-source lookup
    lm_by_id: dict[int, dict] = {}
    landmarks = seed.get("landmarks", []) if isinstance(seed, dict) else []
    if isinstance(landmarks, list):
        for lm in landmarks:
            if not isinstance(lm, dict):
                continue
            if "id" not in lm:
                continue
            lm_by_id[int(lm["id"])] = lm

    # Read the recovered pose
    R = np.asarray(pose_out.get("R", None), dtype=np.float64)
    t = np.asarray(pose_out.get("t", None), dtype=np.float64).reshape(-1)
    if R.shape != (3, 3) or t.size != 3:
        out["reason"] = "bad_pose"
        return out

    # Compute depths and reprojection errors under the recovered pose
    X_c = world_to_camera_points(R, t.reshape(3), X_w)
    z = np.asarray(X_c[2, :], dtype=np.float64).reshape(-1)
    finite_depth = np.isfinite(z)
    positive_depth = finite_depth & (z > float(eps))
    non_positive_depth = finite_depth & ~positive_depth

    err_sq = np.asarray(reprojection_errors_sq(K, R, t.reshape(3), X_w, x_cur), dtype=np.float64).reshape(-1)
    errors_px = np.full((N,), np.inf, dtype=np.float64)
    finite_err_sq = np.isfinite(err_sq) & (err_sq >= 0.0)
    errors_px[finite_err_sq] = np.sqrt(err_sq[finite_err_sq])

    # Split non-PnP correspondences by landmark birth source
    birth_sources = np.full((N,), "unknown", dtype=object)
    for j, lm_id in enumerate(landmark_ids):
        lm = lm_by_id.get(int(lm_id), None)
        if not isinstance(lm, dict):
            continue
        birth_source = lm.get("birth_source", None)
        if isinstance(birth_source, str):
            birth_sources[j] = str(birth_source)

    out["ok"] = True
    out["reason"] = None
    out["all"] = _non_pnp_reprojection_block(errors_px, positive_depth, non_positive_depth, non_pnp_mask)
    out["by_birth_source"] = {
        "bootstrap": _non_pnp_reprojection_block(errors_px, positive_depth, non_positive_depth, non_pnp_mask & (birth_sources == "bootstrap")),
        "map_growth": _non_pnp_reprojection_block(errors_px, positive_depth, non_positive_depth, non_pnp_mask & (birth_sources == "map_growth")),
    }

    return out


# Format the frame-4 non-PnP reprojection diagnostic
def _format_non_pnp_reprojection_diag(diag: dict) -> list[str]:
    if not isinstance(diag, dict) or not bool(diag.get("ok", False)):
        return [f"  frame4_non_pnp_reproj: unavailable reason={diag.get('reason', None) if isinstance(diag, dict) else None}"]

    all_block = diag["all"]
    summary = all_block["error_px"]
    hist = all_block["hist_px"]
    birth = diag["by_birth_source"]
    bootstrap_summary = birth["bootstrap"]["error_px"]
    map_growth_summary = birth["map_growth"]["error_px"]

    line_total = (
        f"  frame4_non_pnp_reproj: n={int(diag['n_non_pnp'])} "
        f"usable={int(summary['count'])} "
        f"positive_depth={int(all_block['n_positive_depth'])} "
        f"non_positive_depth={int(all_block['n_non_positive_depth'])} "
        f"nonfinite_reproj={int(all_block['n_nonfinite_reprojection'])} "
        f"min={_format_optional_px(summary['min'])} "
        f"median={_format_optional_px(summary['median'])} "
        f"p75={_format_optional_px(summary['p75'])} "
        f"p90={_format_optional_px(summary['p90'])} "
        f"p95={_format_optional_px(summary['p95'])} "
        f"max={_format_optional_px(summary['max'])}"
    )

    line_hist = (
        f"  frame4_non_pnp_bins: <=2={int(hist['le_2_px'])} "
        f"2-3={int(hist['gt_2_le_3_px'])} "
        f"3-5={int(hist['gt_3_le_5_px'])} "
        f"5-8={int(hist['gt_5_le_8_px'])} "
        f"8-12={int(hist['gt_8_le_12_px'])} "
        f"12-20={int(hist['gt_12_le_20_px'])} "
        f">20={int(hist['gt_20_px'])}"
    )

    line_birth = (
        f"  frame4_non_pnp_birth: bootstrap_n={int(birth['bootstrap']['n_total'])} "
        f"bootstrap_median={_format_optional_px(bootstrap_summary['median'])} "
        f"map_growth_n={int(birth['map_growth']['n_total'])} "
        f"map_growth_median={_format_optional_px(map_growth_summary['median'])}"
    )

    return [line_total, line_hist, line_birth]


# Format the frame-4 8 px vs 12 px PnP comparison
def _format_frame4_pnp_pose_comparison(diag: dict) -> list[str]:
    if not isinstance(diag, dict):
        return ["  frame4_pnp_8v12: unavailable reason=missing_diag"]

    all_8 = diag["reprojection_all_8px"]["error_px"]
    all_12 = diag["reprojection_all_12px"]["error_px"]
    unique_8 = diag["unique_12px"]["reprojection_under_8px"]["error_px"]
    unique_12 = diag["unique_12px"]["reprojection_under_12px"]["error_px"]
    birth = diag["unique_12px"]["birth_source"]
    comp_8 = diag.get("component_8px", {}) if isinstance(diag.get("component_8px", {}), dict) else {}
    comp_12 = diag.get("component_12px", {}) if isinstance(diag.get("component_12px", {}), dict) else {}
    stability_reasons = diag.get("threshold_stability_reasons", [])
    if isinstance(stability_reasons, list):
        stability_reason_text = ",".join(str(v) for v in stability_reasons)
    else:
        stability_reason_text = str(stability_reasons)
    if stability_reason_text == "":
        stability_reason_text = "None"

    line_pose = (
        f"  frame4_pnp_8v12_pose: ok8={bool(diag.get('pose_8px_ok', False))} "
        f"ok12={bool(diag.get('pose_12px_ok', False))} "
        f"rot_deg={_format_optional_px(diag.get('rotation_diff_deg', None))} "
        f"t_dir_deg={_format_optional_px(diag.get('translation_direction_angle_deg', None))} "
        f"C_dir_deg={_format_optional_px(diag.get('camera_centre_direction_angle_deg', None))}"
    )

    line_inliers = (
        f"  frame4_pnp_8v12_inliers: n8={int(diag.get('inliers_8px', 0))} "
        f"n12={int(diag.get('inliers_12px', 0))} "
        f"overlap={int(diag.get('inlier_overlap', 0))} "
        f"iou={_format_optional_px(diag.get('inlier_iou', None))} "
        f"unique12={int(diag.get('inliers_unique_to_12px', 0))}"
    )

    line_stability = (
        f"  frame4_pnp_8v12_stability: class={diag.get('threshold_stability_classification', 'unavailable')} "
        f"unstable={bool(diag.get('threshold_stability_unstable', False))} "
        f"disjoint={bool(diag.get('threshold_stability_supports_disjoint', False))} "
        f"support_iou_low={bool(diag.get('threshold_stability_support_iou_low', False))} "
        f"t_dir_disagree={bool(diag.get('threshold_stability_translation_direction_disagrees', False))} "
        f"C_dir_disagree={bool(diag.get('threshold_stability_camera_centre_direction_disagrees', False))} "
        f"reasons={stability_reason_text}"
    )

    line_components = (
        f"  frame4_pnp_8v12_components: "
        f"ncomp8={int(comp_8.get('component_count', 0))} "
        f"largest8={int(comp_8.get('largest_component_size', 0))} "
        f"frac8={_format_optional_float(comp_8.get('largest_component_fraction', None), digits=2)} "
        f"area8={_format_optional_float(comp_8.get('largest_component_bbox_area_fraction', None), digits=4)} "
        f"reject8={bool(diag.get('component_gate_8px_rejected', False))} "
        f"reason8={diag.get('component_gate_8px_reason', None)} "
        f"ncomp12={int(comp_12.get('component_count', 0))} "
        f"largest12={int(comp_12.get('largest_component_size', 0))} "
        f"frac12={_format_optional_float(comp_12.get('largest_component_fraction', None), digits=2)} "
        f"area12={_format_optional_float(comp_12.get('largest_component_bbox_area_fraction', None), digits=4)} "
        f"reject12={bool(diag.get('component_gate_12px_rejected', False))} "
        f"reason12={diag.get('component_gate_12px_reason', None)}"
    )

    line_all = (
        f"  frame4_pnp_8v12_all: med8={_format_optional_px(all_8['median'])} "
        f"p75_8={_format_optional_px(all_8['p75'])} "
        f"p90_8={_format_optional_px(all_8['p90'])} "
        f"med12={_format_optional_px(all_12['median'])} "
        f"p75_12={_format_optional_px(all_12['p75'])} "
        f"p90_12={_format_optional_px(all_12['p90'])}"
    )

    line_unique = (
        f"  frame4_pnp_8v12_unique12: count={int(diag['unique_12px']['count'])} "
        f"bootstrap={int(birth['bootstrap'])} "
        f"map_growth={int(birth['map_growth'])} "
        f"med_under8={_format_optional_px(unique_8['median'])} "
        f"med_under12={_format_optional_px(unique_12['median'])} "
        f"p95_under12={_format_optional_px(unique_12['p95'])}"
    )

    return [line_pose, line_inliers, line_stability, line_components, line_all, line_unique]


def main() -> None:
    parser = argparse.ArgumentParser()

    # Default ETH3D profile
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    # Optional dataset override
    parser.add_argument("--dataset_root", type=str, default=None)
    # Optional sequence override
    parser.add_argument("--seq", type=str, default=None)
    # Optional output override
    parser.add_argument("--out_dir", type=str, default=None)

    # Fixed bootstrap frame 0
    parser.add_argument("--i0", type=int, default=0)
    # Fixed bootstrap frame 1
    parser.add_argument("--i1", type=int, default=1)
    # Number of later frames to diagnose
    parser.add_argument("--num_track", type=int, default=5)
    # Threshold sweep in pixels
    parser.add_argument("--thresholds", type=float, nargs="+", default=[3.0, 5.0, 8.0, 12.0])
    # Minimum landmark observation count used before PnP
    parser.add_argument("--min_landmark_observations", type=int, default=2)
    # Tight reprojection gate for appending existing-landmark observations
    parser.add_argument("--max_append_reproj_error_px_existing", type=float, default=2.0)
    # Enable the diagnostic PnP spatial support gate
    parser.add_argument("--enable_pnp_spatial_gate", action=argparse.BooleanOptionalAction, default=True)
    # PnP spatial support grid columns
    parser.add_argument("--pnp_spatial_grid_cols", type=int, default=4)
    # PnP spatial support grid rows
    parser.add_argument("--pnp_spatial_grid_rows", type=int, default=3)
    # Minimum occupied cells required for accepted PnP support
    parser.add_argument("--min_pnp_inlier_cells", type=int, default=1)
    # Maximum fraction of accepted PnP support allowed in one cell
    parser.add_argument("--max_pnp_single_cell_fraction", type=float, default=1.0)
    # Minimum inlier bounding-box image area fraction
    parser.add_argument("--min_pnp_bbox_area_fraction", type=float, default=0.01)
    # Enable the diagnostic PnP component-support gate
    parser.add_argument("--enable_pnp_component_gate", action=argparse.BooleanOptionalAction, default=False)
    # Current-image radius for accepted-inlier component support
    parser.add_argument("--pnp_component_radius_px", type=float, default=80.0)
    # Maximum accepted PnP support allowed in one component
    parser.add_argument("--max_pnp_largest_component_fraction", type=float, default=1.0)
    # Minimum accepted PnP inlier component count
    parser.add_argument("--min_pnp_component_count", type=int, default=0)
    # Minimum largest-component bounding-box image area fraction
    parser.add_argument("--min_pnp_largest_component_bbox_area_fraction", type=float, default=0.0)
    # Enable the diagnostic local displacement-consistency pose filter
    parser.add_argument("--enable_pnp_local_consistency_filter", action=argparse.BooleanOptionalAction, default=False)
    # Apply the local displacement-consistency filter to the live frontend state
    parser.add_argument("--apply_pnp_local_consistency_filter_to_pipeline", action=argparse.BooleanOptionalAction, default=False)
    # Keyframe-image radius for local displacement-consistency neighbours
    parser.add_argument("--pnp_local_consistency_radius_px", type=float, default=80.0)
    # Minimum local neighbours required by the local displacement-consistency filter
    parser.add_argument(
        "--pnp_local_consistency_min_neighbours",
        "--pnp_local_consistency_min_neighbors",
        dest="pnp_local_consistency_min_neighbours",
        type=int,
        default=3,
    )
    # Maximum residual from the local median displacement
    parser.add_argument("--pnp_local_consistency_max_median_residual_px", type=float, default=12.0)
    # Minimum remaining pose correspondences before the filter falls back to keep-all
    parser.add_argument("--pnp_local_consistency_min_keep", type=int, default=0)
    # Enable the diagnostic current-image spatial thinning pose filter
    parser.add_argument("--enable_pnp_spatial_thinning_filter", action=argparse.BooleanOptionalAction, default=False)
    # Apply the current-image spatial thinning filter to the live frontend state
    parser.add_argument("--apply_pnp_spatial_thinning_filter_to_pipeline", action=argparse.BooleanOptionalAction, default=False)
    # Current-image radius for spatial thinning
    parser.add_argument("--pnp_spatial_thinning_radius_px", type=float, default=20.0)
    # Maximum retained pose correspondences allowed within the thinning radius
    parser.add_argument("--pnp_spatial_thinning_max_points_per_radius", type=int, default=16)
    # Minimum remaining pose correspondences before spatial thinning falls back to keep-all
    parser.add_argument("--pnp_spatial_thinning_min_keep", type=int, default=0)
    _add_pnp_threshold_stability_args(parser)

    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)
    profile_pnp_cfg = cfg.get("pnp", {})
    pnp_cfg = _pnp_solver_cfg(profile_pnp_cfg if isinstance(profile_pnp_cfg, dict) else None)
    thresholds = _parse_thresholds(list(args.thresholds))
    min_landmark_observations = check_int_gt0(args.min_landmark_observations, name="min_landmark_observations")
    max_append_reproj_error_px_existing = check_positive(
        args.max_append_reproj_error_px_existing,
        name="max_append_reproj_error_px_existing",
        eps=0.0,
    )
    pnp_cfg["enable_pnp_spatial_gate"] = bool(args.enable_pnp_spatial_gate)
    pnp_cfg["pnp_spatial_grid_cols"] = check_int_gt0(args.pnp_spatial_grid_cols, name="pnp_spatial_grid_cols")
    pnp_cfg["pnp_spatial_grid_rows"] = check_int_gt0(args.pnp_spatial_grid_rows, name="pnp_spatial_grid_rows")
    pnp_cfg["min_pnp_inlier_cells"] = check_int_ge0(args.min_pnp_inlier_cells, name="min_pnp_inlier_cells")
    pnp_cfg["max_pnp_single_cell_fraction"] = check_finite_scalar(args.max_pnp_single_cell_fraction, name="max_pnp_single_cell_fraction")
    pnp_cfg["min_pnp_bbox_area_fraction"] = check_finite_scalar(args.min_pnp_bbox_area_fraction, name="min_pnp_bbox_area_fraction")
    check_in_01(pnp_cfg["max_pnp_single_cell_fraction"], name="max_pnp_single_cell_fraction", eps=0.0)
    check_in_01(pnp_cfg["min_pnp_bbox_area_fraction"], name="min_pnp_bbox_area_fraction", eps=0.0)
    if float(pnp_cfg["max_pnp_single_cell_fraction"]) <= 0.0:
        raise ValueError(f"max_pnp_single_cell_fraction must be > 0; got {pnp_cfg['max_pnp_single_cell_fraction']}")
    pnp_cfg["enable_pnp_component_gate"] = bool(args.enable_pnp_component_gate)
    pnp_cfg["pnp_component_radius_px"] = check_positive(args.pnp_component_radius_px, name="pnp_component_radius_px", eps=0.0)
    pnp_cfg["max_pnp_largest_component_fraction"] = check_finite_scalar(
        args.max_pnp_largest_component_fraction,
        name="max_pnp_largest_component_fraction",
    )
    pnp_cfg["min_pnp_component_count"] = check_int_ge0(args.min_pnp_component_count, name="min_pnp_component_count")
    pnp_cfg["min_pnp_largest_component_bbox_area_fraction"] = check_finite_scalar(
        args.min_pnp_largest_component_bbox_area_fraction,
        name="min_pnp_largest_component_bbox_area_fraction",
    )
    check_in_01(pnp_cfg["max_pnp_largest_component_fraction"], name="max_pnp_largest_component_fraction", eps=0.0)
    check_in_01(
        pnp_cfg["min_pnp_largest_component_bbox_area_fraction"],
        name="min_pnp_largest_component_bbox_area_fraction",
        eps=0.0,
    )
    if float(pnp_cfg["max_pnp_largest_component_fraction"]) <= 0.0:
        raise ValueError(f"max_pnp_largest_component_fraction must be > 0; got {pnp_cfg['max_pnp_largest_component_fraction']}")
    pnp_cfg["enable_pnp_local_consistency_filter"] = bool(args.enable_pnp_local_consistency_filter)
    pnp_cfg["apply_pnp_local_consistency_filter_to_pipeline"] = bool(args.apply_pnp_local_consistency_filter_to_pipeline)
    pnp_cfg["pnp_local_consistency_radius_px"] = check_positive(
        args.pnp_local_consistency_radius_px,
        name="pnp_local_consistency_radius_px",
        eps=0.0,
    )
    pnp_cfg["pnp_local_consistency_min_neighbours"] = check_int_ge0(
        args.pnp_local_consistency_min_neighbours,
        name="pnp_local_consistency_min_neighbours",
    )
    pnp_cfg["pnp_local_consistency_max_median_residual_px"] = check_positive(
        args.pnp_local_consistency_max_median_residual_px,
        name="pnp_local_consistency_max_median_residual_px",
        eps=0.0,
    )
    pnp_cfg["pnp_local_consistency_min_keep"] = check_int_ge0(
        args.pnp_local_consistency_min_keep,
        name="pnp_local_consistency_min_keep",
    )
    pnp_cfg["enable_pnp_spatial_thinning_filter"] = bool(args.enable_pnp_spatial_thinning_filter)
    pnp_cfg["apply_pnp_spatial_thinning_filter_to_pipeline"] = bool(args.apply_pnp_spatial_thinning_filter_to_pipeline)
    pnp_cfg["pnp_spatial_thinning_radius_px"] = check_positive(
        args.pnp_spatial_thinning_radius_px,
        name="pnp_spatial_thinning_radius_px",
        eps=0.0,
    )
    pnp_cfg["pnp_spatial_thinning_max_points_per_radius"] = check_int_gt0(
        args.pnp_spatial_thinning_max_points_per_radius,
        name="pnp_spatial_thinning_max_points_per_radius",
    )
    pnp_cfg["pnp_spatial_thinning_min_keep"] = check_int_ge0(
        args.pnp_spatial_thinning_min_keep,
        name="pnp_spatial_thinning_min_keep",
    )
    pnp_cfg = _apply_pnp_threshold_stability_cli_overrides(pnp_cfg, args)

    dataset_cfg = cfg["dataset"]
    run_cfg = cfg["run"]

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root is not None
        else (ROOT / dataset_cfg["root"]).resolve()
    )
    seq_name = str(args.seq) if args.seq is not None else str(dataset_cfg["seq"])

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (ROOT / str(run_cfg.get("out_dir", "out")) / "diag_pnp_eth3d").resolve()

    check_dir(dataset_root, name="dataset_root")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "pnp_diag.jsonl"

    i0 = check_int_ge0(args.i0, name="i0")
    i1 = check_int_ge0(args.i1, name="i1")
    num_track = check_int_gt0(args.num_track, name="num_track")

    if i1 <= i0:
        raise ValueError(f"Expected i1 > i0 for bootstrap; got i0={i0}, i1={i1}")

    # Load ETH3D sequence
    seq = load_eth3d_sequence(
        dataset_root,
        seq_name,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    max_frames = dataset_cfg.get("max_frames", None)
    min_required_frames = int(i1 + 1 + num_track)
    if max_frames is None:
        n_effective = len(seq)
    else:
        n_effective = min(len(seq), max(int(max_frames), int(min_required_frames)))
    if n_effective <= 0:
        raise ValueError("Loaded ETH3D sequence is empty")

    if i0 >= n_effective or i1 >= n_effective:
        raise IndexError(f"Bootstrap indices out of range for effective sequence length {n_effective}")

    # Read bootstrap images
    im0, ts0, id0 = seq.get(i0)
    im1, ts1, id1 = seq.get(i1)

    # Run two-view bootstrap
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

    print(f"sequence: {seq.name}")
    print(f"dataset_root: {dataset_root}")
    print(f"seq_name: {seq_name}")
    print(f"bootstrap pair: {i0} ({id0}, t={ts0}) -> {i1} ({id1}, t={ts1})")
    print(f"bootstrap ok: {boot['ok']}")
    print(f"bootstrap stats: {boot['stats']}")
    print(f"max_append_reproj_error_px_existing: {max_append_reproj_error_px_existing}")
    print(
        f"pnp_spatial_gate: enabled={bool(pnp_cfg['enable_pnp_spatial_gate'])} "
        f"grid={int(pnp_cfg['pnp_spatial_grid_cols'])}x{int(pnp_cfg['pnp_spatial_grid_rows'])} "
        f"min_cells={int(pnp_cfg['min_pnp_inlier_cells'])} "
        f"max_single_cell_fraction={float(pnp_cfg['max_pnp_single_cell_fraction'])} "
        f"min_bbox_area_fraction={float(pnp_cfg['min_pnp_bbox_area_fraction'])}"
    )
    print(
        f"pnp_component_gate: enabled={bool(pnp_cfg['enable_pnp_component_gate'])} "
        f"radius_px={float(pnp_cfg['pnp_component_radius_px'])} "
        f"max_largest_component_fraction={float(pnp_cfg['max_pnp_largest_component_fraction'])} "
        f"min_component_count={int(pnp_cfg['min_pnp_component_count'])} "
        f"min_largest_component_bbox_area_fraction={float(pnp_cfg['min_pnp_largest_component_bbox_area_fraction'])}"
    )
    print(
        f"pnp_local_consistency_filter: enabled={bool(pnp_cfg['enable_pnp_local_consistency_filter'])} "
        f"apply_to_pipeline={bool(pnp_cfg['apply_pnp_local_consistency_filter_to_pipeline'])} "
        f"radius_px={float(pnp_cfg['pnp_local_consistency_radius_px'])} "
        f"min_neighbours={int(pnp_cfg['pnp_local_consistency_min_neighbours'])} "
        f"max_median_residual_px={float(pnp_cfg['pnp_local_consistency_max_median_residual_px'])} "
        f"min_keep={int(pnp_cfg['pnp_local_consistency_min_keep'])}"
    )
    print(
        f"pnp_spatial_thinning_filter: enabled={bool(pnp_cfg['enable_pnp_spatial_thinning_filter'])} "
        f"apply_to_pipeline={bool(pnp_cfg['apply_pnp_spatial_thinning_filter_to_pipeline'])} "
        f"radius_px={float(pnp_cfg['pnp_spatial_thinning_radius_px'])} "
        f"max_points_per_radius={int(pnp_cfg['pnp_spatial_thinning_max_points_per_radius'])} "
        f"min_keep={int(pnp_cfg['pnp_spatial_thinning_min_keep'])}"
    )
    print(
        f"pnp_threshold_stability: enabled={bool(pnp_cfg['enable_pnp_threshold_stability_diagnostic'])} "
        f"gate={bool(pnp_cfg['enable_pnp_threshold_stability_gate'])} "
        f"compare_px={float(pnp_cfg['pnp_threshold_stability_compare_px'])} "
        f"min_iou={float(pnp_cfg['pnp_threshold_stability_min_support_iou'])} "
        f"max_t_dir_deg={float(pnp_cfg['pnp_threshold_stability_max_translation_direction_deg'])} "
        f"max_C_dir_deg={float(pnp_cfg['pnp_threshold_stability_max_camera_centre_direction_deg'])} "
        f"disjoint_iou={float(pnp_cfg['pnp_threshold_stability_disjoint_iou'])}"
    )

    # Write the bootstrap summary
    _append_jsonl(
        log_path,
        {
            "event": "bootstrap",
            "frame_index_0": int(i0),
            "frame_index_1": int(i1),
            "ok": bool(boot["ok"]),
            "reason": boot["stats"].get("reason", None),
            "n_landmarks": 0 if not isinstance(boot.get("seed"), dict) else int(len(boot["seed"].get("landmarks", []))),
        },
    )

    if not bool(boot["ok"]) or not isinstance(boot.get("seed"), dict):
        print("bootstrap failed; stopping")
        return

    seed = boot["seed"]
    keyframe_feats = seed["feats1"]
    keyframe_index = i1

    print(f"initial landmarks: {len(seed.get('landmarks', []))}")
    print(f"keyframe index: {keyframe_index}")

    start_track = keyframe_index + 1
    stop_track = min(n_effective, start_track + num_track)

    for i in range(start_track, stop_track):
        # Keep the current reference keyframe for this diagnostic step
        ref_keyframe_index = int(keyframe_index)
        ref_keyframe_feats = keyframe_feats

        # Read the current frame image
        cur_im, cur_ts, cur_id = seq.get(i)
        cur_image_shape = (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1]))

        # Track the current frame against the active keyframe
        track_out = track_against_keyframe(
            K,
            ref_keyframe_feats,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
        )

        # Build the shared pose frontend keyword block
        pose_kwargs = pnp_frontend_kwargs_from_cfg(pnp_cfg)
        pose_kwargs.update(
            {
                "threshold_px": float(frontend_kwargs["pnp_threshold_px"]),
                "min_landmark_observations": int(min_landmark_observations),
                "image_shape": cur_image_shape,
            }
        )

        # Build the unfiltered pose output for before/after diagnostics
        base_pose_unfiltered_kwargs = dict(pose_kwargs)
        base_pose_unfiltered_kwargs["enable_pnp_local_consistency_filter"] = False
        base_pose_unfiltered_kwargs["enable_pnp_spatial_thinning_filter"] = False
        base_pose_unfiltered_out = estimate_pose_from_seed(
            K,
            seed,
            track_out,
            **base_pose_unfiltered_kwargs,
        )

        # Build the active pose output with the optional diagnostic filters
        if bool(pnp_cfg["enable_pnp_local_consistency_filter"]) or bool(pnp_cfg["enable_pnp_spatial_thinning_filter"]):
            base_pose_active_kwargs = dict(pose_kwargs)
            base_pose_active_kwargs["enable_pnp_local_consistency_filter"] = bool(pnp_cfg["enable_pnp_local_consistency_filter"])
            base_pose_active_kwargs["enable_pnp_spatial_thinning_filter"] = bool(pnp_cfg["enable_pnp_spatial_thinning_filter"])
            base_pose_out = estimate_pose_from_seed(
                K,
                seed,
                track_out,
                **base_pose_active_kwargs,
            )
        else:
            base_pose_out = base_pose_unfiltered_out

        corrs = base_pose_out["corrs"]
        corrs_before_filters = base_pose_unfiltered_out["corrs"]
        track_stats = track_out.get("stats", {})
        base_pose_stats = base_pose_out.get("stats", {})
        base_pose_unfiltered_stats = base_pose_unfiltered_out.get("stats", {})

        # Run the diagnostic-only non-PnP reprojection analysis on frame 4
        non_pnp_reprojection_diag = None
        local_consistency_diag = None
        spatial_thinning_diag = None
        pnp_pose_comparison_before_diag = None
        pnp_pose_comparison_diag = None
        pnp_spatial_before_diag = None
        pnp_spatial_diag = None
        if int(i) == 4:
            pnp_pose_pair_before = _frame4_pnp_8px_12px_pose_pair(
                corrs_before_filters,
                K,
                pnp_cfg=pnp_cfg,
                image_shape=cur_image_shape,
            )
            pnp_pose_pair = pnp_pose_pair_before
            if bool(pnp_cfg["enable_pnp_local_consistency_filter"]) or bool(pnp_cfg["enable_pnp_spatial_thinning_filter"]):
                pnp_pose_pair = _frame4_pnp_8px_12px_pose_pair(
                    corrs,
                    K,
                    pnp_cfg=pnp_cfg,
                    image_shape=cur_image_shape,
                )
            non_pnp_reprojection_diag = _analyse_non_pnp_pose_eligible_reprojection(
                seed,
                base_pose_out,
                K,
                eps=float(pnp_cfg["eps"]),
            )
            local_consistency_diag = _build_frame4_local_consistency_diag(
                seq,
                ref_keyframe_index,
                i,
                ref_keyframe_feats,
                corrs_before_filters,
                pose_pair=pnp_pose_pair_before,
                pnp_cfg=pnp_cfg,
            )
            spatial_thinning_diag = _build_frame4_spatial_thinning_diag(
                seq,
                ref_keyframe_index,
                i,
                ref_keyframe_feats,
                corrs_before_filters,
                pose_pair=pnp_pose_pair_before,
                pnp_cfg=pnp_cfg,
            )
            pnp_pose_comparison_before_diag = _compare_frame4_pnp_8px_12px(
                seed,
                corrs_before_filters,
                K,
                pnp_cfg=pnp_cfg,
                pose_pair=pnp_pose_pair_before,
            )
            pnp_pose_comparison_diag = _compare_frame4_pnp_8px_12px(
                seed,
                corrs,
                K,
                pnp_cfg=pnp_cfg,
                pose_pair=pnp_pose_pair,
            )
            if bool(pnp_cfg["enable_pnp_local_consistency_filter"]) or bool(pnp_cfg["enable_pnp_spatial_thinning_filter"]):
                pnp_spatial_before_diag = _build_frame4_pnp_spatial_diag(
                    seq,
                    ref_keyframe_index,
                    i,
                    ref_keyframe_feats,
                    track_out,
                    corrs_before_filters,
                    out_dir,
                    pose_pair=pnp_pose_pair_before,
                    name_suffix="before_filters",
                )
            pnp_spatial_diag = _build_frame4_pnp_spatial_diag(
                seq,
                ref_keyframe_index,
                i,
                ref_keyframe_feats,
                track_out,
                corrs,
                out_dir,
                pose_pair=pnp_pose_pair,
                name_suffix="after_filters" if bool(pnp_cfg["enable_pnp_local_consistency_filter"]) or bool(pnp_cfg["enable_pnp_spatial_thinning_filter"]) else "",
            )

        # Sweep PnP thresholds on the same correspondence bundle
        diag_rows = [
            _run_threshold_diag(corrs, K, threshold_px=float(threshold_px), pnp_cfg=pnp_cfg, image_shape=cur_image_shape)
            for threshold_px in thresholds
        ]

        # Advance the real frontend state after recording diagnostics
        seed_landmarks_before = int(len(seed.get("landmarks", [])))
        pipeline_pnp_kwargs = dict(pose_kwargs)
        pipeline_pnp_kwargs["enable_pnp_local_consistency_filter"] = bool(pnp_cfg["apply_pnp_local_consistency_filter_to_pipeline"])
        pipeline_pnp_kwargs["enable_pnp_spatial_thinning_filter"] = bool(pnp_cfg["apply_pnp_spatial_thinning_filter_to_pipeline"])
        frontend_out = process_frame_against_seed(
            K,
            seed,
            ref_keyframe_feats,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            **pipeline_pnp_kwargs,
            keyframe_kf=ref_keyframe_index,
            current_kf=i,
            max_append_reproj_error_px_existing=float(max_append_reproj_error_px_existing),
        )
        frontend_stats = frontend_out.get("stats", {}) if isinstance(frontend_out, dict) else {}
        seed_landmarks_after = int(len(frontend_out.get("seed", {}).get("landmarks", []))) if isinstance(frontend_out, dict) else seed_landmarks_before
        append_stats = {}
        if isinstance(frontend_out, dict) and isinstance(frontend_out.get("seed", None), dict):
            append_stats = frontend_out["seed"].get("last_tracked_observation_append_stats", {})
            if not isinstance(append_stats, dict):
                append_stats = {}
            if int(append_stats.get("current_kf", -1)) != int(i):
                append_stats = {}

        # Record the frame context in the JSONL log
        frame_context = {
            "frame_index": int(i),
            "frame_id": str(cur_id),
            "timestamp": float(cur_ts),
            "reference_keyframe_index": int(ref_keyframe_index),
            "seed_landmarks_before": int(seed_landmarks_before),
            "seed_landmarks_after": int(seed_landmarks_after),
            "track_ok": int(track_stats.get("n_inliers", 0)) > 0,
            "n_track_matches": int(track_stats.get("n_matches", 0)),
            "n_track_inliers": int(track_stats.get("n_inliers", 0)),
            "base_pose_ok": bool(base_pose_out.get("ok", False)),
            "base_pose_reason": base_pose_stats.get("reason", None),
            "base_n_pnp_corr": int(base_pose_stats.get("n_corr", 0)),
            "base_n_pnp_corr_raw": int(base_pose_stats.get("n_corr_raw", 0)),
            "base_n_pnp_corr_bootstrap_born": int(base_pose_stats.get("n_corr_bootstrap_born", 0)),
            "base_n_pnp_corr_post_bootstrap_born": int(base_pose_stats.get("n_corr_post_bootstrap_born", 0)),
            "base_n_pnp_corr_before_local_consistency": int(base_pose_unfiltered_stats.get("n_corr", 0)),
            "base_n_pnp_corr_before_spatial_thinning": int(base_pose_unfiltered_stats.get("n_corr", 0)),
            "n_corr_after_pose_filter": int(base_pose_stats.get("n_corr_after_pose_filter", base_pose_stats.get("n_corr", 0))),
            "n_corr_after_local_consistency_filter": int(
                base_pose_stats.get("n_corr_after_local_consistency_filter", base_pose_stats.get("n_corr", 0))
            ),
            "n_corr_after_spatial_thinning_filter": int(
                base_pose_stats.get("n_corr_after_spatial_thinning_filter", base_pose_stats.get("n_corr", 0))
            ),
            "n_corr_bootstrap_used": int(base_pose_stats.get("n_corr_bootstrap_used", 0)),
            "n_corr_post_bootstrap_used": int(base_pose_stats.get("n_corr_post_bootstrap_used", 0)),
            "n_corr_bootstrap_after_local_consistency": int(base_pose_stats.get("n_corr_bootstrap_after_local_consistency", 0)),
            "n_corr_post_bootstrap_after_local_consistency": int(base_pose_stats.get("n_corr_post_bootstrap_after_local_consistency", 0)),
            "n_corr_bootstrap_after_spatial_thinning": int(base_pose_stats.get("n_corr_bootstrap_after_spatial_thinning", 0)),
            "n_corr_post_bootstrap_after_spatial_thinning": int(base_pose_stats.get("n_corr_post_bootstrap_after_spatial_thinning", 0)),
            "pnp_local_consistency_filter_enabled": bool(base_pose_stats.get("pnp_local_consistency_filter_enabled", False)),
            "pnp_local_consistency_filter_apply_to_pipeline": bool(pnp_cfg["apply_pnp_local_consistency_filter_to_pipeline"]),
            "pnp_local_consistency_filter_evaluated": bool(base_pose_stats.get("pnp_local_consistency_filter_evaluated", False)),
            "pnp_local_consistency_filter_removed": int(base_pose_stats.get("pnp_local_consistency_filter_removed", 0)),
            "pnp_local_consistency_radius_px": float(base_pose_stats.get("pnp_local_consistency_radius_px", pnp_cfg["pnp_local_consistency_radius_px"])),
            "pnp_local_consistency_min_neighbours": int(
                base_pose_stats.get("pnp_local_consistency_min_neighbours", pnp_cfg["pnp_local_consistency_min_neighbours"])
            ),
            "pnp_local_consistency_max_median_residual_px": float(
                base_pose_stats.get(
                    "pnp_local_consistency_max_median_residual_px",
                    pnp_cfg["pnp_local_consistency_max_median_residual_px"],
                )
            ),
            "pnp_spatial_thinning_filter_enabled": bool(base_pose_stats.get("pnp_spatial_thinning_filter_enabled", False)),
            "pnp_spatial_thinning_filter_apply_to_pipeline": bool(pnp_cfg["apply_pnp_spatial_thinning_filter_to_pipeline"]),
            "pnp_spatial_thinning_filter_evaluated": bool(base_pose_stats.get("pnp_spatial_thinning_filter_evaluated", False)),
            "pnp_spatial_thinning_filter_removed": int(base_pose_stats.get("pnp_spatial_thinning_filter_removed", 0)),
            "pnp_spatial_thinning_radius_px": float(
                base_pose_stats.get("pnp_spatial_thinning_radius_px", pnp_cfg["pnp_spatial_thinning_radius_px"])
            ),
            "pnp_spatial_thinning_max_points_per_radius": int(
                base_pose_stats.get(
                    "pnp_spatial_thinning_max_points_per_radius",
                    pnp_cfg["pnp_spatial_thinning_max_points_per_radius"],
                )
            ),
            "base_n_pnp_inliers": int(base_pose_stats.get("n_pnp_inliers", 0)),
            "base_pnp_component_gate_enabled": bool(base_pose_stats.get("pnp_component_gate_enabled", pnp_cfg["enable_pnp_component_gate"])),
            "base_pnp_component_gate_evaluated": bool(base_pose_stats.get("pnp_component_gate_evaluated", False)),
            "base_pnp_component_gate_rejected": bool(base_pose_stats.get("pnp_component_gate_rejected", False)),
            "base_pnp_component_gate_reason": base_pose_stats.get("pnp_component_gate_reason", None),
            "base_pnp_component_radius_px": float(base_pose_stats.get("pnp_component_radius_px", pnp_cfg["pnp_component_radius_px"])),
            "base_pnp_inlier_component_count": int(base_pose_stats.get("pnp_inlier_component_count", 0)),
            "base_pnp_inlier_largest_component_size": int(base_pose_stats.get("pnp_inlier_largest_component_size", 0)),
            "base_pnp_inlier_largest_component_fraction": base_pose_stats.get("pnp_inlier_largest_component_fraction", None),
            "base_pnp_inlier_largest_component_bbox_area_fraction": base_pose_stats.get("pnp_inlier_largest_component_bbox_area_fraction", None),
            "base_pnp_threshold_stability": base_pose_stats.get("pnp_threshold_stability", None),
            "base_pnp_threshold_stability_evaluated": bool(base_pose_stats.get("pnp_threshold_stability_evaluated", False)),
            "base_pnp_threshold_stability_classification": base_pose_stats.get("pnp_threshold_stability_classification", "unavailable"),
            "base_pnp_threshold_stability_unstable": bool(base_pose_stats.get("pnp_threshold_stability_unstable", False)),
            "base_pnp_threshold_stability_ref_inliers": int(base_pose_stats.get("pnp_threshold_stability_ref_inliers", 0)),
            "base_pnp_threshold_stability_compare_inliers": int(base_pose_stats.get("pnp_threshold_stability_compare_inliers", 0)),
            "base_pnp_threshold_stability_support_iou": base_pose_stats.get("pnp_threshold_stability_support_iou", None),
            "base_pnp_threshold_stability_rotation_delta_deg": base_pose_stats.get("pnp_threshold_stability_rotation_delta_deg", None),
            "base_pnp_threshold_stability_translation_direction_delta_deg": base_pose_stats.get("pnp_threshold_stability_translation_direction_delta_deg", None),
            "base_pnp_threshold_stability_camera_centre_direction_delta_deg": base_pose_stats.get("pnp_threshold_stability_camera_centre_direction_delta_deg", None),
            "base_pnp_threshold_stability_looser_solution_only": bool(base_pose_stats.get("pnp_threshold_stability_looser_solution_only", False)),
            "base_pnp_threshold_stability_supports_disjoint": bool(base_pose_stats.get("pnp_threshold_stability_supports_disjoint", False)),
            "base_pnp_threshold_stability_reasons": base_pose_stats.get("pnp_threshold_stability_reasons", []),
            "min_landmark_observations": int(base_pose_stats.get("min_landmark_observations", min_landmark_observations)),
            "allow_bootstrap_landmarks_for_pose": bool(
                base_pose_stats.get("allow_bootstrap_landmarks_for_pose", pnp_cfg["allow_bootstrap_landmarks_for_pose"])
            ),
            "min_post_bootstrap_observations_for_pose": int(
                base_pose_stats.get(
                    "min_post_bootstrap_observations_for_pose",
                    pnp_cfg["min_post_bootstrap_observations_for_pose"],
                )
            ),
            "landmark_observation_histogram": base_pose_stats.get("landmark_observation_histogram", {}),
            "configured_threshold_px": float(frontend_kwargs["pnp_threshold_px"]),
            "max_append_reproj_error_px_existing": float(max_append_reproj_error_px_existing),
            "pnp_spatial_gate_enabled": bool(frontend_stats.get("pnp_spatial_gate_enabled", pnp_cfg["enable_pnp_spatial_gate"])),
            "pnp_spatial_gate_rejected": bool(frontend_stats.get("pnp_spatial_gate_rejected", False)),
            "pnp_spatial_gate_reason": frontend_stats.get("pnp_spatial_gate_reason", None),
            "pnp_inlier_occupied_cells": int(frontend_stats.get("pnp_inlier_occupied_cells", 0)),
            "pnp_inlier_max_cell_fraction": frontend_stats.get("pnp_inlier_max_cell_fraction", None),
            "pnp_inlier_bbox_area_fraction": frontend_stats.get("pnp_inlier_bbox_area_fraction", None),
            "pnp_component_gate_enabled": bool(frontend_stats.get("pnp_component_gate_enabled", pnp_cfg["enable_pnp_component_gate"])),
            "pnp_component_gate_evaluated": bool(frontend_stats.get("pnp_component_gate_evaluated", False)),
            "pnp_component_gate_rejected": bool(frontend_stats.get("pnp_component_gate_rejected", False)),
            "pnp_component_gate_reason": frontend_stats.get("pnp_component_gate_reason", None),
            "pnp_component_radius_px": float(frontend_stats.get("pnp_component_radius_px", pnp_cfg["pnp_component_radius_px"])),
            "pnp_inlier_component_count": int(frontend_stats.get("pnp_inlier_component_count", 0)),
            "pnp_inlier_largest_component_size": int(frontend_stats.get("pnp_inlier_largest_component_size", 0)),
            "pnp_inlier_largest_component_fraction": frontend_stats.get("pnp_inlier_largest_component_fraction", None),
            "pnp_inlier_largest_component_bbox_area_fraction": frontend_stats.get("pnp_inlier_largest_component_bbox_area_fraction", None),
            "pnp_threshold_stability": frontend_stats.get("pnp_threshold_stability", None),
            "pnp_threshold_stability_evaluated": bool(frontend_stats.get("pnp_threshold_stability_evaluated", False)),
            "pnp_threshold_stability_classification": frontend_stats.get("pnp_threshold_stability_classification", "unavailable"),
            "pnp_threshold_stability_unstable": bool(frontend_stats.get("pnp_threshold_stability_unstable", False)),
            "pnp_threshold_stability_ref_inliers": int(frontend_stats.get("pnp_threshold_stability_ref_inliers", 0)),
            "pnp_threshold_stability_compare_inliers": int(frontend_stats.get("pnp_threshold_stability_compare_inliers", 0)),
            "pnp_threshold_stability_support_iou": frontend_stats.get("pnp_threshold_stability_support_iou", None),
            "pnp_threshold_stability_rotation_delta_deg": frontend_stats.get("pnp_threshold_stability_rotation_delta_deg", None),
            "pnp_threshold_stability_translation_direction_delta_deg": frontend_stats.get("pnp_threshold_stability_translation_direction_delta_deg", None),
            "pnp_threshold_stability_camera_centre_direction_delta_deg": frontend_stats.get("pnp_threshold_stability_camera_centre_direction_delta_deg", None),
            "pnp_threshold_stability_looser_solution_only": bool(frontend_stats.get("pnp_threshold_stability_looser_solution_only", False)),
            "pnp_threshold_stability_supports_disjoint": bool(frontend_stats.get("pnp_threshold_stability_supports_disjoint", False)),
            "pnp_threshold_stability_reasons": frontend_stats.get("pnp_threshold_stability_reasons", []),
            "pipeline_ok": bool(frontend_out.get("ok", False)),
            "pipeline_reason": frontend_stats.get("reason", None),
            "pipeline_n_pnp_corr": int(frontend_stats.get("n_pnp_corr", 0)),
            "pipeline_n_pnp_inliers": int(frontend_stats.get("n_pnp_inliers", 0)),
            "pipeline_pnp_local_consistency_filter_removed": int(frontend_stats.get("pnp_local_consistency_filter_removed", 0)),
            "pipeline_pnp_spatial_thinning_filter_removed": int(frontend_stats.get("pnp_spatial_thinning_filter_removed", 0)),
            "n_append_candidates_existing": int(frontend_stats.get("n_append_candidates_existing", 0)),
            "n_append_pnp_inliers": int(frontend_stats.get("n_append_pnp_inliers", 0)),
            "n_append_total": int(frontend_stats.get("n_append_total", 0)),
            "n_append_extra_reproj_pass": int(frontend_stats.get("n_append_extra_reproj_pass", 0)),
            "n_append_duplicates": int(frontend_stats.get("n_append_duplicates", 0)),
            "n_landmarks_with_obs_current_kf_after_append": int(
                frontend_stats.get("n_landmarks_with_obs_current_kf_after_append", 0)
            ),
            "n_append_total_bootstrap_born": int(append_stats.get("n_append_total_bootstrap_born", 0)),
            "n_append_total_map_growth_born": int(append_stats.get("n_append_total_map_growth_born", 0)),
            "n_append_extra_reproj_added_bootstrap_born": int(
                append_stats.get("n_append_extra_reproj_added_bootstrap_born", 0)
            ),
            "n_append_extra_reproj_added_map_growth_born": int(
                append_stats.get("n_append_extra_reproj_added_map_growth_born", 0)
            ),
            "n_linked_landmarks_candidate": int(frontend_stats.get("n_linked_landmarks_candidate", 0)),
            "pipeline_keyframe_promoted": bool(frontend_stats.get("keyframe_promoted", False)),
            "pipeline_keyframe_reason": frontend_stats.get("keyframe_reason", None),
        }

        # Write the frame summary
        _append_jsonl(
            log_path,
            {
                "event": "frame_summary",
                **frame_context,
            },
        )

        # Write the frame-4 non-PnP reprojection diagnostic
        if non_pnp_reprojection_diag is not None:
            _append_jsonl(
                log_path,
                {
                    "event": "frame4_non_pnp_reprojection",
                    **frame_context,
                    "non_pnp_reprojection": non_pnp_reprojection_diag,
                },
            )

        # Write the frame-4 local consistency diagnostic
        if local_consistency_diag is not None:
            _append_jsonl(
                log_path,
                {
                    "event": "frame4_local_consistency",
                    **frame_context,
                    "local_consistency": local_consistency_diag,
                },
            )

        # Write the frame-4 spatial thinning diagnostic
        if spatial_thinning_diag is not None:
            _append_jsonl(
                log_path,
                {
                    "event": "frame4_spatial_thinning",
                    **frame_context,
                    "spatial_thinning": spatial_thinning_diag,
                },
            )

        # Write the frame-4 before-filter 8 px vs 12 px PnP pose comparison
        if pnp_pose_comparison_before_diag is not None and (bool(pnp_cfg["enable_pnp_local_consistency_filter"]) or bool(pnp_cfg["enable_pnp_spatial_thinning_filter"])):
            _append_jsonl(
                log_path,
                {
                    "event": "frame4_pnp_8px_12px_comparison_before_filters",
                    **frame_context,
                    "pnp_8px_12px_comparison": pnp_pose_comparison_before_diag,
                },
            )

        # Write the frame-4 8 px vs 12 px PnP pose comparison
        if pnp_pose_comparison_diag is not None:
            _append_jsonl(
                log_path,
                {
                    "event": "frame4_pnp_8px_12px_comparison",
                    **frame_context,
                    "pnp_8px_12px_comparison": pnp_pose_comparison_diag,
                },
            )

        # Write the frame-4 before-filter spatial diagnostic
        if pnp_spatial_before_diag is not None:
            _append_jsonl(
                log_path,
                {
                    "event": "frame4_pnp_spatial_summary_before_filters",
                    **frame_context,
                    "pnp_spatial": pnp_spatial_before_diag,
                },
            )

        # Write the frame-4 spatial diagnostic
        if pnp_spatial_diag is not None:
            _append_jsonl(
                log_path,
                {
                    "event": "frame4_pnp_spatial_summary",
                    **frame_context,
                    "pnp_spatial": pnp_spatial_diag,
                },
            )

        # Write one diagnostic row per threshold
        for row in diag_rows:
            _append_jsonl(
                log_path,
                {
                    "event": "pnp_threshold",
                    **frame_context,
                    **row,
                },
            )

        print(
            f"frame {i}: reference_keyframe_index={ref_keyframe_index} "
            f"n_track_inliers={int(track_stats.get('n_inliers', 0))} "
            f"diag_n_pnp_corr={int(base_pose_stats.get('n_corr', 0))} "
            f"diag_n_pnp_inliers={int(base_pose_stats.get('n_pnp_inliers', 0))} "
            f"pnp_local_removed={int(base_pose_stats.get('pnp_local_consistency_filter_removed', 0))} "
            f"pnp_spatial_thinned={int(base_pose_stats.get('pnp_spatial_thinning_filter_removed', 0))} "
            f"pnp_cells={int(base_pose_stats.get('pnp_inlier_occupied_cells', 0))} "
            f"pnp_max_cell_frac={_format_optional_float(base_pose_stats.get('pnp_inlier_max_cell_fraction', None), digits=2)} "
            f"pnp_bbox_area_frac={_format_optional_float(base_pose_stats.get('pnp_inlier_bbox_area_fraction', None), digits=3)} "
            f"pnp_spatial_rejected={bool(base_pose_stats.get('pnp_spatial_gate_rejected', False))} "
            f"pnp_spatial_reason={base_pose_stats.get('pnp_spatial_gate_reason', None)} "
            f"pnp_components={int(base_pose_stats.get('pnp_inlier_component_count', 0))} "
            f"pnp_largest_comp={int(base_pose_stats.get('pnp_inlier_largest_component_size', 0))} "
            f"pnp_largest_comp_frac={_format_optional_float(base_pose_stats.get('pnp_inlier_largest_component_fraction', None), digits=2)} "
            f"pnp_largest_comp_area_frac={_format_optional_float(base_pose_stats.get('pnp_inlier_largest_component_bbox_area_fraction', None), digits=4)} "
            f"pnp_component_rejected={bool(base_pose_stats.get('pnp_component_gate_rejected', False))} "
            f"pnp_component_reason={base_pose_stats.get('pnp_component_gate_reason', None)} "
            f"pipeline_n_pnp_corr={int(frontend_stats.get('n_pnp_corr', 0))} "
            f"n_append_total={int(frontend_stats.get('n_append_total', 0))} "
            f"n_append_extra_reproj_pass={int(frontend_stats.get('n_append_extra_reproj_pass', 0))} "
            f"n_linked_landmarks_candidate={int(frontend_stats.get('n_linked_landmarks_candidate', 0))} "
            f"promoted={bool(frontend_stats.get('keyframe_promoted', False))} "
            f"landmarks={seed_landmarks_before}->{seed_landmarks_after}"
        )
        print(f"  {_format_threshold_summary(diag_rows)}")
        print(_format_pnp_threshold_stability_diag(base_pose_stats.get("pnp_threshold_stability", None)))
        if non_pnp_reprojection_diag is not None:
            for line in _format_non_pnp_reprojection_diag(non_pnp_reprojection_diag):
                print(line)
        if local_consistency_diag is not None:
            for line in _format_frame4_local_consistency_diag(local_consistency_diag):
                print(line)
        if spatial_thinning_diag is not None:
            for line in _format_frame4_spatial_thinning_diag(spatial_thinning_diag):
                print(line)
        if pnp_pose_comparison_before_diag is not None and (bool(pnp_cfg["enable_pnp_local_consistency_filter"]) or bool(pnp_cfg["enable_pnp_spatial_thinning_filter"])):
            print("  frame4_before_filters:")
            for line in _format_frame4_pnp_pose_comparison(pnp_pose_comparison_before_diag):
                print(line)
        if pnp_pose_comparison_diag is not None:
            if bool(pnp_cfg["enable_pnp_local_consistency_filter"]) or bool(pnp_cfg["enable_pnp_spatial_thinning_filter"]):
                print("  frame4_after_filters:")
            for line in _format_frame4_pnp_pose_comparison(pnp_pose_comparison_diag):
                print(line)
        if pnp_spatial_before_diag is not None:
            print("  frame4_spatial_before_filters:")
            for line in _format_frame4_spatial_diag(pnp_spatial_before_diag):
                print(line)
        if pnp_spatial_diag is not None:
            if bool(pnp_cfg["enable_pnp_local_consistency_filter"]) or bool(pnp_cfg["enable_pnp_spatial_thinning_filter"]):
                print("  frame4_spatial_after_filters:")
            for line in _format_frame4_spatial_diag(pnp_spatial_diag):
                print(line)

        # Update the live frontend state for the next frame
        seed = frontend_out["seed"]
        if bool(frontend_out.get("stats", {}).get("keyframe_promoted", False)):
            keyframe_feats = frontend_out["track_out"]["cur_feats"]
            keyframe_index = i


if __name__ == "__main__":
    main()
