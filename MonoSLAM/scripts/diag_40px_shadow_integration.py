# scripts/diag_40px_shadow_integration.py
# Diagnostic-only gated 40 px shadow re-solve over real sequence horizons.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from frontend_eth3d_common import ROOT
from frontend_eth3d_common import frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg
from frontend_eth3d_common import load_runtime_cfg as _load_runtime_cfg
from datasets.loader import load_sequence
from geometry.camera import camera_centre, reprojection_errors_sq
from geometry.pose import angle_between_translations
from geometry.pnp import _pnp_inlier_mask_from_pose, estimate_pose_pnp_ransac
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_active_keyframe_kf


_STRICT_EVAL_PX = 8.0
_SHADOW_THRESHOLD_PX = 40.0
_SHADOW_NUM_TRIALS = 5000
_DEFAULT_GATE_MEDIAN_PX = 8.0
_DEFAULT_SEEDS = [0, 1, 2, 3, 7]

_KNOWN_LABELS = {
    "eth3d": {
        12: "bad canonical pose",
        16: "bad canonical pose",
        17: "load-bearing good refresh",
        18: "neutral",
    },
    "kitti": {
        14: "load-bearing good refresh",
        17: "load-bearing good refresh",
        18: "high-residual neutral or bad rescue",
        20: "neutral",
    },
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _pose_from_pose_out(pose_out: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(pose_out, dict):
        return None
    if pose_out.get("R", None) is None or pose_out.get("t", None) is None:
        return None
    return (
        np.asarray(pose_out["R"], dtype=np.float64).reshape(3, 3),
        np.asarray(pose_out["t"], dtype=np.float64).reshape(3),
    )


def _pose_block(pose: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    R, t = pose
    return {
        "R": np.asarray(R, dtype=np.float64).reshape(3, 3),
        "t": np.asarray(t, dtype=np.float64).reshape(3),
    }


def _direction_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        return float(np.degrees(angle_between_translations(a, b)))
    except Exception:
        return None


def _pose_delta(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    R_a, t_a = a
    R_b, t_b = b
    C_a = camera_centre(R_a, t_a)
    C_b = camera_centre(R_b, t_b)
    return {
        "rotation_delta_deg": float(np.degrees(angle_between_rotmats(R_a, R_b))),
        "translation_direction_delta_deg": _direction_deg(t_a, t_b),
        "camera_centre_direction_delta_deg": _direction_deg(C_a, C_b),
        "camera_centre_distance": float(np.linalg.norm(C_b - C_a)),
    }


def _residuals(
    K: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    X_w: np.ndarray,
    xy: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    R, t = pose
    X_w = np.asarray(X_w, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    if X_w.ndim != 2 or xy.ndim != 2 or X_w.shape[0] != 3 or xy.shape[0] != 2:
        return np.zeros((0,), dtype=np.float64)
    if int(X_w.shape[1]) == 0:
        return np.zeros((0,), dtype=np.float64)
    depth = np.asarray((R @ X_w + t.reshape(3, 1))[2, :], dtype=np.float64).reshape(-1)
    err_sq = np.asarray(reprojection_errors_sq(K, R, t, X_w, xy), dtype=np.float64).reshape(-1)
    valid = np.isfinite(depth) & (depth > float(eps)) & np.isfinite(err_sq) & (err_sq >= 0.0)
    out = np.full((int(err_sq.size),), np.nan, dtype=np.float64)
    out[valid] = np.sqrt(err_sq[valid])
    return out


def _summary(errors_px: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(errors_px, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "count": 0,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "squared_error": 0.0,
            "above_8_count": 0,
            "above_8_fraction": None,
        }
    above_8 = int(np.sum(arr > _STRICT_EVAL_PX))
    return {
        "count": n,
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90.0)),
        "max_px": float(np.max(arr)),
        "squared_error": float(np.sum(arr * arr)),
        "above_8_count": int(above_8),
        "above_8_fraction": float(above_8 / max(n, 1)),
    }


def _sq_reduction(base: float, candidate: float) -> float | None:
    if float(base) <= 0.0:
        return None
    return float((float(base) - float(candidate)) / float(base))


def _mask_from_pose(corrs, K: np.ndarray, pose: tuple[np.ndarray, np.ndarray], *, threshold_px: float, eps: float) -> np.ndarray:
    R, t = pose
    mask, _ = _pnp_inlier_mask_from_pose(
        corrs.X_w,
        corrs.x_cur,
        K,
        R,
        t,
        threshold_px=float(threshold_px),
        eps=float(eps),
    )
    return np.asarray(mask, dtype=bool).reshape(-1)


def _summary_on_corrs(
    K: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    corrs,
    *,
    eps: float,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    X_w = np.asarray(corrs.X_w, dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        X_w = X_w[:, mask]
        x_cur = x_cur[:, mask]
    return _summary(_residuals(K, pose, X_w, x_cur, eps=float(eps)))


def _support_ids_from_pose_out(pose_out: dict[str, Any]) -> tuple[list[int], list[int]]:
    if not isinstance(pose_out, dict) or pose_out.get("corrs", None) is None:
        return [], []
    corrs = pose_out["corrs"]
    ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    mask = np.asarray(
        pose_out.get("pnp_inlier_mask", np.zeros((int(ids.size),), dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if int(mask.size) != int(ids.size):
        mask = np.zeros((int(ids.size),), dtype=bool)
    return [int(v) for v in ids], [int(v) for v in ids[mask]]


def _run_shadow_seed(
    corrs,
    K: np.ndarray,
    pnp_cfg: dict[str, Any],
    *,
    seed_value: int,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict[str, Any]]:
    n_corr = int(corrs.X_w.shape[1])
    sample_size = int(pnp_cfg.get("sample_size", 6))
    row: dict[str, Any] = {
        "seed": int(seed_value),
        "ok": False,
        "reason": None,
        "n_corr": int(n_corr),
        "strict_8px_inliers": 0,
        "shadow_40px_inliers": 0,
        "summary_on_full_corrs": _summary(np.zeros((0,), dtype=np.float64)),
    }
    if n_corr < sample_size:
        row["reason"] = "too_few_correspondences"
        return None, row
    try:
        R, t, mask_40, stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=int(_SHADOW_NUM_TRIALS),
            sample_size=sample_size,
            threshold_px=float(_SHADOW_THRESHOLD_PX),
            min_inliers=int(pnp_cfg.get("min_inliers", 8)),
            seed=int(seed_value),
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
        row["reason"] = "shadow_pnp_error"
        row["error"] = str(exc)
        return None, row

    stats = stats if isinstance(stats, dict) else {}
    row["reason"] = stats.get("reason", None)
    if R is None or t is None:
        return None, row

    pose = (
        np.asarray(R, dtype=np.float64).reshape(3, 3),
        np.asarray(t, dtype=np.float64).reshape(3),
    )
    eps = float(pnp_cfg["eps"])
    mask_40 = np.asarray(mask_40, dtype=bool).reshape(-1)
    strict_mask = _mask_from_pose(corrs, K, pose, threshold_px=float(_STRICT_EVAL_PX), eps=eps)
    row.update(
        {
            "ok": True,
            "reason": None,
            "shadow_40px_inliers": int(np.sum(mask_40)),
            "strict_8px_inliers": int(np.sum(strict_mask)),
            "solver_n_model_success": int(stats.get("n_model_success", 0)),
            "solver_refit": bool(stats.get("refit", False)),
            "summary_on_full_corrs": _summary_on_corrs(K, pose, corrs, eps=eps),
        }
    )
    return pose, row


def _select_shadow_pose(
    corrs,
    K: np.ndarray,
    pnp_cfg: dict[str, Any],
    *,
    seeds: list[int],
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    best_pose: tuple[np.ndarray, np.ndarray] | None = None
    best_row: dict[str, Any] | None = None
    best_key: tuple[int, float, float, int] | None = None

    for seed_value in seeds:
        pose, row = _run_shadow_seed(corrs, K, pnp_cfg, seed_value=int(seed_value))
        per_seed.append(row)
        if pose is None:
            continue
        summary = row["summary_on_full_corrs"]
        median_px = summary.get("median_px", None)
        p90_px = summary.get("p90_px", None)
        key = (
            -int(row.get("strict_8px_inliers", 0)),
            float("inf") if median_px is None else float(median_px),
            float("inf") if p90_px is None else float(p90_px),
            int(seed_value),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_pose = pose
            best_row = row

    return {
        "seeds_tried": [int(v) for v in seeds],
        "num_seeds_tried": int(len(seeds)),
        "num_seeds_succeeded": int(sum(1 for row in per_seed if bool(row.get("ok", False)))),
        "selection_rule": "max_strict_8px_inliers_then_lowest_fullset_median",
        "per_seed": per_seed,
        "best_seed": None if best_row is None else int(best_row["seed"]),
        "best_pose": _pose_block(best_pose),
        "best_summary_on_full_corrs": None if best_row is None else best_row["summary_on_full_corrs"],
        "best_strict_8px_inliers": 0 if best_row is None else int(best_row["strict_8px_inliers"]),
        "best_shadow_40px_inliers": 0 if best_row is None else int(best_row["shadow_40px_inliers"]),
    }


def _local_comparison(
    accepted_full: dict[str, Any],
    shadow_full: dict[str, Any] | None,
    accepted_strict_8px: int,
    shadow_strict_8px: int,
) -> str:
    if shadow_full is None:
        return "worse"
    acc_median = accepted_full.get("median_px", None)
    sh_median = shadow_full.get("median_px", None)
    acc_p90 = accepted_full.get("p90_px", None)
    sh_p90 = shadow_full.get("p90_px", None)
    if acc_median is None or sh_median is None:
        return "same"
    strict_gain = int(shadow_strict_8px) - int(accepted_strict_8px)
    median_ratio = float(sh_median) / max(float(acc_median), 1e-12)
    p90_ok = acc_p90 is None or sh_p90 is None or float(sh_p90) <= float(acc_p90)
    if strict_gain > 0 and float(sh_median) < float(acc_median):
        return "better"
    if median_ratio <= 0.80 and bool(p90_ok):
        return "better"
    if strict_gain < 0 or median_ratio > 1.10:
        return "worse"
    return "same"


def _frame_observation_bundle(
    seed: dict[str, Any],
    frame_index: int,
    landmark_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    wanted = set(int(v) for v in landmark_ids)
    X_cols: list[np.ndarray] = []
    xy_cols: list[np.ndarray] = []
    used_ids: list[int] = []
    for lm in seed.get("landmarks", []):
        if not isinstance(lm, dict) or "id" not in lm:
            continue
        lm_id = int(lm["id"])
        if lm_id not in wanted:
            continue
        X_w = np.asarray(lm.get("X_w", None), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue
        for obs in lm.get("obs", []):
            if not isinstance(obs, dict):
                continue
            if int(obs.get("kf", -1)) != int(frame_index):
                continue
            xy = np.asarray(obs.get("xy", None), dtype=np.float64).reshape(-1)
            if xy.size == 2 and np.isfinite(xy).all():
                X_cols.append(X_w.reshape(3, 1))
                xy_cols.append(xy.reshape(2, 1))
                used_ids.append(int(lm_id))
                break
    if len(X_cols) == 0:
        return np.zeros((3, 0), dtype=np.float64), np.zeros((2, 0), dtype=np.float64), []
    return np.hstack(X_cols), np.hstack(xy_cols), used_ids


def _history_impact_estimate(
    final_seed: dict[str, Any],
    K: np.ndarray,
    event: dict[str, Any],
    *,
    eps: float,
) -> dict[str, Any]:
    shadow_pose_raw = event.get("shadow_pose", None)
    accepted_pose_raw = event.get("accepted_pose", None)
    if not isinstance(shadow_pose_raw, dict) or not isinstance(accepted_pose_raw, dict):
        return {
            "available": False,
            "reason": "pose_missing",
        }
    accepted_pose = (
        np.asarray(accepted_pose_raw["R"], dtype=np.float64).reshape(3, 3),
        np.asarray(accepted_pose_raw["t"], dtype=np.float64).reshape(3),
    )
    shadow_pose = (
        np.asarray(shadow_pose_raw["R"], dtype=np.float64).reshape(3, 3),
        np.asarray(shadow_pose_raw["t"], dtype=np.float64).reshape(3),
    )
    X_w, xy, used_ids = _frame_observation_bundle(
        final_seed,
        int(event["frame_index"]),
        [int(v) for v in event.get("full_correspondence_landmark_ids", [])],
    )
    accepted_summary = _summary(_residuals(K, accepted_pose, X_w, xy, eps=float(eps)))
    shadow_summary = _summary(_residuals(K, shadow_pose, X_w, xy, eps=float(eps)))
    return {
        "available": True,
        "observed_count": int(len(used_ids)),
        "landmark_ids": used_ids,
        "accepted_observation_summary": accepted_summary,
        "shadow_observation_summary": shadow_summary,
        "squared_error_reduction": _sq_reduction(
            float(accepted_summary["squared_error"]),
            float(shadow_summary["squared_error"]),
        ),
        "above8_reduction": int(accepted_summary["above_8_count"] - shadow_summary["above_8_count"]),
    }


def _downstream_reuse_estimate(
    records: dict[int, dict[str, Any]],
    event: dict[str, Any],
    *,
    window: int = 3,
) -> dict[str, Any]:
    support_ids = set(int(v) for v in event.get("accepted_support_landmark_ids", []))
    frame_index = int(event["frame_index"])
    rows: list[dict[str, Any]] = []
    for downstream_frame in range(frame_index + 1, frame_index + int(window) + 1):
        record = records.get(int(downstream_frame), None)
        if not isinstance(record, dict):
            rows.append({"frame_index": int(downstream_frame), "available": False})
            continue
        pose_ids = set(int(v) for v in record.get("pose_eligible_ids", []))
        inlier_ids = set(int(v) for v in record.get("accepted_inlier_ids", []))
        rows.append(
            {
                "frame_index": int(downstream_frame),
                "available": True,
                "pipeline_ok": bool(record.get("pipeline_ok", False)),
                "active_basis_before_kf": int(record.get("active_basis_before_kf", -1)),
                "pose_eligible_reuse": int(len(support_ids & pose_ids)),
                "accepted_inlier_reuse": int(len(support_ids & inlier_ids)),
                "final_pnp_inliers": int(record.get("n_pnp_inliers", 0)),
                "final_pnp_correspondences": int(record.get("n_pnp_corr", 0)),
            }
        )
    n_available = int(sum(1 for row in rows if bool(row.get("available", False))))
    n_ok = int(sum(1 for row in rows if bool(row.get("available", False)) and bool(row.get("pipeline_ok", False))))
    any_reused = int(sum(1 for row in rows if int(row.get("accepted_inlier_reuse", 0)) > 0))
    return {
        "window": int(window),
        "support_count": int(len(support_ids)),
        "available_frames": int(n_available),
        "ok_frames": int(n_ok),
        "frames_with_accepted_inlier_reuse": int(any_reused),
        "rows": rows,
    }


def _record_frame(
    dataset_key: str,
    frame_index: int,
    output: dict[str, Any],
    *,
    active_basis_before_kf: int,
) -> dict[str, Any]:
    stats = output.get("stats", {}) if isinstance(output, dict) else {}
    stats = stats if isinstance(stats, dict) else {}
    pose_out = output.get("pose_out", {}) if isinstance(output, dict) else {}
    pose_ids, inlier_ids = _support_ids_from_pose_out(pose_out)
    seed_after = output.get("seed", {}) if isinstance(output, dict) else {}
    return {
        "dataset": str(dataset_key),
        "frame_index": int(frame_index),
        "pipeline_ok": bool(output.get("ok", False)) if isinstance(output, dict) else False,
        "pipeline_reason": stats.get("reason", None),
        "active_basis_before_kf": int(active_basis_before_kf),
        "active_basis_after_kf": int(seed_after.get("active_keyframe_kf", -1)) if isinstance(seed_after, dict) else -1,
        "accepted_pose_type": (
            "failed"
            if not bool(output.get("ok", False))
            else "rescue"
            if bool(stats.get("localisation_only_rescue_frame", False))
            else "normal"
        ),
        "known_label": _KNOWN_LABELS.get(str(dataset_key), {}).get(int(frame_index), "unlabelled"),
        "rescue_attempted": bool(stats.get("pnp_support_rescue_attempted", False)),
        "rescue_succeeded": bool(stats.get("pnp_support_rescue_succeeded", False)),
        "loose_localisation_fallback": bool(stats.get("pnp_support_rescue_loose_localisation_fallback_succeeded", False)),
        "rescue_loose_threshold_px": stats.get("pnp_support_rescue_loose_threshold_px", None),
        "support_refresh_triggered": bool(stats.get("guarded_support_refresh_triggered", False)),
        "support_refresh_reason": stats.get("guarded_support_refresh_reason", None),
        "n_pnp_corr": int(stats.get("n_pnp_corr", 0)),
        "n_pnp_inliers": int(stats.get("n_pnp_inliers", 0)),
        "pose_eligible_ids": sorted(set(int(v) for v in pose_ids)),
        "accepted_inlier_ids": sorted(set(int(v) for v in inlier_ids)),
    }


def _maybe_shadow_event(
    dataset_key: str,
    frame_index: int,
    output: dict[str, Any],
    K: np.ndarray,
    pnp_cfg: dict[str, Any],
    *,
    gate_median_px: float,
    seeds: list[int],
) -> dict[str, Any] | None:
    stats = output.get("stats", {}) if isinstance(output, dict) else {}
    stats = stats if isinstance(stats, dict) else {}
    if not bool(output.get("ok", False)):
        return None
    if not bool(stats.get("pnp_support_rescue_succeeded", False)):
        return None

    pose_out = output.get("pose_out", {})
    if not isinstance(pose_out, dict) or pose_out.get("corrs", None) is None:
        return None
    accepted_pose = _pose_from_pose_out(pose_out)
    if accepted_pose is None:
        return None

    corrs = pose_out["corrs"]
    n_corr = int(corrs.X_w.shape[1])
    eps = float(pnp_cfg["eps"])
    raw_mask = np.asarray(
        pose_out.get("pnp_inlier_mask", np.zeros((n_corr,), dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if int(raw_mask.size) != int(n_corr):
        raw_mask = np.zeros((n_corr,), dtype=bool)

    accepted_full_summary = _summary_on_corrs(K, accepted_pose, corrs, eps=eps)
    accepted_support_summary = _summary_on_corrs(K, accepted_pose, corrs, eps=eps, mask=raw_mask)
    accepted_strict_mask = _mask_from_pose(
        corrs,
        K,
        accepted_pose,
        threshold_px=float(_STRICT_EVAL_PX),
        eps=eps,
    )
    accepted_strict_8px = int(np.sum(accepted_strict_mask))
    trigger_median = accepted_support_summary.get("median_px", None)
    trigger_fired = bool(trigger_median is not None and float(trigger_median) > float(gate_median_px))
    full_ids, accepted_ids = _support_ids_from_pose_out(pose_out)

    event: dict[str, Any] = {
        "dataset": str(dataset_key),
        "frame_index": int(frame_index),
        "known_label": _KNOWN_LABELS.get(str(dataset_key), {}).get(int(frame_index), "unlabelled"),
        "trigger_fired": bool(trigger_fired),
        "trigger_signal": "accepted_inlier_residual_median_px",
        "gate_median_px": float(gate_median_px),
        "trigger_value_px": trigger_median,
        "accepted_pose": _pose_block(accepted_pose),
        "accepted_support_landmark_ids": sorted(set(int(v) for v in accepted_ids)),
        "full_correspondence_landmark_ids": sorted(set(int(v) for v in full_ids)),
        "n_full_correspondences": int(n_corr),
        "n_accepted_support": int(np.sum(raw_mask)),
        "accepted_full_corrs_summary": accepted_full_summary,
        "accepted_inlier_support_summary": accepted_support_summary,
        "accepted_strict_8px_inliers_on_full_corrs": int(accepted_strict_8px),
        "shadow": None,
        "shadow_pose": None,
        "shadow_full_corrs_summary": None,
        "shadow_strict_8px_inliers_on_full_corrs": 0,
        "shadow_vs_accepted_delta": None,
        "local_classification": "not_triggered",
    }

    if not bool(trigger_fired):
        return event

    shadow = _select_shadow_pose(corrs, K, pnp_cfg, seeds=seeds)
    event["shadow"] = shadow
    event["shadow_pose"] = shadow.get("best_pose", None)
    event["shadow_full_corrs_summary"] = shadow.get("best_summary_on_full_corrs", None)
    event["shadow_strict_8px_inliers_on_full_corrs"] = int(shadow.get("best_strict_8px_inliers", 0))
    if isinstance(event["shadow_pose"], dict):
        shadow_pose = (
            np.asarray(event["shadow_pose"]["R"], dtype=np.float64).reshape(3, 3),
            np.asarray(event["shadow_pose"]["t"], dtype=np.float64).reshape(3),
        )
        event["shadow_vs_accepted_delta"] = _pose_delta(accepted_pose, shadow_pose)
    event["local_classification"] = _local_comparison(
        accepted_full_summary,
        event["shadow_full_corrs_summary"],
        int(accepted_strict_8px),
        int(event["shadow_strict_8px_inliers_on_full_corrs"]),
    )
    return event


def _sequence_summary(records: dict[int, dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [event for event in events if bool(event.get("trigger_fired", False))]
    rescues = [row for row in records.values() if bool(row.get("rescue_succeeded", False))]
    verdict_counts: dict[str, int] = {}
    for event in triggered:
        verdict = str(event.get("local_classification", "unknown"))
        verdict_counts[verdict] = int(verdict_counts.get(verdict, 0) + 1)
    return {
        "frames_processed": int(len(records)),
        "pipeline_ok_frames": int(sum(1 for row in records.values() if bool(row.get("pipeline_ok", False)))),
        "pipeline_failed_frames": int(sum(1 for row in records.values() if not bool(row.get("pipeline_ok", False)))),
        "rescue_succeeded_frames": int(len(rescues)),
        "support_refresh_triggered_frames": int(sum(1 for row in records.values() if bool(row.get("support_refresh_triggered", False)))),
        "shadow_candidate_rescue_frames": [int(event["frame_index"]) for event in events],
        "triggered_frames": [int(event["frame_index"]) for event in triggered],
        "trigger_count": int(len(triggered)),
        "triggered_local_verdict_counts": verdict_counts,
    }


def _classify(datasets: dict[str, dict[str, Any]]) -> str:
    triggered: list[dict[str, Any]] = []
    for dataset_result in datasets.values():
        triggered.extend(
            event
            for event in dataset_result.get("shadow_events", [])
            if bool(event.get("trigger_fired", False))
        )
    if len(triggered) == 0:
        return "inconclusive"
    any_worse = any(str(event.get("local_classification", "")) == "worse" for event in triggered)
    any_missing = any(not isinstance(event.get("shadow_pose", None), dict) for event in triggered)
    all_better = all(str(event.get("local_classification", "")) == "better" for event in triggered)
    dataset_keys = set(str(event.get("dataset", "")) for event in triggered)
    low_success = any(
        isinstance(event.get("shadow", None), dict)
        and int(event["shadow"].get("num_seeds_succeeded", 0)) == 0
        for event in triggered
    )
    if bool(any_worse):
        return "gated 40 px re-solve helps bad cases but is risky on good ones"
    if bool(any_missing or low_success):
        return "gated 40 px re-solve is too unstable or noisy"
    if bool(all_better) and len(dataset_keys) >= 2:
        return "gated 40 px re-solve is promising and broadly safe"
    return "inconclusive"


def _run_dataset(
    profile_path: Path,
    dataset_key: str,
    *,
    num_track: int,
    gate_median_px: float,
    seeds: list[int],
) -> dict[str, Any]:
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)
    pnp_cfg = frontend_kwargs["pnp_frontend_kwargs"]
    dataset_cfg = cfg["dataset"]
    run_cfg = cfg.get("run", {})
    dataset_root = (ROOT / str(dataset_cfg["root"])).resolve()
    sequence_name = str(dataset_cfg["seq"])
    sequence = load_sequence(
        str(dataset_cfg["name"]),
        dataset_root,
        sequence_name,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )
    bootstrap_cfg = run_cfg.get("bootstrap", {})
    i0 = int(bootstrap_cfg.get("i0", 0))
    i1 = int(bootstrap_cfg.get("i1", 1))
    image0, timestamp0, frame_id0 = sequence.get(i0)
    image1, timestamp1, frame_id1 = sequence.get(i1)
    boot = bootstrap_from_two_frames(
        K,
        K,
        image0,
        image1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )
    if not bool(boot.get("ok", False)):
        raise RuntimeError(f"Bootstrap failed for {dataset_key}: {boot.get('stats', {}).get('reason', None)}")

    seed = boot["seed"]
    bootstrap_landmark_count = int(len(seed.get("landmarks", []))) if isinstance(seed, dict) else 0
    start_frame = int(i1) + 1
    stop_frame_exclusive = min(len(sequence), start_frame + int(num_track))
    records: dict[int, dict[str, Any]] = {}
    shadow_events: list[dict[str, Any]] = []
    first_failure = None

    for frame_index in range(start_frame, stop_frame_exclusive):
        active_basis_before_kf = int(get_active_keyframe_kf(seed))
        image, _, _ = sequence.get(frame_index)
        output = process_frame_against_seed(
            K,
            seed,
            image,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            current_kf=int(frame_index),
            **pnp_cfg,
        )
        record = _record_frame(
            dataset_key,
            frame_index,
            output,
            active_basis_before_kf=int(active_basis_before_kf),
        )
        records[int(frame_index)] = record
        if first_failure is None and not bool(record["pipeline_ok"]):
            first_failure = int(frame_index)
        event = _maybe_shadow_event(
            dataset_key,
            frame_index,
            output,
            K,
            pnp_cfg,
            gate_median_px=float(gate_median_px),
            seeds=seeds,
        )
        if event is not None:
            shadow_events.append(event)
        seed = output["seed"]

    for event in shadow_events:
        if bool(event.get("trigger_fired", False)):
            event["history_impact_estimate"] = _history_impact_estimate(
                seed,
                K,
                event,
                eps=float(pnp_cfg["eps"]),
            )
            event["downstream_reuse_estimate"] = _downstream_reuse_estimate(records, event, window=3)

    return {
        "profile": str(profile_path),
        "dataset_key": str(dataset_key),
        "dataset_name": str(dataset_cfg["name"]),
        "sequence": sequence_name,
        "bootstrap": {
            "i0": int(i0),
            "i1": int(i1),
            "frame_id0": str(frame_id0),
            "frame_id1": str(frame_id1),
            "timestamp0": float(timestamp0),
            "timestamp1": float(timestamp1),
            "ok": bool(boot.get("ok", False)),
            "n_landmarks": int(bootstrap_landmark_count),
        },
        "num_track_requested": int(num_track),
        "start_frame": int(start_frame),
        "stop_frame_exclusive": int(stop_frame_exclusive),
        "first_failure": first_failure,
        "frame_records": {str(key): value for key, value in sorted(records.items())},
        "shadow_events": shadow_events,
        "sequence_summary": _sequence_summary(records, shadow_events),
    }


def _print_dataset_summary(dataset_key: str, result: dict[str, Any]) -> None:
    summary = result["sequence_summary"]
    print(
        f"{dataset_key}: frames={summary['frames_processed']} "
        f"ok={summary['pipeline_ok_frames']} failed={summary['pipeline_failed_frames']} "
        f"rescues={summary['rescue_succeeded_frames']} "
        f"triggers={summary['triggered_frames']}"
    )
    for event in result["shadow_events"]:
        if not bool(event.get("trigger_fired", False)):
            continue
        acc = event["accepted_full_corrs_summary"]
        shadow = event.get("shadow_full_corrs_summary", {}) or {}
        print(
            f"  frame {int(event['frame_index'])}: "
            f"accepted med/p90={float(acc['median_px']):.2f}/{float(acc['p90_px']):.2f} "
            f"shadow med/p90={float(shadow['median_px']):.2f}/{float(shadow['p90_px']):.2f} "
            f"8px={int(event['accepted_strict_8px_inliers_on_full_corrs'])}"
            f"->{int(event['shadow_strict_8px_inliers_on_full_corrs'])} "
            f"seeds={int(event['shadow']['num_seeds_succeeded'])}/{int(event['shadow']['num_seeds_tried'])} "
            f"{event['local_classification']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eth3d_profile",
        default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"),
    )
    parser.add_argument(
        "--kitti_profile",
        default=str(ROOT / "configs" / "profiles" / "kitti_odometry_00.yaml"),
    )
    parser.add_argument("--eth3d_num_track", type=int, default=40)
    parser.add_argument("--kitti_num_track", type=int, default=30)
    parser.add_argument("--gate_median_px", type=float, default=_DEFAULT_GATE_MEDIAN_PX)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(_DEFAULT_SEEDS))
    parser.add_argument("--out", default="/tmp/diag_40px_shadow_integration.json")
    args = parser.parse_args()

    seeds = [int(v) for v in args.seeds]
    datasets = {
        "eth3d": _run_dataset(
            Path(args.eth3d_profile).expanduser().resolve(),
            "eth3d",
            num_track=int(args.eth3d_num_track),
            gate_median_px=float(args.gate_median_px),
            seeds=seeds,
        ),
        "kitti": _run_dataset(
            Path(args.kitti_profile).expanduser().resolve(),
            "kitti",
            num_track=int(args.kitti_num_track),
            gate_median_px=float(args.gate_median_px),
            seeds=seeds,
        ),
    }
    result = {
        "event": "gated_40px_shadow_integration",
        "gate": {
            "signal": "accepted_inlier_residual_median_px",
            "threshold_px": float(args.gate_median_px),
        },
        "shadow": {
            "threshold_px": float(_SHADOW_THRESHOLD_PX),
            "strict_eval_px": float(_STRICT_EVAL_PX),
            "num_trials": int(_SHADOW_NUM_TRIALS),
            "seeds": seeds,
            "selection_rule": "max_strict_8px_inliers_then_lowest_fullset_median",
        },
        "datasets": datasets,
    }
    result["classification"] = _classify(datasets)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for dataset_key, dataset_result in datasets.items():
        _print_dataset_summary(dataset_key, dataset_result)
    print(f"classification: {result['classification']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
