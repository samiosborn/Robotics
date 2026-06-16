# scripts/diag_bad_vs_useful_fallbacks.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg, load_runtime_cfg

from datasets.loader import load_sequence
from geometry.camera import camera_centre, reprojection_errors_sq
from geometry.pose import angle_between_translations
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_active_keyframe_kf, get_active_landmark_lookup, get_pose_for_kf
from slam.landmark_state import build_landmark_id_index


TARGET_LABELS = {
    "eth3d": {
        12: "bad canonical pose",
        16: "bad canonical pose",
        17: "load-bearing good refresh",
        18: "neutral",
    },
    "kitti": {
        17: "load-bearing good refresh",
        18: "neutral",
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


def _pose(value) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, dict):
        return (
            np.asarray(value["R"], dtype=np.float64).reshape(3, 3),
            np.asarray(value["t"], dtype=np.float64).reshape(3),
        )
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (
            np.asarray(value[0], dtype=np.float64).reshape(3, 3),
            np.asarray(value[1], dtype=np.float64).reshape(3),
        )
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected pose tuple, dict, or 4x4 matrix; got {arr.shape}")
    return arr[:3, :3], arr[:3, 3].reshape(3)


def _pose_copy(value) -> tuple[np.ndarray, np.ndarray]:
    R, t = _pose(value)
    return R.copy(), t.copy()


def _direction_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        return float(np.degrees(angle_between_translations(a, b)))
    except Exception:
        return None


def _pose_delta(a, b) -> dict[str, Any]:
    R_a, t_a = _pose(a)
    R_b, t_b = _pose(b)
    C_a = camera_centre(R_a, t_a)
    C_b = camera_centre(R_b, t_b)
    return {
        "rotation_delta_deg": float(np.degrees(angle_between_rotmats(R_a, R_b))),
        "translation_direction_delta_deg": _direction_deg(t_a, t_b),
        "camera_centre_direction_delta_deg": _direction_deg(C_a, C_b),
        "camera_centre_distance": float(np.linalg.norm(C_b - C_a)),
    }


def _normalise_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    if float(np.linalg.det(U @ Vt)) < 0.0:
        U[:, -1] *= -1.0
    return U @ Vt


def _interpolate_pose(a, b, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    R_a, t_a = _pose(a)
    R_b, t_b = _pose(b)
    R_relative = _normalise_rotation(R_b @ R_a.T)
    theta = float(np.arccos(np.clip((np.trace(R_relative) - 1.0) * 0.5, -1.0, 1.0)))
    if theta <= 1e-12:
        R_step = np.eye(3, dtype=np.float64)
    elif abs(float(np.sin(theta))) <= 1e-12:
        R_step = _normalise_rotation((1.0 - float(alpha)) * np.eye(3, dtype=np.float64) + float(alpha) * R_relative)
    else:
        K_axis = (R_relative - R_relative.T) / (2.0 * np.sin(theta))
        theta_alpha = float(alpha) * theta
        R_step = (
            np.eye(3, dtype=np.float64)
            + np.sin(theta_alpha) * K_axis
            + (1.0 - np.cos(theta_alpha)) * (K_axis @ K_axis)
        )
    R_out = _normalise_rotation(R_step @ R_a)
    C_out = (1.0 - float(alpha)) * camera_centre(R_a, t_a) + float(alpha) * camera_centre(R_b, t_b)
    return R_out, -R_out @ C_out


def _summary(errors_px) -> dict[str, Any]:
    arr = np.asarray(errors_px, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return {
            "count": 0,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "squared_error": 0.0,
            "above_8_count": 0,
            "above_8_fraction": None,
        }
    above_8 = int(np.sum(arr > 8.0))
    return {
        "count": int(arr.size),
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90.0)),
        "max_px": float(np.max(arr)),
        "squared_error": float(np.sum(arr * arr)),
        "above_8_count": int(above_8),
        "above_8_fraction": float(above_8 / max(int(arr.size), 1)),
    }


def _active_lookup_ids(seed: dict[str, Any]) -> set[int]:
    try:
        lookup = np.asarray(get_active_landmark_lookup(seed), dtype=np.int64).reshape(-1)
    except Exception:
        return set()
    return set(int(v) for v in lookup if int(v) >= 0)


def _support_ids_from_pose_out(pose_out: dict[str, Any]) -> tuple[list[int], list[int]]:
    if not isinstance(pose_out, dict) or pose_out.get("corrs", None) is None:
        return [], []
    corrs = pose_out["corrs"]
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    mask = np.asarray(pose_out.get("pnp_inlier_mask", np.zeros((landmark_ids.size,), dtype=bool)), dtype=bool).reshape(-1)
    if int(mask.size) != int(landmark_ids.size):
        mask = np.zeros((int(landmark_ids.size),), dtype=bool)
    return [int(v) for v in landmark_ids], [int(v) for v in landmark_ids[mask]]


def _residuals_for_pose(K: np.ndarray, pose, X_w: np.ndarray, xy: np.ndarray, *, eps: float) -> np.ndarray:
    R, t = _pose(pose)
    X_w = np.asarray(X_w, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    if X_w.ndim != 2 or xy.ndim != 2 or X_w.shape[0] != 3 or xy.shape[0] != 2 or X_w.shape[1] == 0:
        return np.zeros((0,), dtype=np.float64)
    depth = np.asarray((R @ X_w + t.reshape(3, 1))[2, :], dtype=np.float64).reshape(-1)
    err_sq = np.asarray(reprojection_errors_sq(K, R, t, X_w, xy), dtype=np.float64).reshape(-1)
    valid = np.isfinite(depth) & (depth > float(eps)) & np.isfinite(err_sq) & (err_sq >= 0.0)
    out = np.full((int(err_sq.size),), np.nan, dtype=np.float64)
    out[valid] = np.sqrt(err_sq[valid])
    return out


def _support_residuals_current(K: np.ndarray, pose_out: dict[str, Any], *, eps: float) -> dict[str, Any]:
    if not isinstance(pose_out, dict) or pose_out.get("corrs", None) is None:
        return _summary(np.zeros((0,), dtype=np.float64))
    corrs = pose_out["corrs"]
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    mask = np.asarray(pose_out.get("pnp_inlier_mask", np.zeros((landmark_ids.size,), dtype=bool)), dtype=bool).reshape(-1)
    if int(mask.size) != int(landmark_ids.size):
        mask = np.zeros((int(landmark_ids.size),), dtype=bool)
    pose = (pose_out["R"], pose_out["t"])
    return _summary(_residuals_for_pose(K, pose, corrs.X_w[:, mask], corrs.x_cur[:, mask], eps=float(eps)))


def _landmark_history_residuals(
    seed: dict[str, Any],
    K: np.ndarray,
    landmark_ids: list[int],
    *,
    frame_filter,
    eps: float,
) -> dict[str, Any]:
    lm_by_id = build_landmark_id_index(seed, context="bad versus useful fallback history")
    pooled: list[float] = []
    landmark_rows: list[dict[str, Any]] = []
    for landmark_id in sorted(set(int(v) for v in landmark_ids)):
        lm = lm_by_id.get(int(landmark_id), None)
        if not isinstance(lm, dict):
            continue
        X_w = np.asarray(lm.get("X_w", None), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue
        errors: list[float] = []
        obs_frames: list[int] = []
        for observation in lm.get("obs", []):
            if not isinstance(observation, dict):
                continue
            kf = int(observation.get("kf", -1))
            if kf < 0 or not bool(frame_filter(kf)):
                continue
            xy = np.asarray(observation.get("xy", None), dtype=np.float64).reshape(-1)
            if xy.size != 2 or not np.isfinite(xy).all():
                continue
            try:
                pose = get_pose_for_kf(seed, int(kf), context="bad versus useful fallback residuals")
            except Exception:
                continue
            err = _residuals_for_pose(
                K,
                pose,
                X_w.reshape(3, 1),
                xy.reshape(2, 1),
                eps=float(eps),
            )
            if int(err.size) == 1 and np.isfinite(err[0]):
                errors.append(float(err[0]))
                obs_frames.append(int(kf))
        summary = _summary(np.asarray(errors, dtype=np.float64))
        evaluated = bool(int(summary["count"]) >= 2)
        inconsistent = bool(
            evaluated
            and summary["median_px"] is not None
            and summary["p90_px"] is not None
            and summary["max_px"] is not None
            and (
                float(summary["median_px"]) > 3.0
                or float(summary["p90_px"]) > 8.0
                or float(summary["max_px"]) > 12.0
            )
        )
        pooled.extend(errors)
        landmark_rows.append(
            {
                "landmark_id": int(landmark_id),
                "observation_frames": obs_frames,
                "summary": summary,
                "evaluated": bool(evaluated),
                "inconsistent": bool(inconsistent),
            }
        )
    evaluated_count = int(sum(1 for row in landmark_rows if bool(row["evaluated"])))
    inconsistent_count = int(sum(1 for row in landmark_rows if bool(row["inconsistent"])))
    return {
        "support_count": int(len(set(int(v) for v in landmark_ids))),
        "evaluated_count": int(evaluated_count),
        "inconsistent_count": int(inconsistent_count),
        "inconsistent_fraction": float(inconsistent_count / max(int(len(set(int(v) for v in landmark_ids))), 1)),
        "pooled": _summary(np.asarray(pooled, dtype=np.float64)),
        "landmarks": landmark_rows,
    }


def _observed_support_on_frame(
    seed: dict[str, Any],
    K: np.ndarray,
    landmark_ids: list[int],
    frame_index: int,
    *,
    eps: float,
) -> dict[str, Any]:
    lm_by_id = build_landmark_id_index(seed, context="bad versus useful fallback frame observation")
    X_cols: list[np.ndarray] = []
    xy_cols: list[np.ndarray] = []
    used_ids: list[int] = []
    for landmark_id in sorted(set(int(v) for v in landmark_ids)):
        lm = lm_by_id.get(int(landmark_id), None)
        if not isinstance(lm, dict):
            continue
        X_w = np.asarray(lm.get("X_w", None), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue
        for observation in lm.get("obs", []):
            if not isinstance(observation, dict):
                continue
            if int(observation.get("kf", -1)) != int(frame_index):
                continue
            xy = np.asarray(observation.get("xy", None), dtype=np.float64).reshape(-1)
            if xy.size == 2 and np.isfinite(xy).all():
                X_cols.append(X_w.reshape(3, 1))
                xy_cols.append(xy.reshape(2, 1))
                used_ids.append(int(landmark_id))
                break
    if len(X_cols) == 0:
        return {
            "frame_index": int(frame_index),
            "observed_count": 0,
            "observed_fraction": 0.0,
            "summary": _summary(np.zeros((0,), dtype=np.float64)),
            "landmark_ids": [],
        }
    try:
        pose = get_pose_for_kf(seed, int(frame_index), context="bad versus useful fallback frame observation")
    except Exception:
        return {
            "frame_index": int(frame_index),
            "observed_count": int(len(used_ids)),
            "observed_fraction": float(len(used_ids) / max(len(set(int(v) for v in landmark_ids)), 1)),
            "summary": _summary(np.zeros((0,), dtype=np.float64)),
            "landmark_ids": used_ids,
            "pose_available": False,
        }
    errors = _residuals_for_pose(
        K,
        pose,
        np.hstack(X_cols),
        np.hstack(xy_cols),
        eps=float(eps),
    )
    return {
        "frame_index": int(frame_index),
        "observed_count": int(len(used_ids)),
        "observed_fraction": float(len(used_ids) / max(len(set(int(v) for v in landmark_ids)), 1)),
        "summary": _summary(errors),
        "landmark_ids": used_ids,
        "pose_available": True,
    }


def _accepted_pose_support_arrays(pose_out: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(pose_out, dict) or pose_out.get("corrs", None) is None:
        return np.zeros((3, 0), dtype=np.float64), np.zeros((2, 0), dtype=np.float64)
    corrs = pose_out["corrs"]
    ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    mask = np.asarray(pose_out.get("pnp_inlier_mask", np.zeros((ids.size,), dtype=bool)), dtype=bool).reshape(-1)
    if int(mask.size) != int(ids.size):
        mask = np.zeros((int(ids.size),), dtype=bool)
    return np.asarray(corrs.X_w[:, mask], dtype=np.float64), np.asarray(corrs.x_cur[:, mask], dtype=np.float64)


def _frame_time(sequence, frame_index: int) -> float:
    _, timestamp, _ = sequence.get(int(frame_index))
    return float(timestamp)


def _pose_quality(
    sequence,
    K: np.ndarray,
    frame_record: dict[str, Any],
    records: dict[int, dict[str, Any]],
    accepted_frames: list[int],
    *,
    eps: float,
) -> dict[str, Any]:
    frame_index = int(frame_record["frame_index"])
    prev_frames = [int(v) for v in accepted_frames if int(v) < frame_index]
    next_frames = [int(v) for v in accepted_frames if int(v) > frame_index]
    prev_frame = None if len(prev_frames) == 0 else int(prev_frames[-1])
    next_frame = None if len(next_frames) == 0 else int(next_frames[0])
    pose_cur = frame_record["pose"]
    out: dict[str, Any] = {
        "previous_accepted_frame": prev_frame,
        "next_accepted_frame": next_frame,
        "delta_to_previous": None,
        "delta_to_next": None,
        "local_interpolation": None,
        "local_path": None,
        "unusually_displaced": None,
    }
    if prev_frame is not None:
        out["delta_to_previous"] = _pose_delta(records[int(prev_frame)]["pose"], pose_cur)
    if next_frame is not None:
        out["delta_to_next"] = _pose_delta(pose_cur, records[int(next_frame)]["pose"])
    if prev_frame is None or next_frame is None:
        return out

    pose_prev = records[int(prev_frame)]["pose"]
    pose_next = records[int(next_frame)]["pose"]
    ts_prev = _frame_time(sequence, int(prev_frame))
    ts_cur = _frame_time(sequence, int(frame_index))
    ts_next = _frame_time(sequence, int(next_frame))
    alpha = 0.5 if abs(ts_next - ts_prev) <= 1e-12 else float((ts_cur - ts_prev) / (ts_next - ts_prev))
    interp_pose = _interpolate_pose(pose_prev, pose_next, alpha)

    X_support, xy_support = _accepted_pose_support_arrays(frame_record["pose_out"])
    accepted_errors = _residuals_for_pose(K, pose_cur, X_support, xy_support, eps=float(eps))
    interp_errors = _residuals_for_pose(K, interp_pose, X_support, xy_support, eps=float(eps))
    accepted_summary = _summary(accepted_errors)
    interp_summary = _summary(interp_errors)
    accepted_sq = float(accepted_summary["squared_error"])
    interp_sq = float(interp_summary["squared_error"])
    if accepted_sq > 0.0:
        interp_reduction = float((accepted_sq - interp_sq) / accepted_sq)
    else:
        interp_reduction = None

    R_prev, t_prev = _pose(pose_prev)
    R_cur, t_cur = _pose(pose_cur)
    R_next, t_next = _pose(pose_next)
    C_prev = camera_centre(R_prev, t_prev)
    C_cur = camera_centre(R_cur, t_cur)
    C_next = camera_centre(R_next, t_next)
    step_a = C_cur - C_prev
    step_b = C_next - C_cur
    chord = C_next - C_prev
    step_a_norm = float(np.linalg.norm(step_a))
    step_b_norm = float(np.linalg.norm(step_b))
    chord_norm = float(np.linalg.norm(chord))
    chord_norm_sq = float(np.dot(chord, chord))
    chord_alpha = None if chord_norm_sq <= 1e-24 else float(np.dot(C_cur - C_prev, chord) / chord_norm_sq)
    outside_chord = None if chord_alpha is None else bool(chord_alpha < 0.0 or chord_alpha > 1.0)
    path_ratio = None if chord_norm <= 1e-12 else float((step_a_norm + step_b_norm) / chord_norm)
    step_ratio = None
    if min(step_a_norm, step_b_norm) > 1e-12:
        step_ratio = float(max(step_a_norm, step_b_norm) / min(step_a_norm, step_b_norm))
    camera_turn = _direction_deg(step_a, step_b)
    rotation_path_excess = float(
        np.degrees(angle_between_rotmats(R_prev, R_cur))
        + np.degrees(angle_between_rotmats(R_cur, R_next))
        - np.degrees(angle_between_rotmats(R_prev, R_next))
    )
    unusually_displaced = bool(
        (outside_chord is True)
        or (camera_turn is not None and float(camera_turn) > 120.0)
        or float(rotation_path_excess) > 6.0
    )

    out["local_interpolation"] = {
        "alpha": float(alpha),
        "delta_to_interpolated_pose": _pose_delta(interp_pose, pose_cur),
        "accepted_support_residuals_under_accepted_pose": accepted_summary,
        "accepted_support_residuals_under_interpolated_pose": interp_summary,
        "interpolation_squared_error_reduction": interp_reduction,
    }
    out["local_path"] = {
        "camera_motion_turn_deg": camera_turn,
        "adjacent_step_ratio": step_ratio,
        "camera_path_ratio": path_ratio,
        "chord_projection_alpha": chord_alpha,
        "outside_neighbour_chord": outside_chord,
        "rotation_path_excess_deg": rotation_path_excess,
    }
    out["unusually_displaced"] = bool(unusually_displaced)
    return out


def _rescue_stage(stats: dict[str, Any]) -> str:
    if not bool(stats.get("pnp_support_rescue_succeeded", False)):
        return "not_rescue"
    threshold = stats.get("pnp_support_rescue_loose_threshold_px", None)
    if bool(stats.get("pnp_support_rescue_loose_localisation_fallback_succeeded", False)):
        if threshold is None:
            return "loose_localisation_only"
        return f"loose_{float(threshold):.0f}px_localisation_only"
    if bool(stats.get("pnp_support_rescue_second_stage_succeeded", False)):
        return "stage2_strict_refit"
    return str(stats.get("pnp_support_rescue_reason", "rescued"))


def _record_frame(
    dataset_key: str,
    frame_index: int,
    seed_before: dict[str, Any],
    output: dict[str, Any],
    *,
    active_basis_before_kf: int,
    active_lookup_before_ids: set[int],
) -> dict[str, Any]:
    pose_out = output.get("pose_out", {}) if isinstance(output, dict) else {}
    stats = output.get("stats", {}) if isinstance(output, dict) else {}
    stats = stats if isinstance(stats, dict) else {}
    pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
    pose_stats = pose_stats if isinstance(pose_stats, dict) else {}
    pose_ids, inlier_ids = _support_ids_from_pose_out(pose_out)
    accepted = bool(output.get("ok", False)) if isinstance(output, dict) else False
    active_after = -1
    if isinstance(output, dict) and isinstance(output.get("seed", None), dict):
        active_after = int(output["seed"].get("active_keyframe_kf", -1))
    pose = None
    if bool(accepted):
        pose = _pose_copy((pose_out["R"], pose_out["t"]))
    return {
        "dataset": str(dataset_key),
        "frame_index": int(frame_index),
        "pipeline_ok": bool(accepted),
        "pipeline_reason": stats.get("reason", None),
        "active_basis_before_kf": int(active_basis_before_kf),
        "active_basis_after_kf": int(active_after),
        "active_lookup_before_ids": sorted(int(v) for v in active_lookup_before_ids),
        "accepted_pose_type": (
            "failed"
            if not bool(accepted)
            else "rescue"
            if bool(stats.get("localisation_only_rescue_frame", False))
            else "normal"
        ),
        "known_downstream_label": TARGET_LABELS.get(str(dataset_key), {}).get(int(frame_index), "unclear"),
        "rescue_stage": _rescue_stage(pose_stats),
        "loose_threshold_px": pose_stats.get("pnp_support_rescue_loose_threshold_px", None),
        "loose_inlier_count": int(pose_stats.get("pnp_support_rescue_loose_inliers", 0)),
        "final_pnp_inliers": int(stats.get("n_pnp_inliers", 0)),
        "final_pnp_correspondences": int(stats.get("n_pnp_corr", 0)),
        "refresh_triggered": bool(stats.get("guarded_support_refresh_triggered", False)),
        "refresh_reason": stats.get("guarded_support_refresh_reason", None),
        "support_ids": sorted(set(int(v) for v in inlier_ids)),
        "pose_eligible_ids": sorted(set(int(v) for v in pose_ids)),
        "pose": pose,
        "pose_out": pose_out,
        "seed_before_landmark_count": int(len(seed_before.get("landmarks", []))) if isinstance(seed_before, dict) else 0,
    }


def _forward_viability(
    final_seed: dict[str, Any],
    K: np.ndarray,
    target: dict[str, Any],
    records: dict[int, dict[str, Any]],
    accepted_frames: list[int],
    *,
    eps: float,
) -> dict[str, Any]:
    frame_index = int(target["frame_index"])
    support_ids = [int(v) for v in target["support_ids"]]
    support_set = set(support_ids)
    next_frames = [int(v) for v in accepted_frames if int(v) > frame_index]
    next_accepted_frame = None if len(next_frames) == 0 else int(next_frames[0])
    next_support = set()
    if next_accepted_frame is not None:
        next_support = set(int(v) for v in records[int(next_accepted_frame)]["support_ids"])
    overlap = support_set & next_support
    union = support_set | next_support

    downstream_rows: list[dict[str, Any]] = []
    for downstream_frame in range(frame_index + 1, frame_index + 4):
        record = records.get(int(downstream_frame), None)
        if not isinstance(record, dict):
            downstream_rows.append(
                {
                    "frame_index": int(downstream_frame),
                    "available": False,
                }
            )
            continue
        active_ids = set(int(v) for v in record.get("active_lookup_before_ids", []))
        pose_eligible_ids = set(int(v) for v in record.get("pose_eligible_ids", []))
        inlier_ids = set(int(v) for v in record.get("support_ids", []))
        downstream_rows.append(
            {
                "frame_index": int(downstream_frame),
                "available": True,
                "pipeline_ok": bool(record.get("pipeline_ok", False)),
                "active_basis_before_kf": int(record.get("active_basis_before_kf", -1)),
                "mapped_in_active_lookup_before": int(len(support_set & active_ids)),
                "pose_eligible": int(len(support_set & pose_eligible_ids)),
                "accepted_inlier_support": int(len(support_set & inlier_ids)),
                "final_pnp_inliers": int(record.get("final_pnp_inliers", 0)),
                "final_pnp_correspondences": int(record.get("final_pnp_correspondences", 0)),
            }
        )

    future_records = [row for idx, row in records.items() if int(idx) > frame_index]
    future_pose_eligible_any: set[int] = set()
    future_inlier_any: set[int] = set()
    for record in future_records:
        future_pose_eligible_any |= support_set & set(int(v) for v in record.get("pose_eligible_ids", []))
        future_inlier_any |= support_set & set(int(v) for v in record.get("support_ids", []))

    return {
        "historical_residuals_before_frame": _landmark_history_residuals(
            final_seed,
            K,
            support_ids,
            frame_filter=lambda kf: int(kf) < frame_index,
            eps=float(eps),
        ),
        "residuals_at_frame": _observed_support_on_frame(
            final_seed,
            K,
            support_ids,
            frame_index,
            eps=float(eps),
        ),
        "next_frame_reprojection": _observed_support_on_frame(
            final_seed,
            K,
            support_ids,
            frame_index + 1,
            eps=float(eps),
        ),
        "later_residuals_after_frame": _landmark_history_residuals(
            final_seed,
            K,
            support_ids,
            frame_filter=lambda kf: int(kf) > frame_index,
            eps=float(eps),
        ),
        "support_overlap_next_accepted": {
            "next_accepted_frame": next_accepted_frame,
            "overlap_count": int(len(overlap)),
            "target_support_count": int(len(support_set)),
            "next_support_count": int(len(next_support)),
            "overlap_over_target": None if len(support_set) == 0 else float(len(overlap) / len(support_set)),
            "iou": None if len(union) == 0 else float(len(overlap) / len(union)),
        },
        "downstream_window": downstream_rows,
        "future_pose_eligible_any_count": int(len(future_pose_eligible_any)),
        "future_pose_eligible_any_fraction": float(len(future_pose_eligible_any) / max(len(support_set), 1)),
        "future_accepted_inlier_any_count": int(len(future_inlier_any)),
        "future_accepted_inlier_any_fraction": float(len(future_inlier_any) / max(len(support_set), 1)),
    }


def _replay_dataset(profile_path: Path, dataset_key: str, stop_frame: int) -> dict[str, Any]:
    cfg, K = load_runtime_cfg(profile_path)
    frontend_kwargs = frontend_kwargs_from_cfg(cfg)
    pnp_cfg = frontend_kwargs["pnp_frontend_kwargs"]
    dataset_cfg = cfg["dataset"]
    run_cfg = cfg.get("run", {})
    dataset_name = str(dataset_cfg["name"])
    dataset_root = (ROOT / str(dataset_cfg["root"])).resolve()
    sequence_name = str(dataset_cfg["seq"])
    sequence = load_sequence(
        dataset_name,
        dataset_root,
        sequence_name,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )
    bootstrap_cfg = run_cfg.get("bootstrap", {})
    i0 = int(bootstrap_cfg.get("i0", 0))
    i1 = int(bootstrap_cfg.get("i1", 1))
    image0, _, _ = sequence.get(i0)
    image1, _, _ = sequence.get(i1)
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
    records: dict[int, dict[str, Any]] = {}
    accepted_frames: list[int] = []
    for frame_index in range(i1 + 1, min(int(stop_frame) + 1, len(sequence))):
        seed_before = seed
        active_before = int(get_active_keyframe_kf(seed))
        active_lookup_ids = _active_lookup_ids(seed)
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
            seed_before,
            output,
            active_basis_before_kf=active_before,
            active_lookup_before_ids=active_lookup_ids,
        )
        if bool(record["pipeline_ok"]):
            accepted_frames.append(int(frame_index))
        records[int(frame_index)] = record
        seed = output["seed"]

    eps = float(pnp_cfg.get("eps", 1e-12))
    targets: dict[str, Any] = {}
    for frame_index in TARGET_LABELS[str(dataset_key)]:
        record = records.get(int(frame_index), None)
        if not isinstance(record, dict):
            continue
        event = {
            key: value
            for key, value in record.items()
            if key not in {"pose", "pose_out", "active_lookup_before_ids"}
        }
        if bool(record.get("pipeline_ok", False)):
            event["current_support_residuals_under_accepted_pose"] = _support_residuals_current(
                K,
                record["pose_out"],
                eps=float(eps),
            )
            event["pose_quality"] = _pose_quality(
                sequence,
                K,
                record,
                records,
                accepted_frames,
                eps=float(eps),
            )
            event["forward_viability"] = _forward_viability(
                seed,
                K,
                record,
                records,
                accepted_frames,
                eps=float(eps),
            )
        targets[str(frame_index)] = event

    accepted_summary = [
        {
            "frame_index": int(frame_index),
            "accepted_pose_type": records[int(frame_index)]["accepted_pose_type"],
            "support_count": int(len(records[int(frame_index)]["support_ids"])),
            "active_basis_before_kf": int(records[int(frame_index)]["active_basis_before_kf"]),
            "active_basis_after_kf": int(records[int(frame_index)]["active_basis_after_kf"]),
        }
        for frame_index in accepted_frames
    ]

    return {
        "profile": str(profile_path),
        "dataset_key": str(dataset_key),
        "dataset_name": str(dataset_name),
        "sequence": str(sequence_name),
        "bootstrap": {
            "i0": int(i0),
            "i1": int(i1),
            "ok": bool(boot.get("ok", False)),
            "n_landmarks": int(len(seed.get("landmarks", []))) if isinstance(seed, dict) else 0,
        },
        "stop_frame": int(stop_frame),
        "accepted_frames": accepted_summary,
        "targets": targets,
    }


def _classify(results: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset_key, dataset_result in results["datasets"].items():
        for frame_key, row in dataset_result["targets"].items():
            pose_quality = row.get("pose_quality", {})
            forward = row.get("forward_viability", {})
            local_path_raw = pose_quality.get("local_path", {}) if isinstance(pose_quality, dict) else {}
            local_path = local_path_raw if isinstance(local_path_raw, dict) else {}
            local_interp_raw = pose_quality.get("local_interpolation", {}) if isinstance(pose_quality, dict) else {}
            local_interp = local_interp_raw if isinstance(local_interp_raw, dict) else {}
            history = forward.get("historical_residuals_before_frame", {}) if isinstance(forward, dict) else {}
            rows.append(
                {
                    "dataset": str(dataset_key),
                    "frame_index": int(frame_key),
                    "label": str(row.get("known_downstream_label", "unclear")),
                    "unusually_displaced": pose_quality.get("unusually_displaced", None) if isinstance(pose_quality, dict) else None,
                    "rotation_path_excess_deg": local_path.get("rotation_path_excess_deg", None),
                    "outside_neighbour_chord": local_path.get("outside_neighbour_chord", None),
                    "interpolation_squared_error_reduction": local_interp.get("interpolation_squared_error_reduction", None),
                    "future_pose_eligible_any_fraction": forward.get("future_pose_eligible_any_fraction", None) if isinstance(forward, dict) else None,
                    "future_accepted_inlier_any_fraction": forward.get("future_accepted_inlier_any_fraction", None) if isinstance(forward, dict) else None,
                    "history_inconsistent_fraction": history.get("inconsistent_fraction", None) if isinstance(history, dict) else None,
                    "history_p90_px": history.get("pooled", {}).get("p90_px", None) if isinstance(history, dict) else None,
                }
            )
    bad_rows = [row for row in rows if str(row["label"]) == "bad canonical pose"]
    useful_rows = [row for row in rows if str(row["label"]) == "load-bearing good refresh"]
    pose_separates = (
        len(bad_rows) > 0
        and len(useful_rows) > 0
        and all(
            row["rotation_path_excess_deg"] is not None
            and float(row["rotation_path_excess_deg"]) > 6.0
            and row["interpolation_squared_error_reduction"] is not None
            and float(row["interpolation_squared_error_reduction"]) > 0.0
            for row in bad_rows
        )
        and all(
            row["rotation_path_excess_deg"] is not None
            and float(row["rotation_path_excess_deg"]) < 4.0
            and row["interpolation_squared_error_reduction"] is not None
            and float(row["interpolation_squared_error_reduction"]) < 0.0
            for row in useful_rows
        )
    )
    forward_bad = [row.get("future_accepted_inlier_any_fraction", None) for row in bad_rows]
    forward_useful = [row.get("future_accepted_inlier_any_fraction", None) for row in useful_rows]
    history_bad = [row.get("history_p90_px", None) for row in bad_rows]
    history_useful = [row.get("history_p90_px", None) for row in useful_rows]

    forward_separates = False
    if all(v is not None for v in forward_bad + forward_useful):
        forward_separates = bool(min(float(v) for v in forward_useful) > max(float(v) for v in forward_bad))

    history_separates = False
    if all(v is not None for v in history_bad + history_useful):
        history_separates = bool(min(float(v) for v in history_bad) > max(float(v) for v in history_useful))

    if bool(pose_separates) and not bool(forward_separates or history_separates):
        classification = "pose_deviation_from_local_motion"
    elif bool(forward_separates) and not bool(pose_separates or history_separates):
        classification = "forward_support_viability"
    elif bool(history_separates) and not bool(pose_separates or forward_separates):
        classification = "history_residuals_on_accepted_support"
    elif bool(pose_separates or forward_separates or history_separates):
        classification = "mixed"
    else:
        classification = "no_clear_separator_yet"

    return {
        "classification": classification,
        "rows": rows,
        "pose_deviation_separates": bool(pose_separates),
        "forward_support_viability_separates": bool(forward_separates),
        "history_residuals_separate": bool(history_separates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth3d_profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--kitti_profile", type=str, default=str(ROOT / "configs" / "profiles" / "kitti_odometry_00.yaml"))
    parser.add_argument("--eth3d_stop_frame", type=int, default=22)
    parser.add_argument("--kitti_stop_frame", type=int, default=22)
    parser.add_argument("--out", type=str, default="/tmp/bad_vs_useful_fallbacks.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    datasets = {
        "eth3d": _replay_dataset(
            Path(args.eth3d_profile).expanduser().resolve(),
            "eth3d",
            int(args.eth3d_stop_frame),
        ),
        "kitti": _replay_dataset(
            Path(args.kitti_profile).expanduser().resolve(),
            "kitti",
            int(args.kitti_stop_frame),
        ),
    }
    result = {
        "event": "bad_vs_useful_fallback_comparison",
        "datasets": datasets,
    }
    result["classification_probe"] = _classify(result)
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    if bool(args.quiet):
        print(f"wrote {out_path}")
        print(f"classification {result['classification_probe']['classification']}")
    else:
        print(text)


if __name__ == "__main__":
    main()
