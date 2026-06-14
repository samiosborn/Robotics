# scripts/diag_frame16_pose_quality.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_runtime_cfg as _load_runtime_cfg

from datasets.eth3d import load_eth3d_sequence
from geometry.camera import camera_centre, reprojection_errors_sq
from geometry.pose import angle_between_translations
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_active_keyframe_kf, get_pose_for_kf


# Convert diagnostic values for JSON output
def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
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


# Copy one world-to-camera pose
def _pose(value) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(value[0], dtype=np.float64).reshape(3, 3).copy(),
        np.asarray(value[1], dtype=np.float64).reshape(3).copy(),
    )


# Compute a vector direction angle
def _direction_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        return float(np.degrees(angle_between_translations(a, b)))
    except Exception:
        return None


# Compare two canonical poses
def _pose_delta(a, b) -> dict:
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


# Project a matrix onto SO(3)
def _normalise_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    if float(np.linalg.det(U @ Vt)) < 0.0:
        U[:, -1] *= -1.0
    return U @ Vt


# Interpolate rotation and camera centre
def _interpolate_pose(a, b, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    R_a, t_a = _pose(a)
    R_b, t_b = _pose(b)
    R_relative = _normalise_rotation(R_b @ R_a.T)
    theta = float(np.arccos(np.clip((np.trace(R_relative) - 1.0) * 0.5, -1.0, 1.0)))
    if theta <= 1e-12:
        R_step = np.eye(3, dtype=np.float64)
    elif abs(float(np.sin(theta))) <= 1e-12:
        R_step = _normalise_rotation((1.0 - alpha) * np.eye(3) + alpha * R_relative)
    else:
        K_axis = (R_relative - R_relative.T) / (2.0 * np.sin(theta))
        theta_alpha = alpha * theta
        R_step = (
            np.eye(3, dtype=np.float64)
            + np.sin(theta_alpha) * K_axis
            + (1.0 - np.cos(theta_alpha)) * (K_axis @ K_axis)
        )
    R_out = _normalise_rotation(R_step @ R_a)
    C_out = (1.0 - alpha) * camera_centre(R_a, t_a) + alpha * camera_centre(R_b, t_b)
    return R_out, -R_out @ C_out


# Summarise reprojection errors
def _summary(errors_px: np.ndarray) -> dict:
    errors_px = np.asarray(errors_px, dtype=np.float64).reshape(-1)
    errors_px = errors_px[np.isfinite(errors_px)]
    if int(errors_px.size) == 0:
        return {
            "count": 0,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "squared_error": 0.0,
            "above_8_count": 0,
            "above_8_fraction": None,
        }
    above_8 = int(np.sum(errors_px > 8.0))
    return {
        "count": int(errors_px.size),
        "median_px": float(np.median(errors_px)),
        "p90_px": float(np.percentile(errors_px, 90)),
        "max_px": float(np.max(errors_px)),
        "squared_error": float(np.sum(errors_px * errors_px)),
        "above_8_count": int(above_8),
        "above_8_fraction": float(above_8 / errors_px.size),
    }


# Index landmarks by id
def _landmarks(seed: dict) -> dict[int, dict]:
    return {
        int(landmark["id"]): landmark
        for landmark in seed.get("landmarks", [])
        if isinstance(landmark, dict) and "id" in landmark
    }


# Build aligned observations for one frame
def _frame_bundle(seed: dict, landmark_ids: list[int], frame_index: int) -> dict:
    lm_by_id = _landmarks(seed)
    X_rows = []
    xy_rows = []
    used_ids = []
    missing_ids = []
    duplicate_ids = []
    for landmark_id in landmark_ids:
        landmark = lm_by_id.get(int(landmark_id), None)
        X_w = np.asarray(
            None if landmark is None else landmark.get("X_w", None),
            dtype=np.float64,
        ).reshape(-1)
        observations = []
        if isinstance(landmark, dict):
            for observation in landmark.get("obs", []):
                if not isinstance(observation, dict):
                    continue
                if int(observation.get("kf", -1)) != int(frame_index):
                    continue
                xy = np.asarray(observation.get("xy", None), dtype=np.float64).reshape(-1)
                if xy.size == 2 and np.isfinite(xy).all():
                    observations.append(xy.copy())
        if X_w.size != 3 or not np.isfinite(X_w).all() or len(observations) == 0:
            missing_ids.append(int(landmark_id))
            continue
        if len(observations) > 1:
            duplicate_ids.append(int(landmark_id))
        X_rows.append(X_w.reshape(3, 1))
        xy_rows.append(observations[0].reshape(2, 1))
        used_ids.append(int(landmark_id))
    return {
        "X_w": np.hstack(X_rows) if len(X_rows) > 0 else np.zeros((3, 0), dtype=np.float64),
        "xy": np.hstack(xy_rows) if len(xy_rows) > 0 else np.zeros((2, 0), dtype=np.float64),
        "landmark_ids": used_ids,
        "missing_count": int(len(missing_ids)),
        "duplicate_count": int(len(duplicate_ids)),
    }


# Compute positive-depth reprojection errors
def _errors(K: np.ndarray, pose, bundle: dict, *, eps: float) -> np.ndarray:
    R, t = _pose(pose)
    X_w = np.asarray(bundle["X_w"], dtype=np.float64)
    if int(X_w.shape[1]) == 0:
        return np.zeros((0,), dtype=np.float64)
    depth = np.asarray((R @ X_w + t.reshape(3, 1))[2, :], dtype=np.float64)
    error_sq = np.asarray(
        reprojection_errors_sq(K, R, t, X_w, bundle["xy"]),
        dtype=np.float64,
    ).reshape(-1)
    valid = np.isfinite(depth) & (depth > eps) & np.isfinite(error_sq) & (error_sq >= 0.0)
    out = np.full((int(error_sq.size),), np.nan, dtype=np.float64)
    out[valid] = np.sqrt(error_sq[valid])
    return out


# Run the focused frame-16 audit
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
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
    accepted_count = 0
    first_failure = None
    active_basis_before_16 = None
    frame16_acceptance = None
    live_frame19_ids = []
    for frame_index in range(2, 20):
        cur_im, _, _ = seq.get(frame_index)
        if frame_index == 16:
            active_kf = int(get_active_keyframe_kf(seed))
            active_basis_before_16 = {
                "kf": active_kf,
                "pose": _pose(get_pose_for_kf(seed, active_kf, context="frame-16 active basis pose")),
            }
        out = process_frame_against_seed(
            K,
            seed,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            current_kf=frame_index,
            **pnp_cfg,
        )
        stats = out.get("stats", {})
        pose_out = out.get("pose_out", {})
        pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
        if bool(out.get("ok", False)):
            accepted_count += 1
        elif first_failure is None:
            first_failure = int(frame_index)
        if frame_index == 16:
            frame16_acceptance = {
                "pipeline_ok": bool(out.get("ok", False)),
                "n_pnp_corr": int(stats.get("n_pnp_corr", 0)),
                "n_pnp_inliers": int(stats.get("n_pnp_inliers", 0)),
                "rescue_succeeded": bool(pose_stats.get("pnp_support_rescue_succeeded", False)),
                "rescue_reason": pose_stats.get("pnp_support_rescue_reason", None),
                "loose_threshold_px": pose_stats.get("pnp_support_rescue_loose_threshold_px", None),
            }
        if frame_index == 19 and isinstance(pose_out, dict) and pose_out.get("corrs", None) is not None:
            live_frame19_ids = [
                int(value)
                for value in np.asarray(pose_out["corrs"].landmark_ids, dtype=np.int64).reshape(-1)
            ]
        seed = out["seed"]

    pose_by_frame = {
        frame_index: _pose(get_pose_for_kf(seed, frame_index, context="frame-16 pose audit"))
        for frame_index in [15, 16, 17, 18]
    }
    unique_live_ids = sorted(set(live_frame19_ids))
    eps = float(pnp_cfg["eps"])

    bundles = {
        frame_index: _frame_bundle(seed, unique_live_ids, frame_index)
        for frame_index in [15, 16, 17, 18]
    }
    selected_rows = []
    for frame_index in [15, 16, 17, 18]:
        selected_rows.append(
            {
                "frame_index": int(frame_index),
                "missing_count": int(bundles[frame_index]["missing_count"]),
                "duplicate_count": int(bundles[frame_index]["duplicate_count"]),
                **_summary(_errors(K, pose_by_frame[frame_index], bundles[frame_index], eps=eps)),
            }
        )

    lm_by_id = _landmarks(seed)
    history_rows = []
    for landmark_id in unique_live_ids:
        landmark = lm_by_id[int(landmark_id)]
        X_w = np.asarray(landmark["X_w"], dtype=np.float64).reshape(3, 1)
        for observation in landmark.get("obs", []):
            if not isinstance(observation, dict):
                continue
            frame_index = int(observation.get("kf", -1))
            if frame_index not in seed.get("poses", {}):
                continue
            xy = np.asarray(observation.get("xy", None), dtype=np.float64).reshape(-1)
            if xy.size != 2 or not np.isfinite(xy).all():
                continue
            error = _errors(
                K,
                get_pose_for_kf(seed, frame_index, context="frame-16 history pose"),
                {"X_w": X_w, "xy": xy.reshape(2, 1)},
                eps=eps,
            )
            if int(error.size) == 1 and np.isfinite(error[0]):
                history_rows.append(
                    {
                        "landmark_id": int(landmark_id),
                        "frame_index": int(frame_index),
                        "error_px": float(error[0]),
                    }
                )

    history_summary = _summary(np.asarray([row["error_px"] for row in history_rows]))
    selected_squared_error = float(sum(row["squared_error"] for row in selected_rows))
    selected_above_8 = int(sum(row["above_8_count"] for row in selected_rows))
    for row in selected_rows:
        row["squared_error_share_15_18"] = float(row["squared_error"] / selected_squared_error)
        row["above_8_share_15_18"] = float(row["above_8_count"] / selected_above_8)
        row["squared_error_share_all_history"] = float(row["squared_error"] / history_summary["squared_error"])
        row["above_8_share_all_history"] = float(row["above_8_count"] / history_summary["above_8_count"])

    _, ts15, _ = seq.get(15)
    _, ts16, _ = seq.get(16)
    _, ts17, _ = seq.get(17)
    alpha = float((ts16 - ts15) / (ts17 - ts15))
    interpolated_pose = _interpolate_pose(pose_by_frame[15], pose_by_frame[17], alpha)
    frame16_bundle = bundles[16]
    canonical16 = _summary(_errors(K, pose_by_frame[16], frame16_bundle, eps=eps))
    candidate_poses = {
        "canonical_frame_16": pose_by_frame[16],
        "canonical_frame_15": pose_by_frame[15],
        "canonical_frame_17": pose_by_frame[17],
        "canonical_frame_18": pose_by_frame[18],
        "time_interpolated_frame_15_17": interpolated_pose,
        "active_basis_before_frame_16": active_basis_before_16["pose"],
    }
    counterfactual_rows = []
    for label, pose in candidate_poses.items():
        score = _summary(_errors(K, pose, frame16_bundle, eps=eps))
        score["label"] = label
        score["median_change_from_canonical_px"] = float(score["median_px"] - canonical16["median_px"])
        score["squared_error_reduction_from_canonical"] = float(
            (canonical16["squared_error"] - score["squared_error"])
            / canonical16["squared_error"]
        )
        counterfactual_rows.append(score)

    interpolated_errors = _errors(K, interpolated_pose, frame16_bundle, eps=eps)
    interpolated_by_id = {
        int(landmark_id): float(error)
        for landmark_id, error in zip(frame16_bundle["landmark_ids"], interpolated_errors)
        if np.isfinite(error)
    }
    replaced_history = np.asarray(
        [
            interpolated_by_id.get(row["landmark_id"], row["error_px"])
            if row["frame_index"] == 16
            else row["error_px"]
            for row in history_rows
        ],
        dtype=np.float64,
    )
    replaced_summary = _summary(replaced_history)

    R15, t15 = pose_by_frame[15]
    R16, t16 = pose_by_frame[16]
    R17, t17 = pose_by_frame[17]
    C15 = camera_centre(R15, t15)
    C16 = camera_centre(R16, t16)
    C17 = camera_centre(R17, t17)
    step_a = C16 - C15
    step_b = C17 - C16
    chord = C17 - C15
    chord_norm_sq = float(np.dot(chord, chord))
    chord_alpha = float(np.dot(C16 - C15, chord) / chord_norm_sq)
    path_length = float(np.linalg.norm(step_a) + np.linalg.norm(step_b))
    local_path = {
        "camera_motion_turn_deg": _direction_deg(step_a, step_b),
        "adjacent_step_ratio": float(
            max(np.linalg.norm(step_a), np.linalg.norm(step_b))
            / min(np.linalg.norm(step_a), np.linalg.norm(step_b))
        ),
        "camera_path_ratio": float(path_length / np.linalg.norm(chord)),
        "frame16_chord_projection_alpha": chord_alpha,
        "frame16_outside_neighbour_chord": bool(chord_alpha < 0.0 or chord_alpha > 1.0),
        "rotation_path_excess_deg": float(
            np.degrees(angle_between_rotmats(R15, R16))
            + np.degrees(angle_between_rotmats(R16, R17))
            - np.degrees(angle_between_rotmats(R15, R17))
        ),
    }

    result = {
        "event": "frame16_pose_quality",
        "profile": str(profile_path),
        "run_summary": {
            "accepted_count": int(accepted_count),
            "failed_count": int(18 - accepted_count),
            "first_failure": first_failure,
        },
        "frame16_acceptance": frame16_acceptance,
        "frame19_live_landmarks": {
            "correspondence_count": int(len(live_frame19_ids)),
            "unique_count": int(len(unique_live_ids)),
            "landmark_ids": unique_live_ids,
        },
        "pose_neighbours": {
            "frame16_vs_frame15": _pose_delta(pose_by_frame[15], pose_by_frame[16]),
            "frame16_vs_frame17": _pose_delta(pose_by_frame[16], pose_by_frame[17]),
            "frame16_vs_frame18": _pose_delta(pose_by_frame[16], pose_by_frame[18]),
            "active_basis_before_frame16": {
                "kf": int(active_basis_before_16["kf"]),
                "distinct_from_frame15": bool(active_basis_before_16["kf"] != 15),
            },
            "local_path_15_16_17": local_path,
        },
        "landmark_history": {
            "all_history": history_summary,
            "selected_frames": selected_rows,
        },
        "counterfactual_frame16_pose": {
            "time_interpolation_alpha_15_17": alpha,
            "rows": counterfactual_rows,
            "full_history_with_interpolated_frame16": {
                "baseline": history_summary,
                "counterfactual": replaced_summary,
                "squared_error_reduction": float(
                    (history_summary["squared_error"] - replaced_summary["squared_error"])
                    / history_summary["squared_error"]
                ),
                "above_8_count_reduction": int(
                    history_summary["above_8_count"] - replaced_summary["above_8_count"]
                ),
            },
        },
    }

    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.out is not None:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
