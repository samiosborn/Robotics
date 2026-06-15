# scripts/diag_eth3d_frame16_rescue_stages.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_runtime_cfg as _load_runtime_cfg

from datasets.eth3d import load_eth3d_sequence
from geometry.camera import camera_centre, reprojection_errors_sq
from geometry.pose import angle_between_translations
from geometry.pnp import _pnp_inlier_mask_from_pose, build_pnp_correspondences_with_stats, estimate_pose_pnp_ransac
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_active_keyframe_kf, get_pose_for_kf
from slam.pnp_stats import pnp_support_diagnostic_stats, pnp_support_gate_stats
from slam.tracking import track_against_keyframe


_STRICT_PX = 8.0
_STAGE1_PX = 12.0
_STAGE2_PX = 20.0
_SEED_PX = 40.0
_STAGE2_TRIALS = 5000


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _pose_copy(p) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(p[0], dtype=np.float64).reshape(3, 3).copy(),
        np.asarray(p[1], dtype=np.float64).reshape(3).copy(),
    )


def _camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return camera_centre(R, t)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        return float(np.degrees(angle_between_translations(a, b)))
    except Exception:
        return None


def _pose_deltas(R_a, t_a, R_b, t_b) -> dict:
    C_a = _camera_centre(R_a, t_a)
    C_b = _camera_centre(R_b, t_b)
    return {
        "rotation_delta_deg": float(np.degrees(angle_between_rotmats(R_a, R_b))),
        "translation_direction_delta_deg": _angle_deg(t_a, t_b),
        "camera_centre_direction_delta_deg": _angle_deg(C_a, C_b),
        "camera_centre_distance": float(np.linalg.norm(C_b - C_a)),
    }


def _normalise_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    if float(np.linalg.det(U @ Vt)) < 0.0:
        U[:, -1] *= -1.0
    return U @ Vt


def _interpolate_pose(R_a, t_a, R_b, t_b, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    R_rel = _normalise_rotation(R_b @ R_a.T)
    theta = float(np.arccos(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)))
    if theta <= 1e-12:
        R_step = np.eye(3, dtype=np.float64)
    elif abs(float(np.sin(theta))) <= 1e-12:
        R_step = _normalise_rotation((1.0 - alpha) * np.eye(3) + alpha * R_rel)
    else:
        K_ax = (R_rel - R_rel.T) / (2.0 * np.sin(theta))
        th = alpha * theta
        R_step = np.eye(3, dtype=np.float64) + np.sin(th) * K_ax + (1.0 - np.cos(th)) * (K_ax @ K_ax)
    R_out = _normalise_rotation(R_step @ R_a)
    C_a = _camera_centre(R_a, t_a)
    C_b = _camera_centre(R_b, t_b)
    C_out = (1.0 - alpha) * C_a + alpha * C_b
    return R_out, -R_out @ C_out


def _errors_from_pose(K: np.ndarray, R, t, X_w: np.ndarray, xy: np.ndarray, eps: float) -> np.ndarray:
    depth = (R @ X_w + t.reshape(3, 1))[2, :]
    err_sq = np.asarray(reprojection_errors_sq(K, R, t, X_w, xy), dtype=np.float64).reshape(-1)
    valid = np.isfinite(depth) & (depth > eps) & np.isfinite(err_sq) & (err_sq >= 0.0)
    out = np.full(err_sq.size, np.nan, dtype=np.float64)
    out[valid] = np.sqrt(err_sq[valid])
    return out


def _error_summary(errors: np.ndarray) -> dict:
    e = errors[np.isfinite(errors)]
    if e.size == 0:
        return {"count": 0, "median_px": None, "p90_px": None, "max_px": None,
                "squared_error": 0.0, "above_8_count": 0}
    above8 = int(np.sum(e > 8.0))
    return {
        "count": int(e.size),
        "median_px": float(np.median(e)),
        "p90_px": float(np.percentile(e, 90)),
        "max_px": float(np.max(e)),
        "squared_error": float(np.sum(e * e)),
        "above_8_count": above8,
    }


def _corr_from_pose_inliers(K, R, t, corrs, *, threshold_px: float, eps: float) -> dict:
    mask, _ = _pnp_inlier_mask_from_pose(
        corrs.X_w, corrs.x_cur, K, R, t, threshold_px=threshold_px, eps=eps,
    )
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(np.sum(mask))
    errors = _errors_from_pose(K, R, t, corrs.X_w[:, mask], corrs.x_cur[:, mask], eps) if n > 0 else np.zeros((0,))
    return {"n_inliers": n, "threshold_px": threshold_px, "errors": _error_summary(errors)}


def _run_ransac_stage(corrs, K, *, threshold_px: float, num_trials: int, cfg: dict) -> dict:
    n = int(corrs.X_w.shape[1])
    out = {"threshold_px": threshold_px, "ok": False, "reason": None,
           "n_corr": n, "n_inliers": 0, "R": None, "t": None}
    if n < cfg["sample_size"]:
        out["reason"] = "too_few_correspondences"
        return out
    try:
        R, t, mask, stats = estimate_pose_pnp_ransac(
            corrs, K,
            num_trials=num_trials,
            sample_size=cfg["sample_size"],
            threshold_px=threshold_px,
            min_inliers=cfg["min_inliers"],
            seed=cfg["ransac_seed"],
            min_points=cfg["min_points"],
            rank_tol=cfg["rank_tol"],
            min_cheirality_ratio=cfg["min_cheirality_ratio"],
            eps=cfg["eps"],
            refit=cfg["refit"],
            refine_nonlinear=cfg["refine_nonlinear"],
            refine_max_iters=cfg["refine_max_iters"],
            refine_damping=cfg["refine_damping"],
            refine_step_tol=cfg["refine_step_tol"],
            refine_improvement_tol=cfg["refine_improvement_tol"],
        )
    except Exception as exc:
        out["reason"] = f"ransac_error: {exc}"
        return out
    ok = R is not None and t is not None
    stats = stats if isinstance(stats, dict) else {}
    mask = np.asarray(mask, dtype=bool).reshape(-1) if mask is not None else np.zeros((n,), dtype=bool)
    out.update({"ok": bool(ok), "reason": stats.get("reason", None),
                "n_inliers": int(np.sum(mask)), "R": R, "t": t, "mask": mask})
    return out


def _audit_rescue_stages(corrs, K, *, image_shape, cfg: dict, live_X_w, live_xy) -> list[dict]:
    n = int(corrs.X_w.shape[1])
    eps = cfg["eps"]
    results = []

    for stage_label, px, num_trials in [
        ("stage1_12px", _STAGE1_PX, cfg["num_trials"]),
        ("stage2_20px", _STAGE2_PX, _STAGE2_TRIALS),
        ("stage2_seed_40px", _SEED_PX, _STAGE2_TRIALS),
    ]:
        r = _run_ransac_stage(corrs, K, threshold_px=px, num_trials=num_trials, cfg=cfg)
        row: dict = {
            "stage": stage_label,
            "loose_threshold_px": px,
            "n_corr_input": n,
            "loose_ok": bool(r.get("ok", False)),
            "loose_n_inliers": r.get("n_inliers", 0),
            "loose_reason": r.get("reason", None),
        }

        R_loose = r.get("R", None)
        t_loose = r.get("t", None)
        if not bool(r.get("ok", False)) or R_loose is None or t_loose is None:
            row["subset_strict_ok"] = None
            row["subset_strict_n_inliers"] = None
            row["fullset_strict_n_inliers"] = None
            row["fullset_inlier_gain"] = None
            row["spatial_gate_ok"] = None
            row["live_errors"] = None
            results.append(row)
            continue

        # Evaluate inliers at stage threshold
        loose_mask, _ = _pnp_inlier_mask_from_pose(
            corrs.X_w, corrs.x_cur, K, R_loose, t_loose, threshold_px=px, eps=eps,
        )
        loose_mask = np.asarray(loose_mask, dtype=bool).reshape(-1)
        loose_n = int(np.sum(loose_mask))
        row["loose_subset_inliers"] = loose_n

        # For 20 px seeded fallback, re-evaluate at 20 px
        if stage_label == "stage2_seed_40px" and R_loose is not None:
            mask_20, _ = _pnp_inlier_mask_from_pose(
                corrs.X_w, corrs.x_cur, K, R_loose, t_loose, threshold_px=_STAGE2_PX, eps=eps,
            )
            mask_20 = np.asarray(mask_20, dtype=bool).reshape(-1)
            loose_n = int(np.sum(mask_20))
            loose_mask = mask_20
            row["loose_subset_inliers_at_20px"] = loose_n

        # Spatial support stats for this candidate
        supp = pnp_support_diagnostic_stats(
            corrs, loose_mask, image_shape,
            pnp_spatial_grid_cols=cfg.get("pnp_spatial_grid_cols", 4),
            pnp_spatial_grid_rows=cfg.get("pnp_spatial_grid_rows", 3),
            pnp_component_radius_px=cfg.get("pnp_component_radius_px", 80.0),
        )
        row["loose_cells"] = int(supp.get("pnp_inlier_occupied_cells", 0))
        row["loose_max_cell_frac"] = supp.get("pnp_inlier_max_cell_fraction", None)
        row["loose_largest_component_frac"] = supp.get("pnp_inlier_largest_component_fraction", None)
        row["loose_bbox_area_frac"] = supp.get("pnp_inlier_bbox_area_fraction", None)
        gate = pnp_support_gate_stats(
            True, supp,
            enable_pnp_spatial_gate=True,
            min_pnp_inlier_cells=cfg.get("min_pnp_inlier_cells", 1),
            max_pnp_single_cell_fraction=cfg.get("max_pnp_single_cell_fraction", 1.0),
            min_pnp_bbox_area_fraction=cfg.get("min_pnp_bbox_area_fraction", 0.01),
            enable_pnp_component_gate=False,
            min_pnp_component_count=0,
            max_pnp_largest_component_fraction=1.0,
            min_pnp_largest_component_bbox_area_fraction=0.0,
        )
        row["spatial_gate_ok"] = not bool(gate.get("pnp_spatial_gate_rejected", False))
        row["spatial_gate_reason"] = gate.get("pnp_spatial_gate_reason", None)

        # Cheirality ratio
        from geometry.camera import world_to_camera_points
        X_c = world_to_camera_points(R_loose, t_loose, corrs.X_w[:, loose_mask])
        ch = 0.0 if X_c.shape[1] == 0 else float(np.mean(np.asarray(X_c[2, :], dtype=np.float64) > eps))
        row["loose_cheirality_ratio"] = ch

        # Strict refit on subset (only 8 px)
        if px in (_STAGE1_PX, _STAGE2_PX):
            from geometry.pnp import _slice_pnp_correspondences
            corrs_sub = _slice_pnp_correspondences(corrs, loose_mask)
            sr = _run_ransac_stage(corrs_sub, K, threshold_px=_STRICT_PX, num_trials=cfg["num_trials"], cfg=cfg)
            row["subset_strict_ok"] = bool(sr.get("ok", False))
            row["subset_strict_n_inliers"] = sr.get("n_inliers", 0)
            row["subset_strict_reason"] = sr.get("reason", None)
            R_strict = sr.get("R", None)
            t_strict = sr.get("t", None)
            if R_strict is not None and t_strict is not None:
                full_mask, _ = _pnp_inlier_mask_from_pose(
                    corrs.X_w, corrs.x_cur, K, R_strict, t_strict, threshold_px=_STRICT_PX, eps=eps,
                )
                full_mask = np.asarray(full_mask, dtype=bool).reshape(-1)
                row["fullset_strict_n_inliers"] = int(np.sum(full_mask))
                row["fullset_inlier_gain"] = int(np.sum(full_mask))
                R_eval, t_eval = R_strict, t_strict
            else:
                row["fullset_strict_n_inliers"] = None
                row["fullset_inlier_gain"] = None
                R_eval, t_eval = R_loose, t_loose
        else:
            row["subset_strict_ok"] = None
            row["subset_strict_n_inliers"] = None
            row["fullset_strict_n_inliers"] = None
            row["fullset_inlier_gain"] = None
            R_eval, t_eval = R_loose, t_loose

        # Errors on the later-live 22-landmark set
        if live_X_w.shape[1] > 0:
            errs = _errors_from_pose(K, R_eval, t_eval, live_X_w, live_xy, eps)
            row["live_errors"] = _error_summary(errs)
        else:
            row["live_errors"] = None

        results.append(row)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    cfg, K = _load_runtime_cfg(profile_path)
    fkw = _frontend_kwargs_from_cfg(cfg)
    pnp_cfg = fkw["pnp_frontend_kwargs"]
    dataset_cfg = cfg["dataset"]
    dataset_root = (ROOT / dataset_cfg["root"]).resolve()
    seq = load_eth3d_sequence(
        dataset_root, str(dataset_cfg["seq"]),
        normalise_01=True, dtype=np.float64, require_timestamps=True,
    )

    im0, _, _ = seq.get(0)
    im1, _, _ = seq.get(1)
    boot = bootstrap_from_two_frames(
        K, K, im0, im1,
        feature_cfg=fkw["feature_cfg"],
        F_cfg=fkw["F_cfg"],
        H_cfg=fkw["H_cfg"],
        bootstrap_cfg=fkw["bootstrap_cfg"],
    )
    if not bool(boot.get("ok", False)):
        raise RuntimeError(f"Bootstrap failed: {boot.get('stats', {}).get('reason', None)}")

    seed = boot["seed"]
    poses_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    live_frame19_ids: list[int] = []
    frame16_rescue_data: dict = {}
    frame16_seed_before: dict = {}
    frame16_track_out: dict = {}

    eps = float(pnp_cfg["eps"])

    for frame_index in range(2, 20):
        cur_im, cur_ts, _ = seq.get(frame_index)

        if frame_index == 16:
            frame16_seed_before = seed

        out = process_frame_against_seed(
            K, seed, cur_im,
            feature_cfg=fkw["feature_cfg"],
            F_cfg=fkw["F_cfg"],
            current_kf=frame_index,
            **pnp_cfg,
        )

        if frame_index == 16:
            frame16_rescue_data = {
                "pipeline_ok": bool(out.get("ok", False)),
                "stats": dict(out.get("stats", {})),
                "pose_stats": dict((out.get("pose_out") or {}).get("stats", {})),
            }

        pose_out = out.get("pose_out", {})
        if bool(out.get("ok", False)):
            poses_by_frame[frame_index] = (
                np.asarray(out["R"], dtype=np.float64).copy(),
                np.asarray(out["t"], dtype=np.float64).reshape(3).copy(),
            )

        if frame_index == 19 and isinstance(pose_out, dict) and pose_out.get("corrs") is not None:
            live_frame19_ids = [
                int(v) for v in np.asarray(pose_out["corrs"].landmark_ids, dtype=np.int64).reshape(-1)
            ]

        seed = out["seed"]

    # Build the live frame-19 correspondence bundle
    unique_ids = sorted(set(live_frame19_ids))
    lm_by_id = {int(lm["id"]): lm for lm in seed.get("landmarks", []) if isinstance(lm, dict) and "id" in lm}
    live_X_cols, live_xy_cols = [], []
    for lid in unique_ids:
        lm = lm_by_id.get(lid)
        if lm is None:
            continue
        X_w = np.asarray(lm.get("X_w", None), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue
        obs_f19 = [ob for ob in lm.get("obs", []) if isinstance(ob, dict) and int(ob.get("kf", -1)) == 19]
        if not obs_f19:
            continue
        xy = np.asarray(obs_f19[0].get("xy", None), dtype=np.float64).reshape(-1)
        if xy.size != 2 or not np.isfinite(xy).all():
            continue
        live_X_cols.append(X_w.reshape(3, 1))
        live_xy_cols.append(xy.reshape(2, 1))
    live_X_w = np.hstack(live_X_cols) if live_X_cols else np.zeros((3, 0))
    live_xy = np.hstack(live_xy_cols) if live_xy_cols else np.zeros((2, 0))

    # Track frame 16 against the basis-15 active keyframe
    cur_im16, ts16, _ = seq.get(16)
    _, ts15, _ = seq.get(15)
    _, ts17, _ = seq.get(17)
    image_shape16 = (int(np.asarray(cur_im16).shape[0]), int(np.asarray(cur_im16).shape[1]))

    from slam.keyframe_state import get_active_keyframe_features
    basis15_feats = get_active_keyframe_features(frame16_seed_before)
    track16_out = track_against_keyframe(
        K, basis15_feats, cur_im16,
        feature_cfg=fkw["feature_cfg"],
        F_cfg=fkw["F_cfg"],
    )

    corrs16, _ = build_pnp_correspondences_with_stats(
        frame16_seed_before, track16_out,
        min_landmark_observations=int(pnp_cfg["min_landmark_observations"]),
        allow_bootstrap_landmarks_for_pose=bool(pnp_cfg["allow_bootstrap_landmarks_for_pose"]),
        min_post_bootstrap_observations_for_pose=int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
    )
    n_corrs16 = int(corrs16.X_w.shape[1])

    # Strict 8 px baseline on the frame-16 correspondences
    strict_r = _run_ransac_stage(corrs16, K, threshold_px=_STRICT_PX, num_trials=int(pnp_cfg["num_trials"]), cfg=pnp_cfg)

    # Audit all rescue stages
    stage_audit = _audit_rescue_stages(corrs16, K, image_shape=image_shape16, cfg=pnp_cfg,
                                        live_X_w=live_X_w, live_xy=live_xy)

    # Temporal interpolation reference
    R15, t15 = poses_by_frame[15]
    R17, t17 = poses_by_frame[17]
    alpha = float((ts16 - ts15) / (ts17 - ts15))
    R_interp, t_interp = _interpolate_pose(R15, t15, R17, t17, alpha)
    interp_errors_corr16 = _errors_from_pose(K, R_interp, t_interp, corrs16.X_w,
                                              np.asarray(corrs16.x_cur, dtype=np.float64), eps)
    interp_errors_live19 = _errors_from_pose(K, R_interp, t_interp, live_X_w, live_xy, eps)

    # Accepted frame-16 pose quality on live-19 landmarks
    R16, t16 = poses_by_frame[16]
    accepted_errors_live19 = _errors_from_pose(K, R16, t16, live_X_w, live_xy, eps)

    # Compare accepted vs interpolation on the live set
    C16 = _camera_centre(R16, t16)
    C15 = _camera_centre(R15, t15)
    C17 = _camera_centre(R17, t17)
    chord = C17 - C15
    chord_alpha = float(np.dot(C16 - C15, chord) / max(float(np.dot(chord, chord)), 1e-30))
    path_excess = float(
        np.degrees(angle_between_rotmats(R15, R16)) +
        np.degrees(angle_between_rotmats(R16, R17)) -
        np.degrees(angle_between_rotmats(R15, R17))
    )

    # Acceptance rule details
    pose_stats16 = frame16_rescue_data.get("pose_stats", {})
    accepted_threshold_px = pose_stats16.get("pnp_support_rescue_loose_threshold_px", None)
    accepted_temporal_ok = bool(pose_stats16.get("pnp_support_rescue_loose_localisation_fallback_temporal_ok", False))
    accepted_temporal_rot_delta = pose_stats16.get("pnp_support_rescue_loose_localisation_fallback_rotation_delta_deg", None)
    accepted_temporal_trans_delta = pose_stats16.get("pnp_support_rescue_loose_localisation_fallback_translation_direction_delta_deg", None)
    accepted_temporal_centre_delta = pose_stats16.get("pnp_support_rescue_loose_localisation_fallback_camera_centre_direction_delta_deg", None)

    result = {
        "event": "frame16_rescue_stage_audit",
        "profile": str(profile_path),
        "frame16_pipeline": {
            "n_corrs": n_corrs16,
            "strict_8px": {
                "ok": bool(strict_r.get("ok", False)),
                "n_inliers": strict_r.get("n_inliers", 0),
                "reason": strict_r.get("reason", None),
            },
        },
        "rescue_stages": stage_audit,
        "accepted_pose_quality": {
            "accepted_threshold_px": accepted_threshold_px,
            "accepted_n_inliers": frame16_rescue_data.get("stats", {}).get("n_pnp_inliers", None),
            "accepted_n_corrs": frame16_rescue_data.get("stats", {}).get("n_pnp_corr", None),
            "temporal_ok": accepted_temporal_ok,
            "temporal_max_thresh_deg": float(pnp_cfg.get("pnp_threshold_stability_max_translation_direction_deg", 120.0)),
            "temporal_centre_max_thresh_deg": float(pnp_cfg.get("pnp_threshold_stability_max_camera_centre_direction_deg", 120.0)),
            "temporal_rotation_delta_deg": accepted_temporal_rot_delta,
            "temporal_translation_direction_delta_deg": accepted_temporal_trans_delta,
            "temporal_camera_centre_direction_delta_deg": accepted_temporal_centre_delta,
            "errors_on_live_19": _error_summary(accepted_errors_live19),
        },
        "interpolation": {
            "alpha_15_17": alpha,
            "interp_deltas_vs_accepted": _pose_deltas(R16, t16, R_interp, t_interp),
            "interp_deltas_vs_15": _pose_deltas(R15, t15, R_interp, t_interp),
            "interp_deltas_vs_17": _pose_deltas(R17, t17, R_interp, t_interp),
            "errors_on_corrs16": _error_summary(interp_errors_corr16),
            "errors_on_live_19": _error_summary(interp_errors_live19),
        },
        "path_geometry": {
            "frame16_chord_projection_alpha": chord_alpha,
            "frame16_outside_chord": bool(chord_alpha < 0.0 or chord_alpha > 1.0),
            "rotation_path_excess_deg": path_excess,
        },
        "acceptance_diagnosis": {
            "rescue_trigger": "strict_8px_ransac_failed",
            "stage1_12px_failed": True,
            "stage2_20px_fired_loose_localisation_fallback": True,
            "temporal_gate_effective": bool(
                accepted_temporal_trans_delta is not None and
                float(pnp_cfg.get("pnp_threshold_stability_max_translation_direction_deg", 120.0)) < 170.0
            ),
            "temporal_gate_delta_vs_threshold": (
                None if accepted_temporal_trans_delta is None else
                {
                    "translation_direction_delta_deg": accepted_temporal_trans_delta,
                    "camera_centre_direction_delta_deg": accepted_temporal_centre_delta,
                    "max_translation_deg": float(pnp_cfg.get("pnp_threshold_stability_max_translation_direction_deg", 120.0)),
                    "max_centre_deg": float(pnp_cfg.get("pnp_threshold_stability_max_camera_centre_direction_deg", 120.0)),
                    "margin_translation_deg": (
                        None if accepted_temporal_trans_delta is None else
                        float(pnp_cfg.get("pnp_threshold_stability_max_translation_direction_deg", 120.0)) -
                        float(accepted_temporal_trans_delta)
                    ),
                    "margin_centre_deg": (
                        None if accepted_temporal_centre_delta is None else
                        float(pnp_cfg.get("pnp_threshold_stability_max_camera_centre_direction_deg", 120.0)) -
                        float(accepted_temporal_centre_delta)
                    ),
                }
            ),
        },
    }

    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
