# scripts/diag_pnp_eth3d.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from demo_frontend_eth3d import ROOT, _append_jsonl, _frontend_kwargs_from_cfg, _load_runtime_cfg

from core.checks import check_dir, check_int_ge0, check_int_gt0, check_positive
from datasets.eth3d import load_eth3d_sequence
from geometry.camera import reprojection_rmse, world_to_camera_points
from geometry.pnp import estimate_pose_pnp_ransac
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.pnp_frontend import estimate_pose_from_seed
from slam.tracking import track_against_keyframe


# Build the fixed PnP solver settings used by the frontend
def _pnp_solver_cfg() -> dict:
    return {
        "num_trials": 1000,
        "sample_size": 8,
        "min_inliers": 12,
        "ransac_seed": 0,
        "min_points": 6,
        "rank_tol": 1e-10,
        "min_cheirality_ratio": 0.5,
        "allow_bootstrap_landmarks_for_pose": True,
        "min_post_bootstrap_observations_for_pose": 3,
        "eps": 1e-12,
        "refit": True,
        "refine_nonlinear": True,
        "refine_max_iters": 15,
        "refine_damping": 1e-6,
        "refine_step_tol": 1e-9,
        "refine_improvement_tol": 1e-9,
    }


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
def _run_threshold_diag(corrs, K: np.ndarray, *, threshold_px: float, pnp_cfg: dict) -> dict:
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
        R, t, _, stats = estimate_pose_pnp_ransac(
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

    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)
    pnp_cfg = _pnp_solver_cfg()
    thresholds = _parse_thresholds(list(args.thresholds))
    min_landmark_observations = check_int_gt0(args.min_landmark_observations, name="min_landmark_observations")

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

        # Track the current frame against the active keyframe
        track_out = track_against_keyframe(
            K,
            ref_keyframe_feats,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
        )

        # Build the usual pose frontend output once at the configured threshold
        base_pose_out = estimate_pose_from_seed(
            K,
            seed,
            track_out,
            threshold_px=frontend_kwargs["pnp_threshold_px"],
            num_trials=int(pnp_cfg["num_trials"]),
            sample_size=int(pnp_cfg["sample_size"]),
            min_inliers=int(pnp_cfg["min_inliers"]),
            ransac_seed=int(pnp_cfg["ransac_seed"]),
            min_points=int(pnp_cfg["min_points"]),
            rank_tol=float(pnp_cfg["rank_tol"]),
            min_cheirality_ratio=float(pnp_cfg["min_cheirality_ratio"]),
            min_landmark_observations=int(min_landmark_observations),
            allow_bootstrap_landmarks_for_pose=bool(pnp_cfg["allow_bootstrap_landmarks_for_pose"]),
            min_post_bootstrap_observations_for_pose=int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
            eps=float(pnp_cfg["eps"]),
            refit=bool(pnp_cfg["refit"]),
            refine_nonlinear=bool(pnp_cfg["refine_nonlinear"]),
            refine_max_iters=int(pnp_cfg["refine_max_iters"]),
            refine_damping=float(pnp_cfg["refine_damping"]),
            refine_step_tol=float(pnp_cfg["refine_step_tol"]),
            refine_improvement_tol=float(pnp_cfg["refine_improvement_tol"]),
        )

        corrs = base_pose_out["corrs"]
        track_stats = track_out.get("stats", {})
        base_pose_stats = base_pose_out.get("stats", {})

        # Sweep PnP thresholds on the same correspondence bundle
        diag_rows = [
            _run_threshold_diag(corrs, K, threshold_px=float(threshold_px), pnp_cfg=pnp_cfg)
            for threshold_px in thresholds
        ]

        # Advance the real frontend state after recording diagnostics
        frontend_out = process_frame_against_seed(
            K,
            seed,
            ref_keyframe_feats,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            threshold_px=frontend_kwargs["pnp_threshold_px"],
            min_landmark_observations=int(min_landmark_observations),
            keyframe_kf=ref_keyframe_index,
            current_kf=i,
            sample_size=int(pnp_cfg["sample_size"]),
            allow_bootstrap_landmarks_for_pose=bool(pnp_cfg["allow_bootstrap_landmarks_for_pose"]),
            min_post_bootstrap_observations_for_pose=int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
        )

        # Record the frame context in the JSONL log
        frame_context = {
            "frame_index": int(i),
            "frame_id": str(cur_id),
            "timestamp": float(cur_ts),
            "reference_keyframe_index": int(ref_keyframe_index),
            "track_ok": int(track_stats.get("n_inliers", 0)) > 0,
            "n_track_matches": int(track_stats.get("n_matches", 0)),
            "n_track_inliers": int(track_stats.get("n_inliers", 0)),
            "base_pose_ok": bool(base_pose_out.get("ok", False)),
            "base_pose_reason": base_pose_stats.get("reason", None),
            "base_n_pnp_corr": int(base_pose_stats.get("n_corr", 0)),
            "base_n_pnp_corr_raw": int(base_pose_stats.get("n_corr_raw", 0)),
            "base_n_pnp_corr_bootstrap_born": int(base_pose_stats.get("n_corr_bootstrap_born", 0)),
            "base_n_pnp_corr_post_bootstrap_born": int(base_pose_stats.get("n_corr_post_bootstrap_born", 0)),
            "n_corr_after_pose_filter": int(base_pose_stats.get("n_corr_after_pose_filter", base_pose_stats.get("n_corr", 0))),
            "n_corr_bootstrap_used": int(base_pose_stats.get("n_corr_bootstrap_used", 0)),
            "n_corr_post_bootstrap_used": int(base_pose_stats.get("n_corr_post_bootstrap_used", 0)),
            "base_n_pnp_inliers": int(base_pose_stats.get("n_pnp_inliers", 0)),
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
            "pipeline_ok": bool(frontend_out.get("ok", False)),
            "pipeline_reason": frontend_out.get("stats", {}).get("reason", None),
            "pipeline_keyframe_promoted": bool(frontend_out.get("stats", {}).get("keyframe_promoted", False)),
            "pipeline_keyframe_reason": frontend_out.get("stats", {}).get("keyframe_reason", None),
        }

        # Write the frame summary
        _append_jsonl(
            log_path,
            {
                "event": "frame_summary",
                **frame_context,
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
            f"frame {i}: ref_kf={ref_keyframe_index} "
            f"n_track_inliers={int(track_stats.get('n_inliers', 0))} "
            f"n_corr_raw={int(base_pose_stats.get('n_corr_raw', 0))} "
            f"n_corr_after_pose_filter={int(base_pose_stats.get('n_corr_after_pose_filter', base_pose_stats.get('n_corr', 0)))} "
            f"n_corr_bootstrap_used={int(base_pose_stats.get('n_corr_bootstrap_used', 0))} "
            f"n_corr_post_used={int(base_pose_stats.get('n_corr_post_bootstrap_used', 0))} "
            f"min_obs={int(base_pose_stats.get('min_landmark_observations', min_landmark_observations))} "
            f"configured_ok={bool(base_pose_out.get('ok', False))} "
            f"promoted={bool(frontend_out.get('stats', {}).get('keyframe_promoted', False))}"
        )
        print(f"  {_format_threshold_summary(diag_rows)}")

        # Update the live frontend state for the next frame
        seed = frontend_out["seed"]
        if bool(frontend_out.get("stats", {}).get("keyframe_promoted", False)):
            keyframe_feats = frontend_out["track_out"]["cur_feats"]
            keyframe_index = i


if __name__ == "__main__":
    main()
