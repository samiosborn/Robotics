# src/slam/pnp_frontend.py

from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import align_bool_mask_1d, check_finite_scalar, check_in_01, check_int_gt0, check_matrix_3x3, check_positive
from geometry.pnp import build_pnp_correspondences_with_stats, estimate_pose_pnp_ransac


# Build a histogram of observation counts for kept landmarks
def _landmark_observation_histogram(seed: dict[str, Any], landmark_ids: np.ndarray) -> dict[str, int]:
    # Read landmarks
    landmarks = seed.get("landmarks", [])
    if not isinstance(landmarks, list):
        return {}

    # Build landmark lookup
    lm_by_id: dict[int, dict[str, Any]] = {}
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        if "id" not in lm:
            continue
        lm_by_id[int(lm["id"])] = lm

    # Count kept correspondences by landmark observation count
    hist: dict[str, int] = {}
    landmark_ids = np.asarray(landmark_ids, dtype=np.int64).reshape(-1)
    for lm_id in landmark_ids:
        lm = lm_by_id.get(int(lm_id), None)
        if lm is None:
            continue

        obs = lm.get("obs", None)
        if not isinstance(obs, list):
            n_obs = 0
        else:
            n_obs = int(sum(1 for ob in obs if isinstance(ob, dict)))

        key = str(int(n_obs))
        hist[key] = int(hist.get(key, 0) + 1)

    return {str(k): int(hist[k]) for k in sorted(hist, key=lambda s: int(s))}


# Estimate the current pose from a bootstrap seed and tracked observations
def estimate_pose_from_seed(
    K: np.ndarray,
    seed: dict[str, Any],
    track_out: dict[str, Any],
    *,
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
) -> dict[str, Any]:
    # Check intrinsics
    check_matrix_3x3(K, name="K", dtype=float, finite=False)

    # Check containers
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    if not isinstance(track_out, dict):
        raise ValueError("track_out must be a dict")

    # Check RANSAC controls
    num_trials = check_int_gt0(num_trials, name="num_trials")
    sample_size = check_int_gt0(sample_size, name="sample_size")
    min_inliers = check_int_gt0(min_inliers, name="min_inliers")
    min_points = check_int_gt0(min_points, name="min_points")
    ransac_seed = int(ransac_seed)

    # Check linear solver controls
    threshold_px = check_positive(threshold_px, name="threshold_px", eps=0.0)
    rank_tol = check_positive(rank_tol, name="rank_tol", eps=0.0)
    min_cheirality_ratio = check_finite_scalar(min_cheirality_ratio, name="min_cheirality_ratio")
    check_in_01(min_cheirality_ratio, name="min_cheirality_ratio", eps=0.0)
    min_landmark_observations = check_int_gt0(min_landmark_observations, name="min_landmark_observations")
    allow_bootstrap_landmarks_for_pose = bool(allow_bootstrap_landmarks_for_pose)
    min_post_bootstrap_observations_for_pose = check_int_gt0(
        min_post_bootstrap_observations_for_pose,
        name="min_post_bootstrap_observations_for_pose",
    )
    eps = check_positive(eps, name="eps", eps=0.0)

    # Check nonlinear refinement controls
    refine_max_iters = check_int_gt0(refine_max_iters, name="refine_max_iters")
    refine_damping = check_positive(refine_damping, name="refine_damping", eps=0.0)
    refine_step_tol = check_positive(refine_step_tol, name="refine_step_tol", eps=0.0)
    refine_improvement_tol = check_positive(refine_improvement_tol, name="refine_improvement_tol", eps=0.0)

    # Require a valid DLT sample size
    if sample_size < 6:
        raise ValueError(f"sample_size must be >= 6 for linear PnP; got {sample_size}")

    # Require a valid solver minimum
    if min_points < 6:
        raise ValueError(f"min_points must be >= 6 for linear PnP; got {min_points}")

    # Require a meaningful inlier minimum
    if min_inliers < sample_size:
        raise ValueError(f"min_inliers must be >= sample_size; got {min_inliers} < {sample_size}")

    # Build 2D–3D correspondences from the tracked frame
    corrs, corr_stats = build_pnp_correspondences_with_stats(
        seed,
        track_out,
        min_landmark_observations=min_landmark_observations,
        allow_bootstrap_landmarks_for_pose=allow_bootstrap_landmarks_for_pose,
        min_post_bootstrap_observations_for_pose=min_post_bootstrap_observations_for_pose,
    )
    n_corr = int(corrs.X_w.shape[1])
    landmark_observation_histogram = _landmark_observation_histogram(seed, corrs.landmark_ids)

    # Start stats from the correspondence count
    stats: dict[str, Any] = {
        "n_corr": n_corr,
        "n_corr_raw": int(corr_stats.get("n_corr_raw", 0)),
        "n_corr_bootstrap_born": int(corr_stats.get("n_corr_bootstrap_born", 0)),
        "n_corr_post_bootstrap_born": int(corr_stats.get("n_corr_post_bootstrap_born", 0)),
        "n_corr_after_pose_filter": int(corr_stats.get("n_corr_after_pose_filter", n_corr)),
        "n_corr_bootstrap_used": int(corr_stats.get("n_corr_bootstrap_used", 0)),
        "n_corr_post_bootstrap_used": int(corr_stats.get("n_corr_post_bootstrap_used", 0)),
        "min_landmark_observations": int(min_landmark_observations),
        "allow_bootstrap_landmarks_for_pose": bool(allow_bootstrap_landmarks_for_pose),
        "min_post_bootstrap_observations_for_pose": int(min_post_bootstrap_observations_for_pose),
        "landmark_observation_histogram": landmark_observation_histogram,
    }

    # Stop early if nothing survived
    if n_corr == 0:
        stats.update({"reason": "no_pnp_correspondences"})
        return {
            "ok": False,
            "R": None,
            "t": None,
            "corrs": corrs,
            "pnp_inlier_mask": np.zeros((0,), dtype=bool),
            "landmark_ids": np.zeros((0,), dtype=np.int64),
            "stats": stats,
        }

    # Run robust PnP on the correspondence bundle
    try:
        R, t, pnp_inlier_mask, pnp_stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=num_trials,
            sample_size=sample_size,
            threshold_px=threshold_px,
            min_inliers=min_inliers,
            seed=ransac_seed,
            min_points=min_points,
            rank_tol=rank_tol,
            min_cheirality_ratio=min_cheirality_ratio,
            eps=eps,
            refit=refit,
            refine_nonlinear=refine_nonlinear,
            refine_max_iters=refine_max_iters,
            refine_damping=refine_damping,
            refine_step_tol=refine_step_tol,
            refine_improvement_tol=refine_improvement_tol,
        )
    except Exception as exc:
        stats.update({"reason": "pnp_failed", "error": str(exc)})
        return {
            "ok": False,
            "R": None,
            "t": None,
            "corrs": corrs,
            "pnp_inlier_mask": np.zeros((n_corr,), dtype=bool),
            "landmark_ids": np.zeros((0,), dtype=np.int64),
            "stats": stats,
        }

    # Merge solver stats into the frontend stats
    if isinstance(pnp_stats, dict):
        stats.update(pnp_stats)

    # Build an aligned PnP inlier mask
    pnp_inlier_mask = align_bool_mask_1d(pnp_inlier_mask, n_corr)
    stats.update({"n_pnp_inliers": int(pnp_inlier_mask.sum())})

    # Read inlier landmark ids
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    landmark_ids_inlier = landmark_ids[pnp_inlier_mask]

    # Determine success from the returned pose
    ok = (R is not None) and (t is not None)
    if not ok and stats.get("reason") is None:
        stats.update({"reason": "pnp_pose_missing"})

    return {
        "ok": bool(ok),
        "R": None if R is None else np.asarray(R, dtype=np.float64),
        "t": None if t is None else np.asarray(t, dtype=np.float64).reshape(3),
        "corrs": corrs,
        "pnp_inlier_mask": pnp_inlier_mask,
        "landmark_ids": landmark_ids_inlier,
        "stats": stats,
    }
