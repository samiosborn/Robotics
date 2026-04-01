# src/slam/frontend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.checks import as_2xN_points, check_choice, check_finite_scalar, check_in_01, check_int_ge0, check_int_gt0, check_mask_bool_1d, check_matrix_3x3, check_points_xy_N2plus, check_positive, check_required_keys, check_vector_3
from features.matching import match_brief_hamming_with_scale_gate, match_patches_ncc
from features.pipeline import FrameFeatures, detect_and_describe_image
from geometry.pnp import build_pnp_correspondences, estimate_pose_pnp_ransac
from slam.bootstrap import bootstrap_two_view, planar_check
from slam.map_update import grow_map_from_tracking_result
from slam.seed import attach_feature_bookkeeping_to_seed
from slam.two_view_consensus import estimate_fundamental_consensus, estimate_homography_consensus, recover_pose_from_fundamental_consensus, select_two_view_mask


# Matching result
@dataclass(frozen=True)
class MatchBundle:
    ia: np.ndarray
    ib: np.ndarray
    score: np.ndarray


# Two-view estimation result
@dataclass(frozen=True)
class TwoViewEstimation:
    F: np.ndarray | None
    H: np.ndarray | None
    inlier_mask: np.ndarray | None
    R: np.ndarray | None
    t: np.ndarray | None
    E: np.ndarray | None
    cheirality_mask: np.ndarray | None
    stats: dict[str, Any]


# Align a possibly missing mask to an expected length
def _align_bool_mask(mask, N: int) -> np.ndarray:
    # Check requested length
    N = check_int_ge0(N, name="N")

    # Check input mask
    mask = check_mask_bool_1d(mask, name="mask")
    if mask is None:
        return np.zeros((N,), dtype=bool)

    # Align by truncation or zero-padding
    out = np.zeros((N,), dtype=bool)
    n = min(N, int(mask.size))
    out[:n] = np.asarray(mask[:n], dtype=bool)

    return out


# Read the frozen keyframe pose stored inside the seed
def _seed_keyframe_pose(seed: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # Check seed contains the keyframe pose
    seed = check_required_keys(seed, {"T_WC1"}, name="seed")

    # Read stored pose tuple
    T_WC1 = seed["T_WC1"]
    if not isinstance(T_WC1, (tuple, list)) or len(T_WC1) != 2:
        raise ValueError("seed['T_WC1'] must be a length-2 tuple/list (R, t)")

    # Check pose blocks
    R = check_matrix_3x3(T_WC1[0], name="seed['T_WC1'][0]", dtype=float, finite=False)
    t = check_vector_3(T_WC1[1], name="seed['T_WC1'][1]", dtype=float, finite=False)

    return R, t


# Match two feature bundles while preserving original feature indices
def match_frames(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    *,
    mode: str | None = None,
    ncc_min_score: float = 0.7,
    brief_mode: str = "nn",
    brief_max_dist: int | None = 80,
    brief_ratio: float = 0.8,
    mutual: bool = True,
    max_matches: int | None = None,
    scale_gate: int = 1,
) -> MatchBundle:
    # Read keypoints and descriptors
    kpsA = np.asarray(featsA.kps_xy)
    kpsB = np.asarray(featsB.kps_xy)
    descA = np.asarray(featsA.desc)
    descB = np.asarray(featsB.desc)

    # Early exit on empty features
    if kpsA.shape[0] == 0 or kpsB.shape[0] == 0:
        return MatchBundle(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )
    if descA.shape[0] == 0 or descB.shape[0] == 0:
        return MatchBundle(
            ia=np.zeros((0,), dtype=np.int64),
            ib=np.zeros((0,), dtype=np.int64),
            score=np.zeros((0,), dtype=np.float64),
        )

    # Infer the mode from descriptor structure when it is not supplied
    if mode is None:
        if descA.ndim == 3 and descB.ndim == 3:
            mode = "ncc"
        else:
            mode = "brief"

    # Validate common controls
    mode = check_choice(mode, {"ncc", "brief"}, name="match mode")
    mutual = bool(mutual)

    if max_matches is not None:
        max_matches = check_int_gt0(max_matches, name="max_matches")

    # NCC match mode
    if mode == "ncc":
        # Validate NCC controls
        ncc_min_score = check_finite_scalar(ncc_min_score, name="ncc_min_score")
        check_in_01(ncc_min_score, name="ncc_min_score", eps=0.0)

        # Run patch NCC matching
        m = match_patches_ncc(
            descA,
            descB,
            min_score=ncc_min_score,
            mutual=mutual,
            max_matches=max_matches,
        )

        return MatchBundle(
            ia=np.asarray(m.ia, dtype=np.int64),
            ib=np.asarray(m.ib, dtype=np.int64),
            score=np.asarray(m.score, dtype=np.float64),
        )

    # Validate BRIEF controls
    brief_mode = check_choice(brief_mode, {"nn", "ratio"}, name="brief mode")

    if brief_max_dist is not None:
        brief_max_dist = check_int_ge0(brief_max_dist, name="brief_max_dist")

    brief_ratio = check_finite_scalar(brief_ratio, name="brief_ratio")
    if brief_ratio <= 0.0 or brief_ratio >= 1.0:
        raise ValueError(f"brief_ratio must be in (0,1); got {brief_ratio}")

    scale_gate = check_int_ge0(scale_gate, name="scale_gate")

    # Read pyramid levels with a safe zero default
    lvlA = np.asarray(featsA.level, dtype=np.int64).reshape(-1) if featsA.level is not None else np.zeros((descA.shape[0],), dtype=np.int64)
    lvlB = np.asarray(featsB.level, dtype=np.int64).reshape(-1) if featsB.level is not None else np.zeros((descB.shape[0],), dtype=np.int64)

    # Run BRIEF Hamming matching with a scale gate
    m = match_brief_hamming_with_scale_gate(
        descA,
        lvlA,
        descB,
        lvlB,
        mode=brief_mode,
        max_dist=brief_max_dist,
        ratio=brief_ratio,
        mutual=mutual,
        max_matches=max_matches,
        scale_gate_levels=scale_gate,
    )

    return MatchBundle(
        ia=np.asarray(m.ia, dtype=np.int64),
        ib=np.asarray(m.ib, dtype=np.int64),
        score=np.asarray(m.score, dtype=np.float64),
    )


# Convert matched feature indices into matched coordinate pairs
def matched_keypoints_xy(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    matches: MatchBundle,
) -> tuple[np.ndarray, np.ndarray]:
    # Check match index arrays
    ia = np.asarray(matches.ia, dtype=np.int64).reshape(-1)
    ib = np.asarray(matches.ib, dtype=np.int64).reshape(-1)

    # Require matched index arrays to agree
    if ia.size != ib.size:
        raise ValueError(f"matches.ia and matches.ib must have equal size; got {ia.size} and {ib.size}")

    # Early exit on empty matches
    if ia.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    # Check keypoint arrays
    kpsA = check_points_xy_N2plus(featsA.kps_xy, name="featsA.kps_xy", dtype=float, finite=True)
    kpsB = check_points_xy_N2plus(featsB.kps_xy, name="featsB.kps_xy", dtype=float, finite=True)

    # Check feature index bounds
    if int(ia.min()) < 0 or int(ia.max()) >= int(kpsA.shape[0]):
        raise ValueError("matches.ia contains out-of-range feature indices")
    if int(ib.min()) < 0 or int(ib.max()) >= int(kpsB.shape[0]):
        raise ValueError("matches.ib contains out-of-range feature indices")

    # Gather matched image coordinates
    xyA = np.asarray(kpsA[ia, :2], dtype=np.float64)
    xyB = np.asarray(kpsB[ib, :2], dtype=np.float64)

    return xyA, xyB


# Estimate thin two-view geometry from matched features
def estimate_two_view(
    K1: np.ndarray,
    K2: np.ndarray,
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    matches: MatchBundle,
    *,
    F_cfg: dict[str, Any],
    H_cfg: dict[str, Any],
    mask_policy: str = "F",
    planar_gamma: float = 1.2,
    planar_min_H_inliers: int = 20,
) -> TwoViewEstimation:
    # Check intrinsics
    check_matrix_3x3(K1, name="K1", dtype=float, finite=False)
    check_matrix_3x3(K2, name="K2", dtype=float, finite=False)

    # Validate explicit two-view controls
    if not isinstance(F_cfg, dict):
        raise ValueError("F_cfg must be a dict")
    if not isinstance(H_cfg, dict):
        raise ValueError("H_cfg must be a dict")

    mask_policy = str(mask_policy)
    planar_gamma = check_positive(planar_gamma, name="planar_gamma", eps=0.0)
    planar_min_H_inliers = check_int_gt0(planar_min_H_inliers, name="planar_min_H_inliers")

    # Build the minimal two-view config needed by the consensus code
    two_view_cfg = {
        "ransac": {
            "F": dict(F_cfg),
            "H": dict(H_cfg),
        },
        "bootstrap": {
            "mask_policy": mask_policy,
        },
    }

    # Start stats
    stats: dict[str, Any] = {}
    ia = np.asarray(matches.ia, dtype=np.int64).reshape(-1)
    stats.update({"n_matches": int(ia.size)})

    # Build matched image coordinates
    xyA, xyB = matched_keypoints_xy(featsA, featsB, matches)

    # Estimate fundamental consensus
    F, F_mask, F_stats = estimate_fundamental_consensus(xyA, xyB, two_view_cfg)
    stats.update(
        {
            "F_n0": int(F_stats.get("n0", 0)),
            "F_n1": int(F_stats.get("n1", F_stats.get("n0", 0))),
            "F_refit": bool(F_stats.get("refit", False)),
            "F_shrink_ratio": F_stats.get("shrink_ratio", None),
            "F_used_mask": F_stats.get("used_mask", None),
            "nF": int(F_stats.get("n_inliers", 0)),
        }
    )
    if F_stats.get("reason") is not None:
        stats.update({"F_reason": F_stats.get("reason")})

    # Stop if F consensus failed
    if F is None or F_mask is None or not bool(F_stats.get("refit", False)):
        stats.update({"reason": F_stats.get("reason", "fundamental_consensus_failed")})
        return TwoViewEstimation(F, None, None, None, None, None, None, stats)

    # Estimate homography consensus
    H, H_mask, H_stats = estimate_homography_consensus(xyA, xyB, two_view_cfg)
    if H_stats.get("reason") is not None:
        stats.update({"H_reason": H_stats.get("reason")})
    if H_mask is not None:
        stats.update({"nH": int(np.asarray(H_mask, dtype=bool).sum())})

    # Run a planar diagnostic when both masks exist
    if F_mask is not None and H_mask is not None:
        planar_degen, planar_stats = planar_check(
            mask_F=F_mask,
            mask_H=H_mask,
            gamma=planar_gamma,
            min_H_inliers=planar_min_H_inliers,
        )
        stats.update(
            {
                "planar_degenerate": bool(planar_degen),
                "H_over_F_ratio": float(planar_stats.get("ratio", 0.0)),
            }
        )

    # Recover pose from F
    R, t, E, cheir_ratio, cheir_mask, pose_stats = recover_pose_from_fundamental_consensus(
        F,
        F_mask,
        K1,
        K2,
        xyA,
        xyB,
    )
    if pose_stats.get("reason") is not None:
        stats.update({"pose_reason": pose_stats.get("reason")})

    # Stop if pose recovery failed
    if R is None or t is None or E is None or cheir_mask is None:
        stats.update({"reason": pose_stats.get("reason", "pose_recovery_failed")})
        return TwoViewEstimation(F, H, np.asarray(F_mask, dtype=bool), None, None, None, None, stats)

    # Select the final two-view mask
    final_mask = select_two_view_mask(F_mask, cheir_mask, two_view_cfg)

    # Record mask statistics
    stats.update(
        {
            "mask_policy": mask_policy,
            "n_mask": int(final_mask.sum()),
            "cheir_ratio": float(cheir_ratio),
            "nC": int(np.asarray(cheir_mask, dtype=bool).sum()),
        }
    )

    return TwoViewEstimation(
        F=np.asarray(F, dtype=np.float64),
        H=None if H is None else np.asarray(H, dtype=np.float64),
        inlier_mask=np.asarray(final_mask, dtype=bool),
        R=np.asarray(R, dtype=np.float64),
        t=np.asarray(t, dtype=np.float64).reshape(3),
        E=np.asarray(E, dtype=np.float64),
        cheirality_mask=np.asarray(cheir_mask, dtype=bool),
        stats=stats,
    )


# Bootstrap a map seed from two raw images
def bootstrap_from_two_frames(
    K1: np.ndarray,
    K2: np.ndarray,
    im0: np.ndarray,
    im1: np.ndarray,
    *,
    feature_cfg: dict[str, Any],
    match_mode: str | None = None,
    ncc_min_score: float = 0.7,
    brief_mode: str = "nn",
    brief_max_dist: int | None = 80,
    brief_ratio: float = 0.8,
    mutual: bool = True,
    max_matches: int | None = None,
    scale_gate: int = 1,
    F_cfg: dict[str, Any],
    H_cfg: dict[str, Any],
    bootstrap_cfg: dict[str, Any],
) -> dict[str, Any]:
    # Check explicit config blocks
    if not isinstance(feature_cfg, dict):
        raise ValueError("feature_cfg must be a dict")
    if not isinstance(F_cfg, dict):
        raise ValueError("F_cfg must be a dict")
    if not isinstance(H_cfg, dict):
        raise ValueError("H_cfg must be a dict")
    if not isinstance(bootstrap_cfg, dict):
        raise ValueError("bootstrap_cfg must be a dict")

    # Build the minimal bootstrap config required downstream
    two_view_cfg = {
        "ransac": {
            "F": dict(F_cfg),
            "H": dict(H_cfg),
        },
        "bootstrap": dict(bootstrap_cfg),
    }

    # Detect and describe both bootstrap frames
    feats0 = detect_and_describe_image(im0, feature_cfg)
    feats1 = detect_and_describe_image(im1, feature_cfg)

    # Match features across the bootstrap pair
    matches = match_frames(
        feats0,
        feats1,
        mode=match_mode,
        ncc_min_score=ncc_min_score,
        brief_mode=brief_mode,
        brief_max_dist=brief_max_dist,
        brief_ratio=brief_ratio,
        mutual=mutual,
        max_matches=max_matches,
        scale_gate=scale_gate,
    )
    xy0, xy1 = matched_keypoints_xy(feats0, feats1, matches)

    # Convert to the (2,N) convention used by bootstrap geometry
    x0 = as_2xN_points(xy0, name="xy0", finite=True, dtype=float)
    x1 = as_2xN_points(xy1, name="xy1", finite=True, dtype=float)

    # Run the validated two-view bootstrap
    try:
        ok, stats, aux, seed = bootstrap_two_view(K1, K2, x0, x1, two_view_cfg)
    except Exception as exc:
        ok = False
        stats = {"reason": "bootstrap_failed", "error": str(exc)}
        aux = {}
        seed = None

    # Attach feature bookkeeping needed for later PnP
    if ok and isinstance(seed, dict):
        seed = attach_feature_bookkeeping_to_seed(seed, feats0, feats1, matches)

    return {
        "ok": bool(ok),
        "feats0": feats0,
        "feats1": feats1,
        "matches01": matches,
        "xy0": np.asarray(xy0, dtype=np.float64),
        "xy1": np.asarray(xy1, dtype=np.float64),
        "stats": stats,
        "aux": aux,
        "seed": seed,
    }


# Track the current image against a reference keyframe
def track_against_keyframe(
    K: np.ndarray,
    keyframe_feats: FrameFeatures,
    cur_im: np.ndarray,
    *,
    feature_cfg: dict[str, Any],
    match_mode: str | None = None,
    ncc_min_score: float = 0.7,
    brief_mode: str = "nn",
    brief_max_dist: int | None = 80,
    brief_ratio: float = 0.8,
    mutual: bool = True,
    max_matches: int | None = None,
    scale_gate: int = 1,
    F_cfg: dict[str, Any],
) -> dict[str, Any]:
    # Check intrinsics
    check_matrix_3x3(K, name="K", dtype=float, finite=False)

    # Check explicit config blocks
    if not isinstance(feature_cfg, dict):
        raise ValueError("feature_cfg must be a dict")
    if not isinstance(F_cfg, dict):
        raise ValueError("F_cfg must be a dict")

    # Build the minimal fundamental-consensus config required downstream
    fundamental_cfg = {
        "ransac": {
            "F": dict(F_cfg),
        }
    }

    # Detect current frame features
    cur_feats = detect_and_describe_image(cur_im, feature_cfg)

    # Match keyframe features into the current frame
    matches = match_frames(
        keyframe_feats,
        cur_feats,
        mode=match_mode,
        ncc_min_score=ncc_min_score,
        brief_mode=brief_mode,
        brief_max_dist=brief_max_dist,
        brief_ratio=brief_ratio,
        mutual=mutual,
        max_matches=max_matches,
        scale_gate=scale_gate,
    )

    # Convert tentative matches into image coordinates
    xy_kf_all, xy_cur_all = matched_keypoints_xy(keyframe_feats, cur_feats, matches)
    M = int(xy_kf_all.shape[0])

    # Start stats
    stats: dict[str, Any] = {"n_matches": M}

    # Estimate a geometric inlier mask with F consensus
    F, F_mask, F_stats = estimate_fundamental_consensus(xy_kf_all, xy_cur_all, fundamental_cfg)
    stats.update(
        {
            "n0": int(F_stats.get("n0", 0)),
            "n1": int(F_stats.get("n1", F_stats.get("n0", 0))),
            "refit": bool(F_stats.get("refit", False)),
            "shrink_ratio": F_stats.get("shrink_ratio", None),
        }
    )

    # Build an aligned inlier mask
    inlier_mask = _align_bool_mask(F_mask, M)
    stats.update({"n_inliers": int(inlier_mask.sum())})

    # Record failure reason when consensus fails
    if F is None or F_mask is None or not bool(F_stats.get("refit", False)):
        stats.update({"reason": F_stats.get("reason", "fundamental_consensus_failed")})

    # Gather inlier feature indices
    ia = np.asarray(matches.ia, dtype=np.int64).reshape(-1)
    ib = np.asarray(matches.ib, dtype=np.int64).reshape(-1)
    kf_feat_idx = ia[inlier_mask]
    cur_feat_idx = ib[inlier_mask]

    # Gather inlier image coordinates
    xy_kf = np.asarray(xy_kf_all[inlier_mask], dtype=np.float64)
    xy_cur = np.asarray(xy_cur_all[inlier_mask], dtype=np.float64)

    return {
        "cur_feats": cur_feats,
        "matches": matches,
        "inlier_mask": inlier_mask,
        "xy_kf": xy_kf,
        "xy_cur": xy_cur,
        "kf_feat_idx": kf_feat_idx,
        "cur_feat_idx": cur_feat_idx,
        "F": None if F is None else np.asarray(F, dtype=np.float64),
        "stats": stats,
    }


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
    corrs = build_pnp_correspondences(seed, track_out)
    n_corr = int(corrs.X_w.shape[1])

    # Start stats from the correspondence count
    stats: dict[str, Any] = {"n_corr": n_corr}

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
    pnp_inlier_mask = _align_bool_mask(pnp_inlier_mask, n_corr)
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


# Process one new frame against the current seed map and grow the map if pose estimation succeeds
def process_frame_against_seed(
    K: np.ndarray,
    seed: dict[str, Any],
    keyframe_feats: FrameFeatures,
    cur_im: np.ndarray,
    *,
    feature_cfg: dict[str, Any],
    match_mode: str | None = None,
    ncc_min_score: float = 0.7,
    brief_mode: str = "nn",
    brief_max_dist: int | None = 80,
    brief_ratio: float = 0.8,
    mutual: bool = True,
    max_matches: int | None = None,
    scale_gate: int = 1,
    F_cfg: dict[str, Any],
    num_trials: int = 1000,
    sample_size: int = 6,
    threshold_px: float = 3.0,
    min_inliers: int = 12,
    ransac_seed: int = 0,
    min_points: int = 6,
    rank_tol: float = 1e-10,
    min_cheirality_ratio: float = 0.5,
    eps: float = 1e-12,
    refit: bool = True,
    refine_nonlinear: bool = True,
    refine_max_iters: int = 15,
    refine_damping: float = 1e-6,
    refine_step_tol: float = 1e-9,
    refine_improvement_tol: float = 1e-9,
    keyframe_kf: int = 1,
    current_kf: int = -1,
    grow_map: bool = True,
    min_parallax_deg: float = 1.0,
    max_depth_ratio: float = 200.0,
    max_reproj_error_px: float | None = 3.0,
) -> dict[str, Any]:
    # --- Checks ---
    # Check intrinsics
    check_matrix_3x3(K, name="K", dtype=float, finite=False)

    # Check containers
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    if not isinstance(feature_cfg, dict):
        raise ValueError("feature_cfg must be a dict")
    if not isinstance(F_cfg, dict):
        raise ValueError("F_cfg must be a dict")

    # Check map-growth controls
    keyframe_kf = check_int_ge0(keyframe_kf, name="keyframe_kf")
    current_kf = int(current_kf)
    if current_kf < -1:
        raise ValueError(f"current_kf must be >= -1; got {current_kf}")

    min_parallax_deg = check_positive(min_parallax_deg, name="min_parallax_deg", eps=0.0)
    max_depth_ratio = check_positive(max_depth_ratio, name="max_depth_ratio", eps=0.0)
    if max_reproj_error_px is not None:
        max_reproj_error_px = check_positive(max_reproj_error_px, name="max_reproj_error_px", eps=0.0)

    # Track current frame against the reference keyframe
    track_out = track_against_keyframe(
        K,
        keyframe_feats,
        cur_im,
        feature_cfg=feature_cfg,
        match_mode=match_mode,
        ncc_min_score=ncc_min_score,
        brief_mode=brief_mode,
        brief_max_dist=brief_max_dist,
        brief_ratio=brief_ratio,
        mutual=mutual,
        max_matches=max_matches,
        scale_gate=scale_gate,
        F_cfg=F_cfg,
    )

    # Read tracking stats
    track_stats = track_out.get("stats", {}) if isinstance(track_out, dict) else {}
    n_track_inliers = int(track_stats.get("n_inliers", 0))

    # Early exit if tracking produced no geometric inliers
    if n_track_inliers <= 0:
        stats = {
            "ok": False,
            "reason": track_stats.get("reason", "tracking_failed"),
            "n_track_inliers": 0,
            "n_pnp_corr": 0,
            "n_pnp_inliers": 0,
            "n_new_candidates": 0,
            "n_new_triangulated": 0,
            "n_new_added": 0,
        }
        return {
            "ok": False,
            "seed": seed,
            "track_out": track_out,
            "pose_out": None,
            "map_growth_out": None,
            "R": None,
            "t": None,
            "stats": stats,
        }

    # Estimate current pose from the seed map
    pose_out = estimate_pose_from_seed(
        K,
        seed,
        track_out,
        num_trials=num_trials,
        sample_size=sample_size,
        threshold_px=threshold_px,
        min_inliers=min_inliers,
        ransac_seed=ransac_seed,
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

    # Read pose stats
    pose_stats = pose_out.get("stats", {}) if isinstance(pose_out, dict) else {}
    ok = bool(pose_out.get("ok", False)) if isinstance(pose_out, dict) else False

    # Stop if pose estimation failed
    if not ok:
        stats = {
            "ok": False,
            "reason": pose_stats.get("reason", "pnp_failed"),
            "n_track_matches": int(track_stats.get("n_matches", 0)),
            "n_track_inliers": int(track_stats.get("n_inliers", 0)),
            "n_pnp_corr": int(pose_stats.get("n_corr", 0)),
            "n_pnp_inliers": int(pose_stats.get("n_pnp_inliers", 0)),
            "n_new_candidates": 0,
            "n_new_triangulated": 0,
            "n_new_added": 0,
        }
        return {
            "ok": False,
            "seed": seed,
            "track_out": track_out,
            "pose_out": pose_out,
            "map_growth_out": None,
            "R": None,
            "t": None,
            "stats": stats,
        }

    # Default map-growth output
    seed_out = seed
    map_growth_out = None

    # Grow the map only after a valid pose has been recovered
    if bool(grow_map):
        # Read the frozen keyframe pose from the seed
        R_kf, t_kf = _seed_keyframe_pose(seed)

        # Read the current pose
        R_cur = np.asarray(pose_out["R"], dtype=np.float64)
        t_cur = np.asarray(pose_out["t"], dtype=np.float64).reshape(3)

        # Run one map-growth step from the tracked frame
        map_growth_out = grow_map_from_tracking_result(
            seed,
            track_out,
            K,
            K,
            R_kf,
            t_kf,
            R_cur,
            t_cur,
            keyframe_kf=keyframe_kf,
            current_kf=current_kf,
            descriptor_source=track_out.get("cur_feats", None),
            min_parallax_deg=min_parallax_deg,
            max_depth_ratio=max_depth_ratio,
            max_reproj_error_px=max_reproj_error_px,
            eps=eps,
        )

        # Read the updated seed
        seed_out = map_growth_out.seed

    # Read map-growth stats
    map_stats = map_growth_out.stats if map_growth_out is not None else {}

    # Pack a single frontend result
    stats = {
        "ok": True,
        "reason": None,
        "n_track_matches": int(track_stats.get("n_matches", 0)),
        "n_track_inliers": int(track_stats.get("n_inliers", 0)),
        "n_pnp_corr": int(pose_stats.get("n_corr", 0)),
        "n_pnp_inliers": int(pose_stats.get("n_pnp_inliers", 0)),
        "n_new_candidates": int(map_stats.get("n_candidates", 0)),
        "n_new_triangulated": int(map_stats.get("n_triangulated_valid", 0)),
        "n_new_added": int(map_stats.get("n_added", 0)),
        "seed_landmarks_after": int(map_stats.get("seed_landmarks_after", len(seed_out.get("landmarks", [])))),
    }

    return {
        "ok": True,
        "seed": seed_out,
        "track_out": track_out,
        "pose_out": pose_out,
        "map_growth_out": map_growth_out,
        "R": np.asarray(pose_out["R"], dtype=np.float64),
        "t": np.asarray(pose_out["t"], dtype=np.float64).reshape(3),
        "stats": stats,
    }
