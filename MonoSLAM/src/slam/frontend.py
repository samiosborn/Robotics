# src/slam/frontend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.checks import as_2xN_points, check_int_gt0, check_matrix_3x3, check_positive
from features.pipeline import FrameFeatures, detect_and_describe_image
from slam.bootstrap import bootstrap_two_view, planar_check
from slam.matching import MatchBundle, match_frames, matched_keypoints_xy
from slam.seed import attach_feature_bookkeeping_to_seed
from slam.two_view_consensus import estimate_fundamental_consensus, estimate_homography_consensus, recover_pose_from_fundamental_consensus, select_two_view_mask


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
