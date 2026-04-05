# src/slam/tracking.py

from __future__ import annotations

from typing import Any

import numpy as np

from core.checks import align_bool_mask_1d, check_matrix_3x3
from features.pipeline import FrameFeatures, detect_and_describe_image
from slam.matching import MatchBundle, match_frames, matched_keypoints_xy
from slam.two_view_consensus import estimate_fundamental_consensus


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
    matches: MatchBundle = match_frames(
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
    inlier_mask = align_bool_mask_1d(F_mask, M)
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
