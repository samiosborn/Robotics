# src/slam/frontend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.checks import as_2xN_points, check_matrix_3x3
from features.matching import match_brief_hamming_with_scale_gate, match_patches_ncc
from features.pipeline import FrameFeatures, detect_and_describe_image
from slam.bootstrap import bootstrap_two_view, planar_check
from slam.seed import attach_feature_bookkeeping_to_seed
from slam.two_view_consensus import estimate_fundamental_consensus, estimate_homography_consensus, recover_pose_from_fundamental_consensus, select_two_view_mask


# Matching result
@dataclass(frozen=True)
class MatchBundle:
    ia: np.ndarray
    ib: np.ndarray
    score: np.ndarray


# Two view estimation result
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

# Features config
def _feature_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, dict) and isinstance(cfg.get("features"), dict):
        return cfg["features"]
    return cfg

# Matching config
def _match_block(cfg: dict[str, Any]) -> dict[str, Any]:
    c = _feature_cfg(cfg)
    m = c.get("match", cfg.get("match", {}))
    return m if isinstance(m, dict) else {}

# BRIEF config
def _brief_block(cfg: dict[str, Any]) -> dict[str, Any]:
    c = _feature_cfg(cfg)
    b = c.get("brief", cfg.get("brief", {}))
    return b if isinstance(b, dict) else {}

# Bootstrap config
def _bootstrap_block(cfg: dict[str, Any]) -> dict[str, Any]:
    b = cfg.get("bootstrap", {})
    return b if isinstance(b, dict) else {}

# Match mode config
def _match_mode(cfg: dict[str, Any], featsA: FrameFeatures | None = None, featsB: FrameFeatures | None = None) -> str:
    c = _feature_cfg(cfg)
    m = c.get("match", cfg.get("match"))

    if isinstance(m, str):
        return m.lower()
    if isinstance(m, dict):
        mode = m.get("mode", m.get("type"))
        if isinstance(mode, str):
            return mode.lower()

    # Infer from descriptor shapes when mode is absent
    if featsA is not None and featsB is not None:
        dA = np.asarray(featsA.desc)
        dB = np.asarray(featsB.desc)
        if dA.ndim == 3 and dB.ndim == 3:
            return "ncc"

    return "brief"

# Match result (empty)
def _empty_match_bundle() -> MatchBundle:
    return MatchBundle(
        ia=np.zeros((0,), dtype=np.int64),
        ib=np.zeros((0,), dtype=np.int64),
        score=np.zeros((0,), dtype=np.float64),
    )

# Safe integer
def _safe_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)

# Safe float
def _safe_float(value, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


# Detect and describe frame features
def detect_and_describe(im: np.ndarray, cfg: dict[str, Any]) -> FrameFeatures:
    return detect_and_describe_image(im, _feature_cfg(cfg))


# Match two feature bundle (Preserves original feature indices)
def match_frames(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    cfg: dict[str, Any],
) -> MatchBundle:

    # Checks
    kpsA = np.asarray(featsA.kps_xy)
    kpsB = np.asarray(featsB.kps_xy)
    descA = np.asarray(featsA.desc)
    descB = np.asarray(featsB.desc)

    if kpsA.shape[0] == 0 or kpsB.shape[0] == 0:
        return _empty_match_bundle()
    if descA.shape[0] == 0 or descB.shape[0] == 0:
        return _empty_match_bundle()

    mode = _match_mode(cfg, featsA=featsA, featsB=featsB)
    mcfg = _match_block(cfg)


    # NNC match mode
    if mode == "ncc":
        ncc_cfg = mcfg.get("ncc", {}) if isinstance(mcfg, dict) else {}
        min_score = _safe_float(ncc_cfg.get("min_score", mcfg.get("min_score", 0.7)), 0.7)
        mutual = bool(ncc_cfg.get("mutual", mcfg.get("mutual", True)))
        max_matches = ncc_cfg.get("max_matches", mcfg.get("max_matches", None))
        if max_matches is not None:
            max_matches = int(max_matches)

        m = match_patches_ncc(
            descA,
            descB,
            min_score=min_score,
            mutual=mutual,
            max_matches=max_matches,
        )
        return MatchBundle(
            ia=np.asarray(m.ia, dtype=np.int64),
            ib=np.asarray(m.ib, dtype=np.int64),
            score=np.asarray(m.score, dtype=np.float64),
        )

    # BRIEF match mode
    if mode == "brief":
        brief_match_cfg = mcfg.get("brief", {}) if isinstance(mcfg, dict) else {}
        bcfg = _brief_block(cfg)

        brief_mode = str(brief_match_cfg.get("mode", mcfg.get("brief_mode", bcfg.get("mode", "nn")))).lower()
        max_dist = brief_match_cfg.get("max_dist", mcfg.get("max_dist", bcfg.get("max_dist", 80)))
        if max_dist is not None:
            max_dist = int(max_dist)
        ratio = _safe_float(brief_match_cfg.get("ratio", mcfg.get("ratio", bcfg.get("ratio", 0.8))), 0.8)
        mutual = bool(brief_match_cfg.get("mutual", mcfg.get("mutual", bcfg.get("mutual", True))))
        max_matches = brief_match_cfg.get("max_matches", mcfg.get("max_matches", bcfg.get("max_matches", None)))
        if max_matches is not None:
            max_matches = int(max_matches)
        scale_gate = _safe_int(
            brief_match_cfg.get(
                "scale_gate",
                mcfg.get("scale_gate", bcfg.get("scale_gate", bcfg.get("scale_gate_levels", 1))),
            ),
            1,
        )

        lvlA = (
            np.asarray(featsA.level, dtype=np.int64)
            if featsA.level is not None
            else np.zeros((descA.shape[0],), dtype=np.int64)
        )
        lvlB = (
            np.asarray(featsB.level, dtype=np.int64)
            if featsB.level is not None
            else np.zeros((descB.shape[0],), dtype=np.int64)
        )

        m = match_brief_hamming_with_scale_gate(
            descA,
            lvlA,
            descB,
            lvlB,
            mode=brief_mode,
            max_dist=max_dist,
            ratio=ratio,
            mutual=mutual,
            max_matches=max_matches,
            scale_gate=scale_gate,
        )
        return MatchBundle(
            ia=np.asarray(m.ia, dtype=np.int64),
            ib=np.asarray(m.ib, dtype=np.int64),
            score=np.asarray(m.score, dtype=np.float64),
        )

    raise ValueError(f"Unknown match mode: {mode}")


# Convert matched feature indices into matched coordinate pairs
def matched_keypoints_xy(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    matches: MatchBundle,
) -> tuple[np.ndarray, np.ndarray]:

    # Checks
    ia = np.asarray(matches.ia, dtype=np.int64)
    ib = np.asarray(matches.ib, dtype=np.int64)

    if ia.ndim != 1 or ib.ndim != 1:
        raise ValueError(f"matches.ia and matches.ib must be 1D; got {ia.shape} and {ib.shape}")
    if ia.size != ib.size:
        raise ValueError(f"matches.ia and matches.ib must have equal size; got {ia.size} and {ib.size}")
    if ia.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    kpsA = np.asarray(featsA.kps_xy, dtype=np.float64)
    kpsB = np.asarray(featsB.kps_xy, dtype=np.float64)

    if kpsA.ndim != 2 or kpsA.shape[1] < 2:
        raise ValueError(f"featsA.kps_xy must be (N,2+); got {kpsA.shape}")
    if kpsB.ndim != 2 or kpsB.shape[1] < 2:
        raise ValueError(f"featsB.kps_xy must be (N,2+); got {kpsB.shape}")

    if ia.min() < 0 or ia.max() >= kpsA.shape[0]:
        raise ValueError("matches.ia contains out-of-range feature indices")
    if ib.min() < 0 or ib.max() >= kpsB.shape[0]:
        raise ValueError("matches.ib contains out-of-range feature indices")

    xyA = np.asarray(kpsA[ia, :2], dtype=np.float64)
    xyB = np.asarray(kpsB[ib, :2], dtype=np.float64)

    return xyA, xyB


# Estimate (thin) two-view geometry from matched features
def estimate_two_view(
    K1: np.ndarray,
    K2: np.ndarray,
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    matches: MatchBundle,
    cfg: dict[str, Any],
) -> TwoViewEstimation:
    check_matrix_3x3(K1, name="K1", finite=False)
    check_matrix_3x3(K2, name="K2", finite=False)

    stats: dict[str, Any] = {}
    ia = np.asarray(matches.ia, dtype=np.int64)
    stats.update({"n_matches": int(ia.size)})

    xyA, xyB = matched_keypoints_xy(featsA, featsB, matches)

    # Keeps to F/H/pose orchestration and does not apply full bootstrap acceptance
    F, F_mask, F_stats = estimate_fundamental_consensus(xyA, xyB, cfg)
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

    if F is None or F_mask is None or not bool(F_stats.get("refit", False)):
        stats.update({"reason": F_stats.get("reason", "fundamental_consensus_failed")})
        return TwoViewEstimation(F, None, None, None, None, None, None, stats)

    H, H_mask, H_stats = estimate_homography_consensus(xyA, xyB, cfg)
    if H_stats.get("reason") is not None:
        stats.update({"H_reason": H_stats.get("reason")})
    if H_mask is not None:
        stats.update({"nH": int(np.asarray(H_mask, dtype=bool).sum())})

    if F_mask is not None and H_mask is not None:
        b = _bootstrap_block(cfg)
        plan = b.get("planar", {}) if isinstance(b, dict) else {}
        gamma = _safe_float(plan.get("gamma", 1.2), 1.2)
        min_H_inliers = _safe_int(plan.get("min_H_inliers", 20), 20)

        planar_degen, planar_stats = planar_check(
            mask_F=F_mask,
            mask_H=H_mask,
            gamma=gamma,
            min_H_inliers=min_H_inliers,
        )
        stats.update(
            {
                "planar_degenerate": bool(planar_degen),
                "H_over_F_ratio": float(planar_stats.get("ratio", 0.0)),
            }
        )

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

    if R is None or t is None or E is None or cheir_mask is None:
        stats.update({"reason": pose_stats.get("reason", "pose_recovery_failed")})
        return TwoViewEstimation(F, H, np.asarray(F_mask, dtype=bool), None, None, None, None, stats)

    final_mask = select_two_view_mask(F_mask, cheir_mask, cfg)

    b = _bootstrap_block(cfg)
    mask_policy = str(b.get("mask_policy", "F"))
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


# Bootstrap map seed from two raw images
def bootstrap_from_two_frames(
    K1: np.ndarray,
    K2: np.ndarray,
    im0: np.ndarray,
    im1: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:

    # Includes descriptor and feature-index bookkeeping for later 2D-3D linkage
    feats0 = detect_and_describe(im0, cfg)
    feats1 = detect_and_describe(im1, cfg)

    matches = match_frames(feats0, feats1, cfg)
    xy0, xy1 = matched_keypoints_xy(feats0, feats1, matches)

    x0 = as_2xN_points(xy0, name="xy0", finite=True)
    x1 = as_2xN_points(xy1, name="xy1", finite=True)
    
    try:
        ok, stats, aux, seed = bootstrap_two_view(K1, K2, x0, x1, cfg)
    except Exception as exc:
        ok = False
        stats = {"reason": "bootstrap_failed", "error": str(exc)}
        aux = {}
        seed = None

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


# Track current image against a reference keyframe
def track_against_keyframe(
    K: np.ndarray,
    keyframe_feats: FrameFeatures,
    cur_im: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    # Keep signature stable for later pose-aware tracking
    _ = K

    cur_feats = detect_and_describe(cur_im, cfg)
    matches = match_frames(keyframe_feats, cur_feats, cfg)

    xy_kf_all, xy_cur_all = matched_keypoints_xy(keyframe_feats, cur_feats, matches)
    M = int(xy_kf_all.shape[0])

    stats: dict[str, Any] = {"n_matches": M}

    F, F_mask, F_stats = estimate_fundamental_consensus(xy_kf_all, xy_cur_all, cfg)
    stats.update(
        {
            "n0": int(F_stats.get("n0", 0)),
            "n1": int(F_stats.get("n1", F_stats.get("n0", 0))),
            "refit": bool(F_stats.get("refit", False)),
            "shrink_ratio": F_stats.get("shrink_ratio", None),
        }
    )

    # Returns geometrically filtered feature correspondences and index maps
    if F_mask is None:
        inlier_mask = np.zeros((M,), dtype=bool)
    else:
        inlier_mask = np.asarray(F_mask, dtype=bool).reshape(-1)
        if inlier_mask.size != M:
            aligned = np.zeros((M,), dtype=bool)
            n = min(M, inlier_mask.size)
            aligned[:n] = inlier_mask[:n]
            inlier_mask = aligned

    stats.update({"n_inliers": int(inlier_mask.sum())})

    if F is None or F_mask is None or not bool(F_stats.get("refit", False)):
        stats.update({"reason": F_stats.get("reason", "fundamental_consensus_failed")})

    kf_feat_idx = np.asarray(matches.ia, dtype=np.int64)[inlier_mask]
    cur_feat_idx = np.asarray(matches.ib, dtype=np.int64)[inlier_mask]

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
