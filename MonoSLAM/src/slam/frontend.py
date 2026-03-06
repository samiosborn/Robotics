# src/slam/frontend.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.checks import as_2xN_points, check_matrix_3x3
from features.matching import match_brief_hamming_with_scale_gate, match_patches_ncc
from features.pipeline import FrameFeatures, detect_and_describe_image
from geometry.fundamental import estimate_fundamental_ransac, refit_fundamental_on_inliers
from geometry.homography import estimate_homography_ransac
from geometry.pose import pose_from_fundamental
from slam.bootstrap import bootstrap_two_view, planar_check


@dataclass(frozen=True)
class MatchBundle:
    ia: np.ndarray
    ib: np.ndarray
    score: np.ndarray


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


def _feature_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, dict) and isinstance(cfg.get("features"), dict):
        return cfg["features"]
    return cfg


def _match_block(cfg: dict[str, Any]) -> dict[str, Any]:
    c = _feature_cfg(cfg)
    m = c.get("match", cfg.get("match", {}))
    return m if isinstance(m, dict) else {}


def _brief_block(cfg: dict[str, Any]) -> dict[str, Any]:
    c = _feature_cfg(cfg)
    b = c.get("brief", cfg.get("brief", {}))
    return b if isinstance(b, dict) else {}


def _ransac_block(cfg: dict[str, Any]) -> dict[str, Any]:
    r = cfg.get("ransac", {})
    return r if isinstance(r, dict) else {}


def _bootstrap_block(cfg: dict[str, Any]) -> dict[str, Any]:
    b = cfg.get("bootstrap", {})
    return b if isinstance(b, dict) else {}


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


def _empty_match_bundle() -> MatchBundle:
    return MatchBundle(
        ia=np.zeros((0,), dtype=np.int64),
        ib=np.zeros((0,), dtype=np.int64),
        score=np.zeros((0,), dtype=np.float64),
    )


def _safe_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _safe_float(value, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


# Detect and describe frame features
# Uses the shared feature pipeline directly

def detect_and_describe(im: np.ndarray, cfg: dict[str, Any]) -> FrameFeatures:
    return detect_and_describe_image(im, _feature_cfg(cfg))


# Match two feature bundles
# Preserves original feature indices

def match_frames(
    featsA: FrameFeatures,
    featsB: FrameFeatures,
    cfg: dict[str, Any],
) -> MatchBundle:
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

        lvlA = np.asarray(featsA.level, dtype=np.int64) if featsA.level is not None else np.zeros((descA.shape[0],), dtype=np.int64)
        lvlB = np.asarray(featsB.level, dtype=np.int64) if featsB.level is not None else np.zeros((descB.shape[0],), dtype=np.int64)

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


# Estimate thin two-view geometry from matched features
# Keeps to F/H/pose orchestration and does not apply full bootstrap acceptance

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
    ib = np.asarray(matches.ib, dtype=np.int64)
    stats.update({"n_matches": int(ia.size)})

    xyA, xyB = matched_keypoints_xy(featsA, featsB, matches)
    if xyA.shape[0] < 8:
        stats.update({"reason": "too_few_matches_for_fundamental", "n_xy": int(xyA.shape[0])})
        return TwoViewEstimation(None, None, None, None, None, None, None, stats)

    x1 = as_2xN_points(xyA, name="xyA", finite=True)
    x2 = as_2xN_points(xyB, name="xyB", finite=True)

    r = _ransac_block(cfg)
    rF = r.get("F", {}) if isinstance(r, dict) else {}
    seed = _safe_int(r.get("seed", 0), 0)

    F_num_trials = _safe_int(rF.get("num_trials", 2000), 2000)
    F_sample_size = _safe_int(rF.get("sample_size", 8), 8)
    F_thr_px = _safe_float(rF.get("threshold_px", 3.0), 3.0)
    F_min_inliers = _safe_int(rF.get("min_inliers", 8), 8)
    F_shrink_guard = _safe_float(rF.get("shrink_guard", 0.8), 0.8)
    F_recompute = bool(rF.get("recompute_mask", True))

    try:
        F, F_mask = estimate_fundamental_ransac(
            x1,
            x2,
            num_trials=F_num_trials,
            sample_size=F_sample_size,
            threshold=F_thr_px,
            seed=seed,
        )
    except Exception as exc:
        stats.update({"reason": "fundamental_ransac_failed", "error": str(exc)})
        return TwoViewEstimation(None, None, None, None, None, None, None, stats)

    try:
        F, F_mask, refit_stats = refit_fundamental_on_inliers(
            x1,
            x2,
            F=F,
            inlier_mask=F_mask,
            min_inliers=F_min_inliers,
            threshold=F_thr_px,
            shrink_guard=F_shrink_guard,
            recompute_mask=F_recompute,
        )
    except Exception as exc:
        stats.update({"reason": "fundamental_refit_failed", "error": str(exc)})
        return TwoViewEstimation(None, None, None, None, None, None, None, stats)

    stats.update({
        "F_n0": int(refit_stats.get("n0", 0)),
        "F_n1": int(refit_stats.get("n1", refit_stats.get("n0", 0))),
        "F_refit": bool(refit_stats.get("refit", False)),
        "F_shrink_ratio": float(refit_stats.get("shrink_ratio", 1.0)),
        "F_used_mask": refit_stats.get("used_mask", None),
    })
    if not bool(refit_stats.get("refit", False)):
        stats.update({"reason": refit_stats.get("reason", "fundamental_refit_failed")})
        return TwoViewEstimation(F, None, None, None, None, None, None, stats)

    if F_mask is None:
        stats.update({"reason": "fundamental_mask_missing"})
        return TwoViewEstimation(F, None, None, None, None, None, None, stats)

    F_mask = np.asarray(F_mask, dtype=bool)
    stats.update({"nF": int(F_mask.sum())})

    H = None
    H_mask = None
    rH = r.get("H", {}) if isinstance(r, dict) else {}
    if isinstance(rH, dict) and len(rH) > 0:
        H_num_trials = _safe_int(rH.get("num_trials", 2000), 2000)
        H_thr_px = _safe_float(rH.get("threshold_px", 3.0), 3.0)
        H_normalise = bool(rH.get("normalise", True))

        H, H_mask, H_reason = estimate_homography_ransac(
            x1,
            x2,
            num_trials=H_num_trials,
            threshold=H_thr_px,
            normalise=H_normalise,
            seed=seed,
        )
        stats.update({"H_reason": H_reason})

        if H_mask is not None:
            H_mask = np.asarray(H_mask, dtype=bool)
            stats.update({"nH": int(H_mask.sum())})

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

    try:
        R, t, E, cheir_ratio, cheir_mask = pose_from_fundamental(
            F,
            K1,
            K2,
            x1,
            x2,
            F_mask=F_mask,
            enforce_constraints=True,
        )
    except Exception as exc:
        stats.update({"reason": "pose_recovery_failed", "error": str(exc)})
        return TwoViewEstimation(F, H, F_mask, None, None, None, None, stats)

    cheir_mask = np.asarray(cheir_mask, dtype=bool)

    b = _bootstrap_block(cfg)
    mask_policy = str(b.get("mask_policy", "F"))
    if mask_policy == "F":
        final_mask = F_mask
    elif mask_policy == "cheirality":
        final_mask = cheir_mask
    elif mask_policy == "intersection":
        final_mask = F_mask & cheir_mask
    else:
        raise ValueError(f"Unknown mask_policy: {mask_policy}")

    final_mask = np.asarray(final_mask, dtype=bool)

    stats.update(
        {
            "mask_policy": mask_policy,
            "n_mask": int(final_mask.sum()),
            "cheir_ratio": float(cheir_ratio),
            "nC": int(cheir_mask.sum()),
        }
    )

    return TwoViewEstimation(
        F=np.asarray(F, dtype=float),
        H=None if H is None else np.asarray(H, dtype=float),
        inlier_mask=final_mask,
        R=np.asarray(R, dtype=float),
        t=np.asarray(t, dtype=float).reshape(3),
        E=np.asarray(E, dtype=float),
        cheirality_mask=cheir_mask,
        stats=stats,
    )


# Bootstrap map seed from two raw images
# Includes descriptor and feature-index bookkeeping for later 2D-3D linkage

def bootstrap_from_two_frames(
    K1: np.ndarray,
    K2: np.ndarray,
    im0: np.ndarray,
    im1: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
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
        ia = np.asarray(matches.ia, dtype=np.int64)
        ib = np.asarray(matches.ib, dtype=np.int64)
        idx_init = np.asarray(seed.get("idx_init", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        landmarks = list(seed.get("landmarks", []))

        for lm_id, lm in enumerate(landmarks):
            if lm_id >= idx_init.size:
                continue

            j = int(idx_init[lm_id])
            if j < 0 or j >= ia.size or j >= ib.size:
                continue

            feat0 = int(ia[j])
            feat1 = int(ib[j])

            lm["match_idx"] = j
            if "obs" in lm and isinstance(lm["obs"], list) and len(lm["obs"]) >= 2:
                lm["obs"][0]["feat"] = feat0
                lm["obs"][1]["feat"] = feat1
                lm["obs"][0]["xy"] = np.asarray(feats0.kps_xy[feat0], dtype=np.float64)
                lm["obs"][1]["xy"] = np.asarray(feats1.kps_xy[feat1], dtype=np.float64)

            desc1 = np.asarray(feats1.desc)
            if desc1.ndim >= 1 and feat1 < desc1.shape[0]:
                lm["descriptor"] = np.asarray(desc1[feat1]).copy()

        n_feat1 = int(np.asarray(feats1.kps_xy).shape[0])
        landmark_id_by_feat1 = np.full((n_feat1,), -1, dtype=np.int64)
        for lm in landmarks:
            lm_id = int(lm.get("id", -1))
            obs = lm.get("obs", [])
            if not isinstance(obs, list) or len(obs) < 2:
                continue
            feat1 = int(obs[1].get("feat", -1))
            if 0 <= feat1 < n_feat1:
                landmark_id_by_feat1[feat1] = lm_id

        seed["landmark_id_by_feat1"] = landmark_id_by_feat1
        seed["feats0"] = feats0
        seed["feats1"] = feats1
        seed["matches01"] = matches

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
# Returns geometrically filtered feature correspondences and index maps

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
    if M < 8:
        inlier_mask = np.zeros((M,), dtype=bool)
        stats.update({"reason": "too_few_matches_for_fundamental"})
        return {
            "cur_feats": cur_feats,
            "matches": matches,
            "inlier_mask": inlier_mask,
            "xy_kf": np.zeros((0, 2), dtype=np.float64),
            "xy_cur": np.zeros((0, 2), dtype=np.float64),
            "kf_feat_idx": np.zeros((0,), dtype=np.int64),
            "cur_feat_idx": np.zeros((0,), dtype=np.int64),
            "F": None,
            "stats": stats,
        }

    x_kf = as_2xN_points(xy_kf_all, name="xy_kf", finite=True)
    x_cur = as_2xN_points(xy_cur_all, name="xy_cur", finite=True)

    r = _ransac_block(cfg)
    rF = r.get("F", {}) if isinstance(r, dict) else {}
    seed = _safe_int(r.get("seed", 0), 0)

    F_num_trials = _safe_int(rF.get("num_trials", 2000), 2000)
    F_sample_size = _safe_int(rF.get("sample_size", 8), 8)
    F_thr_px = _safe_float(rF.get("threshold_px", 3.0), 3.0)
    F_min_inliers = _safe_int(rF.get("min_inliers", 8), 8)
    F_shrink_guard = _safe_float(rF.get("shrink_guard", 0.8), 0.8)
    F_recompute = bool(rF.get("recompute_mask", True))

    try:
        F, F_mask = estimate_fundamental_ransac(
            x_kf,
            x_cur,
            num_trials=F_num_trials,
            sample_size=F_sample_size,
            threshold=F_thr_px,
            seed=seed,
        )
        F, F_mask, refit_stats = refit_fundamental_on_inliers(
            x_kf,
            x_cur,
            F=F,
            inlier_mask=F_mask,
            min_inliers=F_min_inliers,
            threshold=F_thr_px,
            shrink_guard=F_shrink_guard,
            recompute_mask=F_recompute,
        )

        if F_mask is None:
            raise RuntimeError("fundamental_inlier_mask_missing")

        inlier_mask = np.asarray(F_mask, dtype=bool)
        stats.update(
            {
                "n_inliers": int(inlier_mask.sum()),
                "n0": int(refit_stats.get("n0", 0)),
                "n1": int(refit_stats.get("n1", refit_stats.get("n0", 0))),
                "refit": bool(refit_stats.get("refit", False)),
                "shrink_ratio": float(refit_stats.get("shrink_ratio", 1.0)),
            }
        )

    except Exception as exc:
        inlier_mask = np.zeros((M,), dtype=bool)
        F = None
        stats.update({"reason": "fundamental_ransac_failed", "error": str(exc), "n_inliers": 0})

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
        "F": None if F is None else np.asarray(F, dtype=float),
        "stats": stats,
    }
