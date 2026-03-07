# src/slam/two_view.py
from __future__ import annotations

import numpy as np

from core.checks import as_2xN_points, check_matrix_3x3
from geometry.fundamental import estimate_fundamental_ransac, refit_fundamental_on_inliers
from geometry.homography import estimate_homography_ransac
from geometry.pose import pose_from_fundamental

# As dictionary
def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}

# RANSAC config
def _ransac_blocks(cfg: dict) -> tuple[dict, dict, dict]:
    c = _as_dict(cfg)
    r = _as_dict(c.get("ransac"))
    return r, _as_dict(r.get("F")), _as_dict(r.get("H"))

# xy pair to 2xN
def _xy_pair_to_2xN(xyA, xyB) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    stats: dict = {}

    a = np.asarray(xyA)
    b = np.asarray(xyB)
    nA = int(a.shape[0]) if a.ndim == 2 else 0
    nB = int(b.shape[0]) if b.ndim == 2 else 0
    stats.update({"nA": nA, "nB": nB, "n_matches": min(nA, nB)})

    if a.ndim != 2 or b.ndim != 2:
        stats.update({"reason": "invalid_xy_shape"})
        return None, None, stats
    if a.shape[0] != b.shape[0]:
        stats.update({"reason": "mismatched_match_count"})
        return None, None, stats
    if a.shape[1] < 2 or b.shape[1] < 2:
        stats.update({"reason": "invalid_xy_columns"})
        return None, None, stats

    try:
        x1 = as_2xN_points(a, name="xyA", finite=True)
        x2 = as_2xN_points(b, name="xyB", finite=True)
    except Exception as exc:
        stats.update({"reason": "invalid_xy_values", "error": str(exc)})
        return None, None, stats

    return x1, x2, stats

# Estimate fundamental consensus
def estimate_fundamental_consensus(xyA, xyB, cfg) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    x1, x2, stats = _xy_pair_to_2xN(xyA, xyB)
    if x1 is None or x2 is None:
        return None, None, stats

    n_matches = int(x1.shape[1])
    stats.update({"n_matches": n_matches})
    if n_matches < 8:
        stats.update(
            {
                "reason": "too_few_matches",
                "n0": 0,
                "n1": 0,
                "refit": False,
                "shrink_ratio": None,
                "used_mask": None,
                "n_inliers": 0,
            }
        )
        return None, None, stats

    r, rF, _ = _ransac_blocks(cfg)
    seed = int(r.get("seed", 0))

    num_trials = int(rF.get("num_trials", 2000))
    sample_size = int(rF.get("sample_size", 8))
    threshold_px = float(rF.get("threshold_px", 3.0))
    min_inliers = int(rF.get("min_inliers", 8))
    shrink_guard = float(rF.get("shrink_guard", 0.8))
    recompute_mask = bool(rF.get("recompute_mask", True))

    try:
        F, F_mask = estimate_fundamental_ransac(
            x1,
            x2,
            num_trials=num_trials,
            sample_size=sample_size,
            threshold=threshold_px,
            seed=seed,
        )
    except Exception as exc:
        stats.update(
            {
                "reason": "fundamental_ransac_failed",
                "error": str(exc),
                "n0": 0,
                "n1": 0,
                "refit": False,
                "shrink_ratio": None,
                "used_mask": None,
                "n_inliers": 0,
            }
        )
        return None, None, stats

    try:
        F, F_mask, refit_stats = refit_fundamental_on_inliers(
            x1,
            x2,
            F=F,
            inlier_mask=F_mask,
            min_inliers=min_inliers,
            threshold=threshold_px,
            shrink_guard=shrink_guard,
            recompute_mask=recompute_mask,
        )
    except Exception as exc:
        mask = None if F_mask is None else np.asarray(F_mask, dtype=bool).reshape(-1)
        stats.update(
            {
                "reason": "fundamental_refit_failed",
                "error": str(exc),
                "n0": int(mask.sum()) if mask is not None else 0,
                "n1": int(mask.sum()) if mask is not None else 0,
                "refit": False,
                "shrink_ratio": None,
                "used_mask": None,
                "n_inliers": int(mask.sum()) if mask is not None else 0,
            }
        )
        return np.asarray(F, dtype=np.float64), mask, stats

    mask = None if F_mask is None else np.asarray(F_mask, dtype=bool).reshape(-1)
    n0 = int(refit_stats.get("n0", 0))
    n1 = int(refit_stats.get("n1", n0))
    refit = bool(refit_stats.get("refit", False))
    shrink_ratio = refit_stats.get("shrink_ratio", (float(n1) / max(n0, 1)) if n0 > 0 else None)
    used_mask = refit_stats.get("used_mask", None)
    reason = refit_stats.get("reason", None)

    stats.update(
        {
            "n0": n0,
            "n1": n1,
            "refit": refit,
            "shrink_ratio": None if shrink_ratio is None else float(shrink_ratio),
            "used_mask": used_mask,
            "n_inliers": int(mask.sum()) if mask is not None else 0,
        }
    )
    if reason is not None:
        stats.update({"reason": str(reason)})

    if mask is None:
        if stats.get("reason") is None:
            stats.update({"reason": "fundamental_mask_missing"})
        return np.asarray(F, dtype=np.float64), None, stats

    return np.asarray(F, dtype=np.float64), mask, stats

# Estimate homography consensus
def estimate_homography_consensus(xyA, xyB, cfg) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    x1, x2, stats = _xy_pair_to_2xN(xyA, xyB)
    if x1 is None or x2 is None:
        return None, None, stats

    r, _, rH = _ransac_blocks(cfg)
    if len(rH) == 0:
        stats.update({"reason": "homography_disabled", "nH": 0})
        return None, None, stats

    seed = int(r.get("seed", 0))
    num_trials = int(rH.get("num_trials", 2000))
    threshold_px = float(rH.get("threshold_px", 3.0))
    normalise = bool(rH.get("normalise", True))

    try:
        H, H_mask, reason = estimate_homography_ransac(
            x1,
            x2,
            num_trials=num_trials,
            threshold=threshold_px,
            normalise=normalise,
            seed=seed,
        )
    except Exception as exc:
        stats.update({"reason": "homography_ransac_failed", "error": str(exc), "nH": 0})
        return None, None, stats

    if H_mask is None:
        stats.update({"nH": 0})
        if reason is not None:
            stats.update({"reason": str(reason)})
        elif H is None:
            stats.update({"reason": "homography_failed"})
        return None if H is None else np.asarray(H, dtype=np.float64), None, stats

    mask = np.asarray(H_mask, dtype=bool).reshape(-1)
    stats.update({"nH": int(mask.sum())})
    if reason is not None:
        stats.update({"reason": str(reason)})

    return None if H is None else np.asarray(H, dtype=np.float64), mask, stats

# Recover pose from fundamental consensus
def recover_pose_from_fundamental_consensus(
    F,
    F_mask,
    K1,
    K2,
    xyA,
    xyB,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None, np.ndarray | None, dict]:
    stats: dict = {}

    if F is None:
        stats.update({"reason": "fundamental_missing"})
        return None, None, None, None, None, stats

    try:
        K1 = check_matrix_3x3(K1, name="K1", finite=False)
        K2 = check_matrix_3x3(K2, name="K2", finite=False)
    except Exception as exc:
        stats.update({"reason": "invalid_intrinsics", "error": str(exc)})
        return None, None, None, None, None, stats

    x1, x2, xy_stats = _xy_pair_to_2xN(xyA, xyB)
    stats.update({"n_matches": int(xy_stats.get("n_matches", 0))})
    if x1 is None or x2 is None:
        stats.update({"reason": xy_stats.get("reason", "invalid_xy")})
        if "error" in xy_stats:
            stats.update({"error": xy_stats["error"]})
        return None, None, None, None, None, stats

    mask = None if F_mask is None else np.asarray(F_mask, dtype=bool).reshape(-1)
    if mask is not None:
        stats.update({"n_mask": int(mask.sum())})

    try:
        R, t, E, cheir_ratio, cheir_mask = pose_from_fundamental(
            np.asarray(F, dtype=np.float64),
            K1,
            K2,
            x1,
            x2,
            F_mask=mask,
            enforce_constraints=True,
        )
    except Exception as exc:
        stats.update({"reason": "pose_recovery_failed", "error": str(exc)})
        return None, None, None, None, None, stats

    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    E = np.asarray(E, dtype=np.float64)
    cheir_mask = np.asarray(cheir_mask, dtype=bool).reshape(-1)

    stats.update({"cheir_ratio": float(cheir_ratio), "nC": int(cheir_mask.sum())})
    return R, t, E, float(cheir_ratio), cheir_mask, stats

# Select two-view mask
def select_two_view_mask(F_mask, cheir_mask, cfg) -> np.ndarray:
    c = _as_dict(cfg)
    b = _as_dict(c.get("bootstrap"))
    policy_raw = str(b.get("mask_policy", "F"))
    policy = policy_raw.strip().lower()

    mF = None if F_mask is None else np.asarray(F_mask, dtype=bool).reshape(-1)
    mC = None if cheir_mask is None else np.asarray(cheir_mask, dtype=bool).reshape(-1)

    if mF is None and mC is None:
        mF = np.zeros((0,), dtype=bool)
        mC = np.zeros((0,), dtype=bool)
    elif mF is None:
        mF = np.zeros((mC.size,), dtype=bool)
    elif mC is None:
        mC = np.zeros((mF.size,), dtype=bool)

    if mF.size != mC.size:
        n = min(mF.size, mC.size)
        mF = mF[:n]
        mC = mC[:n]

    if policy == "f":
        return mF
    if policy == "cheirality":
        return mC
    if policy == "intersection":
        return mF & mC

    raise ValueError(f"Unknown mask_policy: {policy_raw}")
