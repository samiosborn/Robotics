# geometry/degeneracy.py
from __future__ import annotations

import numpy as np

from core.checks import check_mask_bool_1d, check_mask_bool_N, check_matrix_3x3
from core.checks import check_2xN_pair
from geometry.parallax import parallax_angle_stats_deg
from geometry.triangulation import depths_two_view, triangulate_points


# Planar degeneracy from F/H inlier masks
def planar_degeneracy_from_masks(
    mask_F,
    mask_H,
    *,
    gamma: float,
    min_H_inliers: int,
) -> tuple[bool, dict]:
    # Handle absent mask(s)
    if mask_H is None or mask_F is None:
        nF = int(np.sum(mask_F)) if mask_F is not None else 0
        return False, {"nF": nF, "nH": 0, "ratio": 0.0}

    # Validate and compare sizes
    mF = check_mask_bool_1d(mask_F, name="mask_F")
    if mF is None:
        return False, {"nF": 0, "nH": 0, "ratio": 0.0}
    mH = check_mask_bool_N(mask_H, mF.size, name="mask_H")
    if mH is None:
        return False, {"nF": int(mF.sum()), "nH": 0, "ratio": 0.0}

    # Inlier counts
    nF = int(mF.sum())
    nH = int(mH.sum())

    # Degeneracy rule
    degenerate = (nH >= float(gamma) * nF) and (nH >= int(min_H_inliers))
    stats = {"nF": nF, "nH": nH, "ratio": float(nH / max(nF, 1))}
    return degenerate, stats


# Parallax degeneracy from angle statistics in degrees
def parallax_degeneracy_deg(
    R,
    K1,
    K2,
    x1,
    x2,
    *,
    mask=None,
    quartile_trim=(0.1, 0.9),
    min_points=8,
    min_p50_deg=1.0,
    min_p25_deg=0.7,
) -> tuple[bool, dict]:
    # Checks
    check_2xN_pair(x1, x2)
    check_matrix_3x3(R, name="R", finite=False)
    check_matrix_3x3(K1, name="K1", finite=False)
    check_matrix_3x3(K2, name="K2", finite=False)
    N = x1.shape[1]
    mask = check_mask_bool_N(mask, N, name="mask")

    # Default
    n_mask = int(mask.sum()) if mask is not None else int(N)
    stats = {"n_mask": n_mask, "n_trim": 0, "p25": None, "p50": None, "p75": None, "reason": None}

    # Too few correspondences after mask
    if mask is not None and n_mask < int(min_points):
        stats.update({"reason": "mask_too_small"})
        return True, stats

    # Robust parallax angle statistics
    par_stats = parallax_angle_stats_deg(
        R,
        K1,
        K2,
        x1,
        x2,
        mask=mask,
        quartile_trim=quartile_trim,
        min_points=min_points,
    )
    stats.update(
        {
            "n_trim": par_stats["n_trim"],
            "p25": par_stats["p25"],
            "p50": par_stats["p50"],
            "p75": par_stats["p75"],
        }
    )

    # Failure from angle statistics
    if par_stats["reason"] is not None:
        stats.update({"reason": par_stats["reason"]})
        return True, stats

    # Threshold checks
    if stats["p50"] < float(min_p50_deg):
        stats.update({"reason": "parallax_p50_too_small"})
        return True, stats
    if stats["p25"] < float(min_p25_deg):
        stats.update({"reason": "parallax_p25_too_small"})
        return True, stats

    return False, stats


# Two-view depth/cheirality degeneracy check
def depth_degeneracy_two_view(
    R,
    t,
    K1,
    K2,
    x1,
    x2,
    *,
    mask=None,
    min_points=20,
    cheirality_min=0.7,
    depth_max_ratio=100.0,
    depth_sanity_min=0.7,
    eps=1e-9,
) -> tuple[bool, dict, dict]:
    # Checks
    check_2xN_pair(x1, x2)
    check_matrix_3x3(R, name="R", finite=False)
    check_matrix_3x3(K1, name="K1", finite=False)
    check_matrix_3x3(K2, name="K2", finite=False)

    # Defaults
    N_full = int(x1.shape[1])
    mask_full = check_mask_bool_N(mask, N_full, name="mask")
    stats = {"N_full": N_full}
    aux = {
        "idx_valid_full": np.array([], dtype=int),
        "X_valid": np.zeros((3, 0), dtype=float),
    }
    t = np.asarray(t, dtype=float).reshape(3)

    # Apply mask
    if mask_full is not None:
        x1 = x1[:, mask_full]
        x2 = x2[:, mask_full]
        idx_mask_full = np.flatnonzero(mask_full)
    else:
        idx_mask_full = np.arange(N_full, dtype=int)

    # Minimum correspondences
    N_mask = int(x1.shape[1])
    stats.update({"N_mask": N_mask})
    if N_mask < int(min_points):
        stats.update({"reason": "too_few_correspondences"})
        return True, stats, aux

    # Projection matrices
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, t.reshape(3, 1)))

    # Triangulate and compute depths
    X = triangulate_points(P1, P2, x1, x2)
    z1, z2 = depths_two_view(R, t, X)

    # Baseline
    B = float(np.linalg.norm(t))
    stats.update({"B": B})
    if B < float(eps):
        stats.update({"reason": "baseline_too_small"})
        return True, stats, aux

    # Cheirality/depth validity
    min_depths = np.minimum(z1, z2)
    cheirality_ratio = float(((z1 > eps) & (z2 > eps)).mean())
    stats.update({"cheirality_ratio": cheirality_ratio})

    valid = (
        np.isfinite(min_depths)
        & (z1 > eps)
        & (z2 > eps)
        & (min_depths <= float(depth_max_ratio) * B)
    )
    n_valid_depth = int(valid.sum())
    depth_sanity_ratio = float(valid.mean())
    stats.update({"n_valid_depth": n_valid_depth, "depth_sanity_ratio": depth_sanity_ratio})

    # Map valid indices back to full correspondence index
    idx_valid_full = idx_mask_full[valid]
    aux.update({"idx_valid_full": idx_valid_full})

    # Degeneracy checks
    if cheirality_ratio < float(cheirality_min):
        stats.update({"reason": "cheirality_ratio_too_low"})
        return True, stats, aux
    if n_valid_depth == 0:
        stats.update({"reason": "too_few_positive_and_finite_depths"})
        return True, stats, aux
    if depth_sanity_ratio < float(depth_sanity_min):
        stats.update({"reason": "depth_sanity_ratio_too_low"})
        return True, stats, aux

    aux.update({"X_valid": X[:, valid]})
    return False, stats, aux
