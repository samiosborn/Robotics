# src/slam/bootstrap.py
import numpy as np

from core.checks import check_2xN_pair, check_matrix_3x3
from geometry.degeneracy import depth_degeneracy_two_view, parallax_degeneracy_deg, planar_degeneracy_from_masks
from slam.seed import build_two_view_seed
from slam.two_view_consensus import estimate_fundamental_consensus, estimate_homography_consensus, recover_pose_from_fundamental_consensus, select_two_view_mask


# Validate two-view bootstrap
def validate_two_view_bootstrap(K1, K2, x1, x2, cfg):
    # Check input dims
    check_2xN_pair(x1, x2)
    xy1 = np.asarray(x1.T, dtype=np.float64)
    xy2 = np.asarray(x2.T, dtype=np.float64)
    check_matrix_3x3(K1, name="K1", finite=False)
    check_matrix_3x3(K2, name="K2", finite=False)


    # Default
    ok = True
    N = int(x1.shape[1])
    stats = {"N": N}

    # --- UNPACK ---
    r = cfg["ransac"]
    b = cfg["bootstrap"]

    # RANSAC blocks
    rF = r["F"]
    rH = r["H"]

    # Degeneracy-check blocks
    plan = b["planar"]
    par = b["parallax"]
    dep = b["depth"]

    # Scalars
    eps = float(b["eps"])

    # Fundamental acceptance
    F_min_inlier_ratio = float(rF["min_inlier_ratio"])

    # Homography acceptance
    H_min_inliers = int(rH["min_inliers"])

    # Planar check
    gamma = float(plan["gamma"])
    min_H_inliers = int(plan["min_H_inliers"])
    require_H_success = bool(plan["require_H_success"])
    min_F_inliers_for_test = int(plan["min_F_inliers_for_test"])

    # Parallax check
    min_parallax_p50_deg = float(par["min_p50_deg"])
    min_parallax_p25_deg = float(par["min_p25_deg"])
    min_parallax_points = int(par["min_points"])
    parallax_quartile_trim = tuple(map(float, par["quartile_trim"]))

    # Depth check
    depth_min_points = int(dep["min_points"])
    cheirality_min = float(dep["cheirality_min"])
    depth_max_ratio = float(dep["depth_max_ratio"])
    depth_sanity_min = float(dep["depth_sanity_min"])
    translation_norm = float(dep["translation_norm"])
    baseline_override = dep["baseline_override"]


    # --- FUNDAMENTAL CONSENSUS ---
    F_best, F_mask, F_stats = estimate_fundamental_consensus(xy1, xy2, cfg)
    aux = {"F_best": F_best, "F_mask": F_mask}

    stats.update(
        {
            "n0": int(F_stats.get("n0", 0)),
            "n1": int(F_stats.get("n1", F_stats.get("n0", 0))),
            "shrink_ratio": F_stats.get("shrink_ratio", None),
        }
    )

    # Reject if F consensus failed or refit rejected
    if F_best is None or F_mask is None or not bool(F_stats.get("refit", False)):
        ok = False
        stats.update({"reason": F_stats.get("reason", "fundamental_consensus_failed")})
        return ok, stats, aux

    # Too few inliers
    nF = int(np.asarray(F_mask, dtype=bool).sum())
    if nF < min_F_inliers_for_test or (nF / max(N, 1)) < F_min_inlier_ratio:
        ok = False
        stats.update({"nF": nF, "reason": "fundamental_insufficient_inliers"})
        return ok, stats, aux

    stats.update({"nF": nF})

    # --- HOMOGRAPHY CONSENSUS ---
    H_best, H_mask, H_stats = estimate_homography_consensus(xy1, xy2, cfg)
    aux.update({"H_best": H_best, "H_mask": H_mask})

    # Failure to find H
    if H_best is None or H_mask is None:
        if require_H_success:
            ok = False
            stats.update({"reason": H_stats.get("reason", "homography_failed")})
            return ok, stats, aux

        # Skip planar check
        stats.update({"nH": 0, "H_inlier_ratio": 0.0, "H_over_F_ratio": None})
    else:
        # H inliers too small
        nH = int(np.asarray(H_mask, dtype=bool).sum())
        stats.update({"nH": nH, "H_inlier_ratio": float(nH / max(N, 1))})
        if nH < H_min_inliers:
            ok = False
            stats.update({"reason": "insufficient_homography_inliers"})
            return ok, stats, aux

        # Planar degeneracy check
        planar_degenerate, planar_stats = planar_degeneracy_from_masks(
            mask_F=F_mask,
            mask_H=H_mask,
            gamma=gamma,
            min_H_inliers=min_H_inliers,
        )

        # Update stats
        stats.update({"H_over_F_ratio": planar_stats["ratio"]})

        # Confirmed planar degeneracy
        if planar_degenerate:
            ok = False
            stats.update({"reason": "planar_degenerate"})
            return ok, stats, aux

    # --- POSE RECOVERY ---
    R, t, E, cheir_ratio, cheir_mask, pose_stats = recover_pose_from_fundamental_consensus(
        F_best,
        F_mask,
        K1,
        K2,
        xy1,
        xy2,
    )

    if R is None or t is None or E is None or cheir_mask is None:
        ok = False
        stats.update({"reason": pose_stats.get("reason", "pose_recovery_failed")})
        if pose_stats.get("error") is not None:
            stats.update({"pose_error": pose_stats.get("error")})
        return ok, stats, aux

    t = np.asarray(t, dtype=np.float64).reshape(3)
    aux.update({"R": R, "t": t, "E": E, "cheir_mask": cheir_mask})
    nC = int(np.asarray(cheir_mask, dtype=bool).sum())
    stats.update({"cheir_ratio": float(cheir_ratio), "nC": nC})

    # Mask selection policy
    mask = select_two_view_mask(F_mask, cheir_mask, cfg)
    aux.update({"mask": mask})
    stats.update({"n_mask": int(mask.sum())})

    # Translation magnitude
    tn = float(np.linalg.norm(t))
    if tn < eps:
        ok = False
        stats.update({"reason": "pose_translation_near_zero"})
        return ok, stats, aux

    # Baseline scaling
    if baseline_override is not None:
        t = t * (float(baseline_override) / tn)
    else:
        t = t * (translation_norm / tn)
    aux.update({"t": t})

    # --- PARALLAX DEGENERACY ---
    parallax_degen, parallax_stats = parallax_degeneracy_deg(
        R,
        K1,
        K2,
        x1,
        x2,
        mask=mask,
        quartile_trim=parallax_quartile_trim,
        min_points=min_parallax_points,
        min_p50_deg=min_parallax_p50_deg,
        min_p25_deg=min_parallax_p25_deg,
    )

    stats.update(
        {
            "parallax_p25_deg": parallax_stats["p25"],
            "parallax_p50_deg": parallax_stats["p50"],
            "parallax_p75_deg": parallax_stats["p75"],
            "parallax_n_trim": parallax_stats["n_trim"],
        }
    )

    if parallax_degen:
        ok = False
        stats.update({"reason": parallax_stats["reason"]})
        return ok, stats, aux

    # --- DEPTH DEGENERACY ---
    depth_degen, depth_stats, depth_aux = depth_degeneracy_two_view(
        R,
        t,
        K1,
        K2,
        x1,
        x2,
        mask=mask,
        min_points=depth_min_points,
        cheirality_min=cheirality_min,
        depth_max_ratio=depth_max_ratio,
        depth_sanity_min=depth_sanity_min,
        eps=eps,
    )

    stats.update({f"depth_{k}": v for k, v in depth_stats.items() if k != "reason"})
    aux.update({"depth_idx_valid_full": depth_aux["idx_valid_full"], "X_valid": depth_aux["X_valid"]})

    if depth_degen:
        ok = False
        stats.update({"reason": depth_stats["reason"]})
        return ok, stats, aux

    return ok, stats, aux


# Bootstrap two-view
def bootstrap_two_view(K1, K2, x1, x2, cfg):
    ok, stats, aux = validate_two_view_bootstrap(K1, K2, x1, x2, cfg)
    if not ok:
        return False, stats, aux, None

    seed = build_two_view_seed(
        x1,
        x2,
        idx_init=aux["depth_idx_valid_full"],
        X_valid=aux["X_valid"],
        R1=aux["R"],
        t1=aux["t"],
    )
    return True, stats, aux, seed
