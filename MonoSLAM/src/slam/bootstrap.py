# slam/bootstrap.py
import numpy as np

from core.checks import check_matrix_3x3
from core.checks import check_2xN_pair
from geometry.degeneracy import depth_degeneracy_two_view, parallax_degeneracy_deg, planar_degeneracy_from_masks
from geometry.fundamental import estimate_fundamental_ransac, refit_fundamental_on_inliers
from geometry.homography import estimate_homography_ransac
from geometry.pose import pose_from_fundamental
from slam.seed import build_two_view_seed


# Planar check compatibility wrapper
def planar_check(mask_F, mask_H, gamma=1.2, min_H_inliers=20):
    return planar_degeneracy_from_masks(
        mask_F=mask_F,
        mask_H=mask_H,
        gamma=float(gamma),
        min_H_inliers=int(min_H_inliers),
    )


# Validate two-view bootstrap
def validate_two_view_bootstrap(K1, K2, x1, x2, cfg):
    # Check input dims
    check_2xN_pair(x1, x2)
    check_matrix_3x3(K1, name="K1", finite=False)
    check_matrix_3x3(K2, name="K2", finite=False)

    # Default
    ok = True
    N = x1.shape[1]
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
    seed = int(r["seed"])
    eps = float(b["eps"])

    # Policy
    mask_policy = b["mask_policy"]

    # Fundamental RANSAC
    F_num_trials = int(rF["num_trials"])
    F_sample_size = int(rF["sample_size"])
    F_thr_px = float(rF["threshold_px"])
    F_min_inliers = int(rF["min_inliers"])
    F_min_inlier_ratio = float(rF["min_inlier_ratio"])
    F_shrink_guard = float(rF["shrink_guard"])
    F_recompute = bool(rF["recompute_mask"])

    # Homography RANSAC
    H_num_trials = int(rH["num_trials"])
    H_thr_px = float(rH["threshold_px"])
    H_min_inliers = int(rH["min_inliers"])
    H_normalise = bool(rH["normalise"])

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
    F_best, F_mask = estimate_fundamental_ransac(
        x1,
        x2,
        num_trials=F_num_trials,
        sample_size=F_sample_size,
        threshold=F_thr_px,
        seed=seed,
    )

    F_best, F_mask, F_refit_stats = refit_fundamental_on_inliers(
        x1,
        x2,
        F=F_best,
        inlier_mask=F_mask,
        min_inliers=F_min_inliers,
        threshold=F_thr_px,
        shrink_guard=F_shrink_guard,
        recompute_mask=F_recompute,
    )

    aux = {"F_best": F_best, "F_mask": F_mask}

    # Update stats with F mask
    stats.update({"n0": F_refit_stats["n0"]})

    # Reject if refit failed
    if F_refit_stats["refit"] is False:
        ok = False
        stats.update({"reason": F_refit_stats["reason"]})
        return ok, stats, aux

    # Update stats with refit mask
    stats.update({"n1": F_refit_stats["n1"], "shrink_ratio": F_refit_stats["shrink_ratio"]})

    # Too few inliers
    nF = int(F_mask.sum())
    if nF < min_F_inliers_for_test or (nF / N) < F_min_inlier_ratio:
        ok = False
        stats.update({"nF": nF, "reason": "fundamental_insufficient_inliers"})
        return ok, stats, aux

    # --- HOMOGRAPHY CONSENSUS ---
    H_best, H_mask, H_reason = estimate_homography_ransac(
        x1,
        x2,
        num_trials=H_num_trials,
        threshold=H_thr_px,
        normalise=H_normalise,
        seed=seed,
    )

    # Update aux
    aux.update({"H_best": H_best, "H_mask": H_mask})

    # Failure to find H
    if H_best is None or H_mask is None:
        if require_H_success:
            ok = False
            stats.update({"reason": H_reason})
            return ok, stats, aux

        # Skip planar check
        stats.update({"nH": 0, "H_inlier_ratio": 0.0, "H_over_F_ratio": None})
    else:
        # H inliers too small
        nH = int(H_mask.sum())
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
    R, t, E, cheir_ratio, cheir_mask = pose_from_fundamental(
        F_best,
        K1,
        K2,
        x1,
        x2,
        F_mask=F_mask,
        enforce_constraints=True,
    )
    t = np.asarray(t, dtype=float).reshape(3)

    # Update stats and aux
    aux.update({"R": R, "t": t, "E": E, "cheir_mask": cheir_mask})
    nC = int(cheir_mask.sum())
    stats.update({"cheir_ratio": float(cheir_ratio), "nC": nC})

    # Mask selection policy
    if mask_policy == "F":
        mask = F_mask
    elif mask_policy == "cheirality":
        mask = cheir_mask
    elif mask_policy == "intersection":
        mask = F_mask & cheir_mask
    else:
        raise ValueError(f"Unknown mask_policy: {mask_policy}")

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
