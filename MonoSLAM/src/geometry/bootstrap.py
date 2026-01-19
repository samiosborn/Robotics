# geometry/bootstrap.py
import numpy as np
from geometry.checks import check_2xN_pair, check_3x3, check_bool_N
from geometry.parallax import parallax_angle_stats_deg
from geometry.triangulation import triangulate_points, depths_two_view
from geometry.fundamental import estimate_fundamental_ransac, refit_fundamental_on_inliers
from geometry.homography import estimate_homography_ransac
from geometry.pose import pose_from_fundamental

# Planar check
def planar_check(mask_F, mask_H, gamma=1.2, min_H_inliers=20): 
    # Early return
    if mask_H is None or mask_F is None:
        return False, {"nF": int(np.sum(mask_F)) if mask_F is not None else 0, "nH": 0, "ratio": 0.0}
    # Default
    degenerate = False
    nH = np.sum(mask_H)
    nF = np.sum(mask_F)
    # Too many homography RANSAC inliers
    if nH >= gamma * nF and nH >= min_H_inliers: 
        degenerate = True
    # Statistics
    stats = dict(nF=int(nF), nH=int(nH), ratio=float(nH/max(nF, 1)))
    return degenerate, stats

# Parallax check (deg)
def parallax_check_deg(R, K1, K2, x1, x2, mask=None, quartile_trim=(0.1, 0.9), min_points=8, min_p50_deg=1.0, min_p25_deg=0.7):
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(K1)
    check_3x3(K2)
    check_3x3(R)
    N = x1.shape[1]
    mask = check_bool_N(mask, N)
    n_mask = int(mask.sum()) if mask is not None else int(N)
    # Default 
    degenerate = False
    stats = {"n_mask": n_mask, "n_trim": 0, "p25": None, "p50": None, "p75": None, "reason": None}
    # Too small mask
    if mask is not None and int(mask.sum()) < min_points:
        degenerate = True
        stats.update({"reason": "mask_too_small"})
        return degenerate, stats
    # Parallax angle stats (degrees)
    par_stats = parallax_angle_stats_deg(R, K1, K2, x1, x2, mask, quartile_trim, min_points)
    # Statistics
    stats.update({"n_trim": par_stats["n_trim"], "p25": par_stats["p25"], "p50": par_stats["p50"], "p75": par_stats["p75"]})
    # Degen
    if par_stats["reason"] is not None: 
        stats.update({"reason": par_stats["reason"]})
        degenerate = True
        return degenerate, stats
    # Parallax
    if stats["p50"] < min_p50_deg: 
        degenerate = True
        stats.update({"reason": "parallax_p50_too_small"})
        return degenerate, stats
    elif stats["p25"] < min_p25_deg:
        degenerate = True
        stats.update({"reason": "parallax_p25_too_small"})
        return degenerate, stats
    else: 
        return degenerate, stats

# Depth check 
def depth_check(R, t, K1, K2, x1, x2, mask=None, min_points=20, cheirality_min=0.7, depth_max_ratio=100.0, depth_sanity_min=0.7, eps=1e-9): 
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(K1)
    check_3x3(K2)
    check_3x3(R)
    N_full = int(x1.shape[1])
    mask = check_bool_N(mask, N_full)
    t = np.asarray(t, float).reshape(3)
    # Default
    degenerate = False
    # Apply mask
    if mask is not None: 
        x1 = x1[:, mask]
        x2 = x2[:, mask]
    # Minimum number of points
    N_mask = int(x1.shape[1])
    stats = {"N_full": N_full, "N_mask": N_mask}
    aux = {"mask": None, "X": None, "X_valid": None}
    if N_mask < min_points: 
        degenerate = True
        stats.update({"reason": "too_few_correspondences"})
        return degenerate, stats, aux
    # Build P
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K2 @ np.hstack((R, t.reshape((3,1))))
    # Triangulate points
    X = triangulate_points(P1, P2, x1, x2)
    aux["X"] = X
    # Depths
    z1, z2 = depths_two_view(R, t, X)
    # Baseline length
    B = float(np.linalg.norm(t))
    stats.update({"B": B})
    # Baseline too small
    if B < eps: 
        degenerate = True
        stats.update({"reason": "baseline_too_small"})
        return degenerate, stats, aux
    # Minimum depth of corresponding points
    min_depths = np.minimum(z1, z2)
    # Cheirality ratio
    cheirality_ratio = float(((z1 > eps) & (z2 > eps)).mean())
    stats.update({"cheirality_ratio": cheirality_ratio})
    # Valid points
    valid = (np.isfinite(min_depths) & 
            (z1 > eps) & (z2 > eps) & 
            (min_depths <= depth_max_ratio * B))
    n_valid_depth = int(valid.sum())
    stats.update({"n_valid_depth": n_valid_depth})
    # Depth sanity ratio
    depth_sanity_ratio = float(valid.mean())
    stats.update({"depth_sanity_ratio": depth_sanity_ratio})
    # Cheirality ratio too low    
    if cheirality_ratio < cheirality_min: 
        degenerate = True
        stats.update({"reason": "cheirality_ratio_too_low"})
        return degenerate, stats, aux    
    # Valid depths
    if n_valid_depth == 0: 
        degenerate = True
        stats.update({"reason": "too_few_positive_and_finite_depths"})
        return degenerate, stats, aux
    # Depth sanity ratio too low
    if depth_sanity_ratio < depth_sanity_min: 
        degenerate = True
        stats.update({"reason": "depth_sanity_ratio_too_low"})
        return degenerate, stats, aux
    else: 
        # Lift to full mask
        full_mask = np.zeros(N_full, dtype=bool)
        if mask is None:
            full_mask[:] = valid
        else:
            full_mask[mask] = valid
        aux["mask"] = full_mask
        # Include X (valid)
        X_valid = X[:, valid]
        aux["X_valid"] = X_valid

        return degenerate, stats, aux

# Validate two-view bootstrap
def validate_two_view_bootstrap(K1, K2, x1, x2, cfg): 
    # Check input dims
    check_2xN_pair(x1, x2)
    check_3x3(K1)
    check_3x3(K2)
    
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
    # Estimate fundamental
    F_best, F_mask = estimate_fundamental_ransac(x1, x2,
    num_trials=F_num_trials,
    sample_size=F_sample_size,
    threshold=F_thr_px,
    seed=seed)
    # Refit fundamental
    F_best, F_mask, F_refit_stats = refit_fundamental_on_inliers(x1, x2,
    F=F_best,
    inlier_mask=F_mask,
    min_inliers=F_min_inliers,
    threshold=F_thr_px,
    shrink_guard=F_shrink_guard,
    recompute_mask=bool(F_recompute))
    aux = {"F_best": F_best, "F_mask": F_mask}
    # Update stats with F mask
    stats.update({"n0":F_refit_stats["n0"]})
    # Reject if refit failed
    if F_refit_stats["refit"] == False: 
        ok = False
        stats.update({"reason": F_refit_stats["reason"]})
        return ok, stats, aux
    # Update stats with refit mask
    stats.update({"n1": F_refit_stats["n1"], "shrink_ratio": F_refit_stats["shrink_ratio"]})
    # Too few inliers
    nF = F_mask.sum() 
    if nF < min_F_inliers_for_test or (nF / N) < F_min_inlier_ratio: 
        ok = False
        stats.update({"nF": nF, "reason": "fundamental_insufficient_inliers"})
        return ok, stats, aux

    # --- HOMOGRAPHY CONSENSUS ---
    # Estimate homography
    H_best, H_mask, H_reason = estimate_homography_ransac(x1, x2,
    num_trials=H_num_trials,
    threshold=H_thr_px,
    normalise=H_normalise,
    seed=seed)
    # Update aux
    aux.update({"H_best": H_best, "H_mask": H_mask})
    # Failure to find H
    if H_best is None: 
        if require_H_success: 
            ok = False
            stats.update({"reason": H_reason})
            return ok, stats, aux
        else: 
            # Skip planar check
            stats.update({"nH": 0, "H_inlier_ratio": 0.0, "H_over_F_ratio": None})
    else: 
        # H inliers too small
        nH = H_mask.sum()
        stats.update({"nH": int(nH), "H_inlier_ratio": float(nH / max(N, 1))})
        if nH < H_min_inliers: 
            ok = False
            stats.update({"reason": "insufficient_homography_inliers"})
            return ok, stats, aux
        # Planar degenerate check
        planar_degenerate, planar_stats =  planar_check(
        mask_F=F_mask,
        mask_H=H_mask, 
        gamma=gamma, 
        min_H_inliers=min_H_inliers)
        # Update stats
        stats.update({"H_over_F_ratio": planar_stats["ratio"]})
        # Confirmed planar degenerate
        if planar_degenerate: 
            ok = False
            stats.update({"reason": "planar_degenerate"})
            return ok, stats, aux
    
    # --- POSE RECOVERY ---
    # Recover pose from F
    R, t, E, cheir_ratio, cheir_mask = pose_from_fundamental(
    F_best, K1, K2, x1, x2,
    F_mask=F_mask,
    enforce_constraints=True)
    t = np.asarray(t, float).reshape(3)
    # Update stats and aux
    aux.update({"R": R, "t": t, "E": E, "cheir_mask": cheir_mask})
    nC = cheir_mask.sum()
    stats.update({"cheir_ratio": float(cheir_ratio), "nC": int(nC)})
    # Mask selection
    if mask_policy == "F":
        mask = F_mask
    elif mask_policy == "cheirality":
        mask = cheir_mask
    elif mask_policy == "intersection":
        mask = F_mask & cheir_mask
    else:
        raise ValueError(f"Unknown mask_policy: {mask_policy}")
    # Update aux and stats
    aux.update({"mask": mask})
    stats.update({"n_mask": int(mask.sum())})
    # Translation distance
    tn = float(np.linalg.norm(t))
    # Near-zero translation
    if tn < eps: 
        ok = False
        stats.update({"reason": "pose_translation_near_zero"})
        return ok, stats, aux
    # Baseline override
    if baseline_override is not None: 
        t = t * (float(baseline_override) / tn)
    else: 
        t = t * (translation_norm / tn)
    aux.update({"t": t})

    # --- PARALLAX DEGENERATE ---
    # Parallax check (deg)
    parallax_degen, parallax_stats_deg = parallax_check_deg(R, K1, K2, x1, x2, mask=mask, quartile_trim=parallax_quartile_trim, min_points=min_parallax_points, min_p50_deg=min_parallax_p50_deg, min_p25_deg=min_parallax_p25_deg)
    # Update stats
    stats.update({"parallax_p25_deg": parallax_stats_deg["p25"], "parallax_p50_deg": parallax_stats_deg["p50"], "parallax_p75_deg": parallax_stats_deg["p75"], "parallax_n_trim": parallax_stats_deg["n_trim"]})
    # Confirmed parallax degenerate
    if parallax_degen: 
        ok = False
        stats.update({"reason": parallax_stats_deg["reason"]})
        return ok, stats, aux

    # --- DEPTH CHECK ---
    # Depth check
    
    depth_degen, depth_stats, depth_aux = depth_check(R, t, K1, K2, x1, x2,
    mask=mask,
    min_points=depth_min_points,
    cheirality_min=cheirality_min,
    depth_max_ratio=depth_max_ratio,
    depth_sanity_min=depth_sanity_min,
    eps=eps)
    # Update stats and aux
    stats.update({f"depth_{k}": v for k, v in depth_stats.items() if k!="reason"})
    aux.update({"depth_mask": depth_aux["mask"], "X": depth_aux["X"], "X_valid": depth_aux["X_valid"]})
    # Depth degenerate
    if depth_degen: 
        ok = False
        stats.update({"reason": depth_stats["reason"]})
        return ok, stats, aux
    else: 
        return ok, stats, aux

# Bootstrap two-view
def bootstrap_two_view(...)
    # Intersection of mask
        mask0 = stats["mask"] & stats["depth_mask"]
        stats.update({"mask0": mask0})
