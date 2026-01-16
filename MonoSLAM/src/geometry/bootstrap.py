# geometry/bootstrap.py
import numpy as np
from geometry.checks import check_2xN_pair, check_3x3, check_bool_N
from geometry.parallax import median_parallax_angle_deg
from geometry.triangulation import triangulate_points, depths_two_view
from geometry.fundamental import estimate_fundamental_ransac, refit_fundamental_on_inliers
from geometry.homography import estimate_homography_ransac

# Planar check
def planar_check(mask_F, mask_H, gamma=1.2, min_H_inliers=20): 
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

# Parallax check
def parallax_check(R, K1, K2, x1, x2, mask=None, min_median_deg=1.0):
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(K1)
    check_3x3(K2)
    mask = check_bool_N(mask, x1.shape[1])
    # Default 
    degenerate = False
    # Too small mask
    if mask is not None and int(mask.sum()) < 8:
        degenerate = True
        stats = dict(angle=None, reason="mask_too_small")
        return degenerate, stats
    # Median parallax angle (degrees)
    angle = median_parallax_angle_deg(R, K1, K2, x1, x2, mask)
    # Parallax
    if angle < min_median_deg: 
        degenerate = True
    # Statistics
    stats = dict(angle=float(angle))
    return degenerate, stats

# Depth check 
def depth_check(R, t, K1, K2, x1, x2, mask=None, min_points=20, cheirality_min=0.7, depth_max_ratio=100.0, depth_sanity_min=0.7, eps=1e-9): 
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(K1)
    check_3x3(K2)
    mask = check_bool_N(mask, x1.shape[1])
    # Default
    degenerate = False
    # Apply mask
    if mask is not None: 
        x1 = x1[:, mask]
        x2 = x2[:, mask]
    # Minimum number of points
    n = int(x1.shape[1])
    if n < min_points: 
        degenerate = True
        stats = dict(n=n, reason="too_few_correspondences")
        return degenerate, stats
    # Build P
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K2 @ np.hstack((R, t.reshape((3,1))))
    # Triangulate points
    X = triangulate_points(P1, P2, x1, x2)
    # Depths
    z1, z2 = depths_two_view(R, t, X)
    # Cheirality mask
    cheirality_mask = (z1 > eps) & (z2 > eps)
    n_cheirality = int(cheirality_mask.sum())
    # Cheirality ratio
    cheirality_ratio = float(cheirality_mask.mean())
    if cheirality_ratio < cheirality_min: 
        degenerate = True
        stats = dict(n=n, n_cheirality=n_cheirality, cheirality_ratio=cheirality_ratio, reason="cheirality_ratio_too_low")
        return degenerate, stats
    # Baseline length
    B = float(np.linalg.norm(t))
    if B < eps: 
        degenerate = True
        stats = dict(n=n, n_cheirality=n_cheirality, cheirality_ratio=cheirality_ratio, B=B, reason="baseline_too_small")
        return degenerate, stats
    # Minimum depth of corresponding points
    min_depths = np.minimum(z1, z2)[cheirality_mask]
    # Finite depth
    finite = np.isfinite(min_depths)
    min_depths = min_depths[finite]
    # Positive depth 
    positive = min_depths > eps
    min_depths = min_depths[positive]
    # Valid depths
    n_depth_valid = int(min_depths.size)
    if n_depth_valid == 0: 
        degenerate = True
        stats = dict(n=n, n_cheirality=n_cheirality, n_depth_valid=n_depth_valid, cheirality_ratio=cheirality_ratio, B=B, reason="too_few_positive_and_finite_depths")
        return degenerate, stats
    # Depth within tolerance
    depth_mask = min_depths <= depth_max_ratio * B
    # Depth sanity ratio
    depth_sanity_ratio = float(depth_mask.mean())
    if depth_sanity_ratio < depth_sanity_min: 
        degenerate = True
        stats = dict(n=n, n_cheirality=n_cheirality, n_depth_valid=n_depth_valid, cheirality_ratio=cheirality_ratio, B=B, depth_sanity_ratio=depth_sanity_ratio, reason="depth_sanity_ratio_too_low")
        return degenerate, stats
    # Statistics
    stats = dict(n=n, n_cheirality=n_cheirality, n_depth_valid=n_depth_valid, cheirality_ratio=cheirality_ratio, B=B, depth_sanity_ratio=depth_sanity_ratio)
    return degenerate, stats

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
    min_median_deg = float(par["min_median_deg"])
    # Depth check
    min_points = int(dep["min_points"])
    cheirality_min = float(dep["cheirality_min"])
    depth_max_ratio = float(dep["depth_max_ratio"])
    depth_sanity_min = float(dep["depth_sanity_min"])
    translation_norm = float(dep["translation_norm"])
    baseline_override = dep["baseline_override"]

    # --- FUNDAMENTAL CONSENSUS ---
    # Estimate fundamental
    F_best, F_mask = estimate_fundamental_ransac(
    x1, x2,
    num_trials=F_num_trials,
    sample_size=F_sample_size,
    threshold=F_thr_px,
    seed=seed,
    )
    # Refit fundamental
    F_best, F_mask, F_refit_stats = refit_fundamental_on_inliers(
    x1, x2,
    F=F_best,
    inlier_mask=F_mask,
    min_inliers=F_min_inliers,
    threshold=F_thr_px,
    shrink_guard=F_shrink_guard,
    recompute_mask=bool(F_recompute),
    )
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
    H_best, H_mask, H_reason = estimate_homography_ransac(
    x1, x2,
    num_trials=H_num_trials,
    threshold=H_thr_px,
    normalise=H_normalise,
    seed=seed,
    )
    # Failure to find H
    if H_best is None: 
        if require_H_success: 
            ok = False
            stats.update({"reason": H_reason})
            return ok, stats, aux
        else: 
            # Skip planar check
            H_mask = None
            nH = 0
    else: 
        # Homography worked
        aux.update({"H_best": H_best, "H_mask": H_mask})
        # H inliers too small
        nH = H_mask.sum()
        stats.update({"nH": nH})
        if nH < H_min_inliers: 
            ok = False
            stats.update({"reason": "insufficient_homography_inliers"})
            return ok, stats, aux
        # Test for planar degen
        planar_degenerate, planar_stats =  planar_check(
        mask_F=F_mask,
        mask_H=H_mask, 
        gamma=gamma, 
        min_H_inliers=min_H_inliers,
        )
        # Update stats
        stats.update({"H_ratio": planar_stats["ratio"]})
        # Planar degenerate
        if planar_degenerate: 
            ok = False
            stats.update({"reason": "planar_degenerate"})
            return ok, stats, aux
    # 


    return ok, stats, aux

