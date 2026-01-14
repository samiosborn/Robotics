# geometry/bootstrap.py
import numpy as np
from geometry.checks import check_2xN_pair, check_3x3, check_bool_N
from geometry.parallax import median_parallax_angle_deg
from geometry.triangulation import triangulate_points, depths_two_view
from geometry.fundamental import estimate_fundamental_ransac
from utils.load_config import two_view_cfg

# Planar check
def planar_check(mask_F, mask_H, gamma=1.2, min_H_inliers=20): 
    # Default
    degenerate = False
    nH = np.sum(mask_H)
    nF = np.sum(mask_F)
    # Insufficient fundamental RANSAC inliers
    if nF < 8: 
        degenerate = True
    else: 
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
        stats = dict(angle=None, reason="Too small mask")
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
def depth_check(R, t, K1, K2, x1, x2, mask=None, min_points=20, cheirality_min=0.7, depth_max_ratio=100.0, depth_sanity_min=0.7): 
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(K1)
    check_3x3(K2)
    mask = check_bool_N(mask, x1.shape[1])
    # Default
    degenerate = False
    eps = 1e-9
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
        stats = dict(n=n, n_cheirality=n_cheirality, cheirality_ratio=cheirality_ratio, reason="too_low_cheirality_ratio")
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
    # Check dims
    check_2xN_pair(x1, x2)
    check_3x3(K1)
    check_3x3(K2)
    # Unpack
    seed = cfg["seed"]
    eps = cfg["eps"]
    cfg_F_ransac = cfg["ransac"]["F"]
    cfg_H_ransac = cfg["ransac"]["H"]
    cfg_planar = cfg["bootstrap"]["planar"]
    cfg_parallax = cfg["bootstrap"]["parallax"]
    cfg_depth = cfg["bootstrap"]["depth"]

    # Estimate fundamental
    F_best, F_mask, _ = estimate_fundamental_ransac(x1, x2, ...
    # Pose from fundamental
    # Estimate homography
    H_best, H_mask, _ = 
    # 

