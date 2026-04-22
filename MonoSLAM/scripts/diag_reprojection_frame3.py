# scripts/diag_reprojection_frame3.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frontend_eth3d_common import (
    ROOT,
    frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg,
    load_runtime_cfg as _load_runtime_cfg,
)
from core.checks import check_dir, check_int_ge0
from datasets.eth3d import load_eth3d_sequence
from geometry.camera import world_to_camera_points, reprojection_errors_sq
from geometry.pose import apply_left_pose_increment_wc
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.tracking import track_against_keyframe
from slam.seed import seed_keyframe_pose
from geometry.pnp import build_pnp_correspondences_with_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--i0", type=int, default=0)
    parser.add_argument("--i1", type=int, default=1)

    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)

    dataset_cfg = cfg["dataset"]

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root is not None
        else (ROOT / dataset_cfg["root"]).resolve()
    )
    seq_name = str(args.seq) if args.seq is not None else str(dataset_cfg["seq"])

    check_dir(dataset_root, name="dataset_root")

    i0 = check_int_ge0(args.i0, name="i0")
    i1 = check_int_ge0(args.i1, name="i1")

    if i1 <= i0:
        raise ValueError(f"Expected i1 > i0 for bootstrap; got i0={i0}, i1={i1}")

    # Load ETH3D sequence
    seq = load_eth3d_sequence(
        dataset_root,
        seq_name,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    max_frames = dataset_cfg.get("max_frames", None)
    n_effective = len(seq) if max_frames is None else min(len(seq), int(max_frames))

    if i0 >= n_effective or i1 >= n_effective or 3 >= n_effective:
        raise IndexError(f"Frame indices out of range for effective sequence length {n_effective}")

    # Bootstrap
    im0, ts0, id0 = seq.get(i0)
    im1, ts1, id1 = seq.get(i1)

    boot = bootstrap_from_two_frames(
        K, K, im0, im1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )

    seed = boot["seed"]
    keyframe_1_feats = seed["feats1"]

    # Process frame 2
    im2, ts2, id2 = seq.get(2)
    out_frame2 = process_frame_against_seed(
        K, seed, keyframe_1_feats, im2,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        keyframe_kf=1,
        current_kf=2,
        **frontend_kwargs["pnp_frontend_kwargs"],
    )

    seed_after_frame2 = out_frame2["seed"]
    keyframe_2_feats = out_frame2["track_out"]["cur_feats"]

    # Track frame 3
    im3, ts3, id3 = seq.get(3)
    track_out = track_against_keyframe(
        K,
        keyframe_2_feats,
        im3,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
    )
    
    # Build correspondences
    corrs, corr_stats = build_pnp_correspondences_with_stats(
        seed_after_frame2,
        track_out,
        min_landmark_observations=2,
        allow_bootstrap_landmarks_for_pose=True,
        min_post_bootstrap_observations_for_pose=2,
    )
    
    # Get the ground-truth pose from ETH3D (if available)
    gt_data = seed_after_frame2.get("last_keyframe_promotion", {})
    
    # Get frame 2 pose (the current keyframe reference)
    R_kf, t_kf = seed_keyframe_pose(seed_after_frame2)
    
    print(f"=== Frame 3 Reprojection Analysis ===\n")
    
    # Get frame 3 features
    cur_feats = track_out.get("cur_feats")
    if cur_feats is None:
        print("No current features available")
        return
    
    cur_kps = cur_feats.kps_xy  # (N, 2)
    
    # For each correspondence, compute reprojection error from frame 2 pose
    print(f"Assuming frame 3 is at an unknown pose, checking reprojection from frame 2...")
    print(f"\nReprojection error stats (from KF2 pose, projecting landmarks to image):")
    
    N = int(corrs.X_w.shape[1])
    X_w = corrs.X_w      # (3, N)
    x_cur = corrs.x_cur  # (2, N)
    cur_feat_idx = corrs.cur_feat_idx  # (N,)
    
    # Project landmarks using frame 2 pose
    X_c = world_to_camera_points(R_kf, t_kf, X_w)
    
    # Check cheirality (points behind camera)
    behind_cam = np.any(X_c[2, :] <= 0)
    n_behind = np.sum(X_c[2, :] <= 0)
    
    print(f"Points behind camera (cheirality): {n_behind} / {N}")
    
    # Reproject valid points
    X_c_valid = X_c[:, X_c[2, :] > 0]
    x_reproj = (K @ X_c_valid / X_c_valid[2, :]).astype(np.float64)[:2, :]
    
    # Get observed points for valid landmarks
    valid_mask = X_c[2, :] > 0
    x_obs = x_cur[:, valid_mask]
    
    # Compute reprojection errors
    reproj_err = np.linalg.norm(x_reproj - x_obs, axis=0)
    
    print(f"\nValid reprojection errors (points in front of camera): {int(valid_mask.sum())} points")
    print(f"  Min: {reproj_err.min():.2f} px")
    print(f"  Max: {reproj_err.max():.2f} px")
    print(f"  Mean: {reproj_err.mean():.2f} px")
    print(f"  Median: {np.median(reproj_err):.2f} px")
    print(f"  Std: {reproj_err.std():.2f} px")
    
    # Count by error threshold
    for thresh in [1.0, 2.0, 3.0, 5.0, 10.0]:
        count = np.sum(reproj_err < thresh)
        print(f"  < {thresh}px: {count} ({100*count/len(reproj_err):.1f}%)")
    
    # Check if the problem is systematic (consistent offset)
    errors_x = x_reproj[0, :] - x_obs[0, :]
    errors_y = x_reproj[1, :] - x_obs[1, :]
    
    print(f"\nReprojection error breakdown:")
    print(f"  X-error mean: {errors_x.mean():.2f} px (std {errors_x.std():.2f})")
    print(f"  Y-error mean: {errors_y.mean():.2f} px (std {errors_y.std():.2f})")
    
    # This could indicate camera motion between frames
    print(f"\nInterpretation:")
    if reproj_err.mean() > 20:
        print(f"  ⚠ Reprojection errors are EXTREMELY HIGH ({reproj_err.mean():.1f}px average)")
        print(f"  This suggests frame 3 has moved significantly from frame 2")
        print(f"  OR the landmarks have errors")
        print(f"  OR the tracking is failing")
    elif reproj_err.mean() > 10:
        print(f"  ⚠ Reprojection errors are HIGH ({reproj_err.mean():.1f}px average)")
        print(f"  The bootstrap map from frames 0-1 may not be accurate enough")
    else:
        print(f"  ✓ Reprojection errors are reasonable ({reproj_err.mean():.1f}px average)")


if __name__ == "__main__":
    main()
