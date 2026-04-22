# scripts/diag_pnp_ransac_frame3.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frontend_eth3d_common import (
    ROOT,
    add_pnp_threshold_stability_args as _add_pnp_threshold_stability_args,
    apply_pnp_threshold_stability_cli_overrides as _apply_pnp_threshold_stability_cli_overrides,
    frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg,
    load_runtime_cfg as _load_runtime_cfg,
)
from core.checks import check_dir, check_int_ge0
from datasets.eth3d import load_eth3d_sequence
from geometry.pnp import build_pnp_correspondences_with_stats, estimate_pose_pnp_ransac
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.tracking import track_against_keyframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--i0", type=int, default=0)
    parser.add_argument("--i1", type=int, default=1)
    _add_pnp_threshold_stability_args(parser)

    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)

    dataset_cfg = cfg["dataset"]
    run_cfg = cfg["run"]

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root is not None
        else (ROOT / dataset_cfg["root"]).resolve()
    )
    seq_name = str(args.seq) if args.seq is not None else str(dataset_cfg["seq"])

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (ROOT / str(run_cfg.get("out_dir", "out")) / "diag_pnp_ransac_frame3").resolve()

    check_dir(dataset_root, name="dataset_root")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    if not boot["ok"]:
        print("Bootstrap failed")
        return

    seed = boot["seed"]
    keyframe_1_feats = seed["feats1"]
    keyframe_1_index = i1

    # Process frame 2
    im2, ts2, id2 = seq.get(2)
    out_frame2 = process_frame_against_seed(
        K, seed, keyframe_1_feats, im2,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        keyframe_kf=keyframe_1_index,
        current_kf=2,
        **frontend_kwargs["pnp_frontend_kwargs"],
    )

    seed_after_frame2 = out_frame2["seed"]
    keyframe_2_feats = out_frame2["track_out"]["cur_feats"]

    # Now process frame 3
    im3, ts3, id3 = seq.get(3)
    
    print(f"=== Frame 3 PnP RANSAC Analysis ===")
    
    # Track frame 3 against keyframe 2
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
        min_post_bootstrap_observations_for_pose=3,
    )
    
    N = int(corrs.X_w.shape[1])
    print(f"Correspondences: {N}")
    
    if N == 0:
        print("No correspondences available!")
        return
    
    # Try PnP RANSAC with different parameters
    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
    
    print(f"\n=== PnP RANSAC with varying thresholds ===")
    for threshold_px in thresholds:
        try:
            R, t, inlier_mask, pose_stats = estimate_pose_pnp_ransac(
                corrs,
                K,
                num_trials=1000,
                sample_size=6,
                threshold_px=threshold_px,
                min_inliers=12,
                seed=0,
                refit=True,
                refine_nonlinear=True,
            )
            
            ok = (R is not None and t is not None)
            n_inliers = int(pose_stats.get("n_inliers", 0))
            reason = pose_stats.get("reason", None)
            
            status = "✓" if ok else "✗"
            print(f"  {status} threshold={threshold_px:5.1f}px: ok={ok}, n_inliers={n_inliers:3d}, reason={reason}")
            
            if ok:
                # Show details of first passing threshold
                print(f"\n    Inlier fraction: {n_inliers} / {N} = {100*n_inliers/N:.1f}%")
                if inlier_mask is not None and np.any(inlier_mask):
                    inlier_x = corrs.x_cur[:, inlier_mask]
                    print(f"    Inlier image coords: x_min={inlier_x[0].min():.1f}, x_max={inlier_x[0].max():.1f}")
                    print(f"                        y_min={inlier_x[1].min():.1f}, y_max={inlier_x[1].max():.1f}")
                    bbox_area = (inlier_x[0].max() - inlier_x[0].min()) * (inlier_x[1].max() - inlier_x[1].min())
                    img_area = 752 * 480  # ETH3D image size
                    print(f"    Inlier bounding box: {bbox_area:.0f} pixels, {100*bbox_area/img_area:.1f}% of image")
                break
        except Exception as e:
            print(f"  ✗ threshold={threshold_px:5.1f}px: ERROR - {e}")
    
    # Check spatial distribution of all correspondences
    print(f"\n=== Correspondence Spatial Distribution ===")
    x_cur = corrs.x_cur
    print(f"X range: [{x_cur[0].min():.1f}, {x_cur[0].max():.1f}]")
    print(f"Y range: [{x_cur[1].min():.1f}, {x_cur[1].max():.1f}]")
    
    # Count by quadrants
    h, w = 480, 752
    mid_x, mid_y = w / 2, h / 2
    
    q1 = np.sum((x_cur[0] >= mid_x) & (x_cur[1] >= mid_y))  # bottom-right
    q2 = np.sum((x_cur[0] < mid_x) & (x_cur[1] >= mid_y))   # bottom-left
    q3 = np.sum((x_cur[0] < mid_x) & (x_cur[1] < mid_y))    # top-left
    q4 = np.sum((x_cur[0] >= mid_x) & (x_cur[1] < mid_y))   # top-right
    
    print(f"Distribution by quadrant:")
    print(f"  Top-left:     {q3:3d}")
    print(f"  Top-right:    {q4:3d}")
    print(f"  Bottom-left:  {q2:3d}")
    print(f"  Bottom-right: {q1:3d}")
    
    # Check depth distribution
    print(f"\n=== Depth Distribution ===")
    depths = np.linalg.norm(corrs.X_w, axis=0)
    print(f"Depth range: [{depths.min():.2f}, {depths.max():.2f}] meters")
    print(f"Median depth: {np.median(depths):.2f}m")
    
    # Count by depth range
    near = np.sum(depths < 30)
    mid = np.sum((depths >= 30) & (depths < 70))
    far = np.sum(depths >= 70)
    print(f"Distribution by depth:")
    print(f"  Near (< 30m):  {near:3d}")
    print(f"  Mid (30-70m):  {mid:3d}")
    print(f"  Far (>= 70m):  {far:3d}")


if __name__ == "__main__":
    main()
