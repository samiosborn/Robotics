# scripts/test_relax_observation_filter.py
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
from geometry.pnp import build_pnp_correspondences_with_stats, estimate_pose_pnp_ransac
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.tracking import track_against_keyframe


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

    # Now test frame 3 with different filter settings
    im3, ts3, id3 = seq.get(3)
    
    print(f"=== Testing Frame 3 with Different Observation Filters ===\n")
    
    # Track frame 3 against keyframe 2
    track_out = track_against_keyframe(
        K,
        keyframe_2_feats,
        im3,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
    )
    
    # Test different filter configurations
    filter_configs = [
        {"allow_bootstrap": True, "min_post_obs": 3, "label": "CURRENT (strict)"},
        {"allow_bootstrap": True, "min_post_obs": 2, "label": "RELAXED (2-obs allowed)"},
        {"allow_bootstrap": True, "min_post_obs": 1, "label": "VERY RELAXED (1-obs allowed)"},
    ]
    
    for config in filter_configs:
        allow_bootstrap = config["allow_bootstrap"]
        min_post_obs = config["min_post_obs"]
        label = config["label"]
        
        # Build correspondences with this config
        corrs, corr_stats = build_pnp_correspondences_with_stats(
            seed_after_frame2,
            track_out,
            min_landmark_observations=2,
            allow_bootstrap_landmarks_for_pose=allow_bootstrap,
            min_post_bootstrap_observations_for_pose=min_post_obs,
        )
        
        N = int(corrs.X_w.shape[1])
        n_bootstrap = int(corr_stats.get("n_corr_bootstrap_used", 0))
        n_post_bootstrap = int(corr_stats.get("n_corr_post_bootstrap_used", 0))
        
        print(f"Config: {label}")
        print(f"  Total correspondences: {N}")
        print(f"    Bootstrap:      {n_bootstrap}")
        print(f"    Post-bootstrap: {n_post_bootstrap}")
        
        # Try PnP with this correspondence set
        if N < 12:
            print(f"  PnP RANSAC: INSUFFICIENT (need >= 12, have {N})")
            print()
            continue
        
        R, t, inlier_mask, pose_stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=1000,
            sample_size=6,
            threshold_px=3.0,
            min_inliers=12,
            seed=0,
            refit=True,
            refine_nonlinear=True,
        )
        
        ok = (R is not None and t is not None)
        n_inliers = int(pose_stats.get("n_inliers", 0))
        reason = pose_stats.get("reason", None)
        
        print(f"  PnP RANSAC (3.0px): {('✓ SUCCESS' if ok else '✗ FAILED')}")
        print(f"    Inliers: {n_inliers} / {N} ({100*n_inliers/N:.1f}%)")
        print(f"    Reason: {reason}")
        print()


if __name__ == "__main__":
    main()
