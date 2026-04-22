# scripts/diag_pnp_corr_frame3.py
from __future__ import annotations

import argparse
import json
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
from geometry.pnp import build_pnp_correspondences_with_stats
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
    frontend_kwargs["pnp_frontend_kwargs"] = _apply_pnp_threshold_stability_cli_overrides(
        frontend_kwargs["pnp_frontend_kwargs"], args
    )

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
        out_dir = (ROOT / str(run_cfg.get("out_dir", "out")) / "diag_pnp_corr_frame3").resolve()

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

    print(f"Frame 2 processed: ok={out_frame2['ok']}, keyframe_promoted={out_frame2['stats']['keyframe_promoted']}")
    print(f"Seed state: {len(seed_after_frame2['landmarks'])} landmarks")
    print(f"New keyframe KF2 has {len(keyframe_2_feats.kps_xy)} features")
    print(f"landmark_id_by_feat1 size: {len(seed_after_frame2.get('landmark_id_by_feat1', []))}")

    # Now process frame 3 - track it first
    im3, ts3, id3 = seq.get(3)
    
    print(f"\n=== Processing Frame 3 ===")
    
    # Track frame 3 against keyframe 2
    track_out = track_against_keyframe(
        K,
        keyframe_2_feats,
        im3,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
    )
    
    track_stats = track_out.get("stats", {})
    print(f"Tracking: n_matches={track_stats.get('n_matches', 0)}, n_inliers={track_stats.get('n_inliers', 0)}")
    
    # Build PnP correspondences
    corrs, corr_stats = build_pnp_correspondences_with_stats(
        seed_after_frame2,
        track_out,
        min_landmark_observations=2,
        allow_bootstrap_landmarks_for_pose=True,
        min_post_bootstrap_observations_for_pose=3,
    )
    
    print(f"\n=== PnP Correspondence Building ===")
    print(f"n_corr_raw: {corr_stats.get('n_corr_raw', 0)}")
    print(f"n_corr_bootstrap_born: {corr_stats.get('n_corr_bootstrap_born', 0)}")
    print(f"n_corr_post_bootstrap_born: {corr_stats.get('n_corr_post_bootstrap_born', 0)}")
    print(f"n_corr_after_pose_filter: {corr_stats.get('n_corr_after_pose_filter', 0)}")
    print(f"n_corr_bootstrap_used: {corr_stats.get('n_corr_bootstrap_used', 0)}")
    print(f"n_corr_post_bootstrap_used: {corr_stats.get('n_corr_post_bootstrap_used', 0)}")
    
    N = int(corrs.X_w.shape[1])
    print(f"\nFinal correspondence count: {N}")
    
    if N > 0:
        print(f"\n=== First 10 Correspondences ===")
        for i in range(min(10, N)):
            X_w = corrs.X_w[:, i]
            x_cur = corrs.x_cur[:, i]
            lm_id = corrs.landmark_ids[i]
            kf_feat = corrs.kf_feat_idx[i]
            cur_feat = corrs.cur_feat_idx[i]
            print(f"  [{i}] lm_id={lm_id}, kf_feat={kf_feat}, cur_feat={cur_feat}, X_w={X_w}, x_cur={x_cur}")
        
        # Check for NaN/Inf
        nan_count = np.sum(~np.isfinite(corrs.X_w))
        inf_count = np.sum(np.isinf(corrs.X_w))
        print(f"\nX_w: {nan_count} NaNs, {inf_count} Infs")
        
        nan_count = np.sum(~np.isfinite(corrs.x_cur))
        inf_count = np.sum(np.isinf(corrs.x_cur))
        print(f"x_cur: {nan_count} NaNs, {inf_count} Infs")
        
        # Check for degenerate world points (at origin)
        distances = np.linalg.norm(corrs.X_w, axis=0)
        print(f"\nWorld point distances from origin: min={distances.min():.4f}, max={distances.max():.4f}")
        zero_count = np.sum(distances < 0.01)
        print(f"Points very close to origin (< 0.01): {zero_count}")
    else:
        print("NO CORRESPONDENCES BUILT!")
        print("\n=== Debugging Info ===")
        
        # Check what happened
        kf_feat_idx = np.asarray(track_out.get("kf_feat_idx", []), dtype=np.int64)
        landmark_id_by_feat1 = np.asarray(seed_after_frame2.get("landmark_id_by_feat1", []), dtype=np.int64)
        
        print(f"Tracked KF feature indices: {kf_feat_idx.size} total")
        print(f"landmark_id_by_feat1 size: {landmark_id_by_feat1.size}")
        
        # Check alignment
        valid_kf = (kf_feat_idx >= 0) & (kf_feat_idx < landmark_id_by_feat1.size)
        print(f"Valid KF feature indices: {np.sum(valid_kf)} / {kf_feat_idx.size}")
        
        if np.any(valid_kf):
            landmark_ids = landmark_id_by_feat1[kf_feat_idx[valid_kf]]
            mapped = np.sum(landmark_ids >= 0)
            print(f"Of valid KF features, mapped to landmarks: {mapped}")
            print(f"Unmapped: {np.sum(landmark_ids < 0)}")


if __name__ == "__main__":
    main()
