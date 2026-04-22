# scripts/diag_landmark_quality.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_runtime_cfg as _load_runtime_cfg
from core.checks import check_dir, check_int_ge0
from datasets.eth3d import load_eth3d_sequence
from geometry.camera import reprojection_errors_sq, world_to_camera_points
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
        out_dir = (ROOT / str(run_cfg.get("out_dir", "out")) / "diag_landmark_quality").resolve()

    check_dir(dataset_root, name="dataset_root")
    out_dir.mkdir(parents=True, exist_ok=True)

    i0 = check_int_ge0(args.i0, name="i0")
    i1 = check_int_ge0(args.i1, name="i1")

    # Load ETH3D sequence with normalized float64 images
    seq = load_eth3d_sequence(
        dataset_root,
        seq_name,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    print(f"sequence: {seq.name}")
    print(f"dataset_root: {dataset_root}")
    print(f"seq_name: {seq_name}")
    print(f"K:\n{K}")

    # Bootstrap
    im0, _ts0, _id0 = seq.get(i0)
    im1, _ts1, _id1 = seq.get(i1)

    bootstrap_out = bootstrap_from_two_frames(
        K, K, im0, im1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )

    if not bootstrap_out["ok"]:
        print(f"Bootstrap failed: {bootstrap_out['stats']}")
        return

    print(f"bootstrap ok: True")
    seed = bootstrap_out["seed"]
    print(f"initial landmarks: {len(seed['landmarks'])}")
    print(f"keyframe index: {i1}")

    # Process frame 2 to create new landmarks
    im2, _ts2, _id2 = seq.get(2)
    frame2_out = process_frame_against_seed(
        K, seed,
        seed["feats1"],
        im2,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        keyframe_kf=i1,
        current_kf=2,
        **frontend_kwargs["pnp_frontend_kwargs"],
    )

    if not frame2_out["ok"]:
        print(f"Frame 2 failed")
        return

    seed = frame2_out["seed"]
    n_bootstrap = sum(1 for lm in seed["landmarks"] if lm.get("birth_source") == "bootstrap")
    n_map_growth = sum(1 for lm in seed["landmarks"] if lm.get("birth_source") == "map_growth")
    print(f"Frame 2: ok={frame2_out['ok']} n_bootstrap={n_bootstrap} n_map_growth={n_map_growth}")

    # Now test frame 3 with and without map_growth landmarks
    im3, _ts3, _id3 = seq.get(3)

    # Get tracking results for frame 3
    keyframe_feats = frame2_out["track_out"]["cur_feats"]
    track_out = track_against_keyframe(
        K, keyframe_feats, im3,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
    )
    
    n_track_inliers = int(track_out.get("stats", {}).get("n_inliers", 0))
    print(f"Frame 3 tracking: n_matches={track_out['stats']['n_matches']} n_inliers={n_track_inliers}")
    
    # Test 1: Full map (bootstrap + map_growth)
    print("\n=== TEST 1: Full map (bootstrap + map_growth) ===")
    analyze_landmark_quality(K, seed, track_out)

    # Test 2: Frozen map (bootstrap only)
    print("\n=== TEST 2: Frozen map (bootstrap only) ===")
    seed_frozen = {
        **seed,
        "landmarks": [lm for lm in seed["landmarks"] if lm.get("birth_source") == "bootstrap"],
    }
    analyze_landmark_quality(K, seed_frozen, track_out)


def analyze_landmark_quality(K, seed, track_out):
    """Analyze reprojection errors by landmark birth source."""

    landmarks = seed.get("landmarks", [])
    if not landmarks:
        print("No landmarks in seed")
        return

    # Use the keyframe pose stored in the seed (frame 2 after processing)
    T_WC1 = seed.get("T_WC1")
    if T_WC1 is None or not isinstance(T_WC1, (tuple, list)) or len(T_WC1) != 2:
        print("No keyframe pose in seed")
        return

    R_kf = np.asarray(T_WC1[0], dtype=np.float64)
    t_kf = np.asarray(T_WC1[1], dtype=np.float64).reshape(3)

    # Get the tracked correspondences
    kf_feat_idx = np.asarray(track_out.get("kf_feat_idx", []), dtype=np.int64)
    cur_feat_idx = np.asarray(track_out.get("cur_feat_idx", []), dtype=np.int64)
    xy_cur = np.asarray(track_out.get("xy_cur", []), dtype=np.float64)

    if kf_feat_idx.size == 0:
        print("No tracked correspondences")
        return

    # Build landmark lookup
    lm_by_id = {int(lm["id"]): lm for lm in landmarks if isinstance(lm, dict) and "id" in lm}

    # Find which tracked features correspond to landmarks via the seed lookup table
    landmark_id_by_feat_kf = np.asarray(seed.get("landmark_id_by_feat1", []), dtype=np.int64)
    
    reproj_by_source = {"bootstrap": [], "map_growth": [], "unknown": []}
    
    for i in range(len(kf_feat_idx)):
        feat_kf = int(kf_feat_idx[i])
        
        # Skip if keyframe feature index is out of bounds
        if feat_kf < 0 or feat_kf >= len(landmark_id_by_feat_kf):
            continue
            
        lm_id = int(landmark_id_by_feat_kf[feat_kf])
        if lm_id < 0:
            continue
            
        lm = lm_by_id.get(lm_id)
        if lm is None:
            continue
            
        X_w = np.asarray(lm.get("X_w"), dtype=np.float64).reshape(3)
        if X_w.size != 3:
            continue
            
        x_cur = np.asarray(xy_cur[i, :2], dtype=np.float64).reshape(2)
        
        # Compute reprojection error
        X_c = world_to_camera_points(R_kf, t_kf, X_w.reshape(3, 1))
        if X_c[2, 0] <= 0:
            continue
            
        err_sq = reprojection_errors_sq(K, R_kf, t_kf, X_w.reshape(3, 1), x_cur.reshape(2, 1))
        err = float(np.sqrt(err_sq[0])) if np.isfinite(err_sq[0]) else np.inf
        
        source = lm.get("birth_source", "unknown")
        if source not in reproj_by_source:
            source = "unknown"
            
        reproj_by_source[source].append(err)
    
    # Print stats
    for source in ["bootstrap", "map_growth", "unknown"]:
        errs = reproj_by_source[source]
        if errs:
            errs = np.array(errs)
            print(f"{source:12s}: count={len(errs):3d} mean={np.mean(errs):6.2f}px p50={np.percentile(errs, 50):6.2f}px p90={np.percentile(errs, 90):6.2f}px")
        else:
            print(f"{source:12s}: count=0")


if __name__ == "__main__":
    main()
