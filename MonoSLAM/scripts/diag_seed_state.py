# scripts/diag_seed_state.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import (
    ROOT,
    add_pnp_threshold_stability_args as _add_pnp_threshold_stability_args,
    append_jsonl as _append_jsonl,
    apply_pnp_threshold_stability_cli_overrides as _apply_pnp_threshold_stability_cli_overrides,
    frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg,
    load_runtime_cfg as _load_runtime_cfg,
)
from core.checks import check_dir, check_int_ge0
from datasets.eth3d import load_eth3d_sequence
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames


def _dump_seed_structure(seed: dict, frame_idx: int, out_path: Path) -> None:
    """Dump compact seed structure diagnostic."""
    out_data = {
        "frame": int(frame_idx),
        "n_landmarks": len(seed.get("landmarks", [])),
        "landmark_id_by_feat1_size": len(seed.get("landmark_id_by_feat1", [])),
        "landmark_id_by_feat1_mapped": int(np.sum(np.asarray(seed.get("landmark_id_by_feat1", []), dtype=np.int64) >= 0)),
        "keyframe_kf": seed.get("keyframe_kf", None),
        "n_feats1": seed.get("feats1").kps_xy.shape[0] if seed.get("feats1") is not None else 0,
    }
    
    # Analyze landmarks
    landmarks = seed.get("landmarks", [])
    birth_sources = {}
    n_obs_histogram = {}
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        
        bs = lm.get("birth_source", "unknown")
        birth_sources[bs] = birth_sources.get(bs, 0) + 1
        
        n_obs = len([o for o in lm.get("obs", []) if isinstance(o, dict)])
        n_obs_str = str(n_obs)
        n_obs_histogram[n_obs_str] = n_obs_histogram.get(n_obs_str, 0) + 1
    
    out_data["birth_sources"] = birth_sources
    out_data["n_obs_histogram"] = n_obs_histogram
    
    # Check for alignment issues
    alignment_issues = []
    landmark_ids_in_lm_list = set()
    for i, lm in enumerate(landmarks):
        if isinstance(lm, dict) and "id" in lm:
            lm_id = int(lm["id"])
            landmark_ids_in_lm_list.add(lm_id)
            
            # Check if this landmark has observations
            obs = lm.get("obs", [])
            if len(obs) == 0:
                alignment_issues.append(f"Landmark {lm_id} has no observations")
    
    # Check landmark_id_by_feat1 for orphaned references
    landmark_id_by_feat1 = np.asarray(seed.get("landmark_id_by_feat1", []), dtype=np.int64)
    for feat_idx, lm_id in enumerate(landmark_id_by_feat1):
        if lm_id >= 0 and lm_id not in landmark_ids_in_lm_list:
            alignment_issues.append(
                f"Feature {feat_idx} references non-existent landmark {lm_id}"
            )
    
    out_data["alignment_issues"] = alignment_issues
    
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)


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
        out_dir = (ROOT / str(run_cfg.get("out_dir", "out")) / "diag_seed_state").resolve()

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

    print(f"Bootstrap OK: {len(seed['landmarks'])} landmarks")
    _dump_seed_structure(seed, -1, out_dir / "seed_after_bootstrap.json")

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
    keyframe_2_index = 2

    print(f"Frame 2 OK: {out_frame2['ok']}, landmarks now {len(seed_after_frame2['landmarks'])}, keyframe promoted: {out_frame2['stats']['keyframe_promoted']}")
    _dump_seed_structure(seed_after_frame2, 2, out_dir / "seed_after_frame2.json")

    # Detailed analysis of landmark_id_by_feat1 after frame 2 processing
    print("\n=== Landmark ID by Feature Lookup Analysis ===")
    
    landmark_id_by_feat1_old = np.asarray(seed.get("landmark_id_by_feat1", []), dtype=np.int64)
    landmark_id_by_feat1_new = np.asarray(seed_after_frame2.get("landmark_id_by_feat1", []), dtype=np.int64)
    
    print(f"Old lookup size (KF1): {landmark_id_by_feat1_old.size}")
    print(f"New lookup size (KF2): {landmark_id_by_feat1_new.size}")
    print(f"Old mapping coverage: {int((landmark_id_by_feat1_old >= 0).sum())} / {landmark_id_by_feat1_old.size}")
    print(f"New mapping coverage: {int((landmark_id_by_feat1_new >= 0).sum())} / {landmark_id_by_feat1_new.size}")
    
    # Find which landmarks have observations in frame 2
    frame2_lm_ids = set()
    for lm in seed_after_frame2["landmarks"]:
        if isinstance(lm, dict):
            obs = lm.get("obs", [])
            for ob in obs:
                if isinstance(ob, dict) and ob.get("kf") == 2:
                    frame2_lm_ids.add(int(lm["id"]))
                    break
    
    print(f"\nLandmarks with observations in frame 2: {len(frame2_lm_ids)}")
    
    # Check if those landmarks are in the lookup
    found_in_lookup = 0
    for feat_idx, lm_id in enumerate(landmark_id_by_feat1_new):
        if lm_id in frame2_lm_ids:
            found_in_lookup += 1
    
    print(f"Of those, found in landmark_id_by_feat1: {found_in_lookup}")
    
    # Detailed mapping diagnostic
    print("\nDetailed landmark_id_by_feat1 mapping (first 50 entries):")
    for feat_idx in range(min(50, landmark_id_by_feat1_new.size)):
        lm_id = int(landmark_id_by_feat1_new[feat_idx])
        status = "mapped" if lm_id >= 0 else "unmapped"
        in_frame2 = lm_id in frame2_lm_ids
        print(f"  Feature {feat_idx}: {status} (lm_id={lm_id}, in_frame2={in_frame2})")


if __name__ == "__main__":
    main()
