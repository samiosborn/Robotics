# scripts/diag_frame3_comparison.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from frontend_eth3d_common import ROOT, add_pnp_threshold_stability_args as _add_pnp_threshold_stability_args, append_jsonl as _append_jsonl, apply_pnp_threshold_stability_cli_overrides as _apply_pnp_threshold_stability_cli_overrides, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_pil_greyscale as _load_pil_greyscale, load_runtime_cfg as _load_runtime_cfg
from core.checks import check_dir, check_int_ge0, check_int_gt0
from datasets.eth3d import load_eth3d_sequence
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames


def main() -> None:
    parser = argparse.ArgumentParser()

    # Default ETH3D profile
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    # Optional dataset override
    parser.add_argument("--dataset_root", type=str, default=None)
    # Optional sequence override
    parser.add_argument("--seq", type=str, default=None)
    # Optional output override
    parser.add_argument("--out_dir", type=str, default=None)

    # Bootstrap source frame index
    parser.add_argument("--i0", type=int, default=0)
    # Bootstrap target frame index
    parser.add_argument("--i1", type=int, default=1)
    # Frame to test against both keyframes
    parser.add_argument("--test_frame", type=int, default=3)
    _add_pnp_threshold_stability_args(parser)

    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)
    frontend_kwargs["pnp_frontend_kwargs"] = _apply_pnp_threshold_stability_cli_overrides(frontend_kwargs["pnp_frontend_kwargs"], args)

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
        out_dir = (ROOT / str(run_cfg.get("out_dir", "out")) / "diag_frame3_comparison").resolve()

    check_dir(dataset_root, name="dataset_root")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the lightweight run log
    log_path = out_dir / "comparison_log.jsonl"

    i0 = check_int_ge0(args.i0, name="i0")
    i1 = check_int_ge0(args.i1, name="i1")
    test_frame = check_int_ge0(args.test_frame, name="test_frame")

    if i1 <= i0:
        raise ValueError(f"Expected i1 > i0 for bootstrap; got i0={i0}, i1={i1}")
    if test_frame <= i1:
        raise ValueError(f"Expected test_frame > i1; got test_frame={test_frame}, i1={i1}")

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
    if n_effective <= 0:
        raise ValueError("Loaded ETH3D sequence is empty")

    if i0 >= n_effective or i1 >= n_effective or test_frame >= n_effective:
        raise IndexError(f"Frame indices out of range for effective sequence length {n_effective}")

    # Read bootstrap images
    im0, ts0, id0 = seq.get(i0)
    im1, ts1, id1 = seq.get(i1)

    # Run two-view bootstrap
    boot = bootstrap_from_two_frames(
        K,
        K,
        im0,
        im1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )

    print(f"sequence: {seq.name}")
    print(f"dataset_root: {dataset_root}")
    print(f"seq_name: {seq_name}")
    print(f"bootstrap pair: {i0} ({id0}, t={ts0}) -> {i1} ({id1}, t={ts1})")
    print(f"bootstrap ok: {boot['ok']}")
    print(f"bootstrap stats: {boot['stats']}")

    # Write the bootstrap summary
    _append_jsonl(
        log_path,
        {
            "event": "bootstrap",
            "frame_index_0": int(i0),
            "frame_index_1": int(i1),
            "ok": bool(boot["ok"]),
            "reason": boot["stats"].get("reason", None),
            "n_landmarks": 0 if not isinstance(boot.get("seed"), dict) else int(len(boot["seed"].get("landmarks", []))),
        },
    )

    if not bool(boot["ok"]) or not isinstance(boot.get("seed"), dict):
        print("bootstrap failed; stopping")
        return

    seed = boot["seed"]
    keyframe_1_feats = seed["feats1"]
    keyframe_1_index = i1

    print(f"initial landmarks: {len(seed.get('landmarks', []))}")
    print(f"keyframe 1 index: {keyframe_1_index}")

    # Process frame 2 to get promoted keyframe 2
    im2, ts2, id2 = seq.get(2)
    out_frame2 = process_frame_against_seed(
        K,
        seed,
        keyframe_1_feats,
        im2,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        keyframe_kf=keyframe_1_index,
        current_kf=2,
        **frontend_kwargs["pnp_frontend_kwargs"],
    )

    seed_after_frame2 = out_frame2["seed"]
    keyframe_2_feats = out_frame2["track_out"]["cur_feats"]
    keyframe_2_index = 2

    print(f"after frame 2: landmarks={len(seed_after_frame2.get('landmarks', []))}")
    print(f"keyframe 2 index: {keyframe_2_index}")

    # Now test frame 3 against both keyframes
    im_test, ts_test, id_test = seq.get(test_frame)

    print(f"testing frame {test_frame} ({id_test}, t={ts_test})")

    # Test against keyframe 1 (original)
    print("Testing against keyframe 1...")
    out_vs_kf1 = process_frame_against_seed(
        K,
        seed,  # Original seed
        keyframe_1_feats,
        im_test,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        keyframe_kf=keyframe_1_index,
        current_kf=test_frame,
        **frontend_kwargs["pnp_frontend_kwargs"],
    )

    # Test against keyframe 2 (promoted)
    print("Testing against keyframe 2...")
    out_vs_kf2 = process_frame_against_seed(
        K,
        seed_after_frame2,  # Seed after frame 2 promotion
        keyframe_2_feats,
        im_test,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        keyframe_kf=keyframe_2_index,
        current_kf=test_frame,
        **frontend_kwargs["pnp_frontend_kwargs"],
    )

    # Extract results
    stats_kf1 = out_vs_kf1.get("stats", {})
    stats_kf2 = out_vs_kf2.get("stats", {})

    print(f"Frame {test_frame} vs keyframe 1: ok={bool(out_vs_kf1.get('ok', False))} n_track_inliers={int(stats_kf1.get('n_track_inliers', 0))} n_pnp_corr={int(stats_kf1.get('n_pnp_corr', 0))} n_pnp_inliers={int(stats_kf1.get('n_pnp_inliers', 0))}")
    print(f"Frame {test_frame} vs keyframe 2: ok={bool(out_vs_kf2.get('ok', False))} n_track_inliers={int(stats_kf2.get('n_track_inliers', 0))} n_pnp_corr={int(stats_kf2.get('n_pnp_corr', 0))} n_pnp_inliers={int(stats_kf2.get('n_pnp_inliers', 0))}")

    # Write results
    _append_jsonl(
        log_path,
        {
            "event": "comparison",
            "test_frame": int(test_frame),
            "vs_keyframe_1": {
                "ok": bool(out_vs_kf1.get("ok", False)),
                "n_track_inliers": int(stats_kf1.get("n_track_inliers", 0)),
                "n_pnp_corr": int(stats_kf1.get("n_pnp_corr", 0)),
                "n_pnp_inliers": int(stats_kf1.get("n_pnp_inliers", 0)),
                "reason": stats_kf1.get("reason", None),
            },
            "vs_keyframe_2": {
                "ok": bool(out_vs_kf2.get("ok", False)),
                "n_track_inliers": int(stats_kf2.get("n_track_inliers", 0)),
                "n_pnp_corr": int(stats_kf2.get("n_pnp_corr", 0)),
                "n_pnp_inliers": int(stats_kf2.get("n_pnp_inliers", 0)),
                "reason": stats_kf2.get("reason", None),
            },
        },
    )


if __name__ == "__main__":
    main()