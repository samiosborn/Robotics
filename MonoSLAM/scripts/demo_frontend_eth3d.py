# scripts/demo_frontend_eth3d.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from frontend_eth3d_common import ROOT, add_pnp_threshold_stability_args as _add_pnp_threshold_stability_args, append_jsonl as _append_jsonl, apply_pnp_threshold_stability_cli_overrides as _apply_pnp_threshold_stability_cli_overrides, frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_pil_greyscale as _load_pil_greyscale, load_runtime_cfg as _load_runtime_cfg
from core.checks import check_dir, check_int_ge0, check_int_gt0
from datasets.eth3d import load_eth3d_sequence
from features.viz import draw_matches
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames


# Build one compact per-frame summary
def _summarise_frontend_frame(frame_index: int, out: dict, seed: dict) -> dict:
    stats = out.get("stats", {}) if isinstance(out, dict) else {}
    return {
        "frame_index": int(frame_index),
        "ok": bool(out.get("ok", False)) if isinstance(out, dict) else False,
        "n_track_inliers": int(stats.get("n_track_inliers", 0)),
        "n_pnp_corr": int(stats.get("n_pnp_corr", 0)),
        "n_pnp_inliers": int(stats.get("n_pnp_inliers", 0)),
        "n_new_added": int(stats.get("n_new_added", 0)),
        "keyframe_promoted": bool(stats.get("keyframe_promoted", False)),
        "current_landmark_count": int(len(seed.get("landmarks", []))) if isinstance(seed, dict) else 0,
        "reason": stats.get("reason", None),
    }


# Draw one match visualisation
def _draw_match_image(
    imgA: Image.Image,
    imgB: Image.Image,
    kpsA: np.ndarray,
    kpsB: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    out_path: Path,
    *,
    max_draw: int,
    inlier_mask: np.ndarray | None = None,
) -> None:
    draw_matches(
        imgA,
        imgB,
        kpsA,
        kpsB,
        ia,
        ib,
        out_path,
        max_draw=int(max_draw),
        draw_topk=int(max_draw),
        draw_inliers_only=(inlier_mask is not None),
        inlier_mask=inlier_mask,
    )


# Draw bootstrap outputs
def _draw_bootstrap_outputs(seq, i0: int, i1: int, boot: dict, out_dir: Path, max_draw: int) -> None:
    img0 = _load_pil_greyscale(seq.frame_info(i0).path)
    img1 = _load_pil_greyscale(seq.frame_info(i1).path)

    feats0 = boot["feats0"]
    feats1 = boot["feats1"]
    matches = boot["matches01"]

    # Draw all descriptor matches
    _draw_match_image(
        img0,
        img1,
        feats0.kps_xy,
        feats1.kps_xy,
        matches.ia,
        matches.ib,
        out_dir / "bootstrap_matches_all.png",
        max_draw=max_draw,
    )

    seed = boot.get("seed")
    if not isinstance(seed, dict):
        return

    idx_init = np.asarray(seed.get("idx_init", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    if idx_init.size == 0:
        return

    ia = np.asarray(matches.ia, dtype=np.int64)
    ib = np.asarray(matches.ib, dtype=np.int64)

    valid = (idx_init >= 0) & (idx_init < ia.size) & (idx_init < ib.size)
    if not np.any(valid):
        return

    # Draw initialised landmark matches
    _draw_match_image(
        img0,
        img1,
        feats0.kps_xy,
        feats1.kps_xy,
        ia[idx_init[valid]],
        ib[idx_init[valid]],
        out_dir / "bootstrap_matches_init.png",
        max_draw=max_draw,
    )


# Draw tracking outputs
def _draw_track_outputs(
    seq,
    keyframe_index: int,
    frame_index: int,
    keyframe_feats,
    track_out: dict,
    out_dir: Path,
    max_draw: int,
) -> None:
    img_kf = _load_pil_greyscale(seq.frame_info(keyframe_index).path)
    img_cur = _load_pil_greyscale(seq.frame_info(frame_index).path)

    cur_feats = track_out["cur_feats"]
    matches = track_out["matches"]

    # Draw all tentative matches
    _draw_match_image(
        img_kf,
        img_cur,
        keyframe_feats.kps_xy,
        cur_feats.kps_xy,
        matches.ia,
        matches.ib,
        out_dir / f"track_{frame_index:04d}_all.png",
        max_draw=max_draw,
    )

    # Draw geometric inliers
    _draw_match_image(
        img_kf,
        img_cur,
        keyframe_feats.kps_xy,
        cur_feats.kps_xy,
        matches.ia,
        matches.ib,
        out_dir / f"track_{frame_index:04d}_inliers.png",
        max_draw=max_draw,
        inlier_mask=np.asarray(track_out["inlier_mask"], dtype=bool),
    )


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
    # Number of subsequent frames to process
    parser.add_argument("--num_track", type=int, default=5)
    # Maximum number of drawn matches
    parser.add_argument("--max_draw", type=int, default=200)
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
        out_dir = (ROOT / str(run_cfg.get("out_dir", "out")) / str(run_cfg.get("run_id", "frontend_eth3d"))).resolve()

    check_dir(dataset_root, name="dataset_root")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the lightweight run log
    log_path = out_dir / "frontend_log.jsonl"

    i0 = check_int_ge0(args.i0, name="i0")
    i1 = check_int_ge0(args.i1, name="i1")
    num_track = check_int_gt0(args.num_track, name="num_track")
    max_draw = check_int_gt0(args.max_draw, name="max_draw")

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
    if n_effective <= 0:
        raise ValueError("Loaded ETH3D sequence is empty")

    if i0 >= n_effective or i1 >= n_effective:
        raise IndexError(f"Bootstrap indices out of range for effective sequence length {n_effective}")

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
    print(f"K:\n{K}")
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

    _draw_bootstrap_outputs(seq, i0, i1, boot, out_dir, max_draw)

    if not bool(boot["ok"]) or not isinstance(boot.get("seed"), dict):
        print("bootstrap failed; stopping")
        return

    seed = boot["seed"]
    keyframe_feats = seed["feats1"]
    keyframe_index = i1

    print(f"initial landmarks: {len(seed.get('landmarks', []))}")
    print(f"keyframe index: {keyframe_index}")

    start_track = keyframe_index + 1
    stop_track = min(n_effective, start_track + num_track)
    consecutive_failures = 0
    max_consecutive_failures = 3

    for i in range(start_track, stop_track):
        # Keep the current reference keyframe for visualisation
        viz_keyframe_feats = keyframe_feats
        viz_keyframe_index = keyframe_index

        # Process the current frame through the frontend pipeline
        cur_im, cur_ts, cur_id = seq.get(i)
        out = process_frame_against_seed(
            K,
            seed,
            keyframe_feats,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            keyframe_kf=keyframe_index,
            current_kf=i,
            **frontend_kwargs["pnp_frontend_kwargs"],
        )
        seed = out["seed"]
        track_out = out["track_out"]
        keyframe_out = out.get("keyframe_out", None)
        summary = _summarise_frontend_frame(i, out, seed)

        print(
            f"frame {summary['frame_index']}: ok={summary['ok']} "
            f"n_track_inliers={summary['n_track_inliers']} "
            f"n_pnp_corr={summary['n_pnp_corr']} "
            f"n_pnp_inliers={summary['n_pnp_inliers']} "
            f"n_new_added={summary['n_new_added']} "
            f"keyframe_promoted={summary['keyframe_promoted']} "
            f"current_landmark_count={summary['current_landmark_count']}"
        )

        # Write the per-frame summary
        _append_jsonl(
            log_path,
            {
                "event": "frame",
                "frame_id": str(cur_id),
                "timestamp": float(cur_ts),
                **summary,
            },
        )

        # Draw tracking outputs against the reference keyframe used for this step
        _draw_track_outputs(seq, viz_keyframe_index, i, viz_keyframe_feats, track_out, out_dir, max_draw)

        # Update the active keyframe after a promotion
        if keyframe_out is not None and bool(keyframe_out.promoted):
            keyframe_feats = out["track_out"]["cur_feats"]
            keyframe_index = i

        # Stop after repeated frontend failures
        if bool(out["ok"]):
            consecutive_failures = 0
            continue

        consecutive_failures += 1
        print(
            f"frame {i}: frontend failed with reason={summary['reason']} "
            f"consecutive_failures={consecutive_failures}"
        )
        if consecutive_failures >= max_consecutive_failures:
            print(f"stopping after {consecutive_failures} consecutive frontend failures")
            _append_jsonl(
                log_path,
                {
                    "event": "stop",
                    "frame_index": int(i),
                    "reason": "repeated_failures",
                    "consecutive_failures": int(consecutive_failures),
                },
            )
            break


if __name__ == "__main__":
    main()
