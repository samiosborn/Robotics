# sripts/demo_frontend_eth3d.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.load_config import load_config
from datasets.eth3d import load_eth3d_sequence
from features.viz import draw_matches
from slam.frontend import bootstrap_from_two_frames, track_against_keyframe

# Resolve directory path
def _resolve_path(p: str | Path, base: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path).resolve()

# Load PIL greyscale image
def _load_pil_greyscale(path: Path) -> Image.Image:
    return Image.open(str(path)).convert("L")

# Camera config load
def _camera_cfg_from_profile(profile_cfg: dict, profile_path: Path) -> dict:

    prof_dir = profile_path.parent

    if isinstance(profile_cfg.get("camera"), dict):
        return profile_cfg["camera"]

    candidates = [
        profile_cfg.get("camera"),
        profile_cfg.get("camera_cfg"),
        profile_cfg.get("paths", {}).get("camera") if isinstance(profile_cfg.get("paths"), dict) else None,
    ]

    for c in candidates:
        if isinstance(c, str) and c.strip() != "":
            cam_path = _resolve_path(c, prof_dir)
            return load_config(cam_path)

    raise ValueError(
        "Could not resolve camera config from profile. "
        "Expected one of: profile['camera'] dict, profile['camera'] path, "
        "profile['camera_cfg'] path, or profile['paths']['camera'] path."
    )

# Camera matrix from config
def _camera_matrix_from_cfg(camera_cfg: dict) -> np.ndarray:

    if isinstance(camera_cfg.get("K"), (list, tuple)):
        K = np.asarray(camera_cfg["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"camera_cfg['K'] must be (3,3); got {K.shape}")
        return K

    intr = camera_cfg.get("intrinsics", camera_cfg)
    if not isinstance(intr, dict):
        raise ValueError("Camera config intrinsics block must be a dict")

    needed = ["fx", "fy", "cx", "cy"]
    if not all(k in intr for k in needed):
        raise ValueError(
            "Camera config must contain fx, fy, cx, cy "
            "either at top level or under 'intrinsics'"
        )

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


# Draw bootstrap outputs
def _draw_bootstrap_outputs(
    seq,
    i0: int,
    i1: int,
    boot: dict,
    out_dir: Path,
    max_draw: int,
) -> None:
    rec0 = seq.frame_info(i0)
    rec1 = seq.frame_info(i1)

    img0 = _load_pil_greyscale(rec0.path)
    img1 = _load_pil_greyscale(rec1.path)

    feats0 = boot["feats0"]
    feats1 = boot["feats1"]
    matches = boot["matches01"]

    draw_matches(
        img0,
        img1,
        feats0.kps_xy,
        feats1.kps_xy,
        matches.ia,
        matches.ib,
        out_dir / "bootstrap_matches_all.png",
        max_draw=int(max_draw),
        draw_topk=int(max_draw),
    )

    seed = boot.get("seed")
    if not isinstance(seed, dict):
        return

    idx_init = np.asarray(seed.get("idx_init", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    if idx_init.size == 0:
        return

    ia_init = np.asarray(matches.ia, dtype=np.int64)[idx_init]
    ib_init = np.asarray(matches.ib, dtype=np.int64)[idx_init]

    draw_matches(
        img0,
        img1,
        feats0.kps_xy,
        feats1.kps_xy,
        ia_init,
        ib_init,
        out_dir / "bootstrap_matches_init.png",
        max_draw=int(max_draw),
        draw_topk=int(max_draw),
    )


# Draw tracked outputs
def _draw_track_outputs(
    seq,
    keyframe_index: int,
    frame_index: int,
    keyframe_feats,
    track_out: dict,
    out_dir: Path,
    max_draw: int,
) -> None:
    rec_kf = seq.frame_info(keyframe_index)
    rec_cur = seq.frame_info(frame_index)

    img_kf = _load_pil_greyscale(rec_kf.path)
    img_cur = _load_pil_greyscale(rec_cur.path)

    cur_feats = track_out["cur_feats"]
    matches = track_out["matches"]
    inlier_mask = track_out["inlier_mask"]

    draw_matches(
        img_kf,
        img_cur,
        keyframe_feats.kps_xy,
        cur_feats.kps_xy,
        matches.ia,
        matches.ib,
        out_dir / f"track_{frame_index:04d}_all.png",
        max_draw=int(max_draw),
        draw_topk=int(max_draw),
    )

    draw_matches(
        img_kf,
        img_cur,
        keyframe_feats.kps_xy,
        cur_feats.kps_xy,
        matches.ia,
        matches.ib,
        out_dir / f"track_{frame_index:04d}_inliers.png",
        max_draw=int(max_draw),
        draw_topk=int(max_draw),
        draw_inliers_only=True,
        inlier_mask=inlier_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Profile YAML, e.g. src/config/profiles/eth3d_c2.yaml",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="ETH3D dataset root, e.g. data/eth3d",
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="ETH3D sequence name, e.g. cables_2_mono",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "out" / "frontend_eth3d"),
    )

    parser.add_argument("--i0", type=int, default=0, help="Bootstrap frame 0 index")
    parser.add_argument("--i1", type=int, default=1, help="Bootstrap frame 1 index")
    parser.add_argument("--num_track", type=int, default=5, help="Number of frames to track after bootstrap")
    parser.add_argument("--max_draw", type=int, default=200, help="Maximum matches to draw per image")

    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(profile_path)
    camera_cfg = _camera_cfg_from_profile(cfg, profile_path)
    K = _camera_matrix_from_cfg(camera_cfg)

    seq = load_eth3d_sequence(
        dataset_root,
        args.seq,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    if len(seq) == 0:
        raise ValueError("Loaded ETH3D sequence is empty")

    if not (0 <= int(args.i0) < len(seq)):
        raise IndexError(f"i0 out of range: {args.i0} for sequence length {len(seq)}")
    if not (0 <= int(args.i1) < len(seq)):
        raise IndexError(f"i1 out of range: {args.i1} for sequence length {len(seq)}")
    if int(args.i1) <= int(args.i0):
        raise ValueError(f"Expected i1 > i0 for bootstrap; got i0={args.i0}, i1={args.i1}")

    im0, ts0, id0 = seq.get(int(args.i0))
    im1, ts1, id1 = seq.get(int(args.i1))

    boot = bootstrap_from_two_frames(K, K, im0, im1, cfg)

    print(f"sequence: {seq.name}")
    print(f"bootstrap pair: {args.i0} ({id0}, t={ts0}) -> {args.i1} ({id1}, t={ts1})")
    print(f"bootstrap ok: {boot['ok']}")
    print(f"bootstrap stats: {boot['stats']}")

    _draw_bootstrap_outputs(
        seq=seq,
        i0=int(args.i0),
        i1=int(args.i1),
        boot=boot,
        out_dir=out_dir,
        max_draw=int(args.max_draw),
    )

    if not bool(boot["ok"]) or not isinstance(boot.get("seed"), dict):
        print("bootstrap failed; stopping")
        return

    seed = boot["seed"]
    keyframe_feats = seed["feats1"]
    keyframe_index = int(args.i1)

    print(f"initial landmarks: {len(seed.get('landmarks', []))}")
    print(f"keyframe index: {keyframe_index}")

    start_track = keyframe_index + 1
    stop_track = min(len(seq), start_track + int(args.num_track))

    for i in range(start_track, stop_track):
        cur_im, cur_ts, cur_id = seq.get(i)

        tr = track_against_keyframe(K, keyframe_feats, cur_im, cfg)

        n_inliers = int(np.asarray(tr["inlier_mask"], dtype=bool).sum())

        print(f"track frame: {i} ({cur_id}, t={cur_ts})")
        print(f"  stats: {tr['stats']}")
        print(f"  inliers: {n_inliers}")

        _draw_track_outputs(
            seq=seq,
            keyframe_index=keyframe_index,
            frame_index=i,
            keyframe_feats=keyframe_feats,
            track_out=tr,
            out_dir=out_dir,
            max_draw=int(args.max_draw),
        )


if __name__ == "__main__":
    main()