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

from core.checks import check_dir, check_file, check_int_ge0, check_int_gt0
from utils.load_config import load_config
from datasets.eth3d import load_eth3d_sequence
from features.viz import draw_matches
from slam.frontend import bootstrap_from_two_frames
from slam.tracking import track_against_keyframe


# Load greyscale image for visualisation
def _load_pil_greyscale(path: Path) -> Image.Image:
    return Image.open(str(check_file(path, name="image"))).convert("L")


# Resolve profile include path
def _resolve_include_path(rel_path: str, profile_path: Path) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        return check_file(p, name="include")
    # Profile includes are written from repo root
    return check_file(ROOT / p, name="include")


# Load profile and build runtime config
def _load_runtime_cfg(profile_path: Path) -> tuple[dict, np.ndarray]:
    profile_cfg = load_config(check_file(profile_path, name="profile"))

    includes = profile_cfg["includes"]
    camera_cfg = load_config(_resolve_include_path(includes["camera"], profile_path))
    features_cfg = load_config(_resolve_include_path(includes["features"], profile_path))
    bootstrap_cfg = load_config(_resolve_include_path(includes["bootstrap"], profile_path))

    K_cfg = camera_cfg["camera"]["K"]
    K = np.array(
        [
            [float(K_cfg["fx"]), 0.0, float(K_cfg["cx"])],
            [0.0, float(K_cfg["fy"]), float(K_cfg["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    runtime_cfg = {
        "profile": profile_cfg.get("profile", {}),
        "dataset": profile_cfg.get("dataset", {}),
        "run": profile_cfg.get("run", {}),
        "features": features_cfg,
        "ransac": bootstrap_cfg["ransac"],
        "bootstrap": bootstrap_cfg["bootstrap"],
    }

    return runtime_cfg, K


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

    # Fixed bootstrap frame 0
    parser.add_argument("--i0", type=int, default=0)
    # Fixed bootstrap frame 1
    parser.add_argument("--i1", type=int, default=1)
    # Number of later frames to track
    parser.add_argument("--num_track", type=int, default=5)
    # Maximum number of drawn matches
    parser.add_argument("--max_draw", type=int, default=200)

    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    cfg, K = _load_runtime_cfg(profile_path)

    dataset_cfg = cfg["dataset"]
    run_cfg = cfg["run"]

    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root is not None else (ROOT / dataset_cfg["root"]).resolve()
    seq_name = str(args.seq) if args.seq is not None else str(dataset_cfg["seq"])

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (ROOT / str(run_cfg.get("out_dir", "out")) / str(run_cfg.get("run_id", "frontend_eth3d"))).resolve()

    check_dir(dataset_root, name="dataset_root")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    boot = bootstrap_from_two_frames(K, K, im0, im1, cfg)

    print(f"sequence: {seq.name}")
    print(f"dataset_root: {dataset_root}")
    print(f"seq_name: {seq_name}")
    print(f"K:\n{K}")
    print(f"bootstrap pair: {i0} ({id0}, t={ts0}) -> {i1} ({id1}, t={ts1})")
    print(f"bootstrap ok: {boot['ok']}")
    print(f"bootstrap stats: {boot['stats']}")

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

    for i in range(start_track, stop_track):
        # Track current frame against the frozen keyframe
        cur_im, cur_ts, cur_id = seq.get(i)
        tr = track_against_keyframe(K, keyframe_feats, cur_im, cfg)
        n_inliers = int(np.asarray(tr["inlier_mask"], dtype=bool).sum())

        print(f"track frame: {i} ({cur_id}, t={cur_ts})")
        print(f"  stats: {tr['stats']}")
        print(f"  inliers: {n_inliers}")

        _draw_track_outputs(seq, keyframe_index, i, keyframe_feats, tr, out_dir, max_draw)


if __name__ == "__main__":
    main()
