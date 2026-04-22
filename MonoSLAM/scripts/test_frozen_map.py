# scripts/test_frozen_map.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from frontend_eth3d_common import (
    frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg,
    load_runtime_cfg as _load_runtime_cfg,
)
from core.checks import check_dir
from datasets.eth3d import load_eth3d_sequence
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames

profile_path = ROOT / "configs" / "profiles" / "eth3d_c2.yaml"
cfg, K = _load_runtime_cfg(profile_path)
frontend_kwargs = _frontend_kwargs_from_cfg(cfg)

dataset_cfg = cfg["dataset"]
dataset_root = ROOT / dataset_cfg["root"]
seq_name = dataset_cfg["seq"]

check_dir(dataset_root, name="dataset_root")

# Load sequence
seq = load_eth3d_sequence(dataset_root, seq_name)

# Read images once
im0, _ts0, _id0 = seq.get(0)
im1, _ts1, _id1 = seq.get(1)
im2, _ts2, _id2 = seq.get(2)
im3, _ts3, _id3 = seq.get(3)


def fresh_bootstrap():
    boot = bootstrap_from_two_frames(
        K, K, im0, im1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )
    if not boot["ok"]:
        print("Bootstrap failed")
        sys.exit(1)
    seed = boot["seed"]
    keyframe_feats = seed["feats1"]
    return seed, keyframe_feats


# Test with grow_map enabled (default)
print("\n=== WITH MAP GROWTH ===")
seed_a, keyframe_feats_a = fresh_bootstrap()
print(f"Bootstrap OK: {len(seed_a['landmarks'])} landmarks")

frame2_out_with_growth = process_frame_against_seed(
    K, seed_a, keyframe_feats_a,
    im2,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=1,
    current_kf=2,
    **frontend_kwargs["pnp_frontend_kwargs"],
)
print(f"Frame 2: ok={frame2_out_with_growth['ok']} n_landmarks={len(frame2_out_with_growth['seed']['landmarks'])}")

seed_after_f2_a = frame2_out_with_growth["seed"]
keyframe_feats_f2_a = frame2_out_with_growth["track_out"]["cur_feats"]

frame3_out_with_growth = process_frame_against_seed(
    K, seed_after_f2_a, keyframe_feats_f2_a,
    im3,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=2,
    current_kf=3,
    **frontend_kwargs["pnp_frontend_kwargs"],
)
print(f"Frame 3: ok={frame3_out_with_growth['ok']} reason={frame3_out_with_growth['stats'].get('reason')}")

# Test WITHOUT map growth using a fresh seed so branch-A mutations don't pollute branch-B
print("\n=== WITHOUT MAP GROWTH ===")
seed_b, keyframe_feats_b = fresh_bootstrap()
print(f"Bootstrap OK: {len(seed_b['landmarks'])} landmarks")

pnp_kwargs_frozen = dict(frontend_kwargs["pnp_frontend_kwargs"])
pnp_kwargs_frozen["grow_map"] = False

frame2_out_frozen = process_frame_against_seed(
    K, seed_b, keyframe_feats_b,
    im2,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=1,
    current_kf=2,
    **pnp_kwargs_frozen,
)
print(f"Frame 2: ok={frame2_out_frozen['ok']} n_landmarks={len(frame2_out_frozen['seed']['landmarks'])}")

seed_frozen_after_f2 = frame2_out_frozen["seed"]
keyframe_feats_frozen_f2 = frame2_out_frozen["track_out"]["cur_feats"]

frame3_out_frozen = process_frame_against_seed(
    K, seed_frozen_after_f2, keyframe_feats_frozen_f2,
    im3,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=2,
    current_kf=3,
    **pnp_kwargs_frozen,
)
print(f"Frame 3: ok={frame3_out_frozen['ok']} reason={frame3_out_frozen['stats'].get('reason')}")

# Summary
print("\n=== SUMMARY ===")
print(f"WITH growth:    Frame 3 ok = {frame3_out_with_growth['ok']}")
print(f"WITHOUT growth: Frame 3 ok = {frame3_out_frozen['ok']}")

if frame3_out_with_growth['ok'] != frame3_out_frozen['ok']:
    print(f"\n⚠️  MAP GROWTH IS THE CULPRIT!")
    print(f"   Frame 3 fails WITH growth but {'would pass' if frame3_out_frozen['ok'] else 'also fails'} without growth")
else:
    print(f"\nℹ️  Map growth is NOT the blocker")
    if not frame3_out_frozen['ok']:
        print(f"   Problem is in the bootstrap or frame 2 itself")
