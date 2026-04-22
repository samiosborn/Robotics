# scripts/test_bootstrap_only.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from frontend_eth3d_common import (
    frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg,
    load_pil_greyscale as _load_pil_greyscale,
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

# Bootstrap
im0, ts0, id0 = seq.get(0)
im1, ts1, id1 = seq.get(1)

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
print(f"Bootstrap OK: {len(seed['landmarks'])} landmarks")

# Disable map growth for all frames
pnp_kwargs = dict(frontend_kwargs["pnp_frontend_kwargs"])
pnp_kwargs["grow_map"] = False

keyframe_feats = seed["feats1"]

frames_to_test = [2, 3, 4, 5]
for frame_idx in frames_to_test:
    im, ts, frame_id = seq.get(frame_idx)
    frame_out = process_frame_against_seed(
        K, seed, keyframe_feats,
        im,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        keyframe_kf=1,
        current_kf=frame_idx,
        **pnp_kwargs,
    )
    
    print(f"Frame {frame_idx}: ok={frame_out['ok']} reason={frame_out['stats'].get('reason', 'None')}")
    
    # Update for next iteration
    if frame_out["ok"]:
        seed = frame_out["seed"]
        keyframe_feats = frame_out["track_out"]["cur_feats"]

print("\n=== CONCLUSION ===")
print("If frames work with bootstrap-only, the problem is map growth / observation appending")
print("If frames still fail, the problem is in the pose recovery or coordinate conventions")
