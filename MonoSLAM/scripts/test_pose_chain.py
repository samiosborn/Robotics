# scripts/test_pose_chain.py
import sys
from pathlib import Path

import numpy as np

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
from geometry.camera import camera_centre, project_points
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames

profile_path = ROOT / "configs" / "profiles" / "eth3d_c2.yaml"
cfg, K = _load_runtime_cfg(profile_path)
frontend_kwargs = _frontend_kwargs_from_cfg(cfg)

dataset_cfg = cfg["dataset"]
dataset_root = ROOT / dataset_cfg["root"]
seq_name = dataset_cfg["seq"]

check_dir(dataset_root, name="dataset_root")

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

# Print bootstrap pose info
R0, t0 = seed["T_WC0"]
R1, t1 = seed["T_WC1"]
C0 = camera_centre(R0, t0)
C1 = camera_centre(R1, t1)
baseline = float(np.linalg.norm(C1 - C0))
print(f"Bootstrap baseline: {baseline:.6f} m")
print(f"Frame 0 camera centre: {C0}")
print(f"Frame 1 camera centre: {C1}")

# Track frame 2
pnp_kwargs = dict(frontend_kwargs["pnp_frontend_kwargs"])
pnp_kwargs["grow_map"] = False

keyframe_feats = seed["feats1"]

im2, ts2, id2 = seq.get(2)
out2 = process_frame_against_seed(
    K, seed, keyframe_feats,
    im2,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=1,
    current_kf=2,
    **pnp_kwargs,
)

print(f"\nFrame 2: ok={out2['ok']}")

if not out2['ok']:
    print("Frame 2 failed, can't continue")
    sys.exit(1)

seed2 = out2["seed"]
R2 = out2.get("R")
t2 = out2.get("t")

if R2 is not None and t2 is not None:
    R2 = np.asarray(R2, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64).reshape(3)
    C2 = camera_centre(R2, t2)
    baseline_01_to_2 = float(np.linalg.norm(C2 - C1))
    print(f"Frame 2 camera centre: {C2}")
    print(f"Baseline F1->F2: {baseline_01_to_2:.6f} m")
    
    # Test: do the bootstrap landmarks still reproject well in frame 2?
    landmarks = seed2.get("landmarks", [])
    reprojection_errors = []
    for lm in landmarks[:10]:  # Test first 10
        if not isinstance(lm, dict):
            continue
        X_w = lm.get("X_w")
        if X_w is None:
            continue
        X_w = np.asarray(X_w, dtype=np.float64).reshape(3)
        if X_w.shape != (3,):
            continue
        
        # Project into frame 2
        x_pred = project_points(K, R2, t2, X_w.reshape(3, 1))
        
        # Find observation in frame 2
        obs = lm.get("obs", [])
        x_obs = None
        for ob in obs:
            if isinstance(ob, dict) and int(ob.get("kf", -1)) == 2:
                x_obs = np.asarray(ob.get("xy"), dtype=np.float64).reshape(2)
                break
        
        if x_obs is not None:
            err = float(np.linalg.norm(x_obs - x_pred.reshape(2)))
            reprojection_errors.append(err)
            print(f"  Landmark {lm.get('id')}: reproj_err = {err:.2f}px")
    
    if reprojection_errors:
        print(f"Mean reprojection error in frame 2: {np.mean(reprojection_errors):.2f}px")

# Now test frame 3
print(f"\nTesting frame 3...")
im3, ts3, id3 = seq.get(3)
out3 = process_frame_against_seed(
    K, seed2, out2["track_out"]["cur_feats"],
    im3,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=2,
    current_kf=3,
    **pnp_kwargs,
)

print(f"Frame 3: ok={out3['ok']} reason={out3['stats'].get('reason')}")

R3 = out3.get("R")
t3 = out3.get("t")

if R3 is not None and t3 is not None:
    R3 = np.asarray(R3, dtype=np.float64)
    t3 = np.asarray(t3, dtype=np.float64).reshape(3)
    C3 = camera_centre(R3, t3)
    baseline_2_to_3 = float(np.linalg.norm(C3 - C2))
    print(f"Frame 3 camera centre: {C3}")
    print(f"Baseline F2->F3: {baseline_2_to_3:.6f} m")
else:
    print("Frame 3 pose not recovered")
