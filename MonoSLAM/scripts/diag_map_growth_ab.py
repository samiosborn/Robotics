# scripts/diag_map_growth_ab.py

from pathlib import Path

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg, load_runtime_cfg
from datasets.eth3d import load_eth3d_sequence
from slam.frontend import bootstrap_from_two_frames
from slam.frame_pipeline import process_frame_against_seed

# --- CONFIG ---
PROFILE = "eth3d_c2"

# Load config
profile_path = ROOT / "configs" / "profiles" / f"{PROFILE}.yaml"
cfg, K = load_runtime_cfg(profile_path)
frontend_kwargs = frontend_kwargs_from_cfg(cfg)

dataset_cfg = cfg["dataset"]
dataset_root = ROOT / dataset_cfg["root"]
seq = load_eth3d_sequence(dataset_root, dataset_cfg["seq"])

# --- UTILS ---
def print_landmark_count(label, seed):
    n = len(seed['landmarks'])
    print(f"{label}: {n} landmarks")
    return n

def fresh_bootstrap():
    # Bootstrap from frames 0 and 1
    im0, ts0, id0 = seq.get(0)
    im1, ts1, id1 = seq.get(1)
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
    if not boot['ok']:
        raise Exception(f"Bootstrap failed: {boot['stats']}")
    seed = boot['seed']
    keyframe_feats = seed['feats1']
    keyframe_index = 1
    return seed, keyframe_feats, keyframe_index

# --- BRANCH A: MAP GROWTH ENABLED ---
print("=== BRANCH A: MAP GROWTH ENABLED ===")
seed_a, keyframe_feats_a, keyframe_index_a = fresh_bootstrap()
count_boot_a = print_landmark_count("After bootstrap", seed_a)

# Frame 2
im2, ts2, id2 = seq.get(2)
out2_a = process_frame_against_seed(
    K,
    seed_a,
    keyframe_feats_a,
    im2,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=keyframe_index_a,
    current_kf=2,
    grow_map=True,
    **frontend_kwargs["pnp_frontend_kwargs"],
)
seed_a = out2_a["seed"]
success2_a = out2_a.get("ok", False)
reason2_a = out2_a.get("stats", {}).get("reason", "")
count_after2_a = print_landmark_count("After frame 2", seed_a)
appended2_a = count_after2_a > count_boot_a
print(f"Frame 2: {'SUCCESS' if success2_a else 'FAIL'} {reason2_a}")
print(f"New landmarks appended at frame 2: {appended2_a}")

# Update keyframe if promoted
if out2_a.get("stats", {}).get("keyframe_promoted", False):
    keyframe_feats_a = out2_a["track_out"]["cur_feats"]
    keyframe_index_a = 2

# Frame 3
im3, ts3, id3 = seq.get(3)
out3_a = process_frame_against_seed(
    K,
    seed_a,
    keyframe_feats_a,
    im3,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=keyframe_index_a,
    current_kf=3,
    grow_map=True,
    **frontend_kwargs["pnp_frontend_kwargs"],
)
seed_a = out3_a["seed"]
success3_a = out3_a.get("ok", False)
reason3_a = out3_a.get("stats", {}).get("reason", "")
print(f"Frame 3: {'SUCCESS' if success3_a else 'FAIL'} {reason3_a}")
print_landmark_count("After frame 3", seed_a)

# --- BRANCH B: MAP GROWTH DISABLED ---
print("\n=== BRANCH B: MAP GROWTH DISABLED ===")
seed_b, keyframe_feats_b, keyframe_index_b = fresh_bootstrap()
count_boot_b = print_landmark_count("After bootstrap", seed_b)

# Frame 2
im2, ts2, id2 = seq.get(2)
out2_b = process_frame_against_seed(
    K,
    seed_b,
    keyframe_feats_b,
    im2,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=keyframe_index_b,
    current_kf=2,
    grow_map=False,
    **frontend_kwargs["pnp_frontend_kwargs"],
)
seed_b = out2_b["seed"]
success2_b = out2_b.get("ok", False)
reason2_b = out2_b.get("stats", {}).get("reason", "")
count_after2_b = print_landmark_count("After frame 2", seed_b)
appended2_b = count_after2_b > count_boot_b
print(f"Frame 2: {'SUCCESS' if success2_b else 'FAIL'} {reason2_b}")
print(f"New landmarks appended at frame 2: {appended2_b}")

# Update keyframe if promoted
if out2_b.get("stats", {}).get("keyframe_promoted", False):
    keyframe_feats_b = out2_b["track_out"]["cur_feats"]
    keyframe_index_b = 2

# Frame 3
im3, ts3, id3 = seq.get(3)
out3_b = process_frame_against_seed(
    K,
    seed_b,
    keyframe_feats_b,
    im3,
    feature_cfg=frontend_kwargs["feature_cfg"],
    F_cfg=frontend_kwargs["F_cfg"],
    keyframe_kf=keyframe_index_b,
    current_kf=3,
    grow_map=False,
    **frontend_kwargs["pnp_frontend_kwargs"],
)
seed_b = out3_b["seed"]
success3_b = out3_b.get("ok", False)
reason3_b = out3_b.get("stats", {}).get("reason", "")
print(f"Frame 3: {'SUCCESS' if success3_b else 'FAIL'} {reason3_b}")
print_landmark_count("After frame 3", seed_b)

# --- CHECK NO-GROWTH INVARIANT ---
if count_after2_b != count_boot_b:
    print("[ERROR] No-growth branch did not keep bootstrap landmark count. Diagnostic INVALID.")
    exit(1)

# --- SUMMARY ---
print("\n=== SUMMARY ===")
print(f"Branch A (growth): Frame 2: {'OK' if success2_a else 'FAIL'}; Frame 3: {'OK' if success3_a else 'FAIL'}")
print(f"Branch B (no growth): Frame 2: {'OK' if success2_b else 'FAIL'}; Frame 3: {'OK' if success3_b else 'FAIL'}")
if not success3_a and success3_b:
    print("map growth is ruled in")
elif success3_a and not success3_b:
    print("map growth is ruled out")
elif not success3_a and not success3_b:
    print("diagnostic still inconclusive")
else:
    print("diagnostic still inconclusive")
