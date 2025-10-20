# src/scripts/preprocess_tactip_numpy.py
import argparse
import json
import numpy as np
from pathlib import Path
from src.preprocessing.rasterise_numpy import load_mat_table, create_label_for_sequence
from src.utils.experiment_numpy import load_config, resolve_data_dir, list_mat_paths

# Save NPZ files
def save_npz(path: Path, X, y, g=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if g is None:
        np.savez_compressed(path, X=X, y=y)
    else:
        np.savez_compressed(path, X=X, y=y, g=g)
    pos = float(y.mean()) if y.size else 0.0
    print(f"saved -> {path}  (X={X.shape}, pos={pos:.3f})")

# Build dataset from paths
def build_from_paths(paths, out_hw, scale, step, channels):
    Xs, ys = [], []
    for p in paths:
        try:
            seq = load_mat_table(p)
            Xi, yi = create_label_for_sequence(
                seq,
                out_hw=tuple(out_hw),
                scale=scale,
                step=step,
                channels=channels
            )
            if Xi.shape[0] > 0:
                Xs.append(Xi); ys.append(yi)
        except Exception:
            # Skip bad sequences
            pass
    if not Xs:
        C = 3 if channels == "pcd" else 2
        H, W = out_hw
        return (np.zeros((0, C, H, W), dtype=np.float64),
                np.zeros((0,), dtype=np.float64))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

# Simple path-level split (sequence-level) of TRAIN into train-sub / val
def split_paths_sequence_level(paths, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    n_val = int(round(n * float(val_frac)))
    val_paths = paths[:n_val]
    tr_paths  = paths[n_val:]
    return tr_paths, val_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data/preprocessed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = resolve_data_dir(args.config, cfg["data_dir"])
    out_dir = Path(args.out_dir)

    # Collect train/test sequence files
    train_paths = list_mat_paths(str(data_dir), "train")
    test_paths  = list_mat_paths(str(data_dir), "test")
    print(f"found {len(train_paths)} train sequences, {len(test_paths)} test sequences")

    # Sequence-level split of TRAIN into train-sub / val
    tr_paths, val_paths = split_paths_sequence_level(
        train_paths,
        val_frac=float(cfg.get("val_frac", 0.1)),
        seed=cfg.get("seed", 0),
    )
    print(f"split -> train-sub sequences: {len(tr_paths)}, val sequences: {len(val_paths)}")

    # Rasterise each split
    Xtr, ytr = build_from_paths(tr_paths, cfg["out_hw"], cfg["scale"], cfg["step"], cfg["channels"])
    Xva, yva = build_from_paths(val_paths, cfg["out_hw"], cfg["scale"], cfg["step"], cfg["channels"])
    Xte, yte = build_from_paths(test_paths, cfg["out_hw"], cfg["scale"], cfg["step"], cfg["channels"])
    print(f"raw shapes -> train-sub: {Xtr.shape}, val: {Xva.shape}, test: {Xte.shape}")

    # Normalise VAL and TEST using TRAIN-SUB statistics
    m = Xtr.mean() if Xtr.size else 0.0
    s = Xtr.std() + 1e-8 if Xtr.size else 1.0
    Xtr = (Xtr - m) / s
    Xva = (Xva - m) / s
    Xte = (Xte - m) / s

    # Save 3 NPZs (TRAIN_SUB, VAL, TEST) and a JSON with stats
    stats_path = Path(out_dir) / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({"mean": float(m), "std": float(s)}, indent=2))

    save_npz(Path(out_dir) / "train_sub.npz", Xtr, ytr)
    save_npz(Path(out_dir) / "val.npz",       Xva, yva)
    save_npz(Path(out_dir) / "test.npz",      Xte, yte)

if __name__ == "__main__":
    main()
