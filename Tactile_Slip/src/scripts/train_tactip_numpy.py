# src/scripts/train_tactip_numpy.py
from pathlib import Path
import json
import argparse
import numpy as np
from src.preprocessing.rasterise_numpy import build_dataset
from src.models.train_cnn_numpy import init_params, fit, train_one_epoch
from src.models.io_params import save_params

# Load configuration
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Create argument parser object
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = ap.parse_args()

    # Load config FIRST
    cfg = load_config(args.config)

    # Resolve data_dir relative to the CONFIG file (robust)
    config_dir = Path(args.config).resolve().parent
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (config_dir / data_dir).resolve()

    # Build train and test labelled dataset
    Xtr, ytr = build_dataset(str(data_dir), "train",
                             tuple(cfg["out_hw"]), cfg["scale"], cfg["step"], cfg["channels"])
    Xte, yte = build_dataset(str(data_dir), "test",
                             tuple(cfg["out_hw"]), cfg["scale"], cfg["step"], cfg["channels"])
    if Xtr.shape[0] == 0:
        raise RuntimeError("Empty training set: check data_dir and preprocessing")
    print(f"Shapes: train: {Xtr.shape}, pos={ytr.mean():.3f}, test: {Xte.shape}, pos={yte.mean():.3f}")

    # Initialise
    rng = np.random.default_rng(cfg.get("seed", 0))
    C_in, H, W = Xtr.shape[1], Xtr.shape[2], Xtr.shape[3]
    params = init_params(rng, C_in=C_in, H=H, W=W,
                         k=cfg.get("k", 3), C1=cfg["C1"], C2=cfg["C2"],
                         stride=cfg["stride"], padding=cfg["padding"],
                         pool_size=cfg["pool_size"], pool_stride=cfg["pool_stride"])

    # Train
    params = fit(params, Xtr, ytr,
                 cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
                 lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                 batch_size=cfg["batch_size"], epochs=cfg["epochs"], verbose=True)

    # Evaluate (no updates)
    eval_bs_train = min(64, max(1, Xtr.shape[0]))
    tr_loss, tr_acc = train_one_epoch(params, Xtr, ytr,
                                      cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
                                      lr=0.0, batch_size=eval_bs_train, weight_decay=0.0)

    if Xte.shape[0] > 0:
        eval_bs_test = min(64, Xte.shape[0])
        te_loss, te_acc = train_one_epoch(params, Xte, yte,
                                          cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
                                          lr=0.0, batch_size=eval_bs_test, weight_decay=0.0)
        print(f"final train: loss={tr_loss:.4f} acc={tr_acc:.3f}")
        print(f"final test: loss={te_loss:.4f} acc={te_acc:.3f}")
    else:
        print(f"final train: loss={tr_loss:.4f} acc={tr_acc:.3f}")
        print("final test: (no test samples)")

    # Save
    save_path = cfg.get("save_params_path")
    if save_path:
        save_params(params, save_path)
        print(f"Parameters saved to: {save_path}")

if __name__ == "__main__":
    main()
