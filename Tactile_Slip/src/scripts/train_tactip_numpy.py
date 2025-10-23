# src/scripts/train_tactip_numpy.py
import argparse
import numpy as np
from src.preprocessing.rasterise_numpy import build_dataset
from src.models.train_cnn_numpy import init_params, fit, train_one_epoch
from src.models.io_params import save_params
from src.utils.experiment_numpy import load_config, resolve_data_dir, sequence_level_split

def main():
    # Create argument parser object
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = ap.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve data_dir relative to the config
    data_dir = resolve_data_dir(args.config, cfg["data_dir"])

    # Build train
    Xtr_all, ytr_all, gtr = build_dataset(
        str(data_dir), "train",
        tuple(cfg["out_hw"]), cfg["scale"], cfg["step"], cfg["channels"],
        return_groups=True
    )

    # Build test
    Xte, yte, _gte = build_dataset(
        str(data_dir), "test",
        tuple(cfg["out_hw"]), cfg["scale"], cfg["step"], cfg["channels"],
        return_groups=True
    )
    if Xtr_all.shape[0] == 0:
        raise RuntimeError("Empty training set: check data_dir and preprocessing")

    # Build train-sub and validation
    Xtr_sub, ytr_sub, Xval, yval = sequence_level_split(
        Xtr_all, ytr_all, gtr, val_frac=float(cfg.get("val_frac", 0.1)), seed=cfg.get("seed", 0)
    )
    print(f"Shapes: train-sub: {Xtr_sub.shape}, pos={ytr_sub.mean():.3f}, "
          f"val: {Xval.shape}, pos={yval.mean():.3f}, "
          f"test: {Xte.shape}, pos={yte.mean():.3f}")
    if Xtr_sub.shape[0] == 0:
        raise RuntimeError("Empty training set after sequence-level split: check data_dir and preprocessing")

    # Initialise
    rng = np.random.default_rng(cfg.get("seed", 0))
    C_in, H, W = Xtr_sub.shape[1], Xtr_sub.shape[2], Xtr_sub.shape[3]
    params = init_params(
        rng, C_in=C_in, H=H, W=W,
        k=cfg.get("k", 3), C1=cfg["C1"], C2=cfg["C2"],
        stride=cfg["stride"], padding=cfg["padding"],
        pool_size=cfg["pool_size"], pool_stride=cfg["pool_stride"]
    )

    # Train
    params = fit(
        params, Xtr_sub, ytr_sub,
        cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        batch_size=cfg["batch_size"], epochs=cfg["epochs"],
        pos_weight=cfg["pos_weight"], verbose=True
    )

    # Evaluate (no updates, no balancing)
    tr_loss, tr_acc = train_one_epoch(
        params, Xtr_sub, ytr_sub,
        cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
        lr=0.0, batch_size=64, weight_decay=0.0,
        pos_weight=cfg["pos_weight"], balanced=False)
    
    va_loss, va_acc = train_one_epoch(
        params, Xval, yval,
        cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
        lr=0.0, batch_size=64, weight_decay=0.0,
        pos_weight=cfg["pos_weight"], balanced=False)
    
    te_loss, te_acc = train_one_epoch(
        params, Xte, yte,
        cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
        lr=0.0, batch_size=64, weight_decay=0.0,
        pos_weight=cfg["pos_weight"], balanced=False)

    print(f"final train-sub: loss={tr_loss:.4f} acc={tr_acc:.3f}")
    print(f"final val: loss={va_loss:.4f} acc={va_acc:.3f}")
    print(f"final test: loss={te_loss:.4f} acc={te_acc:.3f}")

    # Save
    save_path = cfg.get("save_params_path")
    if save_path:
        save_params(params, save_path)
        print(f"Parameters saved to: {save_path}")

if __name__ == "__main__":
    main()
