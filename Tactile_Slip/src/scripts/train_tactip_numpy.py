# src/scripts/train_tactip_numpy.py
import argparse
import numpy as np
from src.preprocessing.rasterise_numpy import build_dataset, load_mat_table, create_label_for_sequence
from src.models.train_cnn_numpy import init_params, fit, train_one_epoch, batch_iter
from src.models.io_params import save_params
from src.utils.experiment_numpy import load_config, resolve_data_dir, list_mat_paths, sequence_level_split

def main():
    # Create argument parser object
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = ap.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve data_dir relative to the config
    data_dir = resolve_data_dir(args.config, cfg["data_dir"])

    # Build test
    Xte, yte = build_dataset(
        str(data_dir), "test",
        tuple(cfg["out_hw"]), cfg["scale"], cfg["step"], cfg["channels"]
    )

    # Build train-sub and validation from a sequence-level split
    train_paths = list_mat_paths(str(data_dir), "train")
    tr_paths, val_paths = sequence_level_split(
        train_paths, val_frac=float(cfg.get("val_frac", 0.1)), seed=cfg.get("seed", 0)
    )

    # Build train-sub
    Xs, ys = [], []
    for p in tr_paths:
        seq = load_mat_table(p)
        Xi, yi = create_label_for_sequence(
            seq,
            out_hw=tuple(cfg["out_hw"]),
            scale=cfg["scale"],
            step=cfg["step"],
            channels=cfg["channels"]
        )
        if Xi.shape[0] > 0:
            Xs.append(Xi); ys.append(yi)
    C = 3 if cfg["channels"] == "pcd" else 2
    H, W = cfg["out_hw"]
    Xtr_sub = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, C, H, W), dtype=np.float64)
    ytr_sub = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float64)

    # Build validation set
    Xs, ys = [], []
    for p in val_paths:
        seq = load_mat_table(p)
        Xi, yi = create_label_for_sequence(
            seq,
            out_hw=tuple(cfg["out_hw"]),
            scale=cfg["scale"],
            step=cfg["step"],
            channels=cfg["channels"]
        )
        if Xi.shape[0] > 0:
            Xs.append(Xi); ys.append(yi)
    Xval = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, C, H, W), dtype=np.float64)
    yval = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float64)

    print(f"Shapes: train: {Xtr_sub.shape}, pos={ytr_sub.mean():.3f}, test: {Xte.shape}, pos={yte.mean():.3f}")
    print(f"Train-sub shape: {Xtr_sub.shape}, pos={ytr_sub.mean():.3f} & Val shape: {Xval.shape}, pos={yval.mean():.3f}")

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
        batch_size=cfg["batch_size"], epochs=cfg["epochs"], verbose=True
    )

    # Evaluate (no updates)
    tr_loss, tr_acc = train_one_epoch(
        params, Xtr_sub, ytr_sub,
        cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
        lr=0.0, batch_size=64, weight_decay=0.0
    )
    va_loss, va_acc = train_one_epoch(
        params, Xval, yval,
        cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
        lr=0.0, batch_size=64, weight_decay=0.0
    )
    te_loss, te_acc = train_one_epoch(
        params, Xte, yte,
        cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"],
        lr=0.0, batch_size=64, weight_decay=0.0
    )
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
