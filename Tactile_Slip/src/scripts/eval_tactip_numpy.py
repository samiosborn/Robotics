# src/scripts/eval_tactip_numpy.py
from pathlib import Path
import json
import argparse
import numpy as np
from src.preprocessing.rasterise_numpy import build_dataset
from src.models.io_params import load_params
from src.models.cnn_numpy import forward_numpy
from src.models.train_cnn_numpy import batch_iter
from src.models.metrics_numpy import binary_metrics_from_logits, best_threshold
from src.utils.experiment_numpy import load_config, sequence_level_split

# Run forward (no updates) and collect logits with labels for metrics / threshold tuning
def collect_logits(params, X, y, padding, stride, pool_size, pool_stride, batch_size=256):
    zs, ys = [], []
    processed = 0
    total = X.shape[0]

    for xb, yb in batch_iter(X, y, batch_size, shuffle=False):
        # Forward pass each item
        for n in range(xb.shape[0]):
            p, cache = forward_numpy(
                xb[n],
                params["conv1_w"], params["conv1_b"],
                params["conv2_w"], params["conv2_b"],
                params["lin_w"],   params["lin_b"],
                padding, stride, pool_size, pool_stride, True
            )
            zs.append(float(cache["z"][0]))
            ys.append(float(yb[n]))
        # Progress print
        processed += xb.shape[0]
        if processed % 500 == 0 or processed >= total:
            print(f"processed {processed}/{total}", flush=True)

    return np.array(zs, dtype=np.float64), np.array(ys, dtype=np.float64)

# Print quantiles of score distributions
def print_quantiles(name, s):
    qs = [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
    qv = np.quantile(s, qs)
    print(name, " ".join(f"{q:.2f}:{v:.3f}" for q,v in zip(qs,qv)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config (same as training)")
    ap.add_argument("--params", type=str, required=True, help="Path to saved params .npz")
    ap.add_argument("--val_frac", type=float, default=None, help="Override val fraction; default to config or 0.1")
    ap.add_argument("--criterion", type=str, default="f1", choices=["f1","precision","recall","acc"], help="Metric to optimize for threshold")
    ap.add_argument("--out", type=str, default="configs/tactip_metrics.json", help="Where to save metrics + threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    params = load_params(args.params)

    # Resolve data_dir relative to config
    config_dir = Path(args.config).resolve().parent
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (config_dir / data_dir).resolve()

    # Rebuild datasets as in training (expects build_dataset to return (X, y, g))
    Xtr, ytr, gtr = build_dataset(str(data_dir), "train",
                                  tuple(cfg["out_hw"]), cfg["scale"], cfg["step"], cfg["channels"])
    Xte, yte, gte = build_dataset(str(data_dir), "test",
                                  tuple(cfg["out_hw"]), cfg["scale"], cfg["step"], cfg["channels"])
    if Xtr.shape[0] == 0:
        raise RuntimeError("Empty training set: check data_dir and preprocessing")
    print(f"Shapes: train: {Xtr.shape}, pos={ytr.mean():.3f}, test: {Xte.shape}, pos={yte.mean():.3f}")

    # Use the same splitting strategy as training
    val_frac = float(args.val_frac) if args.val_frac is not None else float(cfg.get("val_frac", 0.1))
    Xtr_sub, ytr_sub, Xval, yval = sequence_level_split(Xtr, ytr, gtr, val_frac=val_frac, seed=cfg.get("seed", 0))
    print(f"Train-sub shape: {Xtr_sub.shape}, pos={ytr_sub.mean():.3f} & Val shape: {Xval.shape}, pos={yval.mean():.3f}")

    # Collect logits
    z_val, y_val = collect_logits(params, Xval, yval,
                                  cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"])
    # Choose probability threshold to maximize chosen criterion on validation
    th = best_threshold(z_val, y_val, criterion=args.criterion)

    # Report metrics on train-sub & test at that fixed threshold
    z_trs, y_trs = collect_logits(params, Xtr_sub, ytr_sub,
                                  cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"])
    m_trs = binary_metrics_from_logits(z_trs, y_trs, thresh=th)

    z_te, y_te = collect_logits(params, Xte, yte,
                                cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"])
    m_te = binary_metrics_from_logits(z_te, y_te, thresh=th)

    print(f"chosen threshold (val, max {args.criterion.upper()}): {th:.3f}")
    print(f"train-sub PRF1/acc: P={m_trs['precision']:.3f} R={m_trs['recall']:.3f} F1={m_trs['f1']:.3f} acc={m_trs['acc']:.3f}")
    print(f"test PRF1/acc: P={m_te['precision']:.3f} R={m_te['recall']:.3f} F1={m_te['f1']:.3f} acc={m_te['acc']:.3f}")

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    out_path.write_text(json.dumps({
        "threshold": float(th),
        "criterion": args.criterion,
        "train_sub": m_trs,
        "test": m_te,
    }, indent=2))
    print(f"Saved metrics & threshold to: {out_path}")

    # Save quintiles (optional)
    cfg_dir = out_path.parent
    (cfg_dir).mkdir(parents=True, exist_ok=True)
    np.save(cfg_dir / "z_te.npy", z_te)
    np.save(cfg_dir / "y_te.npy", y_te)

if __name__ == "__main__":
    main()
