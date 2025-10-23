# src/scripts/eval_tactip_numpy.py
from pathlib import Path
import json
import argparse
import numpy as np
from src.models.io_params import load_params
from src.models.cnn_numpy import forward_numpy
from src.utils.experiment_numpy import load_config, sequential_batch_iter
from src.models.metrics_numpy import binary_metrics_from_logits, best_threshold
from src.utils.experiment_numpy import load_config

# Load (X,y) from a preprocessed NPZ
def load_npz_pair(path: str):
    d = np.load(path)
    return d["X"].astype(np.float64), d["y"].astype(np.float64)

# Run forward (no updates) and collect logits with labels for metrics / threshold tuning
def collect_logits(params, X, y, padding, stride, pool_size, pool_stride, batch_size=256):
    zs, ys = [], []
    processed = 0
    total = X.shape[0]

    for xb, yb in sequential_batch_iter(X, y, batch_size, shuffle=False):
        # Forward pass each item
        for n in range(xb.shape[0]):
            p, cache = forward_numpy(
                xb[n],
                params["conv1_w"], params["conv1_b"],
                params["conv2_w"], params["conv2_b"],
                params["lin_w"],   params["lin_b"],
                padding, stride, pool_size, pool_stride, True)
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

# Print threshold sweep table
def print_sweep_table(title, sweep_rows, split_key="val"):
    # split_key: "val" or "test"
    print(f"\n{title}")
    print("-" * 68)
    print("thr: P_{:3} R_{:3} F1_{:3} Acc_{:3}".format(split_key, split_key, split_key, split_key))
    print("-" * 68)
    for r in sweep_rows:
        m = r[split_key]
        print(f"{r['thr']:0.2f}: {m['precision']:0.3f} {m['recall']:0.3f} {m['f1']:0.3f} {m['acc']:0.3f}")
    print("-" * 68)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config (same as training)")
    ap.add_argument("--params", type=str, required=True, help="Path to saved params .npz")
    ap.add_argument("--criterion", type=str, default="f1", choices=["f1","precision","recall","acc"], help="Metric to optimize threshold")
    ap.add_argument("--out", type=str, default="configs/tactip_metrics.json", help="Where to save metrics + threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    params = load_params(args.params)

    # Load preprocessed NPZs
    pre_dir = Path(cfg.get("preprocessed_dir", "data/preprocessed"))
    Xtr, ytr = load_npz_pair(str(pre_dir / "train_sub.npz"))
    Xva, yva = load_npz_pair(str(pre_dir / "val.npz"))
    Xte, yte = load_npz_pair(str(pre_dir / "test.npz"))
    print(f"Shapes: train-sub: {Xtr.shape}, pos={ytr.mean():.3f}, val: {Xva.shape}, pos={yva.mean():.3f}, test: {Xte.shape}, pos={yte.mean():.3f}")

    # Collect logits
    z_val, y_val = collect_logits(params, Xva, yva, cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"])
    z_trs, y_trs = collect_logits(params, Xtr, ytr, cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"])
    z_te, y_te = collect_logits(params, Xte, yte, cfg["padding"], cfg["stride"], cfg["pool_size"], cfg["pool_stride"])

    # Choose probability threshold to maximize chosen criterion on validation
    th = best_threshold(z_val, y_val, criterion=args.criterion)

    # Report metrics on train-sub & test at that fixed threshold
    m_trs = binary_metrics_from_logits(z_trs, y_trs, thresh=th)
    m_te = binary_metrics_from_logits(z_te, y_te, thresh=th)

    print(f"\nchosen threshold (val, max {args.criterion.upper()}): {th:.3f}")
    print(f"train-sub PRF1/acc: P={m_trs['precision']:.3f} R={m_trs['recall']:.3f} F1={m_trs['f1']:.3f} acc={m_trs['acc']:.3f}")
    print(f"test PRF1/acc: P={m_te['precision']:.3f} R={m_te['recall']:.3f} F1={m_te['f1']:.3f} acc={m_te['acc']:.3f}")

    # Threshold sweep every 0.05 
    sweep_rows = []
    for thr in np.round(np.arange(0.05, 1.00, 0.05), 2):
        mv = binary_metrics_from_logits(z_val, y_val, thresh=thr)
        mt = binary_metrics_from_logits(z_te, y_te, thresh=thr)
        sweep_rows.append({"thr": float(thr), "val": mv, "test": mt})

    print_sweep_table("Threshold sweep on VAL", sweep_rows, split_key="val")
    print_sweep_table("Threshold sweep on TEST", sweep_rows, split_key="test")

    # Save results (+ full sweep) to JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": float(th),
        "criterion": args.criterion,
        "train_sub": m_trs,
        "test": m_te,
        "sweep": sweep_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics & threshold to: {out_path}")

if __name__ == "__main__":
    main()
