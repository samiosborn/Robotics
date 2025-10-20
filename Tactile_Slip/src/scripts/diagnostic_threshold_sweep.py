# src/scrips/diagnostic_threshold_sweep.py
import numpy as np, json, argparse
from pathlib import Path
from src.models.metrics_numpy import binary_metrics_from_logits

ap = argparse.ArgumentParser()
ap.add_argument("--z", type=str, required=True)   # npy of logits on test
ap.add_argument("--y", type=str, required=True)   # npy of labels on test
args = ap.parse_args()

z = np.load(args.z); y = np.load(args.y)
ths = np.linspace(0.05, 0.95, 19)
for th in ths:
    m = binary_metrics_from_logits(z, y, th)
    print(f"th={th:.2f}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} acc={m['acc']:.3f}")
