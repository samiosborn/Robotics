# src/models/metrics_numpy.py
import numpy as np

# Numerically stable sigmoid for metrics (tanh form)
def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))

# Compute precision/recall/F1/acc from raw logits and labels at a given threshold
def binary_metrics_from_logits(z_list, y_list, thresh: float = 0.5):

    z = np.asarray(z_list).reshape(-1)
    y = np.asarray(y_list).reshape(-1)
    p = stable_sigmoid(z)
    yhat = (p >= thresh).astype(np.float64)

    tp = np.sum((yhat == 1) & (y == 1))
    tn = np.sum((yhat == 0) & (y == 0))
    fp = np.sum((yhat == 1) & (y == 0))
    fn = np.sum((yhat == 0) & (y == 1))

    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "acc": float(acc),
    }

# Sweep thresholds in [0.01,0.99] and pick the one that maximizes the criterion on (z,y)
def best_threshold(z, y, criterion: str = "f1") -> float:
    ths = np.linspace(0.01, 0.99, 99)
    best, best_th = -1.0, 0.5
    for th in ths:
        m = binary_metrics_from_logits(z, y, th)
        score = m[criterion]
        if score > best:
            best, best_th = score, th
    return best_th
