#  src/tests/diag_preproc.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_mat_table(path: str):
    m = loadmat(path)
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
            return v.astype(np.float64)
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 2 and np.issubdtype(v.dtype, np.number):
            T, M, _ = v.shape
            v2 = v.reshape(T, M*2)
            return v2.astype(np.float64)
    raise ValueError(f"No 2D numeric array in {path}")

def normalise_coordinate_sequence(seq):
    T, D = seq.shape
    assert D % 2 == 0
    M = D // 2
    xs = seq[:, 0::2]
    ys = seq[:, 1::2]
    xs = xs - xs.mean(axis=0, keepdims=True)
    ys = ys - ys.mean(axis=0, keepdims=True)
    std = np.std(np.concatenate([xs, ys], axis=1))
    if std < 1e-8:
        std = 1.0
    xs = xs / std
    ys = ys / std
    out = np.empty_like(seq)
    out[:, 0::2] = xs
    out[:, 1::2] = ys
    return out, M

def compute_velocities(seq_norm):
    # seq_norm shape (T, 2*M)
    dx = np.diff(seq_norm[:, 0::2], axis=0)
    dy = np.diff(seq_norm[:, 1::2], axis=0)
    speeds = np.sqrt(dx*dx + dy*dy).mean(axis=1)
    return speeds

def label_by_threshold(speeds, k_med=3.0):
    med = np.median(speeds)
    mad = np.median(np.abs(speeds - med))
    thresh = med + k_med * mad
    labels = (speeds >= thresh).astype(int)
    return labels, thresh, med, mad

def rasterise_dummy(seq_norm, out_hw=(48,48), scale=10.0):
    # Dummy raster to visualise sequence motions (just sum positions into heatmap)
    H, W = out_hw
    img = np.zeros((H, W), dtype=np.float32)
    # scale and shift
    xs = seq_norm[:, 0::2]*scale + W/2.0
    ys = seq_norm[:, 1::2]*scale + H/2.0
    for t in range(xs.shape[0]):
        for i in range(xs.shape[1]):
            x = int(xs[t, i])
            y = int(ys[t, i])
            if 0 <= x < W and 0 <= y < H:
                img[y, x] += 1
    return img

def run_diagnostics(base_dir, split="train", num_examples=5):
    paths = sorted(glob.glob(os.path.join(base_dir, split, "**", "*.mat"), recursive=True))
    print(f"Found {len(paths)} sequences in {split}")
    all_speeds = []
    all_labels = []
    for idx, p in enumerate(paths[:num_examples]):
        print("\nSequence", idx, p)
        seq = load_mat_table(p)
        seqn, M = normalise_coordinate_sequence(seq)
        speeds = compute_velocities(seqn)
        labels, thresh, med, mad = label_by_threshold(speeds)
        print(f"Median speed={med:.4f}, MAD={mad:.4f}, threshold={thresh:.4f}")
        print(f"Label counts: {np.bincount(labels)} (0=non-slip,1=slip)")
        all_speeds.append(speeds)
        all_labels.append(labels)
        # Plot speeds vs labels
        plt.figure(figsize=(8,4))
        plt.plot(speeds, label="mean pin speed")
        plt.scatter(np.where(labels==1)[0], speeds[labels==1], color='red', label="label=slip")
        plt.axhline(thresh, color='black', linestyle='--', label="threshold")
        plt.legend()
        plt.title(os.path.basename(p))
        plt.xlabel("time step")
        plt.ylabel("speed")
        plt.show()
        # Show rasterised image of first and last frame
        img0 = rasterise_dummy(seqn, out_hw=(48,48))
        plt.figure()
        plt.imshow(img0, cmap='hot')
        plt.title("Rasterised (dummy) sequence heatmap")
        plt.colorbar()
        plt.show()
    # Concatenate all and show distribution
    all_speeds_concat = np.concatenate(all_speeds)
    plt.figure()
    plt.hist(all_speeds_concat, bins=50)
    plt.title("Histogram of mean pin speeds (all example sequences)")
    plt.xlabel("speed")
    plt.show()
    # Print overall slip ratio
    all_labels_concat = np.concatenate(all_labels)
    print(f"Overall slip ratio in these {num_examples} sequences: {all_labels_concat.mean():.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for .mat files (train/test split folders)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()
    run_diagnostics(args.base_dir, split=args.split, num_examples=args.num_examples)
