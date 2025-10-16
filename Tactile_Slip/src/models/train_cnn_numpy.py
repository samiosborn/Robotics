# src/models/train_cnn_numpy.py
import numpy as np
from typing import Dict, Tuple
from src.layers.functional_numpy import out_dim_hw
from src.models.cnn_numpy import forward_numpy
from src.layers.functional_numpy_backward import (
    bce_with_logits_loss, bce_with_logits_backward, 
    linear_backward, relu_backward, maxpool3d_backward,
    conv3d_backward, flatten3d_backward
)

# Pre-allocate
Array = np.ndarray
Params = Dict[str, Array]
Grads  = Dict[str, Array]

# He initialisation for convolution layers (with RELU)
def he_init_conv(rng: np.random.Generator, shape: Tuple[int, ...]):
    # Shape = (C_out, C_in, k, k)
    fan_in = shape[1] * shape[2] * shape[3]
    # Standard deviation
    std = np.sqrt(2.0 / fan_in)
    # Gaussian init
    return rng.normal(0.0, std, size = shape).astype(np.float64)

# Xavier initialisation for linear head
def xavier_init_linear(rng: np.random.Generator, out_dim: int, in_dim: int):
    # Standard deviation
    std = np.sqrt(2.0 / (in_dim + out_dim))
    # Gaussian init
    return rng.normal(0.0, std, size = (out_dim, in_dim)).astype(np.float64)

# Initialise parameters
def init_params(
    rng: np.random.Generator,
    C_in: int, H: int, W: int,
    k: int = 3, C1: int = 8, C2: int = 8,
    stride: int = 1, padding: int = 1,
    pool_size: int = 2, pool_stride: int = 2) -> Params:
    # Block 1
    Hc1, Wc1 = out_dim_hw(H, W, k, stride, padding)
    Hp1, Wp1 = out_dim_hw(Hc1, Wc1, pool_size, pool_stride, 0)
    # Block 2
    Hc2, Wc2 = out_dim_hw(Hp1, Wp1, k, stride, padding)
    Hp2, Wp2 = out_dim_hw(Hc2, Wc2, pool_size, pool_stride, 0)
    in_dim = C2 * Hp2 * Wp2
    # Learned params
    return {
        "conv1_w": he_init_conv(rng, (C1, C_in, k, k)),
        "conv1_b": np.zeros((C1,), dtype=np.float64),
        "conv2_w": he_init_conv(rng, (C2, C1, k, k)),
        "conv2_b": np.zeros((C2,), dtype=np.float64),
        "lin_w": xavier_init_linear(rng, 1, in_dim),
        "lin_b": np.zeros((1,), dtype=np.float64),
    }

# Per-batch grad accumulator with zero arrays matching param shapes
def zero_like_params(params: Params) -> Grads:
    return {
        "conv1_w": np.zeros_like(params["conv1_w"]),
        "conv1_b": np.zeros_like(params["conv1_b"]),
        "conv2_w": np.zeros_like(params["conv2_w"]),
        "conv2_b": np.zeros_like(params["conv2_b"]),
        "lin_w":   np.zeros_like(params["lin_w"]),
        "lin_b":   np.zeros_like(params["lin_b"]),
    }

# SGD pptimiser with weight decay
def sgd_update(params: Params, grads: Grads, lr: float, weight_decay: float = 0.0):
    # Weight decay
    if weight_decay != 0.0:
        grads["conv1_w"] = grads["conv1_w"] + weight_decay * params["conv1_w"]
        grads["conv2_w"] = grads["conv2_w"] + weight_decay * params["conv2_w"]
        grads["lin_w"] = grads["lin_w"] + weight_decay * params["lin_w"]
    # Update params
    params["conv1_w"] -= lr * grads["conv1_w"]
    params["conv1_b"] -= lr * grads["conv1_b"]
    params["conv2_w"] -= lr * grads["conv2_w"]
    params["conv2_b"] -= lr * grads["conv2_b"]
    params["lin_w"] -= lr * grads["lin_w"]
    params["lin_b"] -= lr * grads["lin_b"]

# Backward (with cache)
def backward_from_cache(cache: dict, y: np.ndarray):
    # Unpack
    x = cache["x"]
    conv1_w, conv1_b = cache["conv1_w"], cache["conv1_b"]
    conv2_w, conv2_b = cache["conv2_w"], cache["conv2_b"]
    lin_w, lin_b = cache["lin_w"], cache["lin_b"]
    padding = cache["padding"]
    stride = cache["stride"]
    pool_size = cache["pool_size"]
    pool_stride = cache["pool_stride"]
    z1, a1, p1 = cache["z1"], cache["a1"], cache["p1"]
    z2, a2, p2 = cache["z2"], cache["a2"], cache["p2"]
    v, z, p = cache["v"], cache["z"], cache["p"]
    C2, Hp2, Wp2 = cache["C2Hp2Wp2"] 
    # Head
    dLdz = bce_with_logits_backward(cache["z"], y)
    dv, dW_lin, dB_lin = linear_backward(cache["v"], lin_w, lin_b, dLdz)
    # Unflatten
    dP2 = flatten3d_backward(dv, C2, Hp2, Wp2)
    # Block 2
    dA2 = maxpool3d_backward(a2, pool_size, pool_stride, dP2)
    dZ2 = relu_backward(z2, dA2)
    dP1, dW_conv2, dB_conv2 = conv3d_backward(p1, conv2_w, dZ2, stride, padding)
    # Block 1
    dA1 = maxpool3d_backward(a1, pool_size, pool_stride, dP1)
    dZ1 = relu_backward(z1, dA1)
    dX, dW_conv1, dB_conv1 = conv3d_backward(x, conv1_w, dZ1, stride, padding)
    # Gradients
    grads = {
        "conv1_w": dW_conv1, "conv1_b": dB_conv1,
        "conv2_w": dW_conv2, "conv2_b": dB_conv2,
        "lin_w": dW_lin, "lin_b": dB_lin,
    }
    return grads

# Mini batch data loader
def batch_iter(X: Array, y: Array, batch_size: int, shuffle: bool = True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, N, batch_size):
        b = idx[s:s+batch_size]
        yield X[b], y[b]

# Train one epoch
def train_one_epoch(
    params: Params,
    X: Array, y: Array,
    padding: int, stride: int,
    pool_size: int, pool_stride: int,
    lr: float, batch_size: int = 8, weight_decay: float = 0.0) -> Tuple[float, float]:
    # Initialise
    N = X.shape[0]
    total_loss = 0.0
    running_correct = 0
    running_loss = 0.0
    # SGD on mini batch
    for batch_idx, (xb, yb) in enumerate(batch_iter(X, y, batch_size, shuffle=True), start=1):
        # Accumulate grads across the batch
        acc_grads = zero_like_params(params)
        batch_loss = 0.0
        # Loop through each datapiece
        for n in range(xb.shape[0]):
            x_n = xb[n] 
            y_n = np.array([yb[n]], dtype=np.float64)
            # Forward with cache
            p, cache = forward_numpy(
                x_n,
                params["conv1_w"], params["conv1_b"],
                params["conv2_w"], params["conv2_b"],
                params["lin_w"], params["lin_b"],
                padding, stride, pool_size, pool_stride, True)
            # Loss
            L = bce_with_logits_loss(cache["z"], y_n)
            # Increment loss for batch
            batch_loss += L
            # Accuracy stat
            pred = 1.0 if p[0] >= 0.5 else 0.0
            running_correct += int(pred == y_n[0])
            # Backpropagation
            grads_n = backward_from_cache(cache, y_n)
            # Accumulate grads
            for k in ("conv1_w","conv1_b","conv2_w","conv2_b","lin_w","lin_b"):
                acc_grads[k] += grads_n[k]
        # Average grads across the batch
        for k in ("conv1_w","conv1_b","conv2_w","conv2_b","lin_w","lin_b"):
            acc_grads[k] /= xb.shape[0]
        # Update step
        sgd_update(params, acc_grads, lr=lr, weight_decay=weight_decay)
        # Accumulate loss
        total_loss += batch_loss
        # Track avg-per-batch for display
        running_loss += (batch_loss / xb.shape[0])
        if batch_idx % 20 == 0:
            print(f"batch {batch_idx}: running_loss ~ {running_loss / batch_idx:.4f}")
    # Statistics
    avg_loss = total_loss / N
    avg_acc  = running_correct / N
    return float(avg_loss), float(avg_acc)

# Fit loop
def fit(params: Params,
    X: Array, y: Array,
    padding: int, stride: int, pool_size: int, pool_stride: int,
    lr: float, weight_decay: float,
    batch_size: int, epochs: int, verbose: bool = False) -> Params:
    # Run for several epochs
    for epoch in range(1, epochs+1):
        loss, acc = train_one_epoch(
            params, X, y, padding, stride, pool_size, pool_stride,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay
        )
        if verbose:
            print(f"Epoch number {epoch:02d}, loss: {loss:.4f}, accuracy: {acc:.3f}")
    return params
