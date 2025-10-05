# src/models/train_cnn_numpy.py
import numpy as np
from typing import Dict, Tuple
from src.models.cnn_numpy import forward_numpy
from src.layers.functional_numpy_backward import (
    bce_loss, bce_sigmoid_backward, linear_backward, 
    relu_backward, maxpool3d_backward, 
    conv3d_backward, flatten3d_backward
)

# Pre-allocate
Array = np.ndarray
Params = Dict[str, Array]
Grads  = Dict[str, Array]

# Initialise params
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
    params["lin_w"]   -= lr * grads["lin_w"]
    params["lin_b"]   -= lr * grads["lin_b"]

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
    dLdz = bce_sigmoid_backward(p, y)
    dv, dW_lin, dB_lin = linear_backward(v, lin_w, lin_b, dLdz)
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

# Mini batch generation
def batch_iter(X: Array, y: Array, batch_size: int, shuffle: bool = True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, N, batch_size):
        b = idx[s:s+batch_size]
        yield X[b], y[b]

# Train step
def train_one_epoch(
    params: Params,
    X: Array, y: Array,
    padding: int, stride: int,
    pool_size: int, pool_stride: int,
    lr: float, batch_size: int = 8, weight_decay: float = 0.0) -> Tuple[float, float]:
    # Initialise
    N = X.shape[0]
    running_loss = 0.0
    running_correct = 0
    # SGD on mini batch
    for xb, yb in batch_iter(X, y, batch_size, shuffle=True):
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
                params["lin_w"],   params["lin_b"],
                padding, stride, pool_size, pool_stride, True)
            # Loss
            L = bce_loss(p, y_n)
            # Increment loss for batch
            batch_loss += L
            # accuracy stat
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
        running_loss += batch_loss / xb.shape[0]
    # Statistics
    avg_loss = running_loss / max(1, int(np.ceil(N / batch_size)))
    avg_acc  = running_correct / N
    return float(avg_loss), float(avg_acc)

# Fit loop
def fit(
    params: Params,
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
