# src/layers/functional_numpy_backward.py
import numpy as np
from src.layers.functional_numpy import pad2d, pad3d

# Binary cross entropy (BCE) loss function (L) given probability p
def bce_loss(p: np.ndarray, y: np.ndarray):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(y*np.log(p) + (1-y)*np.log(1-p))

# BCE loss (L) from logits input to sigmoid z (weighted)
def bce_with_logits_loss(z: np.ndarray, y: np.ndarray, pos_weight: float = 1.0):
    a = np.maximum(z, 0.0)
    sp = np.log1p(np.exp(-np.abs(z))) + a
    # standard BCE: L = sp - y*z
    # weighted BCE: L = (1-y)*sp + y*pos_weight*(sp - z)
    loss = (1 - y) * sp + y * pos_weight * (sp - z)
    return float(loss)

# Derivative of BCE loss L w.r.t. logits input to sigmoid z (dL/dz) from BCE loss
def bce_sigmoid_backward(p: np.ndarray, y: np.ndarray):
    return (p - y)

# Derivative of BCE loss L w.r.t. logits input to sigmoid z (dL/dz) from logits
def bce_with_logits_backward(z: np.ndarray, y: np.ndarray):
    # dL/dz = sigmoid(z) - y
    # Stable sigmoid via tanh: sigma(z) = 0.5 * (1 + tanh(z/2))
    p = 0.5 * (1.0 + np.tanh(0.5 * z))
    return (p - y)

# Derivatives from linear function (z = W @ x + b)
def linear_backward(x: np.ndarray, W: np.ndarray, b: np.ndarray, dLdz: np.ndarray):
    # Single example
    if x.ndim == 1:
        dLdb = dLdz
        dLdW = dLdz[:, None] * x[None, :]
        dLdx = W.T @ dLdz
        return dLdx, dLdW, dLdb
    else:
        # dLdb = sum_b dLdz[b]
        dLdb = dLdz.sum(axis=0)
        # dLdW = sum_b dLdz[b]^T x[b] == dLdz^T @ x
        dLdW = dLdz.T @ x
        # dLdx = dLdz @ W
        dLdx = dLdz @ W
        return dLdx, dLdW, dLdb

# Derivative of ReLU
def relu_backward(x: np.ndarray, dLdx: np.ndarray):
    return dLdx * (x > 0)

# Derivative of flatten (3D to 1D)
def flatten3d_backward(grad_out: np.ndarray, C: int, H: int, W: int):
    assert grad_out.size == C*H*W
    return grad_out.reshape(C, H, W)

# Derivative of Max Pooling (single-channel, 2D)
def maxpool2d_backward(x: np.ndarray, pool_size: int, pool_stride: int, grad_out: np.ndarray):
    # Dimensions 
    H_in, W_in = x.shape
    H_out, W_out = grad_out.shape
    # Preallocate
    grad_in = np.zeros((H_in, W_in))
    # Loop over x
    for i in range(H_out):
        i_step = i * pool_stride
        for j in range(W_out):
            j_step = j * pool_stride
            window = x[i_step:i_step+pool_size, j_step:j_step+pool_size]
            a, b = np.unravel_index(np.argmax(window), window.shape)
            grad_in[i_step+a, j_step+b] += grad_out[i, j]
    return grad_in

# Derivative of Convolution (single-channel)
def conv2d_backward(image: np.ndarray, kernel: np.ndarray, grad_out: np.ndarray, stride: int, padding: int):
    # Input dimensions 
    H_in, W_in = image.shape
    k, _ = kernel.shape
    # Pad input to mirror forward
    if padding > 0:
        x = pad2d(image, padding)
    else:
        x = image
    # Output dimensions
    H_out, W_out = grad_out.shape
    H_x, W_x = x.shape
    # Kernel bias
    dLdkb = float(np.sum(grad_out))
    # Preallocate
    dLdk = np.zeros((k, k))
    dLdx = np.zeros((H_x, W_x))
    # Accumulate gradients
    for i in range(H_out):
        i_step = i * stride
        for j in range(W_out):
            j_step = j * stride
            patch = x[i_step:i_step+k, j_step:j_step+k]
            dLdk += grad_out[i, j] * patch
            dLdx[i_step:i_step+k, j_step:j_step+k] += grad_out[i, j] * kernel
    # Unpad dLdx
    if padding > 0:
        dLdimg = dLdx[padding:padding+H_in, padding:padding+W_in]
    else: 
        dLdimg = dLdx
    return dLdimg, dLdk, dLdkb

# Derivative of Max Pooling (multi-channel, 3D)
def maxpool3d_backward(x: np.ndarray, pool_size: int, pool_stride: int, grad_out: np.ndarray):
    # Dimensions
    C, H_in, W_in = x.shape
    _, H_out, W_out = grad_out.shape
    grad_in = np.zeros_like(x, dtype=np.float64)
    # Loop over channels
    for c in range(C):
        # Accumulate gradients
        for i in range(H_out):
            i_index = i * pool_stride
            for j in range(W_out):
                j_index = j * pool_stride
                window = x[c, i_index:i_index+pool_size, j_index:j_index+pool_size]
                a, b = np.unravel_index(np.argmax(window), window.shape)
                grad_in[c, i_index + a, j_index + b] += grad_out[c, i, j]
    return grad_in

# Derivative of Convolution (multi-channel)
def conv3d_backward(image: np.ndarray, kernel: np.ndarray, grad_out: np.ndarray, stride: int, padding: int):
    # Dimensions
    C_in, H_in, W_in = image.shape
    C_out, Ck_in, k, _ = kernel.shape
    H_out, W_out = grad_out.shape[1], grad_out.shape[2]
    # Pad input
    x = pad3d(image, padding) if padding > 0 else image
    _, Hx, Wx = x.shape
    # Pre allocate
    dLdx_pad = np.zeros_like(x, dtype=np.float64)
    dLdW = np.zeros_like(kernel, dtype=np.float64)
    dLdb = np.sum(grad_out, axis=(1, 2)).astype(np.float64)
    # Loop over each channel
    for c in range(C_out):
        # Kernel weights (C_in, k, k)
        Wco = kernel[c]
        # Accumulate gradients
        for i in range(H_out):
            i_index = i * stride
            for j in range(W_out):
                j_index = j * stride
                patch = x[:, i_index:i_index+k, j_index:j_index+k]
                dLdW[c] += grad_out[c, i, j] * patch
                dLdx_pad[:, i_index:i_index+k, j_index:j_index+k] += grad_out[c, i, j] * Wco
    # Unpad gradient
    if padding > 0:
        dLdx = dLdx_pad[:, padding:padding+H_in, padding:padding+W_in]
    else:
        dLdx = dLdx_pad
    return dLdx, dLdW, dLdb

