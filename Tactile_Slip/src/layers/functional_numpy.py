# src/layers/functional_numpy.py
import numpy as np

# Output dimensions (H, W) from 2D convolution
def out_dim_hw(H_in: int, W_in: int, k: int, stride: int = 1, padding: int = 0):
    H_out = (H_in + 2*padding - k) // stride + 1
    W_out = (W_in + 2*padding - k) // stride + 1
    return int(H_out), int(W_out)

# Pad a 2D numpy image
def pad2d(image: np.ndarray, padding: int):
    return np.pad(image, ((padding, padding), (padding, padding)))

# Pad a 3D numpy image
def pad3d(image: np.ndarray, padding: int):
    return np.pad(image, ((0,0), (padding, padding), (padding, padding)))

# Apply rectified linear unit (ReLU) on a 2D image
def relu2d(image: np.ndarray):
    H, W = image.shape
    out = np.zeros((H, W))
    for i in range(H): 
        for j in range(W):
            out[i,j] = max(0, image[i,j])
    return out

# Apply rectified linear unit (ReLU) on a 3D image
def relu3d(image: np.ndarray):
    C, H, W = image.shape
    out = np.zeros((C, H, W))
    for c in range(C):
        for i in range(H): 
            for j in range(W):
                out[c, i, j] = max(0, image[c, i, j])
    return out

# Max pooling (square) on a 2D image
def maxpool2d(image: np.ndarray, pool_size: int = 2, pool_stride: int = 2):
    H_in, W_in = image.shape
    H_out, W_out = out_dim_hw(H_in, W_in, pool_size, pool_stride, 0)
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = image[i*pool_stride:i*pool_stride+pool_size, j*pool_stride:j*pool_stride+pool_size]
            out[i, j] = np.max(patch)
    return out

# Max pooling (square) on a 3D image
def maxpool3d(image: np.ndarray, pool_size: int = 2, pool_stride: int = 2):
    C, H_in, W_in = image.shape
    H_out, W_out = out_dim_hw(H_in, W_in, pool_size, pool_stride, 0)
    out = np.zeros((C, H_out, W_out))
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                patch = image[c, i*pool_stride:i*pool_stride+pool_size, j*pool_stride:j*pool_stride+pool_size]
                out[c, i, j] = np.max(patch)
    return out

# Convolution in 2D for a single-channel image
def conv2d(image: np.ndarray, kernel: np.ndarray, bias: float = 0.0, stride: int = 1, padding: int = 0):
    # Add padding
    if padding > 0:
        image = pad2d(image, padding)
    # Dimensions
    H_in, W_in = image.shape
    k, _ = kernel.shape
    H_out, W_out = out_dim_hw(H_in, W_in, k, stride, 0)
    # Convolution
    out = np.zeros((H_out, W_out), dtype = float)
    for i in range(H_out):
        i_step = i * stride
        for j in range(W_out):
            j_step = j * stride
            patch = image[i_step:i_step+k, j_step:j_step+k]
            out[i, j] = np.sum(patch * kernel) + bias
    return out

# Convolution in 3D for multi-channel
def conv3d(image: np.ndarray, kernel: np.ndarray, bias: np.ndarray = None, stride: int = 1, padding: int = 0):
    # Padding
    if padding > 0: 
        image = pad3d(image, padding)
    # Dimensions
    C_in_img, H_in, W_in = image.shape
    C_out, C_in, k, _ = kernel.shape
    H_out, W_out = out_dim_hw(H_in, W_in, k, stride, 0)
    out = np.zeros((C_out, H_out, W_out), dtype = float)
    # Assertions
    assert C_in == C_in_img, "kernel C_in must match input image C_in"
    assert bias is None or bias.shape == (C_out,)
    # Per channel
    for c in range(C_out):
        # Convolution
        filt = kernel[c]
        b = 0.0 if bias is None else (bias[c])
        for i in range(H_out):
            i_step = i * stride
            for j in range(W_out):
                j_step = j * stride
                patch = image[:, i_step:i_step+k, j_step:j_step+k]
                out[c, i, j] = np.sum(patch * filt) + b
    return out

# Flatten 3D image to 1D
def flatten3d(image: np.ndarray):
    # Dimensions
    C, H, W = image.shape
    # Pre allocate
    out = np.empty(C * H * W, dtype=image.dtype)
    # Loop
    idx = 0
    for c in range(C):
        for i in range(H):
            for j in range(W):
                out[idx] = image[c, i, j]
                idx += 1
    return out

# Linear transformation
def linear(A: np.ndarray, W: np.ndarray, b: np.ndarray):
    return W @ A + b

# Sigmoid activation
def sigmoid(Z: np.ndarray):
    return 1.0 / (1.0 + np.exp(-Z))
