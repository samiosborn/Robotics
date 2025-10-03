# src/layers/functional_numpy.py
import numpy as np

# Output dimensions (H, W) from 2D convolution
def out_dim_hw(H_in: int, W_in: int, k: int, stride: int = 1, padding: int = 0):
    H_out = np.floor(((H_in + 2*padding - k) / stride) + 1)
    W_out = np.floor(((W_in + 2*padding - k) / stride) + 1)
    return int(H_out), int(W_out)

# Pad a numpy image
def pad(image: np.ndarray, padding: int):
    return np.pad(image, ((padding, padding), (padding, padding)))

# Apply rectified linear unit (ReLU) on a 2D image
def relu(A: np.ndarray):
    y, x = A.shape
    B = np.zeros((y, x))
    for i in range(y): 
        for j in range(x):
            B[i,j] = np.max(0, A[i,j])
    return B

# Max pooling (square) on a 2D image
def maxpool(A: np.ndarray, pool_size: int = 2, pool_stride: int = 2):
    H, W = A.shape
    H_out, W_out = out_dim_hw(H, W, pool_size, pool_stride, 0)
    B = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            C = A[i*pool_stride:i*pool_stride+pool_size, j*pool_stride:j*pool_stride+pool_size]
            B[i, j] = np.max(C)
    return B

# Convolution in 2D for a single-channel image (stride = 1)
def conv(image: np.ndarray, kernel: np.ndarray, bias: float = 0.0, stride: int = 1, padding: int = 0):
    # Add padding
    if padding > 0:
        image = pad(image, padding)
    # Dimensions
    H_in, W_in = image.shape
    k, _ = kernel.shape
    H_out, W_out = out_dim_hw(H_in, W_in, k, stride, 0)
    # Convolution
    out = np.zeros((H_out, W_out), dtype = float)
    for i in range(H_out):
        for j in range(W_out):
            patch = image[i:i+k, j:j+k]
            out[i, j] = np.sum(patch * kernel) + bias
    return out