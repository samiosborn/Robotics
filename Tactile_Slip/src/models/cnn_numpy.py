# src/models/cnn_numpy.py
import numpy as np
from src.layers.functional_numpy import conv3d, relu3d, maxpool3d, flatten3d, linear, sigmoid

# Forward pass
def forward_numpy(x: np.ndarray, 
                   conv1_w: np.ndarray, 
                   conv1_b: np.ndarray, 
                   conv2_w: np.ndarray, 
                   conv2_b: np.ndarray, 
                   lin_w: np.ndarray,
                   lin_b: np.ndarray,
                   padding: int, 
                   stride: int, 
                   pool_size: int, 
                   pool_stride: int, 
                   cache_return: bool = False):
    
    # Convolution 1
    z1 = conv3d(x, conv1_w, conv1_b, stride, padding)
    # ReLU activation 1
    a1 = relu3d(z1)
    # Max pooling 1
    p1 = maxpool3d(a1, pool_size, pool_stride)
    # Convolution 2
    z2 = conv3d(p1, conv2_w, conv2_b, stride, padding)
    # ReLU activation 2
    a2 = relu3d(z2)
    # Max pooling 2
    p2 = maxpool3d(a2, pool_size, pool_stride)
    # Head dimensions
    C2, Hp2, Wp2 = p2.shape
    # Flatten
    v = flatten3d(p2)
    # Linear function
    z = linear(v, lin_w, lin_b)
    # Sigmoid activation
    p = sigmoid(z)
    # Build cache
    if cache_return:    
        cache = {
        # Inputs
        "x": x,
        "conv1_w": conv1_w, "conv1_b": conv1_b,
        "conv2_w": conv2_w, "conv2_b": conv2_b,
        "lin_w": lin_w, "lin_b": lin_b,
        # Hyperparams
        "padding": padding, "stride": stride,
        "pool_size": pool_size, "pool_stride": pool_stride,
        # Tensors per layer
        "z1": z1, "a1": a1, "p1": p1,
        "z2": z2, "a2": a2, "p2": p2,
        "v": v, "z": z, "p": p,
        # Shapes
        "C2Hp2Wp2": (C2, Hp2, Wp2),}
        return p, cache
    else:
        return p