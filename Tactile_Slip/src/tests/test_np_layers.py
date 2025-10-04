# tests/test_np_layers.py
import numpy as np
from src.layers.functional_numpy import pad2d, pad3d, relu2d, relu3d, maxpool2d, maxpool3d, conv2d, conv3d

# Test data image (2D)
I = np.array([
    [1,1,2,2,0],
    [1,2,2,3,0],
    [0,1,2,3,1],
    [0,1,1,2,1],
    [0,0,1,1,1]
], dtype=float)
print("2D image, I: ")
print(I)
print(I.shape)

# Kernel filter (1 channel)
K = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
], dtype=float)
print("Kernel filter, K: ")
print(K)

# Pad
B = pad2d(I, 1)
print("Padded data matrix, B: ")
print(B)
print(B.shape)

# Convolution
C = conv2d(B, K, 0.0, 1, 0)
print("Convolution of I and K, C: ")
print(C)
print(C.shape)

# ReLu 
D = relu2d(C)
print("ReLU on C, D: ")
print(D)
print(D.shape)

# Max pooling
E = maxpool2d(D, 2, 2)
print("Max pooling on D, E: ")
print(E)
print(E.shape)

# 3D test image
X = np.stack([
    np.array([
        [1,1,2,2,0],
        [1,2,2,3,0],
        [0,1,2,3,1],
        [0,1,1,2,1],
        [0,0,1,1,1]
    ], dtype=float),
    np.array([
        [0,1,1,0,0],
        [1,2,2,1,0],
        [1,1,2,2,1],
        [0,1,1,1,0],
        [0,0,1,1,0]
    ], dtype=float)
], axis=0)
print(X)
print("3D image X shape:", X.shape)

# 2 channel kernel 
K1 = np.array([
    [[-1,0,1],
     [-1,0,1],
     [-1,0,1]],
    [[-1,0,1],
     [-1,0,1],
     [-1,0,1]]
], dtype=float)
K2 = np.array([
    [[-1,-1,-1],
     [ 0, 0, 0],
     [ 1, 1, 1]],
    [[-1,-1,-1],
     [ 0, 0, 0],
     [ 1, 1, 1]]
], dtype=float)
W = np.stack([K1, K2], axis=0)
print("Shape of kernel", W.shape)

# Convolution (stride length = 2)
Y = conv3d(X, W, None, 2, 1)
print(Y)
print("2-channel convolution: ", Y)
print("Convolution shape: ", Y.shape)

# ReLU (3D)
Z = relu3d(Y)
print(Z)
print("Shape after ReLU: ", Z.shape)

# Max Pooling (3D)
M = maxpool3d(Z, 2, 2)
print(M)
print("Shape after max pooling: ", M.shape)