# src/tests/test_np_backprop.py
import numpy as np
from src.layers.functional_numpy import maxpool2d, conv2d, maxpool3d, conv3d
from src.layers.functional_numpy_backward import conv2d_backward, conv3d_backward, maxpool2d_backward, maxpool3d_backward

# Test derivative of Max Pooling (2D)
def test_maxpool2d_backwards():
    # Test data
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 6)).astype(np.float64)

    # Hyperparams
    pool_size = 2
    pool_stride = 2

    # Forward and scalar loss
    y = maxpool2d(x, pool_size, pool_stride)
    L = np.sum(y)
    grad_out = np.ones_like(y)

    # Analytic grad
    g_analytic = maxpool2d_backward(x, pool_size, pool_stride, grad_out)

    # Numerical grad
    eps = 1e-6
    g_num = np.zeros_like(x)
    for p in range(x.shape[0]):
        for q in range(x.shape[1]):
            xp = x.copy(); xp[p, q] += eps
            xm = x.copy(); xm[p, q] -= eps
            Lp = np.sum(maxpool2d(xp, pool_size, pool_stride))
            Lm = np.sum(maxpool2d(xm, pool_size, pool_stride))
            g_num[p, q] = (Lp - Lm) / (2 * eps)

    diff = np.max(np.abs(g_analytic - g_num))
    print("maxpool2d_backward max abs diff:", diff)
    assert diff < 1e-5

# Test derivative of Convolution (2D)
def test_conv2d_backwards():
    # Assumptions
    rng = np.random.default_rng(1)
    H, W, k = 7, 8, 3
    x = rng.normal(size=(H, W)).astype(np.float64)
    K = rng.normal(size=(k, k)).astype(np.float64)
    b = 0.2
    stride = 2
    padding = 1

    # Forward
    y = conv2d(x, K, b, stride, padding)
    # Loss
    L = np.sum(y)
    # Pre allocate
    grad_out = np.ones_like(y)

    # Analytic grads
    gx, gK, gb = conv2d_backward(x, K, grad_out, stride, padding)

    # Numerical grads
    eps = 1e-6

    # w.r.t. x
    gx_num = np.zeros_like(x)
    for p in range(H):
        for q in range(W):
            xp = x.copy(); xp[p, q] += eps
            xm = x.copy(); xm[p, q] -= eps
            Lp = np.sum(conv2d(xp, K, b, stride, padding))
            Lm = np.sum(conv2d(xm, K, b, stride, padding))
            gx_num[p, q] = (Lp - Lm) / (2 * eps)

    # w.r.t. K
    gK_num = np.zeros_like(K)
    for u in range(k):
        for v in range(k):
            Kp = K.copy(); Kp[u, v] += eps
            Km = K.copy(); Km[u, v] -= eps
            Lp = np.sum(conv2d(x, Kp, b, stride, padding))
            Lm = np.sum(conv2d(x, Km, b, stride, padding))
            gK_num[u, v] = (Lp - Lm) / (2 * eps)

    # w.r.t. b
    Lp = np.sum(conv2d(x, K, b + eps, stride, padding))
    Lm = np.sum(conv2d(x, K, b - eps, stride, padding))
    gb_num = (Lp - Lm) / (2 * eps)

    dx_diff = np.max(np.abs(gx - gx_num))
    dK_diff = np.max(np.abs(gK - gK_num))
    db_diff = abs(gb - gb_num)

    print("conv2d_backward max|dx diff|:", dx_diff)
    print("conv2d_backward max|dK diff|:", dK_diff)
    print("conv2d_backward |db diff|:", db_diff)

    assert dx_diff < 1e-5
    assert dK_diff < 1e-5
    assert db_diff < 1e-7

# Test derivative of Max Pooling (3D)
def test_maxpool3d_backwards():
    # Assumptions
    rng = np.random.default_rng(3)
    C, H, W = 2, 6, 6
    x = rng.normal(size=(C, H, W)).astype(np.float64)
    pool_size, pool_stride = 2, 2

    # Forward
    y = maxpool3d(x, pool_size, pool_stride)
    L = np.sum(y)
    grad_out = np.ones_like(y)

    # Analytic grad
    g_analytic = maxpool3d_backward(x, pool_size, pool_stride, grad_out)

    # Numeric grad
    eps = 1e-6
    g_num = np.zeros_like(x)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                xp = x.copy(); xp[c, i, j] += eps
                xm = x.copy(); xm[c, i, j] -= eps
                Lp = np.sum(maxpool3d(xp, pool_size, pool_stride))
                Lm = np.sum(maxpool3d(xm, pool_size, pool_stride))
                g_num[c, i, j] = (Lp - Lm) / (2*eps)

    diff = np.max(np.abs(g_analytic - g_num))
    print("maxpool3d_backward max abs diff:", diff)
    assert diff < 1e-5

# Test derivative of Convolution (3D)
def test_conv3d_backwards():
    # Assumptions
    rng = np.random.default_rng(2)
    C_in, H, W = 2, 6, 7
    C_out, k = 3, 3
    x = rng.normal(size=(C_in, H, W)).astype(np.float64)
    Wk = rng.normal(size=(C_out, C_in, k, k)).astype(np.float64)
    b  = rng.normal(size=(C_out,)).astype(np.float64)
    stride, padding = 2, 1

    # forward & loss
    y = conv3d(x, Wk, b, stride, padding)
    L = np.sum(y)
    grad_out = np.ones_like(y)

    # analytic grads
    gx, gW, gb = conv3d_backward(x, Wk, grad_out, stride, padding)

    # numeric grads
    eps = 1e-6

    # w.r.t x
    gx_num = np.zeros_like(x)
    for c in range(C_in):
        for i in range(H):
            for j in range(W):
                xp = x.copy(); xp[c, i, j] += eps
                xm = x.copy(); xm[c, i, j] -= eps
                Lp = np.sum(conv3d(xp, Wk, b, stride, padding))
                Lm = np.sum(conv3d(xm, Wk, b, stride, padding))
                gx_num[c, i, j] = (Lp - Lm) / (2*eps)

    # w.r.t Wk
    gW_num = np.zeros_like(Wk)
    for co in range(C_out):
        for ci in range(C_in):
            for u in range(k):
                for v in range(k):
                    Wp = Wk.copy(); Wp[co, ci, u, v] += eps
                    Wm = Wk.copy(); Wm[co, ci, u, v] -= eps
                    Lp = np.sum(conv3d(x, Wp, b, stride, padding))
                    Lm = np.sum(conv3d(x, Wm, b, stride, padding))
                    gW_num[co, ci, u, v] = (Lp - Lm) / (2*eps)

    # w.r.t b
    gb_num = np.zeros_like(b)
    for co in range(C_out):
        bp = b.copy(); bp[co] += eps
        bm = b.copy(); bm[co] -= eps
        Lp = np.sum(conv3d(x, Wk, bp, stride, padding))
        Lm = np.sum(conv3d(x, Wk, bm, stride, padding))
        gb_num[co] = (Lp - Lm) / (2*eps)

    dx_diff = np.max(np.abs(gx - gx_num))
    dW_diff = np.max(np.abs(gW - gW_num))
    db_diff = np.max(np.abs(gb - gb_num))
    print("conv3d_backward max|dx diff|:", dx_diff)
    print("conv3d_backward max|dW diff|:", dW_diff)
    print("conv3d_backward max|db diff|:", db_diff)

    assert dx_diff < 1e-5
    assert dW_diff < 1e-5
    assert db_diff < 1e-7


if __name__ == "__main__":
    test_maxpool2d_backwards()
    test_conv2d_backwards()
    test_maxpool3d_backwards()
    test_conv3d_backwards()