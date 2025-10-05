# tests/test_np_cnn.py
import numpy as np
from src.models.cnn_numpy import forward_numpy
from src.layers.functional_numpy import out_dim_hw

def test_forward_cnn_numpy():
    rng = np.random.default_rng(0)

    # Input
    C_in, H, W = 2, 5, 5
    x = rng.normal(size=(C_in, H, W)).astype(np.float64)

    # Hyperparams
    k = 3
    padding = 1
    stride = 1
    pool_size = 2
    pool_stride = 2

    # Conv1
    C1 = 4
    Hc1, Wc1 = out_dim_hw(H, W, k, stride, padding)
    Hp1, Wp1 = out_dim_hw(Hc1, Wc1, pool_size, pool_stride, 0)
    conv1_w = rng.normal(size=(C1, C_in, k, k)).astype(np.float64)
    conv1_b = rng.normal(size=(C1,)).astype(np.float64)

    # Conv2
    C2 = 8
    Hc2, Wc2 = out_dim_hw(Hp1, Wp1, k, stride, padding)
    Hp2, Wp2 = out_dim_hw(Hc2, Wc2, pool_size, pool_stride, 0)
    conv2_w = rng.normal(size=(C2, C1, k, k)).astype(np.float64)
    conv2_b = rng.normal(size=(C2,)).astype(np.float64)

    # Linear head
    in_dim = C2 * Hp2 * Wp2
    lin_w  = rng.normal(size=(1, in_dim)).astype(np.float64)
    lin_b  = rng.normal(size=(1,)).astype(np.float64)

    # Forward pass
    p = forward_numpy(
        x,
        conv1_w, conv1_b,
        conv2_w, conv2_b,
        lin_w, lin_b,
        padding, stride,
        pool_size, pool_stride,
    )

    print("Output p:", p, "shape:", p.shape)
    assert p.shape == (1,), f"Expected (1,), got {p.shape}"
    assert np.all((p >= 0) & (p <= 1)), "Sigmoid output must be in [0,1]"

if __name__ == "__main__":
    test_forward_cnn_numpy()
