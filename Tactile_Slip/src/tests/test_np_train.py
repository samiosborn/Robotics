# src/tests/test_np_train.py
import numpy as np
from src.models.train_cnn_numpy import init_params, fit, train_one_epoch

# Mini dataset
def make_toy_patches(N=64, C=1, H=16, W=16, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.05, size=(N, C, H, W)).astype(np.float64)
    y = np.zeros((N,), dtype=np.float64)
    for n in range(N):
        if n % 2 == 0:
            y[n] = 1.0
            r0, c0 = H//2 - 1, W//2 - 1
        else:
            r0, c0 = 1, 1
        X[n, 0, r0:r0+3, c0:c0+3] += 1.0
    # Normalise
    X = (X - X.mean()) / (X.std() + 1e-8)
    return X, y

# Test if training reduces loss and increases accuracy
def test_training_loop_improves_loss_and_acc():
    rng = np.random.default_rng(123)
    X, y = make_toy_patches(N=64, C=1, H=16, W=16, seed=1)

    padding, stride = 1, 1
    pool_size, pool_stride = 2, 2

    # Initialise parameters
    params = init_params(rng, C_in=1, H=16, W=16,
                         k=3, C1=8, C2=8,
                         stride=stride, padding=padding,
                         pool_size=pool_size, pool_stride=pool_stride)

    loss0, acc0 = train_one_epoch(params, X, y, padding, stride, pool_size, pool_stride,
                                  lr=0.0, batch_size=8, weight_decay=0.0)

    # Fit to data
    fit(params, X, y, padding, stride, pool_size, pool_stride,
        lr=0.1, weight_decay=1e-4,
        batch_size=8, epochs=10, verbose=True)

    loss1, acc1 = train_one_epoch(params, X, y, padding, stride, pool_size, pool_stride,
                                  lr=0.0, batch_size=8, weight_decay=0.0)

    print(f"baseline: loss = {loss0:.4f}, acc = {acc0:.3f}")
    print(f"after : loss = {loss1:.4f}, acc = {acc1:.3f}")

    assert loss1 < loss0 - 0.1
    assert acc1 > acc0 + 0.2
    assert acc1 >= 0.8

if __name__ == "__main__":
    test_training_loop_improves_loss_and_acc()
