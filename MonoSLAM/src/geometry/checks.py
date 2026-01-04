# geometry/checks.py

# Check if shape (2, N)
def check_2xN(x):
    if x.ndim != 2:
        raise ValueError(f"Must be (2,N) same shape; got {x.shape}")

# Check pair is shape (2, N)
def check_2xN_pair(x1, x2):
    if x1.ndim != 2 or x1.shape[0] != 2 or x1.shape != x2.shape:
        raise ValueError(f"Must be (2,N) same shape; got {x1.shape} and {x2.shape}")

# Check if K matrix is (3, 3)
def check_K(K, name="K"):
    K = np.asarray(K)
    if K.shape != (3,3):
        raise ValueError(f"{name} must be (3,3); got {K.shape}")
