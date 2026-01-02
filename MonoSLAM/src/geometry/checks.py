# geometry/checks.py

# Check pair is shape (2, N)
def check_2xN_pair(x1, x2):
    if x1.ndim != 2 or x1.shape[0] != 2 or x1.shape != x2.shape:
        raise ValueError(f"Must be (2,N) same shape; got {x1.shape} and {x2.shape}")
