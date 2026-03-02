# geometry/checks.py

import numpy as np

from core.checks import (
    check_points_xy_N2plus,
    check_points_2xN,
    check_points_3xN,
)

# Check for shape-matched (2, N) point pairs 
def check_2xN_pair(x1, x2):
    a = check_points_2xN(x1, name="x1", finite=False)
    b = check_points_2xN(x2, name="x2", finite=False)
    if a.shape != b.shape:
        raise ValueError(f"Must be (2,N) same shape; got {a.shape} and {b.shape}")


# Check shape-matched (3, N) point pairs
def check_3xN_pair(x1, x2):
    a = check_points_3xN(x1, name="x1", finite=False)
    b = check_points_3xN(x2, name="x2", finite=False)
    if a.shape != b.shape:
        raise ValueError(f"Must be (3,N) same shape; got {a.shape} and {b.shape}")


# Convert keypoints (N,2) / (N,>=2) to a (2,N) matrix.
def as_2xN_points(xy, name="xy", finite=True):
    xy = check_points_xy_N2plus(xy, name=name, finite=finite)
    return np.vstack([xy[:, 0], xy[:, 1]])
