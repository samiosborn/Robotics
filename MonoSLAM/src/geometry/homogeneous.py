# src/geometry/homogeneous.py
import numpy as np


# Homogenise (2xN -> 3xN) or (DxN -> (D+1)xN)
def homogenise(X):
    # Convert to array
    X = np.asarray(X)
    # Validate 2D
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (D,N); got {X.shape}")
    # Append ones row
    ones = np.ones((1, X.shape[1]), dtype=X.dtype)
    # Stack
    return np.vstack([X, ones])


# Dehomogenise ((D+1)xN -> DxN) safely
def dehomogenise(X_h, *, eps=1e-12):
    # Convert to array
    X_h = np.asarray(X_h)
    # Validate 2D
    if X_h.ndim != 2:
        raise ValueError(f"X_h must be 2D (D+1,N); got {X_h.shape}")
    # Require at least 2 rows (so there is a w row)
    if X_h.shape[0] < 2:
        raise ValueError(f"X_h must have at least 2 rows; got {X_h.shape}")
    # Read scale row
    w = X_h[-1]
    # Prepare output
    Y = np.empty_like(X_h[:-1], dtype=float)
    # Mark valid scales
    ok = np.abs(w) > float(eps)
    # Default everything to +inf (points at infinity / invalid)
    Y[...] = np.inf
    # Safe divide only where ok (avoids RuntimeWarning)
    np.divide(X_h[:-1], w[None, :], out=Y, where=ok[None, :])
    # Return
    return Y
