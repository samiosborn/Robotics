import numpy as np

import geometry.triangulation as _triangulation
from geometry.homogeneous import homogenise, dehomogenise


def install_two_view_compat_shims():
    # Compatibility shim for triangulation point-shape handling.
    def _triangulate_point_compat(P1, P2, x1, x2):
        x1_h = homogenise(np.asarray(x1, dtype=float).reshape(2, 1)).reshape(3,)
        x2_h = homogenise(np.asarray(x2, dtype=float).reshape(2, 1)).reshape(3,)
        A = np.vstack([
            x1_h[0] * P1[2] - P1[0],
            x1_h[1] * P1[2] - P1[1],
            x2_h[0] * P2[2] - P2[0],
            x2_h[1] * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1].reshape(4, 1)
        w = X_h[-1, 0]
        if abs(w) < 1e-12:
            return np.array([np.nan, np.nan, np.nan])
        return dehomogenise(X_h).reshape(3,)

    _triangulation.triangulate_point = _triangulate_point_compat
