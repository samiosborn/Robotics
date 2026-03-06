# slam/seed.py
from __future__ import annotations

import numpy as np

from core.checks import check_matrix_3x3
from core.checks import check_2xN_pair


# Build a two-view seed map from validated bootstrap outputs
def build_two_view_seed(x1, x2, *, idx_init, X_valid, R1, t1) -> dict:
    # Checks
    check_2xN_pair(x1, x2)
    R1 = check_matrix_3x3(R1, name="R1", finite=False)
    t1 = np.asarray(t1, dtype=float).reshape(3)
    idx_init = np.asarray(idx_init, dtype=int).reshape(-1)
    X_valid = np.asarray(X_valid, dtype=float)
    if X_valid.ndim != 2 or X_valid.shape[0] != 3:
        raise ValueError(f"X_valid must be (3,N); got {X_valid.shape}")
    if X_valid.shape[1] != idx_init.size:
        raise ValueError(f"Mismatch: X_valid has {X_valid.shape[1]} cols but idx_init has {idx_init.size}")

    # Validate index bounds
    n_cols = x1.shape[1]
    if idx_init.size > 0 and (int(idx_init.min()) < 0 or int(idx_init.max()) >= n_cols):
        raise ValueError(f"idx_init contains values outside [0,{n_cols - 1}]")

    # Poses
    R0 = np.eye(3, dtype=float)
    t0 = np.zeros(3, dtype=float)

    # Landmarks
    landmarks = []
    for lm_id, (j, X_w) in enumerate(zip(idx_init, X_valid.T)):
        j = int(j)
        landmarks.append(
            {
                "id": lm_id,
                "X_w": X_w,
                "obs": [
                    {"kf": 0, "feat": j, "xy": x1[:, j]},
                    {"kf": 1, "feat": j, "xy": x2[:, j]},
                ],
                "descriptor": None,
                "quality": {"reproj0_px": None, "reproj1_px": None},
            }
        )

    # Initialised correspondence mask
    mask_init = np.zeros(n_cols, dtype=bool)
    if idx_init.size > 0:
        mask_init[idx_init] = True

    return {
        "T_WC0": (R0, t0),
        "T_WC1": (R1, t1),
        "landmarks": landmarks,
        "mask_init": mask_init,
        "idx_init": idx_init,
    }
