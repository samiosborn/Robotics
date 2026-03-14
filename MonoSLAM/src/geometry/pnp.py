# src/geometry/pnp.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import check_points_2xN, check_points_3xN


# Bundle of 2D–3D correspondences for PnP
@dataclass(frozen=True)
class PnPCorrespondences:
    # World points as (3,N)
    X_w: np.ndarray
    # Current image points as (2,N)
    x_cur: np.ndarray
    # Landmark id per correspondence as (N,)
    landmark_ids: np.ndarray
    # Current feature index per correspondence as (N,)
    cur_feat_idx: np.ndarray
    # Keyframe feature index per correspondence as (N,)
    kf_feat_idx: np.ndarray


# Build lookup from landmark id to landmark dict
def _landmark_dict_by_id(seed: dict) -> dict[int, dict]:
    # Read landmarks
    landmarks = seed.get("landmarks", [])
    if not isinstance(landmarks, list):
        raise ValueError("seed['landmarks'] must be a list")

    # Build lookup
    out: dict[int, dict] = {}
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        if "id" not in lm:
            continue
        out[int(lm["id"])] = lm

    return out


# Build 2D–3D correspondences from seed and tracking output
def build_pnp_correspondences(seed: dict, track_out: dict) -> PnPCorrespondences:
    # --- Checks ---
    # Seed must be a dict
    if not isinstance(seed, dict):
        raise ValueError("seed must be a dict")
    # Track output must be a dict
    if not isinstance(track_out, dict):
        raise ValueError("track_out must be a dict")

    # Read landmark id map from keyframe features to landmarks
    landmark_id_by_feat1 = np.asarray(
        seed.get("landmark_id_by_feat1", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)

    # Read tracked feature indices
    kf_feat_idx = np.asarray(
        track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)

    cur_feat_idx = np.asarray(
        track_out.get("cur_feat_idx", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)

    # Read current image points
    xy_cur = np.asarray(
        track_out.get("xy_cur", np.zeros((0, 2), dtype=np.float64)),
        dtype=np.float64,
    )

    # Require aligned tracking outputs
    if kf_feat_idx.ndim != 1:
        raise ValueError(f"kf_feat_idx must be 1D; got {kf_feat_idx.shape}")
    if cur_feat_idx.ndim != 1:
        raise ValueError(f"cur_feat_idx must be 1D; got {cur_feat_idx.shape}")
    if kf_feat_idx.size != cur_feat_idx.size:
        raise ValueError(f"kf_feat_idx and cur_feat_idx must have same size; got {kf_feat_idx.size} and {cur_feat_idx.size}")
    if xy_cur.ndim != 2 or xy_cur.shape[1] != 2:
        raise ValueError(f"xy_cur must be (N,2); got {xy_cur.shape}")
    if xy_cur.shape[0] != kf_feat_idx.size:
        raise ValueError(f"xy_cur and tracked feature indices must have same N; got {xy_cur.shape[0]} and {kf_feat_idx.size}")

    # Build landmark lookup
    lm_by_id = _landmark_dict_by_id(seed)

    # Collect valid correspondences
    X_cols: list[np.ndarray] = []
    x_cols: list[np.ndarray] = []
    landmark_ids: list[int] = []
    cur_idx_keep: list[int] = []
    kf_idx_keep: list[int] = []

    # Walk tracked correspondences
    for i in range(kf_feat_idx.size):
        # Keyframe feature index
        feat1 = int(kf_feat_idx[i])

        # Skip invalid keyframe feature index
        if feat1 < 0 or feat1 >= landmark_id_by_feat1.size:
            continue

        # Map keyframe feature to landmark id
        lm_id = int(landmark_id_by_feat1[feat1])

        # Skip unmatched features
        if lm_id < 0:
            continue

        # Lookup landmark
        lm = lm_by_id.get(lm_id, None)
        if lm is None:
            continue

        # Read world point
        X_w = np.asarray(lm.get("X_w", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        if X_w.size != 3:
            continue
        if not np.isfinite(X_w).all():
            continue

        # Read current image point
        x = np.asarray(xy_cur[i], dtype=np.float64).reshape(-1)
        if x.size != 2:
            continue
        if not np.isfinite(x).all():
            continue

        # Append valid correspondence
        X_cols.append(X_w.reshape(3, 1))
        x_cols.append(x.reshape(2, 1))
        landmark_ids.append(lm_id)
        cur_idx_keep.append(int(cur_feat_idx[i]))
        kf_idx_keep.append(feat1)

    # Return empty bundle if nothing survived
    if len(X_cols) == 0:
        return PnPCorrespondences(
            X_w=np.zeros((3, 0), dtype=np.float64),
            x_cur=np.zeros((2, 0), dtype=np.float64),
            landmark_ids=np.zeros((0,), dtype=np.int64),
            cur_feat_idx=np.zeros((0,), dtype=np.int64),
            kf_feat_idx=np.zeros((0,), dtype=np.int64),
        )

    # Stack into canonical arrays
    X_w = np.hstack(X_cols)
    x_cur = np.hstack(x_cols)
    landmark_ids_arr = np.asarray(landmark_ids, dtype=np.int64)
    cur_feat_idx_arr = np.asarray(cur_idx_keep, dtype=np.int64)
    kf_feat_idx_arr = np.asarray(kf_idx_keep, dtype=np.int64)

    # Final checks
    X_w = check_points_3xN(X_w, name="X_w", dtype=float, finite=True)
    x_cur = check_points_2xN(x_cur, name="x_cur", dtype=float, finite=True)

    return PnPCorrespondences(
        X_w=X_w,
        x_cur=x_cur,
        landmark_ids=landmark_ids_arr,
        cur_feat_idx=cur_feat_idx_arr,
        kf_feat_idx=kf_feat_idx_arr,
    )

