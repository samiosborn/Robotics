# src/geometry/pnp.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import check_3xN_2xN_cols, check_matrix_3x3, check_points_3xN
from geometry.camera import pixel_to_normalised, reprojection_rmse, world_to_camera_points


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


# Build DLT design matrix for linear PnP
def _build_pnp_dlt_matrix(X_w, x_hat):
    # --- Checks ---
    # Require same correspondence count
    X_w, x_hat = check_3xN_pair(X_w, x_hat, dtype=float, finite=True)

    # Read sizes
    N = int(X_w.shape[1])

    # Homogeneous world points
    X_h = np.vstack([X_w, np.ones((1, N), dtype=float)])

    # Dehomogenise normalised image coordinates
    w = x_hat[2, :]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("x_hat contains points with near-zero homogeneous scale")

    u = x_hat[0, :] / w
    v = x_hat[1, :] / w

    # Allocate design matrix
    A = np.zeros((2 * N, 12), dtype=float)

    # Fill two equations per correspondence
    for i in range(N):
        Xi = X_h[:, i]

        # p1^T X - u p3^T X = 0
        A[2 * i, 0:4] = Xi
        A[2 * i, 8:12] = -float(u[i]) * Xi

        # p2^T X - v p3^T X = 0
        A[2 * i + 1, 4:8] = Xi
        A[2 * i + 1, 8:12] = -float(v[i]) * Xi

    return A


# Recover a metric pose from a projective DLT camera matrix
def _pose_from_dlt_projection(P_tilde, eps=1e-12):
    # --- Checks ---
    P_tilde = np.asarray(P_tilde, dtype=float)
    if P_tilde.shape != (3, 4):
        raise ValueError(f"P_tilde must be (3,4); got {P_tilde.shape}")

    # Split into linear and translation parts
    M = P_tilde[:, :3]
    p4 = P_tilde[:, 3]

    # Project M onto the nearest rotation
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    # Enforce proper rotation
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt

    # Recover signed common scale
    scale = float(np.trace(M @ R.T) / 3.0)
    if abs(scale) < float(eps):
        raise ValueError("Recovered DLT scale is near zero")

    # Recover translation
    t = p4 / scale

    stats = {
        "M_singular_values": np.asarray(S, dtype=float),
        "scale": float(scale),
        "det_R": float(np.linalg.det(R)),
    }

    return np.asarray(R, dtype=float), np.asarray(t, dtype=float).reshape(3), stats


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
        raise ValueError(
            f"kf_feat_idx and cur_feat_idx must have same size; got {kf_feat_idx.size} and {cur_feat_idx.size}"
        )
    if xy_cur.ndim != 2 or xy_cur.shape[1] != 2:
        raise ValueError(f"xy_cur must be (N,2); got {xy_cur.shape}")
    if xy_cur.shape[0] != kf_feat_idx.size:
        raise ValueError(
            f"xy_cur and tracked feature indices must have same N; got {xy_cur.shape[0]} and {kf_feat_idx.size}"
        )

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


# Estimate camera pose from 2D–3D correspondences with linear DLT PnP
def estimate_pose_pnp(
    corrs: PnPCorrespondences,
    K,
    *,
    min_points: int = 6,
    rank_tol: float = 1e-10,
    min_cheirality_ratio: float = 0.5,
    eps: float = 1e-12,
):
    # --- Checks ---
    # Check intrinsics
    K = check_matrix_3x3(K, name="K", dtype=float, finite=False)
    # Check correspondences
    X_w, x_cur = check_3xN_2xN_cols(corrs.X_w, corrs.x_cur, nameX="corrs.X_w", namex="corrs.x_cur", dtype=float, finite=True)
    # Validate scalar parameters
    min_points = max(6, int(min_points))
    rank_tol = float(rank_tol)
    min_cheirality_ratio = float(min_cheirality_ratio)
    eps = float(eps)

    # Initialise stats
    N = int(X_w.shape[1])
    stats = {
        "n_corr": N,
        "min_points": int(min_points),
        "method": "dlt",
    }

    # Require enough correspondences
    if N < min_points:
        stats.update({"reason": "too_few_correspondences"})
        return None, None, stats

    # Convert pixels to normalised homogeneous image coordinates
    x_hat = pixel_to_normalised(K, x_cur)

    # Build DLT design matrix
    try:
        A = _build_pnp_dlt_matrix(X_w, x_hat)
    except Exception as exc:
        stats.update({"reason": "dlt_matrix_failed", "error": str(exc)})
        return None, None, stats

    # Solve homogeneous least squares
    try:
        _, S_A, Vt_A = np.linalg.svd(A)
    except Exception as exc:
        stats.update({"reason": "dlt_svd_failed", "error": str(exc)})
        return None, None, stats

    # Estimate numerical rank
    if S_A.size == 0:
        stats.update({"reason": "dlt_empty_spectrum"})
        return None, None, stats

    s0 = float(S_A[0])
    tol = float(rank_tol) * max(s0, eps)
    rank = int(np.sum(S_A > tol))
    stats.update({"A_rank": rank, "A_singular_values": np.asarray(S_A, dtype=float)})

    # Require near-full rank
    if rank < 11:
        stats.update({"reason": "dlt_rank_deficient"})
        return None, None, stats

    # Recover projective camera matrix
    P_tilde = Vt_A[-1, :].reshape(3, 4)

    # Recover metric pose
    try:
        R, t, pose_stats = _pose_from_dlt_projection(P_tilde, eps=eps)
    except Exception as exc:
        stats.update({"reason": "pose_from_dlt_failed", "error": str(exc)})
        return None, None, stats

    stats.update(pose_stats)

    # Check cheirality under recovered pose
    X_c = world_to_camera_points(R, t, X_w)
    z = X_c[2, :]
    cheirality_ratio = float(np.mean(z > eps))
    stats.update({"cheirality_ratio": cheirality_ratio})

    # Reject poses with too many points behind the camera
    if cheirality_ratio < min_cheirality_ratio:
        stats.update({"reason": "cheirality_ratio_too_low"})
        return None, None, stats

    # Score by reprojection error
    try:
        rmse_px = float(reprojection_rmse(K, R, t, X_w, x_cur))
    except Exception as exc:
        stats.update({"reason": "reprojection_eval_failed", "error": str(exc)})
        return None, None, stats

    stats.update({"reprojection_rmse_px": rmse_px})

    return np.asarray(R, dtype=float), np.asarray(t, dtype=float).reshape(3), stats

