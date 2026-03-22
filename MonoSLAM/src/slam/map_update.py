# src/slam/map_update.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import as_2xN_points, check_1d_pair_same_length, check_2xN_pair, check_dict, check_index_array_1d, check_matrix_3x3, check_points_xy_N2_rows, check_positive, check_required_keys, check_vector_3
from geometry.camera import projection_matrix, reprojection_errors_sq, world_to_camera_points
from geometry.triangulation import triangulate_points


# Bundle of tracked correspondences that are eligible to become new landmarks
@dataclass(frozen=True)
class NewLandmarkCandidates:
    # Indices into the tracked correspondence arrays
    track_idx: np.ndarray
    # Keyframe feature index per candidate
    kf_feat_idx: np.ndarray
    # Current-frame feature index per candidate
    cur_feat_idx: np.ndarray
    # Keyframe image points as (2,N)
    x_kf: np.ndarray
    # Current-frame image points as (2,N)
    x_cur: np.ndarray


# Bundle of successfully triangulated new landmarks
@dataclass(frozen=True)
class TriangulatedLandmarkBatch:
    # Valid candidate indices back into the tracked correspondence arrays
    track_idx: np.ndarray
    # Keyframe feature index per valid landmark
    kf_feat_idx: np.ndarray
    # Current-frame feature index per valid landmark
    cur_feat_idx: np.ndarray
    # Keyframe image points as (2,N_valid)
    x_kf: np.ndarray
    # Current-frame image points as (2,N_valid)
    x_cur: np.ndarray
    # Triangulated world points as (3,N_valid)
    X_w: np.ndarray
    # Validity mask over the input candidate bundle
    valid_mask: np.ndarray
    # Debug / acceptance statistics
    stats: dict


# Build candidate 2D-2D correspondences for new landmark triangulation
def build_new_landmark_candidates(seed: dict, track_out: dict) -> NewLandmarkCandidates:
    # --- Checks ---
    # Check containers
    seed = check_required_keys(seed, {"landmark_id_by_feat1"}, name="seed")
    track_out = check_dict(track_out, name="track_out")

    # Read landmark lookup from keyframe feature index to landmark id
    landmark_id_by_feat1 = check_index_array_1d(
        seed["landmark_id_by_feat1"],
        name="seed['landmark_id_by_feat1']",
        dtype=np.int64,
        allow_negative=True,
    )

    # Read tracked feature indices
    kf_feat_idx = check_index_array_1d(
        track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)),
        name="track_out['kf_feat_idx']",
        dtype=np.int64,
        allow_negative=True,
    )

    cur_feat_idx = check_index_array_1d(
        track_out.get("cur_feat_idx", np.zeros((0,), dtype=np.int64)),
        name="track_out['cur_feat_idx']",
        dtype=np.int64,
        allow_negative=True,
    )

    # Require aligned tracked index arrays
    kf_feat_idx, cur_feat_idx = check_1d_pair_same_length(
        kf_feat_idx,
        cur_feat_idx,
        nameA="track_out['kf_feat_idx']",
        nameB="track_out['cur_feat_idx']",
    )

    # Read tracked correspondence count
    M = int(kf_feat_idx.size)

    # Read tracked image points in (N,2) form
    xy_kf = check_points_xy_N2_rows(
        track_out.get("xy_kf", np.zeros((0, 2), dtype=np.float64)),
        M,
        name="track_out['xy_kf']",
        dtype=float,
        finite=True,
    )

    xy_cur = check_points_xy_N2_rows(
        track_out.get("xy_cur", np.zeros((0, 2), dtype=np.float64)),
        M,
        name="track_out['xy_cur']",
        dtype=float,
        finite=True,
    )

    # Early exit on no tracked correspondences
    if M == 0:
        return NewLandmarkCandidates(
            track_idx=np.zeros((0,), dtype=np.int64),
            kf_feat_idx=np.zeros((0,), dtype=np.int64),
            cur_feat_idx=np.zeros((0,), dtype=np.int64),
            x_kf=np.zeros((2, 0), dtype=np.float64),
            x_cur=np.zeros((2, 0), dtype=np.float64),
        )

    # Valid keyframe feature indices must lie inside the seed lookup
    valid_kf = (kf_feat_idx >= 0) & (kf_feat_idx < int(landmark_id_by_feat1.size))

    # Valid current-frame feature indices must be non-negative
    valid_cur = (cur_feat_idx >= 0)

    # A tracked keyframe feature is a candidate only if it is currently unmapped
    unmapped = np.zeros((M,), dtype=bool)
    if np.any(valid_kf):
        unmapped[valid_kf] = (landmark_id_by_feat1[kf_feat_idx[valid_kf]] < 0)

    # Keep only valid, currently unmapped correspondences
    keep = valid_kf & valid_cur & unmapped
    track_idx = np.nonzero(keep)[0].astype(np.int64, copy=False)

    # Early exit when nothing new is available
    if track_idx.size == 0:
        return NewLandmarkCandidates(
            track_idx=np.zeros((0,), dtype=np.int64),
            kf_feat_idx=np.zeros((0,), dtype=np.int64),
            cur_feat_idx=np.zeros((0,), dtype=np.int64),
            x_kf=np.zeros((2, 0), dtype=np.float64),
            x_cur=np.zeros((2, 0), dtype=np.float64),
        )

    # Gather candidate indices
    kf_feat_idx_keep = np.asarray(kf_feat_idx[track_idx], dtype=np.int64)
    cur_feat_idx_keep = np.asarray(cur_feat_idx[track_idx], dtype=np.int64)

    # Gather candidate image points and convert to (2,N)
    x_kf = as_2xN_points(
        xy_kf[track_idx],
        name="xy_kf_candidates",
        finite=True,
        dtype=float,
    )

    x_cur = as_2xN_points(
        xy_cur[track_idx],
        name="xy_cur_candidates",
        finite=True,
        dtype=float,
    )

    return NewLandmarkCandidates(
        track_idx=track_idx,
        kf_feat_idx=kf_feat_idx_keep,
        cur_feat_idx=cur_feat_idx_keep,
        x_kf=x_kf,
        x_cur=x_cur,
    )


# Triangulate candidate landmarks between a keyframe and the current frame
def triangulate_new_landmarks(
    K_kf,
    K_cur,
    R_kf,
    t_kf,
    R_cur,
    t_cur,
    candidates: NewLandmarkCandidates,
    *,
    min_parallax_deg: float = 1.0,
    max_depth_ratio: float = 200.0,
    max_reproj_error_px: float | None = 3.0,
    eps: float = 1e-12,
) -> TriangulatedLandmarkBatch:
    # --- Checks ---
    # Check intrinsics
    K_kf = check_matrix_3x3(K_kf, name="K_kf", dtype=float, finite=False)
    K_cur = check_matrix_3x3(K_cur, name="K_cur", dtype=float, finite=False)

    # Check poses
    R_kf = check_matrix_3x3(R_kf, name="R_kf", dtype=float, finite=False)
    t_kf = check_vector_3(t_kf, name="t_kf", dtype=float, finite=False)
    R_cur = check_matrix_3x3(R_cur, name="R_cur", dtype=float, finite=False)
    t_cur = check_vector_3(t_cur, name="t_cur", dtype=float, finite=False)

    # Check candidate bundle
    if not isinstance(candidates, NewLandmarkCandidates):
        raise ValueError("candidates must be a NewLandmarkCandidates bundle")

    # Check scalar gates
    min_parallax_deg = check_positive(min_parallax_deg, name="min_parallax_deg", eps=0.0)
    max_depth_ratio = check_positive(max_depth_ratio, name="max_depth_ratio", eps=0.0)
    eps = check_positive(eps, name="eps", eps=0.0)

    if max_reproj_error_px is not None:
        max_reproj_error_px = check_positive(max_reproj_error_px, name="max_reproj_error_px", eps=0.0)

    # Check candidate indices
    track_idx = check_index_array_1d(candidates.track_idx, name="candidates.track_idx", dtype=np.int64, allow_negative=False)
    kf_feat_idx = check_index_array_1d(candidates.kf_feat_idx, name="candidates.kf_feat_idx", dtype=np.int64, allow_negative=False)
    cur_feat_idx = check_index_array_1d(candidates.cur_feat_idx, name="candidates.cur_feat_idx", dtype=np.int64, allow_negative=False)

    # Require aligned candidate index arrays
    track_idx, kf_feat_idx = check_1d_pair_same_length(
        track_idx,
        kf_feat_idx,
        nameA="candidates.track_idx",
        nameB="candidates.kf_feat_idx",
    )

    track_idx, cur_feat_idx = check_1d_pair_same_length(
        track_idx,
        cur_feat_idx,
        nameA="candidates.track_idx",
        nameB="candidates.cur_feat_idx",
    )

    # Check candidate image points
    x_kf, x_cur = check_2xN_pair(candidates.x_kf, candidates.x_cur, dtype=float, finite=True)

    # Require image points to align with the index arrays
    N = int(track_idx.size)
    if int(x_kf.shape[1]) != N:
        raise ValueError(
            f"candidates.x_kf must have {N} columns to match candidates.track_idx; got {x_kf.shape[1]}"
        )

    # Early exit on no candidates
    if N == 0:
        return TriangulatedLandmarkBatch(
            track_idx=np.zeros((0,), dtype=np.int64),
            kf_feat_idx=np.zeros((0,), dtype=np.int64),
            cur_feat_idx=np.zeros((0,), dtype=np.int64),
            x_kf=np.zeros((2, 0), dtype=np.float64),
            x_cur=np.zeros((2, 0), dtype=np.float64),
            X_w=np.zeros((3, 0), dtype=np.float64),
            valid_mask=np.zeros((0,), dtype=bool),
            stats={
                "N_in": 0,
                "baseline": 0.0,
                "n_finite": 0,
                "n_cheirality": 0,
                "n_parallax": 0,
                "n_depth": 0,
                "n_reproj": 0,
                "n_valid": 0,
                "reason": "no_candidates",
            },
        )

    # Camera centres in world coordinates
    C_kf = -R_kf.T @ t_kf
    C_cur = -R_cur.T @ t_cur

    # Baseline length
    baseline = float(np.linalg.norm(C_cur - C_kf))
    if baseline <= float(eps):
        return TriangulatedLandmarkBatch(
            track_idx=np.zeros((0,), dtype=np.int64),
            kf_feat_idx=np.zeros((0,), dtype=np.int64),
            cur_feat_idx=np.zeros((0,), dtype=np.int64),
            x_kf=np.zeros((2, 0), dtype=np.float64),
            x_cur=np.zeros((2, 0), dtype=np.float64),
            X_w=np.zeros((3, 0), dtype=np.float64),
            valid_mask=np.zeros((N,), dtype=bool),
            stats={
                "N_in": N,
                "baseline": baseline,
                "n_finite": 0,
                "n_cheirality": 0,
                "n_parallax": 0,
                "n_depth": 0,
                "n_reproj": 0,
                "n_valid": 0,
                "reason": "baseline_too_small",
            },
        )

    # Build projection matrices
    P_kf = projection_matrix(K_kf, R_kf, t_kf)
    P_cur = projection_matrix(K_cur, R_cur, t_cur)

    # Triangulate candidate world points
    X_w = triangulate_points(P_kf, P_cur, x_kf, x_cur)
    X_w = np.asarray(X_w, dtype=np.float64)

    # Finite-point gate
    finite_mask = np.isfinite(X_w).all(axis=0)

    # Depths in both cameras
    X_kf = world_to_camera_points(R_kf, t_kf, X_w)
    X_cur = world_to_camera_points(R_cur, t_cur, X_w)

    z_kf = np.asarray(X_kf[2, :], dtype=np.float64)
    z_cur = np.asarray(X_cur[2, :], dtype=np.float64)

    # Cheirality gate
    cheirality_mask = finite_mask & (z_kf > float(eps)) & (z_cur > float(eps))

    # Parallax gate from viewing rays in world coordinates
    v1 = X_w - C_kf.reshape(3, 1)
    v2 = X_w - C_cur.reshape(3, 1)

    n1 = np.linalg.norm(v1, axis=0)
    n2 = np.linalg.norm(v2, axis=0)

    valid_ray_norms = (n1 > float(eps)) & (n2 > float(eps))
    cos_parallax = np.ones((N,), dtype=np.float64)
    cos_parallax[valid_ray_norms] = np.sum(v1[:, valid_ray_norms] * v2[:, valid_ray_norms], axis=0) / (n1[valid_ray_norms] * n2[valid_ray_norms])
    cos_parallax = np.clip(cos_parallax, -1.0, 1.0)

    parallax_deg = np.degrees(np.arccos(cos_parallax))
    parallax_mask = cheirality_mask & valid_ray_norms & (parallax_deg >= float(min_parallax_deg))

    # Depth sanity gate relative to the baseline
    depth_ratio = np.maximum(z_kf, z_cur) / max(baseline, float(eps))
    depth_mask = cheirality_mask & np.isfinite(depth_ratio) & (depth_ratio <= float(max_depth_ratio))

    # Optional reprojection gate
    if max_reproj_error_px is None:
        reproj_mask = np.ones((N,), dtype=bool)
        reproj_err_sq = np.full((N,), np.nan, dtype=np.float64)
    else:
        err_kf_sq = np.asarray(reprojection_errors_sq(K_kf, R_kf, t_kf, X_w, x_kf), dtype=np.float64).reshape(-1)
        err_cur_sq = np.asarray(reprojection_errors_sq(K_cur, R_cur, t_cur, X_w, x_cur), dtype=np.float64).reshape(-1)

        err_kf_sq[~np.isfinite(err_kf_sq)] = np.inf
        err_cur_sq[~np.isfinite(err_cur_sq)] = np.inf

        reproj_err_sq = np.maximum(err_kf_sq, err_cur_sq)
        reproj_mask = reproj_err_sq <= float(max_reproj_error_px ** 2)

    # Final validity mask
    valid_mask = cheirality_mask & parallax_mask & depth_mask & reproj_mask

    # Gather valid outputs
    X_w_valid = np.asarray(X_w[:, valid_mask], dtype=np.float64)
    track_idx_valid = np.asarray(track_idx[valid_mask], dtype=np.int64)
    kf_feat_idx_valid = np.asarray(kf_feat_idx[valid_mask], dtype=np.int64)
    cur_feat_idx_valid = np.asarray(cur_feat_idx[valid_mask], dtype=np.int64)
    x_kf_valid = np.asarray(x_kf[:, valid_mask], dtype=np.float64)
    x_cur_valid = np.asarray(x_cur[:, valid_mask], dtype=np.float64)

    # Summary stats
    parallax_use = parallax_deg[cheirality_mask]
    stats = {
        "N_in": N,
        "baseline": baseline,
        "n_finite": int(finite_mask.sum()),
        "n_cheirality": int(cheirality_mask.sum()),
        "n_parallax": int(parallax_mask.sum()),
        "n_depth": int(depth_mask.sum()),
        "n_reproj": int(reproj_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "parallax_p25_deg": None if parallax_use.size == 0 else float(np.percentile(parallax_use, 25)),
        "parallax_p50_deg": None if parallax_use.size == 0 else float(np.percentile(parallax_use, 50)),
        "parallax_p75_deg": None if parallax_use.size == 0 else float(np.percentile(parallax_use, 75)),
        "max_reproj_error_px": None if max_reproj_error_px is None else float(max_reproj_error_px),
        "reason": None if np.any(valid_mask) else "no_valid_triangulated_landmarks",
    }

    return TriangulatedLandmarkBatch(
        track_idx=track_idx_valid,
        kf_feat_idx=kf_feat_idx_valid,
        cur_feat_idx=cur_feat_idx_valid,
        x_kf=x_kf_valid,
        x_cur=x_cur_valid,
        X_w=X_w_valid,
        valid_mask=np.asarray(valid_mask, dtype=bool),
        stats=stats,
    )

