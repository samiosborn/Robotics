# src/slam/map_update.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import as_2xN_points, check_1d_pair_same_length, check_dict, check_index_array_1d, check_points_xy_N2_rows, check_required_keys


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

