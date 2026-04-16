# src/slam/map_update.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import align_bool_mask_1d, as_2xN_points, check_1d_pair_same_length, check_2xN_pair, check_dict, check_index_array_1d, check_int_ge0, check_matrix_3x3, check_points_2xN, check_points_xy_N2_rows, check_positive, check_required_keys, check_vector_3
from geometry.camera import camera_centre, projection_matrix, reprojection_errors_sq, world_to_camera_points
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


# Bundle for one map-growth step
@dataclass(frozen=True)
class MapGrowthResult:
    # Updated seed after attempting map growth
    seed: dict
    # Candidate 2D-2D tracks considered for triangulation
    candidates: NewLandmarkCandidates
    # Triangulated batch after geometric filtering
    batch: TriangulatedLandmarkBatch
    # Step-level summary stats
    stats: dict


# Append current-frame observations for already-known tracked landmarks
def append_tracked_observations_to_seed(
    seed: dict,
    pose_out: dict,
    *,
    current_kf: int,
    K: np.ndarray | None = None,
    track_out: dict | None = None,
    max_append_reproj_error_px_existing: float = 2.0,
    eps: float = 1e-12,
) -> tuple[dict, dict]:
    # --- Checks ---
    # Check containers
    seed = check_required_keys(seed, {"landmarks"}, name="seed")
    pose_out = check_required_keys(pose_out, {"corrs", "pnp_inlier_mask"}, name="pose_out")

    # Check frame index
    current_kf = check_int_ge0(current_kf, name="current_kf")

    # Check append controls
    max_append_reproj_error_px_existing = check_positive(
        max_append_reproj_error_px_existing,
        name="max_append_reproj_error_px_existing",
        eps=0.0,
    )
    eps = check_positive(eps, name="eps", eps=0.0)

    # Read correspondence bundle
    corrs = pose_out["corrs"]
    if not hasattr(corrs, "landmark_ids"):
        raise ValueError("pose_out['corrs'] must have attribute 'landmark_ids'")
    if not hasattr(corrs, "cur_feat_idx"):
        raise ValueError("pose_out['corrs'] must have attribute 'cur_feat_idx'")
    if not hasattr(corrs, "x_cur"):
        raise ValueError("pose_out['corrs'] must have attribute 'x_cur'")

    # Read correspondence arrays
    landmark_ids = check_index_array_1d(
        getattr(corrs, "landmark_ids"),
        name="pose_out['corrs'].landmark_ids",
        dtype=np.int64,
        allow_negative=False,
    )

    cur_feat_idx = check_index_array_1d(
        getattr(corrs, "cur_feat_idx"),
        name="pose_out['corrs'].cur_feat_idx",
        dtype=np.int64,
        allow_negative=False,
    )

    landmark_ids, cur_feat_idx = check_1d_pair_same_length(
        landmark_ids,
        cur_feat_idx,
        nameA="pose_out['corrs'].landmark_ids",
        nameB="pose_out['corrs'].cur_feat_idx",
    )

    # Read current-frame image points
    x_cur = check_points_2xN(
        getattr(corrs, "x_cur"),
        name="pose_out['corrs'].x_cur",
        dtype=float,
        finite=True,
    )

    # Require aligned correspondence arrays
    N = int(landmark_ids.size)
    if int(x_cur.shape[1]) != N:
        raise ValueError(
            f"pose_out['corrs'].x_cur must have {N} columns to match pose_out['corrs'].landmark_ids; got {x_cur.shape[1]}"
        )

    # Align the PnP inlier mask to the correspondence count
    pnp_inlier_mask = align_bool_mask_1d(
        pose_out["pnp_inlier_mask"],
        N,
        name="pose_out['pnp_inlier_mask']",
    )

    # Read seed landmarks
    landmarks_raw = seed["landmarks"]
    if not isinstance(landmarks_raw, list):
        raise ValueError("seed['landmarks'] must be a list")
    landmarks = list(landmarks_raw)

    # Build a landmark-id to list-index lookup
    landmark_pos_by_id: dict[int, int] = {}
    for i, lm in enumerate(landmarks):
        if not isinstance(lm, dict):
            continue
        if "id" not in lm:
            continue
        landmark_pos_by_id[int(lm["id"])] = int(i)

    # Start stats
    stats = {
        "n_corr": int(N),
        "n_inlier_corr": int(pnp_inlier_mask.sum()),
        "n_added": 0,
        "n_duplicate": 0,
        "n_missing_landmark": 0,
        "current_kf": int(current_kf),
        "max_append_reproj_error_px_existing": float(max_append_reproj_error_px_existing),
        "n_append_candidates_existing": 0,
        "n_append_pnp_inliers": int(pnp_inlier_mask.sum()),
        "n_append_pnp_inliers_added": 0,
        "n_append_extra_reproj_tested": 0,
        "n_append_extra_reproj_pass": 0,
        "n_append_extra_reproj_added": 0,
        "n_append_total": 0,
        "n_append_duplicates": 0,
        "n_append_total_bootstrap_born": 0,
        "n_append_total_map_growth_born": 0,
        "n_append_pnp_inliers_added_bootstrap_born": 0,
        "n_append_pnp_inliers_added_map_growth_born": 0,
        "n_append_extra_reproj_added_bootstrap_born": 0,
        "n_append_extra_reproj_added_map_growth_born": 0,
        "n_landmarks_with_obs_current_kf_after_append": 0,
    }

    # Record landmark birth-source stats for appended observations
    def _record_birth_source_append(lm: dict, *, pnp_inlier: bool, extra_reproj: bool) -> None:
        birth_source = lm.get("birth_source", None)
        if birth_source == "bootstrap":
            stats["n_append_total_bootstrap_born"] += 1
            if bool(pnp_inlier):
                stats["n_append_pnp_inliers_added_bootstrap_born"] += 1
            if bool(extra_reproj):
                stats["n_append_extra_reproj_added_bootstrap_born"] += 1
        elif birth_source == "map_growth":
            stats["n_append_total_map_growth_born"] += 1
            if bool(pnp_inlier):
                stats["n_append_pnp_inliers_added_map_growth_born"] += 1
            if bool(extra_reproj):
                stats["n_append_extra_reproj_added_map_growth_born"] += 1

    # Count landmarks linked to this frame after the append step
    def _count_landmarks_with_current_observation() -> int:
        n_linked = 0
        for lm in landmarks:
            if not isinstance(lm, dict):
                continue

            obs = lm.get("obs", None)
            if not isinstance(obs, list):
                continue

            linked = False
            for ob in obs:
                if not isinstance(ob, dict):
                    continue
                if int(ob.get("kf", -1)) != int(current_kf):
                    continue
                linked = True
                break

            if linked:
                n_linked += 1

        return int(n_linked)

    # Append one current-frame observation if it is not already present
    def _append_current_observation(
        lm_id: int,
        feat_idx: int,
        xy,
        *,
        pnp_inlier: bool,
        extra_reproj: bool,
    ) -> bool:
        lm_pos = landmark_pos_by_id.get(int(lm_id), None)
        if lm_pos is None:
            stats["n_missing_landmark"] += 1
            return False

        lm = landmarks[lm_pos]
        obs = lm.get("obs", None)
        if not isinstance(obs, list):
            obs = []

        duplicate = False
        for ob in obs:
            if not isinstance(ob, dict):
                continue
            if int(ob.get("kf", -1)) != int(current_kf):
                continue
            if int(ob.get("feat", -1)) != int(feat_idx):
                continue
            duplicate = True
            break

        if duplicate:
            stats["n_duplicate"] += 1
            stats["n_append_duplicates"] += 1
            lm["obs"] = obs
            return False

        obs.append(
            {
                "kf": int(current_kf),
                "feat": int(feat_idx),
                "xy": np.asarray(xy, dtype=np.float64).reshape(2,),
            }
        )
        lm["obs"] = obs
        stats["n_added"] += 1
        stats["n_append_total"] += 1
        if bool(pnp_inlier):
            stats["n_append_pnp_inliers_added"] += 1
        if bool(extra_reproj):
            stats["n_append_extra_reproj_added"] += 1
        _record_birth_source_append(lm, pnp_inlier=bool(pnp_inlier), extra_reproj=bool(extra_reproj))

        return True

    # Append one current-frame observation per inlier landmark track
    pnp_inlier_pairs: set[tuple[int, int]] = set()
    for i in np.flatnonzero(pnp_inlier_mask):
        lm_id = int(landmark_ids[i])
        feat_idx = int(cur_feat_idx[i])
        xy = np.asarray(x_cur[:, i], dtype=np.float64).reshape(2,)

        pnp_inlier_pairs.add((int(lm_id), int(feat_idx)))
        _append_current_observation(
            lm_id,
            feat_idx,
            xy,
            pnp_inlier=True,
            extra_reproj=False,
        )

    # Append additional existing-landmark tracks that fit the recovered pose
    if track_out is not None and K is not None:
        track_out = check_dict(track_out, name="track_out")

        K_checked = check_matrix_3x3(K, name="K", dtype=float, finite=False)
        R_cur = check_matrix_3x3(pose_out.get("R", None), name="pose_out['R']", dtype=float, finite=True)
        t_cur = check_vector_3(pose_out.get("t", None), name="pose_out['t']", dtype=float, finite=True)

        landmark_id_by_feat1 = check_index_array_1d(
            seed.get("landmark_id_by_feat1", np.zeros((0,), dtype=np.int64)),
            name="seed['landmark_id_by_feat1']",
            dtype=np.int64,
            allow_negative=True,
        )

        kf_feat_idx_track = check_index_array_1d(
            track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)),
            name="track_out['kf_feat_idx']",
            dtype=np.int64,
            allow_negative=True,
        )

        cur_feat_idx_track = check_index_array_1d(
            track_out.get("cur_feat_idx", np.zeros((0,), dtype=np.int64)),
            name="track_out['cur_feat_idx']",
            dtype=np.int64,
            allow_negative=True,
        )

        kf_feat_idx_track, cur_feat_idx_track = check_1d_pair_same_length(
            kf_feat_idx_track,
            cur_feat_idx_track,
            nameA="track_out['kf_feat_idx']",
            nameB="track_out['cur_feat_idx']",
        )

        M = int(kf_feat_idx_track.size)
        xy_cur_track = check_points_xy_N2_rows(
            track_out.get("xy_cur", np.zeros((0, 2), dtype=np.float64)),
            M,
            name="track_out['xy_cur']",
            dtype=float,
            finite=True,
        )

        valid_kf = (kf_feat_idx_track >= 0) & (kf_feat_idx_track < int(landmark_id_by_feat1.size))
        valid_cur = cur_feat_idx_track >= 0
        mapped_existing = np.zeros((M,), dtype=bool)
        if np.any(valid_kf):
            mapped_existing[valid_kf] = landmark_id_by_feat1[kf_feat_idx_track[valid_kf]] >= 0

        append_candidate_mask = valid_kf & valid_cur & mapped_existing
        append_candidate_idx = np.flatnonzero(append_candidate_mask).astype(np.int64, copy=False)
        stats["n_append_candidates_existing"] = int(append_candidate_idx.size)

        X_cols: list[np.ndarray] = []
        x_cols: list[np.ndarray] = []
        lm_ids_extra: list[int] = []
        feat_idx_extra: list[int] = []

        for track_i in append_candidate_idx:
            lm_id = int(landmark_id_by_feat1[int(kf_feat_idx_track[track_i])])
            feat_idx = int(cur_feat_idx_track[track_i])
            if (int(lm_id), int(feat_idx)) in pnp_inlier_pairs:
                continue

            lm_pos = landmark_pos_by_id.get(int(lm_id), None)
            if lm_pos is None:
                stats["n_missing_landmark"] += 1
                continue

            lm = landmarks[lm_pos]
            X_w_i = np.asarray(lm.get("X_w", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
            if X_w_i.size != 3:
                continue
            if not np.isfinite(X_w_i).all():
                continue

            x_i = np.asarray(xy_cur_track[track_i, :2], dtype=np.float64).reshape(2,)
            if not np.isfinite(x_i).all():
                continue

            X_cols.append(X_w_i.reshape(3, 1))
            x_cols.append(x_i.reshape(2, 1))
            lm_ids_extra.append(int(lm_id))
            feat_idx_extra.append(int(feat_idx))

        stats["n_append_extra_reproj_tested"] = int(len(lm_ids_extra))

        if len(lm_ids_extra) > 0:
            X_extra = np.hstack(X_cols)
            x_extra = np.hstack(x_cols)

            X_c_extra = world_to_camera_points(R_cur, t_cur, X_extra)
            err_sq = np.asarray(reprojection_errors_sq(K_checked, R_cur, t_cur, X_extra, x_extra), dtype=np.float64).reshape(-1)
            err_sq[~np.isfinite(err_sq)] = np.inf

            pass_mask = (
                np.isfinite(err_sq)
                & (err_sq <= float(max_append_reproj_error_px_existing ** 2))
                & np.isfinite(X_c_extra).all(axis=0)
                & (X_c_extra[2, :] > float(eps))
            )
            stats["n_append_extra_reproj_pass"] = int(np.sum(pass_mask))

            for extra_i in np.flatnonzero(pass_mask):
                _append_current_observation(
                    int(lm_ids_extra[int(extra_i)]),
                    int(feat_idx_extra[int(extra_i)]),
                    np.asarray(x_extra[:, int(extra_i)], dtype=np.float64),
                    pnp_inlier=False,
                    extra_reproj=True,
                )

    # Pack back into the seed
    seed["landmarks"] = landmarks
    stats["n_landmarks_with_obs_current_kf_after_append"] = _count_landmarks_with_current_observation()
    seed["last_tracked_observation_append_stats"] = stats

    return seed, stats


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
    C_kf = camera_centre(R_kf, t_kf)
    C_cur = camera_centre(R_cur, t_cur)

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
    cos_parallax[valid_ray_norms] = np.sum(v1[:, valid_ray_norms] * v2[:, valid_ray_norms], axis=0) / (
        n1[valid_ray_norms] * n2[valid_ray_norms]
    )
    cos_parallax = np.clip(cos_parallax, -1.0, 1.0)

    parallax_deg = np.degrees(np.arccos(cos_parallax))
    parallax_mask = cheirality_mask & valid_ray_norms & (parallax_deg >= float(min_parallax_deg))

    # Depth sanity gate relative to the baseline
    depth_ratio = np.maximum(z_kf, z_cur) / max(baseline, float(eps))
    depth_mask = cheirality_mask & np.isfinite(depth_ratio) & (depth_ratio <= float(max_depth_ratio))

    # Optional reprojection gate
    if max_reproj_error_px is None:
        reproj_mask = np.ones((N,), dtype=bool)
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


# Append newly triangulated landmarks into the seed map
def append_new_landmarks_to_seed(
    seed: dict,
    batch: TriangulatedLandmarkBatch,
    *,
    keyframe_kf: int = 1,
    current_kf: int = -1,
    descriptor_source=None,
) -> dict:
    # --- Checks ---
    # Check containers
    seed = check_required_keys(seed, {"landmarks", "landmark_id_by_feat1"}, name="seed")

    # Check triangulated batch
    if not isinstance(batch, TriangulatedLandmarkBatch):
        raise ValueError("batch must be a TriangulatedLandmarkBatch bundle")

    # Check frame indices
    keyframe_kf = check_int_ge0(keyframe_kf, name="keyframe_kf")
    current_kf = int(current_kf)
    if current_kf < -1:
        raise ValueError(f"current_kf must be >= -1; got {current_kf}")

    # Read seed landmarks
    landmarks_raw = seed["landmarks"]
    if not isinstance(landmarks_raw, list):
        raise ValueError("seed['landmarks'] must be a list")
    landmarks = list(landmarks_raw)

    # Read and check lookup
    landmark_id_by_feat1 = check_index_array_1d(
        seed["landmark_id_by_feat1"],
        name="seed['landmark_id_by_feat1']",
        dtype=np.int64,
        allow_negative=True,
    ).copy()

    # Check batch arrays
    track_idx = check_index_array_1d(batch.track_idx, name="batch.track_idx", dtype=np.int64, allow_negative=False)
    kf_feat_idx = check_index_array_1d(batch.kf_feat_idx, name="batch.kf_feat_idx", dtype=np.int64, allow_negative=False)
    cur_feat_idx = check_index_array_1d(batch.cur_feat_idx, name="batch.cur_feat_idx", dtype=np.int64, allow_negative=False)

    track_idx, kf_feat_idx = check_1d_pair_same_length(
        track_idx,
        kf_feat_idx,
        nameA="batch.track_idx",
        nameB="batch.kf_feat_idx",
    )
    track_idx, cur_feat_idx = check_1d_pair_same_length(
        track_idx,
        cur_feat_idx,
        nameA="batch.track_idx",
        nameB="batch.cur_feat_idx",
    )

    # Check 2D/3D data
    x_kf, x_cur = check_2xN_pair(batch.x_kf, batch.x_cur, dtype=float, finite=True)
    X_w = np.asarray(batch.X_w, dtype=np.float64)
    if X_w.ndim != 2 or X_w.shape[0] != 3:
        raise ValueError(f"batch.X_w must be (3,N); got {X_w.shape}")
    if X_w.shape[1] != int(track_idx.size):
        raise ValueError(
            f"batch.X_w must have {track_idx.size} columns to match batch.track_idx; got {X_w.shape[1]}"
        )
    if x_kf.shape[1] != int(track_idx.size):
        raise ValueError(
            f"batch.x_kf must have {track_idx.size} columns to match batch.track_idx; got {x_kf.shape[1]}"
        )

    # Check feature-index bounds against the seed lookup
    if track_idx.size > 0:
        if int(kf_feat_idx.max()) >= int(landmark_id_by_feat1.size):
            raise ValueError(
                f"batch.kf_feat_idx contains index {int(kf_feat_idx.max())} outside seed lookup size {landmark_id_by_feat1.size}"
            )

    # Read descriptor source if supplied
    desc = None
    if descriptor_source is not None:
        desc = np.asarray(getattr(descriptor_source, "desc", np.zeros((0,), dtype=np.float64)))

    # Next landmark id
    existing_ids = [int(lm["id"]) for lm in landmarks if isinstance(lm, dict) and "id" in lm]
    next_id = 0 if len(existing_ids) == 0 else (max(existing_ids) + 1)

    # Append each triangulated landmark
    n_added = 0
    added_ids = []

    for i in range(int(track_idx.size)):
        # Read feature indices
        feat_kf = int(kf_feat_idx[i])
        feat_cur = int(cur_feat_idx[i])

        # Skip if the keyframe feature is already assigned
        if int(landmark_id_by_feat1[feat_kf]) >= 0:
            continue

        # Read world point and image observations
        X_i = np.asarray(X_w[:, i], dtype=np.float64)
        x_kf_i = np.asarray(x_kf[:, i], dtype=np.float64)
        x_cur_i = np.asarray(x_cur[:, i], dtype=np.float64)

        # Optional descriptor copy from the current-frame feature index
        descriptor_i = None
        if desc.ndim >= 1 and feat_cur < int(desc.shape[0]):
            descriptor_i = np.asarray(desc[feat_cur]).copy()

        # Build observation list
        obs = [
            {"kf": int(keyframe_kf), "feat": feat_kf, "xy": x_kf_i},
            {"kf": int(current_kf), "feat": feat_cur, "xy": x_cur_i},
        ]

        # Build landmark dict in the same style as seed.py
        lm_id = int(next_id)
        landmark = {
            "id": lm_id,
            "X_w": X_i,
            "birth_source": "map_growth",
            "birth_kf": int(current_kf),
            "obs": obs,
            "descriptor": descriptor_i,
            "quality": {
                "reproj0_px": None,
                "reproj1_px": None,
            },
        }

        # Append to the landmark list
        landmarks.append(landmark)

        # Update the keyframe feature lookup
        landmark_id_by_feat1[feat_kf] = lm_id

        # Advance counters
        added_ids.append(lm_id)
        next_id += 1
        n_added += 1

    # Pack back into the seed
    seed["landmarks"] = landmarks
    seed["landmark_id_by_feat1"] = landmark_id_by_feat1

    # Store append stats for debugging
    seed["last_append_stats"] = {
        "n_in_batch": int(track_idx.size),
        "n_added": int(n_added),
        "added_ids": np.asarray(added_ids, dtype=np.int64),
        "keyframe_kf": int(keyframe_kf),
        "current_kf": int(current_kf),
    }

    return seed


# Run one complete map-growth step from a tracked frame
def grow_map_from_tracking_result(
    seed: dict,
    track_out: dict,
    K_kf,
    K_cur,
    R_kf,
    t_kf,
    R_cur,
    t_cur,
    *,
    keyframe_kf: int = 1,
    current_kf: int = -1,
    descriptor_source=None,
    min_parallax_deg: float = 1.0,
    max_depth_ratio: float = 200.0,
    max_reproj_error_px: float | None = 3.0,
    eps: float = 1e-12,
) -> MapGrowthResult:
    # --- Checks ---
    # Check containers
    seed = check_required_keys(seed, {"landmarks", "landmark_id_by_feat1"}, name="seed")
    track_out = check_dict(track_out, name="track_out")

    # Use current-frame features as the default descriptor source when available
    if descriptor_source is None and "cur_feats" in track_out:
        descriptor_source = track_out["cur_feats"]

    # Build candidate 2D-2D correspondences that are not already mapped
    candidates = build_new_landmark_candidates(seed, track_out)

    # Triangulate and filter the candidate set
    batch = triangulate_new_landmarks(
        K_kf,
        K_cur,
        R_kf,
        t_kf,
        R_cur,
        t_cur,
        candidates,
        min_parallax_deg=min_parallax_deg,
        max_depth_ratio=max_depth_ratio,
        max_reproj_error_px=max_reproj_error_px,
        eps=eps,
    )

    # Append valid landmarks into the seed
    seed_before = int(len(seed["landmarks"]))
    seed = append_new_landmarks_to_seed(
        seed,
        batch,
        keyframe_kf=keyframe_kf,
        current_kf=current_kf,
        descriptor_source=descriptor_source,
    )
    seed_after = int(len(seed["landmarks"]))

    # Read append stats
    append_stats = seed.get("last_append_stats", {})
    n_added = int(append_stats.get("n_added", max(seed_after - seed_before, 0)))

    # Pack step-level stats
    stats = {
        "n_candidates": int(candidates.track_idx.size),
        "n_triangulated_valid": int(batch.stats.get("n_valid", 0)),
        "n_added": int(n_added),
        "seed_landmarks_before": int(seed_before),
        "seed_landmarks_after": int(seed_after),
        "reason": batch.stats.get("reason", None),
    }

    return MapGrowthResult(
        seed=seed,
        candidates=candidates,
        batch=batch,
        stats=stats,
    )
