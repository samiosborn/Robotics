# src/slam/map_update.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import as_2xN_points, check_1d_pair_same_length, check_2xN_pair, check_dict, check_index_array_1d, check_int_ge0, check_mask_bool_N, check_matrix_3x3, check_points_2xN, check_points_xy_N2_rows, check_positive, check_required_keys, check_vector_3
from geometry.camera import camera_centre, projection_matrix, reprojection_errors_sq, world_to_camera_points
from geometry.triangulation import triangulate_points
from slam.keyframe_state import get_active_keyframe_kf, get_rebuilt_active_landmark_lookup, has_active_keyframe_state, rebuild_active_landmark_lookup
from slam.landmark_state import add_landmark_observation, build_landmark_id_index, build_observation_indexes, count_valid_landmark_observations, create_landmark_record, get_landmarks, next_landmark_id
from slam.map_mutation import new_map_mutation_report


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
    # Diagnostic and acceptance statistics
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
    # Explicit map mutation report
    mutation_report: dict


# Resolve public keyframe ids against canonical active state
def _resolve_active_keyframe_kf_argument(seed: dict, keyframe_kf: int, *, context: str) -> int:
    keyframe_kf = check_int_ge0(keyframe_kf, name="keyframe_kf")
    if not has_active_keyframe_state(seed):
        return int(keyframe_kf)

    active_kf = get_active_keyframe_kf(seed)
    if int(active_kf) != int(keyframe_kf):
        raise ValueError(
            f"{context} keyframe_kf argument must match active keyframe state; got {int(keyframe_kf)} and seed active {int(active_kf)}"
        )

    return int(active_kf)


# Rebuild active lookup cache before runtime use
def _active_lookup_for_runtime(seed: dict, *, context: str) -> np.ndarray:
    if not has_active_keyframe_state(seed):
        return np.zeros((0,), dtype=np.int64)

    return get_rebuilt_active_landmark_lookup(seed, context=str(context))


# Append current-frame observations for already-known tracked landmarks
def append_tracked_observations_to_seed(
    seed: dict,
    pose_out: dict,
    *,
    keyframe_kf: int = 1,
    current_kf: int,
    K: np.ndarray | None = None,
    track_out: dict | None = None,
    max_append_reproj_error_px_existing: float = 2.0,
    prune_stale_map_growth: bool = True,
    eps: float = 1e-12,
    return_report: bool = False,
) -> tuple[dict, dict] | tuple[dict, dict, dict]:
    # --- Checks ---
    # Check containers
    seed = check_required_keys(seed, {"landmarks"}, name="seed")
    pose_out = check_required_keys(pose_out, {"corrs", "pnp_inlier_mask"}, name="pose_out")

    # Check frame index
    keyframe_kf = _resolve_active_keyframe_kf_argument(seed, keyframe_kf, context="tracked observation append")
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

    # Require the PnP inlier mask to match the correspondence count
    pnp_inlier_mask = check_mask_bool_N(
        pose_out["pnp_inlier_mask"],
        N,
        name="pose_out['pnp_inlier_mask']",
    )
    if pnp_inlier_mask is None:
        pnp_inlier_mask = np.zeros((N,), dtype=bool)

    # Read seed landmarks and disposable graph indexes
    landmarks = list(get_landmarks(seed))
    landmark_seed = {"landmarks": landmarks}
    landmark_by_id = build_landmark_id_index(landmark_seed, context="seed['landmarks']")
    observation_indexes = build_observation_indexes(landmark_seed, context="seed['landmarks']")
    assignment_by_feature = dict(observation_indexes["landmark_id_by_feature"])
    active_lookup_raw = None
    if has_active_keyframe_state(seed):
        active_lookup_raw = _active_lookup_for_runtime(seed, context="tracked observation append active lookup")
    active_lookup = None
    if active_lookup_raw is not None:
        active_lookup = check_index_array_1d(
            active_lookup_raw,
            name="active keyframe lookup",
            dtype=np.int64,
            allow_negative=True,
        ).copy()

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
        "n_stale_map_growth_removed": 0,
    }
    mutation_report = new_map_mutation_report(context="tracked_observation_append")

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
        lm = landmark_by_id.get(int(lm_id), None)
        if lm is None:
            stats["n_missing_landmark"] += 1
            mutation_report["missing_landmarks"] += 1
            return False

        try:
            added = add_landmark_observation(
                lm,
                int(current_kf),
                int(feat_idx),
                xy,
                assignment_by_feature=assignment_by_feature,
                context="tracked observation",
            )
        except ValueError as exc:
            if "feature assignment conflict" in str(exc):
                mutation_report["feature_assignment_conflicts"] += 1
            raise
        if not bool(added):
            stats["n_duplicate"] += 1
            stats["n_append_duplicates"] += 1
            mutation_report["skipped_duplicate_observations"] += 1
            return False

        stats["n_added"] += 1
        stats["n_append_total"] += 1
        mutation_report["added_observations"] += 1
        if bool(pnp_inlier):
            stats["n_append_pnp_inliers_added"] += 1
        if bool(extra_reproj):
            stats["n_append_extra_reproj_added"] += 1
        _record_birth_source_append(lm, pnp_inlier=bool(pnp_inlier), extra_reproj=bool(extra_reproj))

        return True

    # Hard-remove stale two-view map-growth landmarks and clear the active lookup
    def _prune_stale_map_growth_landmarks() -> None:
        nonlocal landmarks, active_lookup

        kept_landmarks: list = []
        n_removed = 0
        n_lookup_cleared = 0

        for lm in landmarks:
            if not isinstance(lm, dict):
                kept_landmarks.append(lm)
                continue

            birth_source = lm.get("birth_source", None)
            n_obs = count_valid_landmark_observations(lm, context="stale map-growth landmark")
            obs = lm.get("obs", None)

            should_remove = (
                birth_source == "map_growth"
                and n_obs == 2
                and (int(current_kf) - int(lm.get("birth_kf", -1)) >= 2)
            )
            if not should_remove:
                kept_landmarks.append(lm)
                continue

            lm_id = int(lm.get("id", -1))
            if active_lookup is not None and isinstance(obs, list) and lm_id >= 0:
                for ob in obs:
                    if not isinstance(ob, dict):
                        continue
                    if int(ob.get("kf", -1)) != int(keyframe_kf):
                        continue
                    feat = int(ob.get("feat", -1))
                    if feat < 0 or feat >= int(active_lookup.size):
                        continue
                    if int(active_lookup[feat]) == lm_id:
                        active_lookup[feat] = -1
                        n_lookup_cleared += 1

            n_removed += 1

        landmarks = kept_landmarks
        stats["n_stale_map_growth_removed"] = int(n_removed)
        mutation_report["removed_landmarks"] += int(n_removed)
        mutation_report["updated_active_lookup_entries"] += int(n_lookup_cleared)

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

        active_lookup_extra_raw = _active_lookup_for_runtime(seed, context="tracked observation extra active lookup")
        active_lookup_extra = check_index_array_1d(
            active_lookup_extra_raw,
            name="active keyframe lookup",
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

        valid_kf = (kf_feat_idx_track >= 0) & (kf_feat_idx_track < int(active_lookup_extra.size))
        valid_cur = cur_feat_idx_track >= 0
        mapped_existing = np.zeros((M,), dtype=bool)
        if np.any(valid_kf):
            mapped_existing[valid_kf] = active_lookup_extra[kf_feat_idx_track[valid_kf]] >= 0

        append_candidate_mask = valid_kf & valid_cur & mapped_existing
        append_candidate_idx = np.flatnonzero(append_candidate_mask).astype(np.int64, copy=False)
        stats["n_append_candidates_existing"] = int(append_candidate_idx.size)

        X_cols: list[np.ndarray] = []
        x_cols: list[np.ndarray] = []
        lm_ids_extra: list[int] = []
        feat_idx_extra: list[int] = []

        for track_i in append_candidate_idx:
            lm_id = int(active_lookup_extra[int(kf_feat_idx_track[track_i])])
            feat_idx = int(cur_feat_idx_track[track_i])
            if (int(lm_id), int(feat_idx)) in pnp_inlier_pairs:
                continue

            lm = landmark_by_id.get(int(lm_id), None)
            if lm is None:
                stats["n_missing_landmark"] += 1
                mutation_report["missing_landmarks"] += 1
                continue

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

    if bool(prune_stale_map_growth):
        _prune_stale_map_growth_landmarks()

    # Pack back into the seed
    seed["landmarks"] = landmarks
    if active_lookup is not None:
        rebuild_active_landmark_lookup(seed, context="tracked observation append active lookup")
    stats["n_landmarks_with_obs_current_kf_after_append"] = _count_landmarks_with_current_observation()
    stats["mutation_report"] = mutation_report
    seed["last_tracked_observation_append_stats"] = stats
    seed["last_tracked_observation_append_report"] = mutation_report

    if bool(return_report):
        return seed, stats, mutation_report

    return seed, stats


# Build candidate 2D-2D correspondences for new landmark triangulation
def build_new_landmark_candidates(seed: dict, track_out: dict) -> NewLandmarkCandidates:
    # --- Checks ---
    # Check containers
    seed = check_dict(seed, name="seed")
    track_out = check_dict(track_out, name="track_out")

    # Read landmark lookup from keyframe feature index to landmark id
    active_lookup = check_index_array_1d(
        _active_lookup_for_runtime(seed, context="new landmark candidate active lookup"),
        name="active keyframe lookup",
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
    valid_kf = (kf_feat_idx >= 0) & (kf_feat_idx < int(active_lookup.size))

    # Valid current-frame feature indices must be non-negative
    valid_cur = (cur_feat_idx >= 0)

    # A tracked keyframe feature is a candidate only if it is currently unmapped
    unmapped = np.zeros((M,), dtype=bool)
    if np.any(valid_kf):
        unmapped[valid_kf] = (active_lookup[kf_feat_idx[valid_kf]] < 0)

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
    return_report: bool = False,
) -> dict | tuple[dict, dict]:
    # --- Checks ---
    # Check containers
    seed = check_required_keys(seed, {"landmarks"}, name="seed")

    # Check triangulated batch
    if not isinstance(batch, TriangulatedLandmarkBatch):
        raise ValueError("batch must be a TriangulatedLandmarkBatch bundle")

    # Check frame indices
    keyframe_kf = _resolve_active_keyframe_kf_argument(seed, keyframe_kf, context="map-growth append")
    current_kf = int(current_kf)
    if current_kf < -1:
        raise ValueError(f"current_kf must be >= -1; got {current_kf}")

    # Read seed landmarks and disposable graph indexes
    landmarks = list(get_landmarks(seed))
    landmark_seed = {"landmarks": landmarks}
    landmark_by_id = build_landmark_id_index(landmark_seed, context="seed['landmarks']")
    observation_indexes = build_observation_indexes(landmark_seed, context="seed['landmarks']")
    assignment_by_feature = dict(observation_indexes["landmark_id_by_feature"])
    mutation_report = new_map_mutation_report(context="map_growth_append")

    # Read and check lookup
    active_lookup = check_index_array_1d(
        _active_lookup_for_runtime(seed, context="map-growth append active lookup"),
        name="active keyframe lookup",
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
        if int(kf_feat_idx.max()) >= int(active_lookup.size):
            raise ValueError(
                f"batch.kf_feat_idx contains index {int(kf_feat_idx.max())} outside active lookup size {active_lookup.size}"
            )

    # Read descriptor source if supplied
    desc = None
    if descriptor_source is not None:
        desc = np.asarray(getattr(descriptor_source, "desc", np.zeros((0,), dtype=np.float64)))

    # Next landmark id
    next_id = next_landmark_id(landmark_seed)

    # Append each triangulated landmark
    n_added = 0
    added_ids = []

    for i in range(int(track_idx.size)):
        # Read feature indices
        feat_kf = int(kf_feat_idx[i])
        feat_cur = int(cur_feat_idx[i])

        # Skip if the keyframe feature is already assigned
        if int(active_lookup[feat_kf]) >= 0:
            mutation_report["skipped_landmark_candidates"] += 1
            mutation_report["skipped_mapped_keyframe_features"] += 1
            mutation_report["feature_assignment_conflicts"] += 1
            continue

        # Read world point and image observations
        X_i = np.asarray(X_w[:, i], dtype=np.float64)
        x_kf_i = np.asarray(x_kf[:, i], dtype=np.float64)
        x_cur_i = np.asarray(x_cur[:, i], dtype=np.float64)

        # Optional descriptor copy from the current-frame feature index
        descriptor_i = None
        if desc is not None and desc.ndim >= 1 and feat_cur < int(desc.shape[0]):
            descriptor_i = np.asarray(desc[feat_cur]).copy()

        # Build landmark dict in the same style as seed.py
        lm_id = int(next_id)
        if int(lm_id) in landmark_by_id:
            raise ValueError(f"seed['landmarks'] already contains landmark id {lm_id}")
        landmark = create_landmark_record(
            lm_id,
            X_i,
            birth_source="map_growth",
            birth_kf=int(current_kf),
            descriptor=descriptor_i,
            quality={
                "reproj0_px": None,
                "reproj1_px": None,
            },
            context="map-growth landmark",
        )
        try:
            added_keyframe_obs = add_landmark_observation(
                landmark,
                int(keyframe_kf),
                feat_kf,
                x_kf_i,
                assignment_by_feature=assignment_by_feature,
                context="map-growth keyframe observation",
            )
            added_current_obs = add_landmark_observation(
                landmark,
                int(current_kf),
                feat_cur,
                x_cur_i,
                assignment_by_feature=assignment_by_feature,
                context="map-growth current observation",
            )
        except ValueError as exc:
            if "feature assignment conflict" in str(exc):
                mutation_report["feature_assignment_conflicts"] += 1
            raise
        if not bool(added_keyframe_obs and added_current_obs):
            mutation_report["skipped_duplicate_observations"] += int(not bool(added_keyframe_obs))
            mutation_report["skipped_duplicate_observations"] += int(not bool(added_current_obs))
            raise ValueError("map-growth landmark observations must be unique")

        # Append to the landmark list
        landmarks.append(landmark)
        landmark_by_id[int(lm_id)] = landmark

        # Update the keyframe feature lookup
        active_lookup[feat_kf] = lm_id

        # Advance counters
        added_ids.append(lm_id)
        next_id += 1
        n_added += 1
        mutation_report["added_landmarks"] += 1
        mutation_report["added_observations"] += 2
        mutation_report["updated_active_lookup_entries"] += 1

    # Pack back into the seed
    seed["landmarks"] = landmarks
    rebuild_active_landmark_lookup(seed, context="map-growth append active lookup")

    # Store append presentation diagnostics
    seed["last_append_stats"] = {
        "n_in_batch": int(track_idx.size),
        "n_added": int(n_added),
        "n_skipped": int(mutation_report["skipped_landmark_candidates"]),
        "n_feature_assignment_conflicts": int(mutation_report["feature_assignment_conflicts"]),
        "added_ids": np.asarray(added_ids, dtype=np.int64),
        "keyframe_kf": int(keyframe_kf),
        "current_kf": int(current_kf),
        "mutation_report": mutation_report,
    }
    seed["last_append_mutation_report"] = mutation_report

    if bool(return_report):
        return seed, mutation_report

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
    seed = check_required_keys(seed, {"landmarks"}, name="seed")
    track_out = check_dict(track_out, name="track_out")
    keyframe_kf = _resolve_active_keyframe_kf_argument(seed, keyframe_kf, context="map growth")

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
    seed_before = int(len(get_landmarks(seed)))
    seed, mutation_report = append_new_landmarks_to_seed(
        seed,
        batch,
        keyframe_kf=keyframe_kf,
        current_kf=current_kf,
        descriptor_source=descriptor_source,
        return_report=True,
    )
    seed_after = int(len(get_landmarks(seed)))

    # Read mutation count from the explicit report
    n_added = int(mutation_report["added_landmarks"])

    # Pack step-level stats
    stats = {
        "n_candidates": int(candidates.track_idx.size),
        "n_triangulated_valid": int(batch.stats.get("n_valid", 0)),
        "n_added": int(n_added),
        "seed_landmarks_before": int(seed_before),
        "seed_landmarks_after": int(seed_after),
        "reason": batch.stats.get("reason", None),
        "mutation_report": mutation_report,
    }

    return MapGrowthResult(
        seed=seed,
        candidates=candidates,
        batch=batch,
        stats=stats,
        mutation_report=mutation_report,
    )
