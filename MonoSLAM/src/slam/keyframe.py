# src/slam/keyframe.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import check_int_ge0, check_matrix_3x3, check_points_xy_N2plus, check_positive, check_required_keys, check_vector_3
from geometry.camera import camera_centre
from geometry.rotation import angle_between_rotmats
from slam.map_update import MapGrowthResult
from slam.seed import seed_keyframe_pose


_PROMOTION_COVERAGE_GRID_COLS = 4
_PROMOTION_COVERAGE_GRID_ROWS = 3
_MIN_LINKED_OBS_CELLS_FOR_PROMOTION = 4
_MIN_LINKED_OBS_BBOX_AREA_FRACTION_FOR_PROMOTION = 0.15


# Keyframe decision bundle
@dataclass(frozen=True)
class KeyframeDecision:
    # Whether the current frame should become a new keyframe
    make_keyframe: bool
    # Short reason for the decision
    reason: str | None
    # Diagnostic statistics used by the decision rule
    stats: dict


# Keyframe update result
@dataclass(frozen=True)
class KeyframeUpdateResult:
    # Updated seed after the keyframe decision
    seed: dict
    # Decision bundle
    decision: KeyframeDecision
    # Whether promotion actually happened
    promoted: bool
    # Step-level stats
    stats: dict


# Build the feature-index to landmark-id lookup for a given keyframe id
def _build_landmark_id_by_feat_for_kf(seed: dict, n_feat: int, kf_index: int) -> np.ndarray:
    # Check seed contains landmarks
    seed = check_required_keys(seed, {"landmarks"}, name="seed")

    # Check sizes
    n_feat = check_int_ge0(n_feat, name="n_feat")
    kf_index = check_int_ge0(kf_index, name="kf_index")

    # Read landmarks
    landmarks = seed["landmarks"]
    if not isinstance(landmarks, list):
        raise ValueError("seed['landmarks'] must be a list")

    # Initialise lookup as unmapped
    landmark_id_by_feat = np.full((n_feat,), -1, dtype=np.int64)

    # Scan all landmarks for observations in this keyframe
    for lm in landmarks:
        # Skip malformed landmarks
        if not isinstance(lm, dict):
            continue
        if "id" not in lm:
            continue

        # Read landmark id
        lm_id = int(lm["id"])

        # Read observation list
        obs = lm.get("obs", None)
        if not isinstance(obs, list):
            continue

        # Scan observations for this keyframe
        for ob in obs:
            # Skip malformed observations
            if not isinstance(ob, dict):
                continue

            # Keep only observations from the requested keyframe
            if int(ob.get("kf", -1)) != kf_index:
                continue

            # Read feature index
            feat = int(ob.get("feat", -1))
            if feat < 0 or feat >= n_feat:
                continue

            # Reject conflicting assignments
            prev = int(landmark_id_by_feat[feat])
            if prev >= 0 and prev != lm_id:
                raise ValueError(
                    f"Feature {feat} in keyframe {kf_index} is assigned to multiple landmarks: {prev} and {lm_id}"
                )

            # Store the landmark id for this feature
            landmark_id_by_feat[feat] = lm_id

    return landmark_id_by_feat


# Count landmarks that have an observation in a given keyframe
def count_linked_landmarks_for_kf(seed: dict, kf_index: int) -> int:
    # Check seed contains landmarks
    seed = check_required_keys(seed, {"landmarks"}, name="seed")

    # Check keyframe index
    kf_index = check_int_ge0(kf_index, name="kf_index")

    # Read landmarks
    landmarks = seed["landmarks"]
    if not isinstance(landmarks, list):
        raise ValueError("seed['landmarks'] must be a list")

    # Count linked landmarks
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
            if int(ob.get("kf", -1)) != int(kf_index):
                continue
            linked = True
            break

        if linked:
            n_linked += 1

    return int(n_linked)


# Summarise spatial coverage of landmark observations in one frame
def _linked_landmark_observation_coverage(seed: dict, kf_index: int, image_shape) -> dict:
    # Check seed contains landmarks
    seed = check_required_keys(seed, {"landmarks"}, name="seed")

    # Check keyframe index
    kf_index = check_int_ge0(kf_index, name="kf_index")

    if image_shape is None:
        return {
            "evaluated": False,
            "reason": "image_shape_unavailable",
            "n_points": 0,
            "bbox_area_fraction": None,
            "occupied_cells": 0,
            "grid_cols": int(_PROMOTION_COVERAGE_GRID_COLS),
            "grid_rows": int(_PROMOTION_COVERAGE_GRID_ROWS),
        }

    shape = tuple(image_shape)
    if len(shape) < 2:
        raise ValueError(f"image_shape must have at least two entries; got {image_shape}")

    H = int(shape[0])
    W = int(shape[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"image_shape height and width must be positive; got {image_shape}")

    landmarks = seed["landmarks"]
    if not isinstance(landmarks, list):
        raise ValueError("seed['landmarks'] must be a list")

    xy_rows: list[np.ndarray] = []
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue

        obs = lm.get("obs", None)
        if not isinstance(obs, list):
            continue

        for ob in obs:
            if not isinstance(ob, dict):
                continue
            if int(ob.get("kf", -1)) != int(kf_index):
                continue

            xy = np.asarray(ob.get("xy", np.zeros((2,), dtype=np.float64)), dtype=np.float64).reshape(-1)
            if xy.size < 2:
                continue
            xy = np.asarray(xy[:2], dtype=np.float64)
            if not np.isfinite(xy).all():
                continue

            xy_rows.append(xy.reshape(1, 2))
            break

    if len(xy_rows) == 0:
        return {
            "evaluated": True,
            "reason": "no_linked_observations",
            "n_points": 0,
            "bbox_area_fraction": 0.0,
            "occupied_cells": 0,
            "grid_cols": int(_PROMOTION_COVERAGE_GRID_COLS),
            "grid_rows": int(_PROMOTION_COVERAGE_GRID_ROWS),
        }

    xy_all = np.vstack(xy_rows)
    x = xy_all[:, 0]
    y = xy_all[:, 1]

    bbox_w = float(np.max(x) - np.min(x))
    bbox_h = float(np.max(y) - np.min(y))
    bbox_area_fraction = float((bbox_w * bbox_h) / max(float(W * H), 1.0))

    cols = np.floor((x / max(float(W), 1.0)) * int(_PROMOTION_COVERAGE_GRID_COLS)).astype(np.int64)
    rows = np.floor((y / max(float(H), 1.0)) * int(_PROMOTION_COVERAGE_GRID_ROWS)).astype(np.int64)
    cols = np.clip(cols, 0, int(_PROMOTION_COVERAGE_GRID_COLS) - 1)
    rows = np.clip(rows, 0, int(_PROMOTION_COVERAGE_GRID_ROWS) - 1)
    occupied = {(int(row), int(col)) for row, col in zip(rows, cols)}

    return {
        "evaluated": True,
        "reason": None,
        "n_points": int(xy_all.shape[0]),
        "bbox_area_fraction": float(bbox_area_fraction),
        "occupied_cells": int(len(occupied)),
        "grid_cols": int(_PROMOTION_COVERAGE_GRID_COLS),
        "grid_rows": int(_PROMOTION_COVERAGE_GRID_ROWS),
    }


# Read map-growth stats from either a dict or a typed result bundle
def _map_growth_stats(map_growth_out: dict | MapGrowthResult | None) -> dict:
    if map_growth_out is None:
        return {}
    if isinstance(map_growth_out, MapGrowthResult):
        return map_growth_out.stats if isinstance(map_growth_out.stats, dict) else {}
    if isinstance(map_growth_out, dict):
        stats = map_growth_out.get("stats", {})
        return stats if isinstance(stats, dict) else {}
    raise ValueError("map_growth_out must be a dict, MapGrowthResult, or None")


# Decide whether the current frame should become a new keyframe
def should_make_keyframe(
    seed: dict,
    pose_out: dict,
    track_out: dict,
    *,
    map_growth_out: dict | MapGrowthResult | None = None,
    current_kf: int,
    image_shape: tuple[int, int] | None = None,
    min_track_inliers: int = 80,
    min_pnp_inliers: int = 40,
    min_landmark_growth: int = 20,
    min_linked_landmarks_for_promotion: int = 100,
    min_translation_m: float = 0.10,
    min_rotation_deg: float = 5.0,
    require_pose: bool = True,
) -> KeyframeDecision:
    # --- Checks ---
    # Check required seed structure
    seed = check_required_keys(seed, {"T_WC1", "landmarks"}, name="seed")

    # Check container types
    if not isinstance(pose_out, dict):
        raise ValueError("pose_out must be a dict")
    if not isinstance(track_out, dict):
        raise ValueError("track_out must be a dict")
    _map_growth_stats(map_growth_out)

    # Check frame index
    current_kf = check_int_ge0(current_kf, name="current_kf")

    # Check scalar thresholds
    min_track_inliers = check_int_ge0(min_track_inliers, name="min_track_inliers")
    min_pnp_inliers = check_int_ge0(min_pnp_inliers, name="min_pnp_inliers")
    min_landmark_growth = check_int_ge0(min_landmark_growth, name="min_landmark_growth")
    min_linked_landmarks_for_promotion = check_int_ge0(
        min_linked_landmarks_for_promotion,
        name="min_linked_landmarks_for_promotion",
    )
    min_translation_m = check_positive(min_translation_m, name="min_translation_m", eps=0.0)
    min_rotation_deg = check_positive(min_rotation_deg, name="min_rotation_deg", eps=0.0)

    # Read stats dicts
    track_stats = track_out.get("stats", {})
    pose_stats = pose_out.get("stats", {})
    map_stats = _map_growth_stats(map_growth_out)

    # Read counts
    n_track_inliers = int(track_stats.get("n_inliers", 0))
    n_pnp_inliers = int(pose_stats.get("n_pnp_inliers", 0))
    n_added = int(map_stats.get("n_added", 0))
    n_landmarks = int(len(seed.get("landmarks", [])))
    n_linked_landmarks_candidate = count_linked_landmarks_for_kf(seed, current_kf)
    linked_coverage = _linked_landmark_observation_coverage(seed, current_kf, image_shape)

    # Read pose status
    pose_ok = bool(pose_out.get("ok", False))

    # Pack initial stats
    stats = {
        "pose_ok": bool(pose_ok),
        "n_track_inliers": int(n_track_inliers),
        "n_pnp_inliers": int(n_pnp_inliers),
        "n_added": int(n_added),
        "n_landmarks": int(n_landmarks),
        "n_linked_landmarks_candidate": int(n_linked_landmarks_candidate),
        "promotion_vetoed_for_low_links": False,
        "promotion_vetoed_for_low_coverage": False,
        "promotion_linked_coverage_evaluated": bool(linked_coverage.get("evaluated", False)),
        "promotion_linked_coverage_reason": linked_coverage.get("reason", None),
        "promotion_linked_coverage_n_points": int(linked_coverage.get("n_points", 0)),
        "promotion_linked_coverage_bbox_area_fraction": linked_coverage.get("bbox_area_fraction", None),
        "promotion_linked_coverage_occupied_cells": int(linked_coverage.get("occupied_cells", 0)),
        "promotion_min_linked_bbox_area_fraction": float(_MIN_LINKED_OBS_BBOX_AREA_FRACTION_FOR_PROMOTION),
        "promotion_min_linked_occupied_cells": int(_MIN_LINKED_OBS_CELLS_FOR_PROMOTION),
        "translation_m": None,
        "rotation_deg": None,
    }

    # Apply the linked-landmark promotion gate
    def _decision(make_keyframe: bool, reason: str | None) -> KeyframeDecision:
        stats_out = dict(stats)
        if bool(make_keyframe) and int(n_linked_landmarks_candidate) < int(min_linked_landmarks_for_promotion):
            stats_out["promotion_vetoed_for_low_links"] = True
            return KeyframeDecision(
                make_keyframe=False,
                reason="linked_landmarks_low",
                stats=stats_out,
            )
        if bool(make_keyframe) and bool(linked_coverage.get("evaluated", False)):
            bbox_area_fraction = linked_coverage.get("bbox_area_fraction", None)
            bbox_low = bbox_area_fraction is None or (
                float(bbox_area_fraction) < float(_MIN_LINKED_OBS_BBOX_AREA_FRACTION_FOR_PROMOTION)
            )
            cells_low = int(linked_coverage.get("occupied_cells", 0)) < int(_MIN_LINKED_OBS_CELLS_FOR_PROMOTION)
            if bool(bbox_low) or bool(cells_low):
                stats_out["promotion_vetoed_for_low_coverage"] = True
                return KeyframeDecision(
                    make_keyframe=False,
                    reason="linked_landmark_coverage_low",
                    stats=stats_out,
                )
        return KeyframeDecision(
            make_keyframe=bool(make_keyframe),
            reason=reason,
            stats=stats_out,
        )

    # If a valid pose is required, stop early when it is unavailable
    if bool(require_pose) and not pose_ok:
        return _decision(False, "pose_not_available")

    # If pose is unavailable, fall back to track quality only
    if not pose_ok:
        if n_track_inliers < min_track_inliers:
            return _decision(True, "track_inliers_low")

        return _decision(False, None)

    # Read the stored keyframe pose
    R_kf, t_kf = seed_keyframe_pose(seed)

    # Read the current pose
    R_cur = check_matrix_3x3(pose_out["R"], name="pose_out['R']", dtype=float, finite=False)
    t_cur = check_vector_3(pose_out["t"], name="pose_out['t']", dtype=float, finite=False)

    # Compute camera centres
    C_kf = camera_centre(R_kf, t_kf)
    C_cur = camera_centre(R_cur, t_cur)

    # Compute motion since the current reference keyframe
    translation_m = float(np.linalg.norm(C_cur - C_kf))
    rotation_deg = float(np.degrees(angle_between_rotmats(R_kf, R_cur)))

    # Update stats with motion
    stats["translation_m"] = float(translation_m)
    stats["rotation_deg"] = float(rotation_deg)

    # Promote when both translation and rotation are clearly significant
    if translation_m >= min_translation_m and rotation_deg >= min_rotation_deg:
        return _decision(True, "translation_and_rotation_large")

    # Promote when translation alone is significant
    if translation_m >= min_translation_m:
        return _decision(True, "translation_large")

    # Promote when rotation alone is significant
    if rotation_deg >= min_rotation_deg:
        return _decision(True, "rotation_large")

    # Promote when the current frame added many new landmarks
    if n_added >= min_landmark_growth:
        return _decision(True, "landmark_growth_high")

    # Promote when keyframe tracking quality has degraded
    if n_track_inliers < min_track_inliers:
        return _decision(True, "track_inliers_low")

    # Promote when pose support has degraded
    if n_pnp_inliers < min_pnp_inliers:
        return _decision(True, "pnp_inliers_low")

    # Otherwise keep the existing keyframe
    return _decision(False, None)


# Promote the current processed frame to become the new reference keyframe
def promote_frame_to_keyframe(
    seed: dict,
    cur_feats,
    R_cur,
    t_cur,
    *,
    current_kf: int,
) -> dict:
    # --- Checks ---
    # Check required seed structure
    seed = check_required_keys(seed, {"landmarks"}, name="seed")

    # Check frame index
    current_kf = check_int_ge0(current_kf, name="current_kf")

    # Check pose
    R_cur = check_matrix_3x3(R_cur, name="R_cur", dtype=float, finite=False)
    t_cur = check_vector_3(t_cur, name="t_cur", dtype=float, finite=False)

    # Check current feature bundle has keypoints
    if not hasattr(cur_feats, "kps_xy"):
        raise ValueError("cur_feats must have attribute 'kps_xy'")

    # Read current keypoints
    kps_xy = check_points_xy_N2plus(cur_feats.kps_xy, name="cur_feats.kps_xy", dtype=float, finite=True)
    n_feat = int(kps_xy.shape[0])

    # Build a fresh feature-to-landmark lookup for this new keyframe
    landmark_id_by_feat1 = _build_landmark_id_by_feat_for_kf(seed, n_feat, current_kf)

    # Store the new keyframe pose
    seed["T_WC1"] = (
        np.asarray(R_cur, dtype=np.float64),
        np.asarray(t_cur, dtype=np.float64).reshape(3,),
    )

    # Store the new reference feature bundle
    seed["feats1"] = cur_feats

    # Store the new keyframe feature-to-landmark lookup
    seed["landmark_id_by_feat1"] = landmark_id_by_feat1

    # Store the current keyframe id for bookkeeping
    seed["keyframe_kf"] = int(current_kf)

    # Store promotion diagnostics
    seed["last_keyframe_promotion"] = {
        "current_kf": int(current_kf),
        "n_feat": int(n_feat),
        "n_linked_landmarks": int(np.sum(landmark_id_by_feat1 >= 0)),
    }

    return seed


# Apply the keyframe decision and consider promoting the current frame
def consider_promote_keyframe(
    seed: dict,
    pose_out: dict,
    track_out: dict,
    *,
    map_growth_out: dict | MapGrowthResult | None = None,
    current_kf: int,
    image_shape: tuple[int, int] | None = None,
    min_track_inliers: int = 80,
    min_pnp_inliers: int = 40,
    min_landmark_growth: int = 20,
    min_linked_landmarks_for_promotion: int = 100,
    min_translation_m: float = 0.10,
    min_rotation_deg: float = 5.0,
    require_pose: bool = True,
) -> KeyframeUpdateResult:
    # --- Checks ---
    # Check required seed structure
    seed = check_required_keys(seed, {"landmarks", "T_WC1"}, name="seed")

    # Check containers
    if not isinstance(pose_out, dict):
        raise ValueError("pose_out must be a dict")
    if not isinstance(track_out, dict):
        raise ValueError("track_out must be a dict")
    _map_growth_stats(map_growth_out)

    # Check frame index
    current_kf = check_int_ge0(current_kf, name="current_kf")

    # Run the decision rule
    decision = should_make_keyframe(
        seed,
        pose_out,
        track_out,
        map_growth_out=map_growth_out,
        current_kf=current_kf,
        image_shape=image_shape,
        min_track_inliers=min_track_inliers,
        min_pnp_inliers=min_pnp_inliers,
        min_landmark_growth=min_landmark_growth,
        min_linked_landmarks_for_promotion=min_linked_landmarks_for_promotion,
        min_translation_m=min_translation_m,
        min_rotation_deg=min_rotation_deg,
        require_pose=require_pose,
    )

    # Default outcome is to keep the current keyframe
    promoted = False
    seed_out = seed

    # Promote only when requested by the decision rule
    if bool(decision.make_keyframe):
        # Promotion requires a valid pose
        if not bool(pose_out.get("ok", False)):
            raise ValueError("Cannot promote a keyframe without a valid pose in pose_out")

        # Promotion requires current-frame features
        if "cur_feats" not in track_out:
            raise ValueError("track_out must contain 'cur_feats' to promote a new keyframe")

        # Promote the current frame to the new keyframe
        seed_out = promote_frame_to_keyframe(
            seed,
            track_out["cur_feats"],
            pose_out["R"],
            pose_out["t"],
            current_kf=current_kf,
        )
        promoted = True

    # Pack update stats
    stats = {
        "make_keyframe": bool(decision.make_keyframe),
        "promoted": bool(promoted),
        "reason": decision.reason,
        "current_kf": int(current_kf),
        "n_landmarks": int(len(seed_out.get("landmarks", []))),
        "n_linked_landmarks_candidate": int(decision.stats.get("n_linked_landmarks_candidate", 0)),
        "promotion_vetoed_for_low_links": bool(decision.stats.get("promotion_vetoed_for_low_links", False)),
        "promotion_vetoed_for_low_coverage": bool(decision.stats.get("promotion_vetoed_for_low_coverage", False)),
        "promotion_linked_coverage_bbox_area_fraction": decision.stats.get(
            "promotion_linked_coverage_bbox_area_fraction",
            None,
        ),
        "promotion_linked_coverage_occupied_cells": int(
            decision.stats.get("promotion_linked_coverage_occupied_cells", 0)
        ),
    }

    return KeyframeUpdateResult(
        seed=seed_out,
        decision=decision,
        promoted=bool(promoted),
        stats=stats,
    )
