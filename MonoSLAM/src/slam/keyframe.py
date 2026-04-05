# src/slam/keyframe.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.checks import check_int_ge0, check_matrix_3x3, check_points_xy_N2plus, check_positive, check_required_keys, check_vector_3
from geometry.camera import camera_center
from geometry.rotation import angle_between_rotmats
from slam.seed import seed_keyframe_pose


# Keyframe decision bundle
@dataclass(frozen=True)
class KeyframeDecision:
    # Whether the current frame should become a new keyframe
    make_keyframe: bool
    # Short reason for the decision
    reason: str | None
    # Debug statistics used by the decision rule
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


# Decide whether the current frame should become a new keyframe
def should_make_keyframe(
    seed: dict,
    pose_out: dict,
    track_out: dict,
    *,
    map_growth_out: dict | None = None,
    min_track_inliers: int = 80,
    min_pnp_inliers: int = 40,
    min_landmark_growth: int = 20,
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
    if map_growth_out is not None and not isinstance(map_growth_out, dict):
        raise ValueError("map_growth_out must be a dict or None")

    # Check scalar thresholds
    min_track_inliers = check_int_ge0(min_track_inliers, name="min_track_inliers")
    min_pnp_inliers = check_int_ge0(min_pnp_inliers, name="min_pnp_inliers")
    min_landmark_growth = check_int_ge0(min_landmark_growth, name="min_landmark_growth")
    min_translation_m = check_positive(min_translation_m, name="min_translation_m", eps=0.0)
    min_rotation_deg = check_positive(min_rotation_deg, name="min_rotation_deg", eps=0.0)

    # Read stats dicts
    track_stats = track_out.get("stats", {})
    pose_stats = pose_out.get("stats", {})
    map_stats = map_growth_out.get("stats", {}) if map_growth_out is not None else {}

    # Read counts
    n_track_inliers = int(track_stats.get("n_inliers", 0))
    n_pnp_inliers = int(pose_stats.get("n_pnp_inliers", 0))
    n_added = int(map_stats.get("n_added", 0))
    n_landmarks = int(len(seed.get("landmarks", [])))

    # Read pose status
    pose_ok = bool(pose_out.get("ok", False))

    # Pack initial stats
    stats = {
        "pose_ok": bool(pose_ok),
        "n_track_inliers": int(n_track_inliers),
        "n_pnp_inliers": int(n_pnp_inliers),
        "n_added": int(n_added),
        "n_landmarks": int(n_landmarks),
        "translation_m": None,
        "rotation_deg": None,
    }

    # If a valid pose is required, stop early when it is unavailable
    if bool(require_pose) and not pose_ok:
        return KeyframeDecision(
            make_keyframe=False,
            reason="pose_not_available",
            stats=stats,
        )

    # If pose is unavailable, fall back to track quality only
    if not pose_ok:
        if n_track_inliers < min_track_inliers:
            return KeyframeDecision(
                make_keyframe=True,
                reason="track_inliers_low",
                stats=stats,
            )

        return KeyframeDecision(
            make_keyframe=False,
            reason=None,
            stats=stats,
        )

    # Read the stored keyframe pose
    R_kf, t_kf = seed_keyframe_pose(seed)

    # Read the current pose
    R_cur = check_matrix_3x3(pose_out["R"], name="pose_out['R']", dtype=float, finite=False)
    t_cur = check_vector_3(pose_out["t"], name="pose_out['t']", dtype=float, finite=False)

    # Compute camera centres
    C_kf = camera_center(R_kf, t_kf)
    C_cur = camera_center(R_cur, t_cur)

    # Compute motion since the current reference keyframe
    translation_m = float(np.linalg.norm(C_cur - C_kf))
    rotation_deg = float(np.degrees(angle_between_rotmats(R_kf, R_cur)))

    # Update stats with motion
    stats["translation_m"] = float(translation_m)
    stats["rotation_deg"] = float(rotation_deg)

    # Promote when both translation and rotation are clearly significant
    if translation_m >= min_translation_m and rotation_deg >= min_rotation_deg:
        return KeyframeDecision(
            make_keyframe=True,
            reason="translation_and_rotation_large",
            stats=stats,
        )

    # Promote when translation alone is significant
    if translation_m >= min_translation_m:
        return KeyframeDecision(
            make_keyframe=True,
            reason="translation_large",
            stats=stats,
        )

    # Promote when rotation alone is significant
    if rotation_deg >= min_rotation_deg:
        return KeyframeDecision(
            make_keyframe=True,
            reason="rotation_large",
            stats=stats,
        )

    # Promote when the current frame added many new landmarks
    if n_added >= min_landmark_growth:
        return KeyframeDecision(
            make_keyframe=True,
            reason="landmark_growth_high",
            stats=stats,
        )

    # Promote when keyframe tracking quality has degraded
    if n_track_inliers < min_track_inliers:
        return KeyframeDecision(
            make_keyframe=True,
            reason="track_inliers_low",
            stats=stats,
        )

    # Promote when pose support has degraded
    if n_pnp_inliers < min_pnp_inliers:
        return KeyframeDecision(
            make_keyframe=True,
            reason="pnp_inliers_low",
            stats=stats,
        )

    # Otherwise keep the existing keyframe
    return KeyframeDecision(
        make_keyframe=False,
        reason=None,
        stats=stats,
    )


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

    # Store promotion stats for debugging
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
    map_growth_out: dict | None = None,
    current_kf: int,
    min_track_inliers: int = 80,
    min_pnp_inliers: int = 40,
    min_landmark_growth: int = 20,
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
    if map_growth_out is not None and not isinstance(map_growth_out, dict):
        raise ValueError("map_growth_out must be a dict or None")

    # Check frame index
    current_kf = check_int_ge0(current_kf, name="current_kf")

    # Run the decision rule
    decision = should_make_keyframe(
        seed,
        pose_out,
        track_out,
        map_growth_out=map_growth_out,
        min_track_inliers=min_track_inliers,
        min_pnp_inliers=min_pnp_inliers,
        min_landmark_growth=min_landmark_growth,
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
    }

    return KeyframeUpdateResult(
        seed=seed_out,
        decision=decision,
        promoted=bool(promoted),
        stats=stats,
    )
