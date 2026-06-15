from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg, load_runtime_cfg

from datasets.eth3d import load_eth3d_sequence
from geometry.camera import camera_centre, pixel_to_normalised, reprojection_errors_sq, world_to_camera_points
from geometry.rotation import quat_to_rotmat
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import (
    get_active_keyframe_features,
    get_active_keyframe_kf,
    get_active_landmark_lookup,
    get_pose_for_kf,
    set_active_keyframe_record,
)
from slam.landmark_state import build_landmark_id_index, count_valid_landmark_observations
from slam.pnp_frontend import estimate_pose_from_seed
from slam.tracking import track_against_keyframe


# Convert numpy values into JSON-safe values
def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


# Summarise finite scalar values
def _summary(values) -> dict:
    if not isinstance(values, np.ndarray):
        values = list(values)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "mean": None,
            "std": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


# Count stringified categorical values
def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = int(out.get(key, 0) + 1)
    return {key: int(out[key]) for key in sorted(out)}


# Return pairwise angular separations in degrees
def _pairwise_angles_deg(unit_vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(unit_vectors, dtype=np.float64)
    if vectors.ndim != 2 or int(vectors.shape[1]) != 3 or int(vectors.shape[0]) < 2:
        return np.zeros((0,), dtype=np.float64)
    dots = np.clip(vectors @ vectors.T, -1.0, 1.0)
    rows, cols = np.triu_indices(int(vectors.shape[0]), k=1)
    return np.degrees(np.arccos(dots[rows, cols]))


# Normalise row vectors and discard zero-length rows
def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1)
    keep = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > 1e-12)
    return vectors[keep] / norms[keep, None]


# Summarise centred point-cloud singular values
def _point_cloud_spectrum(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if int(points.shape[0]) == 0:
        return {
            "count": 0,
            "singular_values": [],
            "s2_over_s1": None,
            "s3_over_s1": None,
            "condition_s1_over_s3": None,
        }
    centred = points - np.mean(points, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    singular_values = np.pad(singular_values, (0, max(0, 3 - int(singular_values.size))))[:3]
    s1 = float(singular_values[0])
    s2 = float(singular_values[1])
    s3 = float(singular_values[2])
    return {
        "count": int(points.shape[0]),
        "singular_values": [s1, s2, s3],
        "s2_over_s1": None if s1 <= 1e-12 else float(s2 / s1),
        "s3_over_s1": None if s1 <= 1e-12 else float(s3 / s1),
        "condition_s1_over_s3": None if s3 <= 1e-12 else float(s1 / s3),
    }


# Summarise camera-ray spread
def _ray_spread(vectors: np.ndarray) -> dict:
    unit = _unit_rows(vectors)
    pairwise = _pairwise_angles_deg(unit)
    if int(unit.shape[0]) == 0:
        centre_angles = np.zeros((0,), dtype=np.float64)
    else:
        mean_ray = np.mean(unit, axis=0)
        mean_norm = float(np.linalg.norm(mean_ray))
        if mean_norm <= 1e-12:
            centre_angles = np.zeros((0,), dtype=np.float64)
        else:
            mean_ray = mean_ray / mean_norm
            centre_angles = np.degrees(np.arccos(np.clip(unit @ mean_ray, -1.0, 1.0)))
    return {
        "count": int(unit.shape[0]),
        "pairwise_angle_deg": _summary(pairwise),
        "angle_from_mean_deg": _summary(centre_angles),
        "spectrum": _point_cloud_spectrum(unit),
    }


# Read a compact landmark record
def _landmark_row(seed: dict, landmark: dict, installed_frame: int) -> dict:
    landmark_id = int(landmark["id"])
    X_w = np.asarray(landmark["X_w"], dtype=np.float64).reshape(3)
    observations = landmark.get("obs", [])
    observation_rows = []
    camera_centres = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        obs_kf = int(observation.get("kf", -1))
        if obs_kf < 0 or obs_kf > int(installed_frame):
            continue
        xy = np.asarray(observation.get("xy", None), dtype=np.float64).reshape(-1)
        if xy.size != 2 or not np.isfinite(xy).all():
            continue
        try:
            R_obs, t_obs = get_pose_for_kf(seed, obs_kf, context="basis geometry observation pose")
        except ValueError:
            continue
        centre = camera_centre(R_obs, t_obs)
        camera_centres.append(np.asarray(centre, dtype=np.float64).reshape(3))
        observation_rows.append(
            {
                "kf": int(obs_kf),
                "feat": int(observation.get("feat", -1)),
                "xy": xy,
            }
        )

    viewpoint_angles = np.zeros((0,), dtype=np.float64)
    baseline_ratios = np.zeros((0,), dtype=np.float64)
    if len(camera_centres) >= 2:
        centres = np.vstack(camera_centres)
        view_rays = _unit_rows(X_w.reshape(1, 3) - centres)
        viewpoint_angles = _pairwise_angles_deg(view_rays)
        rows, cols = np.triu_indices(int(centres.shape[0]), k=1)
        baselines = np.linalg.norm(centres[rows] - centres[cols], axis=1)
        depths = 0.5 * (
            np.linalg.norm(X_w.reshape(1, 3) - centres[rows], axis=1)
            + np.linalg.norm(X_w.reshape(1, 3) - centres[cols], axis=1)
        )
        valid = np.isfinite(baselines) & np.isfinite(depths) & (depths > 1e-12)
        baseline_ratios = baselines[valid] / depths[valid]

    return {
        "landmark_id": int(landmark_id),
        "X_w": X_w,
        "birth_source": str(landmark.get("birth_source", "unknown")),
        "birth_kf": None if landmark.get("birth_kf", None) is None else int(landmark["birth_kf"]),
        "observation_count": int(count_valid_landmark_observations(landmark, context=f"basis landmark {landmark_id}")),
        "observation_count_at_install": int(len(observation_rows)),
        "max_viewpoint_angle_deg": None if int(viewpoint_angles.size) == 0 else float(np.max(viewpoint_angles)),
        "median_viewpoint_angle_deg": None if int(viewpoint_angles.size) == 0 else float(np.median(viewpoint_angles)),
        "max_baseline_depth_ratio": None if int(baseline_ratios.size) == 0 else float(np.max(baseline_ratios)),
        "observations": observation_rows,
    }


# Snapshot the installed active basis and its landmark state
def _basis_snapshot(seed: dict, K: np.ndarray, installed_frame: int) -> dict:
    active_kf = int(get_active_keyframe_kf(seed))
    if active_kf != int(installed_frame):
        raise RuntimeError(f"Expected active basis {installed_frame}, got {active_kf}")
    R, t = get_pose_for_kf(seed, active_kf, context="basis geometry installed pose")
    lookup = np.asarray(get_active_landmark_lookup(seed), dtype=np.int64).reshape(-1).copy()
    mapped_features = np.flatnonzero(lookup >= 0)
    landmark_ids = np.asarray(lookup[mapped_features], dtype=np.int64)
    landmark_by_id = build_landmark_id_index(seed, context="basis geometry landmarks")
    rows = [
        _landmark_row(seed, landmark_by_id[int(landmark_id)], int(installed_frame))
        for landmark_id in landmark_ids
        if int(landmark_id) in landmark_by_id
    ]
    X_w = np.column_stack([np.asarray(row["X_w"], dtype=np.float64) for row in rows])
    X_c = world_to_camera_points(R, t, X_w)
    depths = np.asarray(X_c[2], dtype=np.float64)
    positive_depth = depths > 1e-12
    extents = np.ptp(X_c[:, positive_depth], axis=1) if np.any(positive_depth) else np.zeros((3,), dtype=np.float64)

    features = get_active_keyframe_features(seed)
    keypoints = np.asarray(features.kps_xy, dtype=np.float64)
    installed_xy = keypoints[mapped_features]
    installed_rays = pixel_to_normalised(K, installed_xy.T).T

    return {
        "installed_frame": int(installed_frame),
        "active_kf": int(active_kf),
        "R": np.asarray(R, dtype=np.float64).copy(),
        "t": np.asarray(t, dtype=np.float64).reshape(3).copy(),
        "features": copy.deepcopy(features),
        "lookup": lookup,
        "mapped_feature_indices": mapped_features,
        "landmark_ids": landmark_ids,
        "unique_landmark_ids": sorted(set(int(value) for value in landmark_ids)),
        "landmark_rows": rows,
        "geometry": {
            "depth": _summary(depths),
            "positive_depth_count": int(np.sum(positive_depth)),
            "depth_coefficient_of_variation": (
                None
                if not np.any(positive_depth) or abs(float(np.mean(depths[positive_depth]))) <= 1e-12
                else float(np.std(depths[positive_depth]) / abs(float(np.mean(depths[positive_depth]))))
            ),
            "camera_coordinate_extent_xyz": extents,
            "camera_coordinate_spectrum": _point_cloud_spectrum(X_c.T),
            "geometric_camera_ray_spread": _ray_spread(X_c.T),
            "installed_feature_ray_spread": _ray_spread(installed_rays),
            "viewpoint_angle_max_by_landmark_deg": _summary(
                [row["max_viewpoint_angle_deg"] for row in rows if row["max_viewpoint_angle_deg"] is not None]
            ),
            "viewpoint_angle_median_by_landmark_deg": _summary(
                [row["median_viewpoint_angle_deg"] for row in rows if row["median_viewpoint_angle_deg"] is not None]
            ),
            "baseline_depth_ratio_max_by_landmark": _summary(
                [row["max_baseline_depth_ratio"] for row in rows if row["max_baseline_depth_ratio"] is not None]
            ),
        },
        "set_summary": {
            "installed_feature_count": int(mapped_features.size),
            "installed_landmark_count": int(landmark_ids.size),
            "unique_landmark_count": int(len(set(int(value) for value in landmark_ids))),
            "duplicate_landmark_count": int(landmark_ids.size - len(set(int(value) for value in landmark_ids))),
            "birth_source_counts": _counts(row["birth_source"] for row in rows),
            "birth_kf_counts": _counts(row["birth_kf"] for row in rows),
            "observation_count": _summary(row["observation_count"] for row in rows),
            "observation_count_at_install": _summary(row["observation_count_at_install"] for row in rows),
        },
    }


# Read and interpolate ETH3D ground-truth camera poses
def _load_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    centres = []
    quaternions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        values = [float(value) for value in stripped.split()]
        if len(values) != 8:
            continue
        timestamps.append(values[0])
        centres.append(values[1:4])
        quaternions.append([values[7], values[4], values[5], values[6]])
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(centres, dtype=np.float64),
        np.asarray(quaternions, dtype=np.float64),
    )


# Interpolate one unit quaternion
def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + float(alpha) * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    return (
        np.sin((1.0 - float(alpha)) * theta) / sin_theta * q0
        + np.sin(float(alpha) * theta) / sin_theta * q1
    )


# Interpolate one camera-to-world ground-truth pose
def _groundtruth_pose(
    timestamps: np.ndarray,
    centres: np.ndarray,
    quaternions: np.ndarray,
    timestamp: float,
) -> tuple[np.ndarray, np.ndarray]:
    upper = int(np.searchsorted(timestamps, float(timestamp), side="left"))
    if upper <= 0:
        return centres[0].copy(), quat_to_rotmat(quaternions[0])
    if upper >= int(timestamps.size):
        return centres[-1].copy(), quat_to_rotmat(quaternions[-1])
    lower = int(upper - 1)
    span = float(timestamps[upper] - timestamps[lower])
    alpha = 0.0 if abs(span) <= 1e-12 else float((float(timestamp) - timestamps[lower]) / span)
    centre = (1.0 - alpha) * centres[lower] + alpha * centres[upper]
    quaternion = _slerp(quaternions[lower], quaternions[upper], alpha)
    return centre, quat_to_rotmat(quaternion)


# Compute a rotation disagreement in degrees
def _rotation_delta_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


# Compute an angular disagreement between non-zero vectors
def _vector_angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float | None:
    vector_a = np.asarray(vector_a, dtype=np.float64).reshape(3)
    vector_b = np.asarray(vector_b, dtype=np.float64).reshape(3)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    cosine = float(np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


# Build a frame-19 reference pose from local ETH3D relative motion
def _frame19_reference_pose(
    seed: dict,
    sequence,
    groundtruth_path: Path,
) -> dict:
    timestamps, centres, quaternions = _load_groundtruth(groundtruth_path)
    gt = {}
    for frame_index in (17, 18, 19):
        timestamp = float(sequence.frame_info(frame_index).timestamp)
        centre, Q_direct = _groundtruth_pose(timestamps, centres, quaternions, timestamp)
        gt[frame_index] = {
            "timestamp": timestamp,
            "centre": centre,
            "Q_direct": Q_direct,
        }

    R17, t17 = get_pose_for_kf(seed, 17, context="frame-19 reference pose 17")
    R18, t18 = get_pose_for_kf(seed, 18, context="frame-19 reference pose 18")
    Q17_est = np.asarray(R17, dtype=np.float64).T
    Q18_est = np.asarray(R18, dtype=np.float64).T
    C17_est = camera_centre(R17, t17)
    C18_est = camera_centre(R18, t18)
    relative_est = Q17_est.T @ Q18_est

    direct_relative = gt[17]["Q_direct"].T @ gt[18]["Q_direct"]
    inverse_relative = gt[17]["Q_direct"] @ gt[18]["Q_direct"].T
    direct_error = _rotation_delta_deg(relative_est, direct_relative)
    inverse_error = _rotation_delta_deg(relative_est, inverse_relative)
    use_inverse = bool(inverse_error < direct_error)

    for frame_index in (17, 18, 19):
        gt[frame_index]["Q"] = (
            gt[frame_index]["Q_direct"].T
            if use_inverse
            else gt[frame_index]["Q_direct"]
        )

    gt_step = float(np.linalg.norm(gt[18]["centre"] - gt[17]["centre"]))
    estimated_step = float(np.linalg.norm(C18_est - C17_est))
    scale = None if gt_step <= 1e-12 else float(estimated_step / gt_step)
    if scale is None:
        raise RuntimeError("Cannot recover local ETH3D scale from frames 17 and 18")

    delta_C_est_17_18 = C18_est - C17_est
    delta_C_gt_17_18_camera17 = gt[17]["Q"].T @ (gt[18]["centre"] - gt[17]["centre"])
    delta_C_predicted_17_18 = float(scale) * (Q17_est @ delta_C_gt_17_18_camera17)
    predicted_C18 = C17_est + delta_C_predicted_17_18
    predicted_Q18 = Q17_est @ (gt[17]["Q"].T @ gt[18]["Q"])

    relative_Q_18_19 = gt[18]["Q"].T @ gt[19]["Q"]
    delta_C_gt_in_camera18 = gt[18]["Q"].T @ (gt[19]["centre"] - gt[18]["centre"])
    Q19_est = Q18_est @ relative_Q_18_19
    C19_est = C18_est + float(scale) * (Q18_est @ delta_C_gt_in_camera18)
    R19_est = Q19_est.T
    t19_est = -R19_est @ C19_est

    return {
        "R": R19_est,
        "t": t19_est,
        "C": C19_est,
        "scale_from_frame17_18": float(scale),
        "groundtruth_quaternion_convention": "world_to_camera" if use_inverse else "camera_to_world",
        "frame17_18_relative_rotation_error_direct_deg": float(direct_error),
        "frame17_18_relative_rotation_error_inverse_deg": float(inverse_error),
        "groundtruth_step_17_18": float(gt_step),
        "estimated_step_17_18": float(estimated_step),
        "groundtruth_step_18_19": float(np.linalg.norm(gt[19]["centre"] - gt[18]["centre"])),
        "frame17_18_translation_direction_error_deg": _vector_angle_deg(
            delta_C_est_17_18,
            delta_C_predicted_17_18,
        ),
        "frame17_18_predicted_centre_error": float(np.linalg.norm(predicted_C18 - C18_est)),
        "frame17_18_predicted_rotation_error_deg": _rotation_delta_deg(
            predicted_Q18,
            Q18_est,
        ),
    }


# Build the normalised DLT design spectrum
def _dlt_spectrum(K: np.ndarray, X_w: np.ndarray, x_cur: np.ndarray) -> dict:
    X_w = np.asarray(X_w, dtype=np.float64)
    x_cur = np.asarray(x_cur, dtype=np.float64)
    count = int(X_w.shape[1])
    if count < 6:
        return {
            "count": int(count),
            "rank": 0,
            "singular_values": [],
            "smallest_over_largest": None,
            "second_smallest_over_largest": None,
        }
    centre = np.mean(X_w, axis=1)
    centred = X_w - centre[:, None]
    scale = float(np.sqrt(np.mean(np.sum(centred * centred, axis=0))))
    if scale <= 1e-12:
        scale = 1.0
    X_norm = centred / scale
    X_h = np.vstack([X_norm, np.ones((1, count), dtype=np.float64)])
    x_hat = pixel_to_normalised(K, x_cur)
    u = x_hat[0] / x_hat[2]
    v = x_hat[1] / x_hat[2]
    A = np.zeros((2 * count, 12), dtype=np.float64)
    for index in range(count):
        A[2 * index, 0:4] = X_h[:, index]
        A[2 * index, 8:12] = -float(u[index]) * X_h[:, index]
        A[2 * index + 1, 4:8] = X_h[:, index]
        A[2 * index + 1, 8:12] = -float(v[index]) * X_h[:, index]
    singular_values = np.linalg.svd(A, compute_uv=False)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    second_smallest = float(singular_values[-2])
    tolerance = 1e-10 * max(largest, 1e-12)
    return {
        "count": int(count),
        "rank": int(np.sum(singular_values > tolerance)),
        "singular_values": singular_values,
        "smallest_over_largest": None if largest <= 1e-12 else float(smallest / largest),
        "second_smallest_over_largest": None if largest <= 1e-12 else float(second_smallest / largest),
    }


# Build the pose Jacobian spectrum at a reference pose
def _pose_jacobian_spectrum(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    X_w: np.ndarray,
) -> dict:
    X_c = world_to_camera_points(R, t, X_w)
    valid = np.isfinite(X_c).all(axis=0) & (X_c[2] > 1e-12)
    X_c = X_c[:, valid]
    if int(X_c.shape[1]) == 0:
        return {
            "count": 0,
            "singular_values": [],
            "smallest_over_largest": None,
            "condition": None,
        }
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    blocks = []
    for index in range(int(X_c.shape[1])):
        X, Y, Z = [float(value) for value in X_c[:, index]]
        J_projection = np.array(
            [
                [fx / Z, 0.0, -fx * X / (Z * Z)],
                [0.0, fy / Z, -fy * Y / (Z * Z)],
            ],
            dtype=np.float64,
        )
        skew = np.array(
            [
                [0.0, -Z, Y],
                [Z, 0.0, -X],
                [-Y, X, 0.0],
            ],
            dtype=np.float64,
        )
        blocks.append(J_projection @ np.hstack([np.eye(3), -skew]))
    J = np.vstack(blocks)
    singular_values = np.linalg.svd(J, compute_uv=False)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    return {
        "count": int(X_c.shape[1]),
        "singular_values": singular_values,
        "smallest_over_largest": None if largest <= 1e-12 else float(smallest / largest),
        "condition": None if smallest <= 1e-12 else float(largest / smallest),
    }


# Summarise current-image spatial support
def _image_spread(xy: np.ndarray, image_shape: tuple[int, int]) -> dict:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if int(xy.shape[0]) == 0:
        return {
            "count": 0,
            "bbox": None,
            "bbox_area_fraction": None,
            "occupied_cells": 0,
        }
    height = int(image_shape[0])
    width = int(image_shape[1])
    xmin = float(np.min(xy[:, 0]))
    ymin = float(np.min(xy[:, 1]))
    xmax = float(np.max(xy[:, 0]))
    ymax = float(np.max(xy[:, 1]))
    cells = set()
    for point in xy:
        col = int(np.clip(np.floor(float(point[0]) / max(float(width), 1.0) * 4.0), 0, 3))
        row = int(np.clip(np.floor(float(point[1]) / max(float(height), 1.0) * 3.0), 0, 2))
        cells.add((row, col))
    return {
        "count": int(xy.shape[0]),
        "bbox": [xmin, ymin, xmax, ymax],
        "bbox_area_fraction": float(max(xmax - xmin, 0.0) * max(ymax - ymin, 0.0) / max(float(width * height), 1.0)),
        "occupied_cells": int(len(cells)),
    }


# Evaluate one frozen basis against frame 19
def _frame19_viability(
    K: np.ndarray,
    seed_after_frame18: dict,
    basis: dict,
    frame19_image: np.ndarray,
    frontend_kwargs: dict,
    reference_pose: dict,
) -> dict:
    seed = copy.deepcopy(seed_after_frame18)
    set_active_keyframe_record(
        seed,
        int(basis["active_kf"]),
        (
            np.asarray(basis["R"], dtype=np.float64).copy(),
            np.asarray(basis["t"], dtype=np.float64).reshape(3).copy(),
        ),
        copy.deepcopy(basis["features"]),
        np.asarray(basis["lookup"], dtype=np.int64).copy(),
    )
    track_out = track_against_keyframe(
        K,
        get_active_keyframe_features(seed),
        frame19_image,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
    )
    pose_out = estimate_pose_from_seed(
        K,
        seed,
        track_out,
        image_shape=(int(frame19_image.shape[0]), int(frame19_image.shape[1])),
        **frontend_kwargs["pnp_frontend_kwargs"],
    )
    corrs = pose_out["corrs"]
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    X_w = np.asarray(corrs.X_w, dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    R_ref = np.asarray(reference_pose["R"], dtype=np.float64)
    t_ref = np.asarray(reference_pose["t"], dtype=np.float64).reshape(3)
    X_c_ref = world_to_camera_points(R_ref, t_ref, X_w)
    positive_depth = X_c_ref[2] > 1e-12
    errors_sq = reprojection_errors_sq(K, R_ref, t_ref, X_w, x_cur)
    errors_px = np.sqrt(np.maximum(np.asarray(errors_sq, dtype=np.float64), 0.0))
    valid_errors = errors_px[np.isfinite(errors_px) & positive_depth]
    residual_rows = [
        {
            "landmark_id": int(landmark_ids[index]),
            "positive_depth": bool(positive_depth[index]),
            "error_px": None if not np.isfinite(errors_px[index]) else float(errors_px[index]),
            "x_cur": np.asarray(x_cur[:, index], dtype=np.float64),
        }
        for index in range(int(landmark_ids.size))
    ]
    installed_ids = set(int(value) for value in basis["unique_landmark_ids"])
    live_ids = set(int(value) for value in landmark_ids)
    pose_stats = pose_out.get("stats", {})
    pnp_mask = np.asarray(
        pose_out.get("pnp_inlier_mask", np.zeros((landmark_ids.size,), dtype=bool)),
        dtype=bool,
    ).reshape(-1)

    return {
        "basis_frame": int(basis["installed_frame"]),
        "tracking": {
            "matches": int(track_out.get("stats", {}).get("n_matches", 0)),
            "fundamental_inliers": int(track_out.get("stats", {}).get("n_inliers", 0)),
            "reason": track_out.get("stats", {}).get("reason", None),
        },
        "pose_attempt": {
            "ok": bool(pose_out.get("ok", False)),
            "reason": pose_stats.get("reason", None),
            "correspondence_count": int(landmark_ids.size),
            "inlier_count": int(np.sum(pnp_mask)),
            "best_strict_consensus_count": int(pose_stats.get("n_inliers", 0)),
            "solver_model_successes": int(pose_stats.get("n_model_success", 0)),
            "rescue_loose_inliers": int(pose_stats.get("pnp_support_rescue_loose_inliers", 0)),
            "rescue_second_stage_succeeded": bool(
                pose_stats.get("pnp_support_rescue_second_stage_succeeded", False)
            ),
        },
        "support": {
            "live_landmark_ids": sorted(live_ids),
            "installed_landmarks_reused": int(len(installed_ids & live_ids)),
            "installed_reuse_fraction": float(len(installed_ids & live_ids) / max(len(installed_ids), 1)),
            "unique_live_landmark_count": int(len(live_ids)),
            "current_image_spread": _image_spread(x_cur.T, frame19_image.shape[:2]),
        },
        "reference_pose_reprojection": {
            "positive_depth_count": int(np.sum(positive_depth)),
            "error_px": _summary(valid_errors),
            "within_3_px": int(np.sum(valid_errors <= 3.0)),
            "within_8_px": int(np.sum(valid_errors <= 8.0)),
            "within_12_px": int(np.sum(valid_errors <= 12.0)),
            "by_landmark": residual_rows,
        },
        "reference_pose_geometry": {
            "depth": _summary(X_c_ref[2]),
            "camera_coordinate_extent_xyz": np.ptp(X_c_ref[:, positive_depth], axis=1) if np.any(positive_depth) else np.zeros((3,)),
            "camera_coordinate_spectrum": _point_cloud_spectrum(X_c_ref.T),
            "camera_ray_spread": _ray_spread(X_c_ref.T),
            "dlt_spectrum": _dlt_spectrum(K, X_w, x_cur),
            "pose_jacobian_spectrum": _pose_jacobian_spectrum(K, R_ref, t_ref, X_w),
        },
    }


# Compare the two installed landmark sets
def _set_comparison(basis17: dict, basis18: dict) -> dict:
    ids17 = set(int(value) for value in basis17["unique_landmark_ids"])
    ids18 = set(int(value) for value in basis18["unique_landmark_ids"])
    intersection = ids17 & ids18
    union = ids17 | ids18
    rows17 = {int(row["landmark_id"]): row for row in basis17["landmark_rows"]}
    rows18 = {int(row["landmark_id"]): row for row in basis18["landmark_rows"]}
    return {
        "basis17_count": int(len(ids17)),
        "basis18_count": int(len(ids18)),
        "intersection_count": int(len(intersection)),
        "union_count": int(len(union)),
        "jaccard": None if len(union) == 0 else float(len(intersection) / len(union)),
        "only_basis17": sorted(ids17 - ids18),
        "only_basis18": sorted(ids18 - ids17),
        "shared_landmarks": sorted(intersection),
        "only_basis17_rows": [rows17[landmark_id] for landmark_id in sorted(ids17 - ids18)],
        "only_basis18_rows": [rows18[landmark_id] for landmark_id in sorted(ids18 - ids17)],
    }


# Locate the sequence ground-truth file
def _groundtruth_path(dataset_root: Path, sequence_name: str) -> Path:
    candidates = sorted((dataset_root / sequence_name).rglob("groundtruth.txt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one groundtruth.txt under {dataset_root / sequence_name}, found {len(candidates)}")
    return candidates[0]


# Run the focused basis geometry diagnosis
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--output", type=str, default="/tmp/eth3d_basis17_18_geometry.json")
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cfg, K = load_runtime_cfg(profile_path)
    frontend_kwargs = frontend_kwargs_from_cfg(cfg)
    dataset_cfg = cfg["dataset"]
    dataset_root = (ROOT / str(dataset_cfg["root"])).resolve()
    sequence_name = str(dataset_cfg["seq"])
    sequence = load_eth3d_sequence(
        dataset_root,
        sequence_name,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    image0, _, _ = sequence.get(0)
    image1, _, _ = sequence.get(1)
    bootstrap = bootstrap_from_two_frames(
        K,
        K,
        image0,
        image1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )
    if not bool(bootstrap.get("ok", False)):
        raise RuntimeError(f"Bootstrap failed: {bootstrap.get('stats', {}).get('reason', None)}")

    seed = bootstrap["seed"]
    basis_snapshots = {}
    frame_rows = []
    seed_after_frame18 = None
    frame19_image = None
    for frame_index in range(2, 20):
        current_image, timestamp, frame_id = sequence.get(frame_index)
        if frame_index == 19:
            seed_after_frame18 = copy.deepcopy(seed)
            frame19_image = current_image
        output = process_frame_against_seed(
            K,
            seed,
            current_image,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            current_kf=int(frame_index),
            **frontend_kwargs["pnp_frontend_kwargs"],
        )
        stats = output.get("stats", {})
        pose_stats = output.get("pose_out", {}).get("stats", {})
        frame_rows.append(
            {
                "frame_index": int(frame_index),
                "frame_id": str(frame_id),
                "timestamp": float(timestamp),
                "ok": bool(output.get("ok", False)),
                "active_basis_before": int(get_active_keyframe_kf(seed)),
                "active_basis_after": int(get_active_keyframe_kf(output["seed"])),
                "rescue_succeeded": bool(pose_stats.get("pnp_support_rescue_succeeded", False)),
                "refresh_triggered": bool(stats.get("guarded_support_refresh_triggered", False)),
                "n_corr": int(pose_stats.get("n_corr", 0)),
                "n_inliers": int(pose_stats.get("n_pnp_inliers", 0)),
                "reason": pose_stats.get("reason", None),
            }
        )
        seed = output["seed"]
        if frame_index in (17, 18):
            if not bool(stats.get("guarded_support_refresh_triggered", False)):
                raise RuntimeError(f"Expected refresh at frame {frame_index}")
            basis_snapshots[frame_index] = _basis_snapshot(seed, K, frame_index)

    if seed_after_frame18 is None or frame19_image is None:
        raise RuntimeError("Frame-19 input state was not captured")
    if 17 not in basis_snapshots or 18 not in basis_snapshots:
        raise RuntimeError("Basis snapshots 17 and 18 were not captured")

    groundtruth_path = _groundtruth_path(dataset_root, sequence_name)
    reference_pose = _frame19_reference_pose(seed_after_frame18, sequence, groundtruth_path)
    viability17 = _frame19_viability(
        K,
        seed_after_frame18,
        basis_snapshots[17],
        frame19_image,
        frontend_kwargs,
        reference_pose,
    )
    viability18 = _frame19_viability(
        K,
        seed_after_frame18,
        basis_snapshots[18],
        frame19_image,
        frontend_kwargs,
        reference_pose,
    )
    live17 = set(int(value) for value in viability17["support"]["live_landmark_ids"])
    live18 = set(int(value) for value in viability18["support"]["live_landmark_ids"])

    result = {
        "profile": str(profile_path),
        "groundtruth_path": str(groundtruth_path),
        "frame_rows": frame_rows,
        "set_comparison": _set_comparison(basis_snapshots[17], basis_snapshots[18]),
        "basis17": {
            "set_summary": basis_snapshots[17]["set_summary"],
            "geometry": basis_snapshots[17]["geometry"],
            "landmark_rows": basis_snapshots[17]["landmark_rows"],
        },
        "basis18": {
            "set_summary": basis_snapshots[18]["set_summary"],
            "geometry": basis_snapshots[18]["geometry"],
            "landmark_rows": basis_snapshots[18]["landmark_rows"],
        },
        "frame19_reference_pose": reference_pose,
        "frame19_viability": {
            "basis17": viability17,
            "basis18": viability18,
            "live_set_overlap": {
                "basis17_count": int(len(live17)),
                "basis18_count": int(len(live18)),
                "intersection_count": int(len(live17 & live18)),
                "union_count": int(len(live17 | live18)),
                "jaccard": None if len(live17 | live18) == 0 else float(len(live17 & live18) / len(live17 | live18)),
                "only_basis17": sorted(live17 - live18),
                "only_basis18": sorted(live18 - live17),
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
