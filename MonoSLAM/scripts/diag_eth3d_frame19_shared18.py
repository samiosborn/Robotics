# scripts/diag_eth3d_frame19_shared18.py
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg, load_runtime_cfg

from datasets.eth3d import load_eth3d_sequence
from geometry.camera import camera_centre, reprojection_errors_sq, world_to_camera_points
from geometry.pnp import PnPCorrespondences, _slice_pnp_correspondences, build_pnp_correspondences_with_stats, estimate_pose_pnp, estimate_pose_pnp_ransac, _pnp_inlier_mask_from_pose
from geometry.rotation import angle_between_rotmats
from geometry.pose import angle_between_translations
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_active_keyframe_features, get_active_keyframe_kf, get_active_landmark_lookup, get_pose_for_kf, set_active_keyframe_record
from slam.landmark_state import build_landmark_id_index
from slam.pnp_frontend import estimate_pose_from_seed
from slam.tracking import track_against_keyframe


# Four basis-18-only live landmark IDs from the previous geometry run
_EXTRA_LANDMARK_IDS = frozenset({181, 226, 360, 588})

_THRESHOLDS_PX = [8.0, 12.0, 20.0, 40.0]

_RANSAC_SEED = 0
_RANSAC_TRIALS = 1000
_SAMPLE_SIZE = 6
_MIN_INLIERS = 6
_MIN_POINTS = 6
_RANK_TOL = 1e-10
_MIN_CHEIRALITY = 0.5
_EPS = 1e-12


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _best_inliers_at_threshold(corrs: PnPCorrespondences, K: np.ndarray, threshold_px: float) -> int:
    N = int(corrs.X_w.shape[1])
    if N < _SAMPLE_SIZE:
        return 0
    try:
        _, _, _, stats = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=_RANSAC_TRIALS,
            sample_size=_SAMPLE_SIZE,
            threshold_px=threshold_px,
            min_inliers=_MIN_INLIERS,
            seed=_RANSAC_SEED,
            min_points=_MIN_POINTS,
            rank_tol=_RANK_TOL,
            min_cheirality_ratio=_MIN_CHEIRALITY,
            eps=_EPS,
            refit=True,
            refine_nonlinear=True,
            refine_max_iters=15,
            refine_damping=1e-6,
            refine_step_tol=1e-9,
            refine_improvement_tol=1e-9,
        )
        return int(stats.get("n_inliers", 0)) if isinstance(stats, dict) else 0
    except Exception:
        return 0


def _inlier_sweep(corrs: PnPCorrespondences, K: np.ndarray) -> dict:
    return {
        str(int(t)): _best_inliers_at_threshold(corrs, K, t)
        for t in _THRESHOLDS_PX
    }


# Collect all successful DLT hypotheses from many minimal samples
def _hypothesis_sample_analysis(corrs: PnPCorrespondences, K: np.ndarray, *, n_trials: int = 2000) -> dict:
    N = int(corrs.X_w.shape[1])
    if N < _SAMPLE_SIZE:
        return {"n_trials": n_trials, "n_success": 0, "n_valid": 0, "note": "too_few_correspondences"}

    rng = np.random.default_rng(42)
    rotations = []
    translations = []
    camera_centres = []
    n_success = 0
    n_valid = 0

    for _ in range(n_trials):
        idx = rng.choice(N, size=_SAMPLE_SIZE, replace=False)
        corrs_sub = _slice_pnp_correspondences(corrs, idx)
        try:
            R_t, t_t, _ = estimate_pose_pnp(
                corrs_sub,
                K,
                min_points=_MIN_POINTS,
                rank_tol=_RANK_TOL,
                min_cheirality_ratio=_MIN_CHEIRALITY,
                eps=_EPS,
            )
        except Exception:
            continue
        if R_t is None or t_t is None:
            continue
        n_success += 1
        C = camera_centre(R_t, t_t)
        if not np.isfinite(C).all():
            continue
        n_valid += 1
        rotations.append(R_t.copy())
        translations.append(t_t.copy())
        camera_centres.append(C.copy())

    if n_valid < 2:
        return {
            "n_trials": n_trials,
            "n_success": n_success,
            "n_valid": n_valid,
            "note": "insufficient_valid_hypotheses",
        }

    # Pairwise rotation angle distribution
    rot_angles = []
    trans_angles = []
    centre_angles = []
    for i in range(n_valid):
        for j in range(i + 1, min(i + 51, n_valid)):
            try:
                rot_angles.append(float(np.degrees(angle_between_rotmats(rotations[i], rotations[j]))))
            except Exception:
                pass
            try:
                ta = float(np.degrees(angle_between_translations(translations[i], translations[j])))
                trans_angles.append(ta)
            except Exception:
                pass
            try:
                ca = float(np.degrees(angle_between_translations(camera_centres[i], camera_centres[j])))
                centre_angles.append(ca)
            except Exception:
                pass

    def _stat(arr):
        if len(arr) == 0:
            return {"count": 0, "median": None, "p90": None, "max": None}
        a = np.asarray(arr, dtype=np.float64)
        return {
            "count": int(a.size),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "max": float(np.max(a)),
        }

    # How many hypotheses are within 8 px of at least N_shared/2 correspondences
    inlier_counts_8px = []
    inlier_counts_20px = []
    for k in range(n_valid):
        mask_8, _ = _pnp_inlier_mask_from_pose(
            corrs.X_w, corrs.x_cur, K, rotations[k], translations[k],
            threshold_px=8.0, eps=_EPS,
        )
        mask_20, _ = _pnp_inlier_mask_from_pose(
            corrs.X_w, corrs.x_cur, K, rotations[k], translations[k],
            threshold_px=20.0, eps=_EPS,
        )
        inlier_counts_8px.append(int(np.sum(mask_8)))
        inlier_counts_20px.append(int(np.sum(mask_20)))

    ic8 = np.asarray(inlier_counts_8px, dtype=np.int64)
    ic20 = np.asarray(inlier_counts_20px, dtype=np.int64)

    return {
        "n_trials": n_trials,
        "n_success": n_success,
        "n_valid": n_valid,
        "pairwise_rotation_deg": _stat(rot_angles),
        "pairwise_translation_direction_deg": _stat(trans_angles),
        "pairwise_camera_centre_direction_deg": _stat(centre_angles),
        "inlier_counts_8px": {
            "min": int(np.min(ic8)),
            "median": float(np.median(ic8)),
            "max": int(np.max(ic8)),
            "n_above_6": int(np.sum(ic8 > 6)),
        },
        "inlier_counts_20px": {
            "min": int(np.min(ic20)),
            "median": float(np.median(ic20)),
            "max": int(np.max(ic20)),
            "n_above_6": int(np.sum(ic20 > 6)),
        },
    }


def _slice_by_landmark_ids(corrs: PnPCorrespondences, landmark_ids_to_keep: set) -> PnPCorrespondences:
    landmark_ids = np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
    keep = np.asarray([int(lm_id) in landmark_ids_to_keep for lm_id in landmark_ids], dtype=bool)
    return _slice_pnp_correspondences(corrs, keep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--output", type=str, default="/tmp/eth3d_frame19_shared18.json")
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
        K, K, image0, image1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )
    if not bool(bootstrap.get("ok", False)):
        raise RuntimeError(f"Bootstrap failed: {bootstrap.get('stats', {}).get('reason', None)}")

    seed = bootstrap["seed"]
    seed_before_frame19 = None
    frame19_image = None
    frame19_track_out = None

    # Run production pipeline through frame 18, then track frame 19
    for frame_index in range(2, 20):
        current_image, _, _ = sequence.get(frame_index)
        if frame_index == 19:
            seed_before_frame19 = copy.deepcopy(seed)
            frame19_image = current_image
            # Track against the current active basis (basis 18)
            frame19_track_out = track_against_keyframe(
                K,
                get_active_keyframe_features(seed),
                frame19_image,
                feature_cfg=frontend_kwargs["feature_cfg"],
                F_cfg=frontend_kwargs["F_cfg"],
            )
            break
        output = process_frame_against_seed(
            K, seed, current_image,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            current_kf=int(frame_index),
            **frontend_kwargs["pnp_frontend_kwargs"],
        )
        seed = output["seed"]

    if seed_before_frame19 is None or frame19_image is None or frame19_track_out is None:
        raise RuntimeError("Frame-19 state was not captured")

    # Build the full 22-correspondence bundle
    pnp_kwargs = frontend_kwargs["pnp_frontend_kwargs"]
    corrs_all, _ = build_pnp_correspondences_with_stats(
        seed_before_frame19,
        frame19_track_out,
        min_landmark_observations=int(pnp_kwargs.get("min_landmark_observations", 2)),
        allow_bootstrap_landmarks_for_pose=bool(pnp_kwargs.get("allow_bootstrap_landmarks_for_pose", True)),
        min_post_bootstrap_observations_for_pose=int(pnp_kwargs.get("min_post_bootstrap_observations_for_pose", 3)),
        enable_local_consistency_filter=False,
        enable_spatial_thinning_filter=False,
    )

    all_ids = set(int(lm) for lm in corrs_all.landmark_ids)
    shared_ids = all_ids - _EXTRA_LANDMARK_IDS
    extra_ids = all_ids & _EXTRA_LANDMARK_IDS
    extra_ids_sorted = sorted(extra_ids)

    print(f"Total live correspondences: {int(corrs_all.X_w.shape[1])}")
    print(f"Shared IDs ({len(shared_ids)}): {sorted(shared_ids)}")
    print(f"Extra IDs ({len(extra_ids)}): {extra_ids_sorted}")

    # Part A: shared-18 consensus fragility
    print("\n--- Part A: shared-18 only ---")
    corrs_shared = _slice_by_landmark_ids(corrs_all, shared_ids)
    n_shared = int(corrs_shared.X_w.shape[1])
    print(f"Correspondences in shared set: {n_shared}")

    shared_sweep = _inlier_sweep(corrs_shared, K)
    print("Inlier counts by threshold:", shared_sweep)

    shared_hypothesis = _hypothesis_sample_analysis(corrs_shared, K, n_trials=2000)
    print(f"Hypothesis analysis: n_trials={shared_hypothesis['n_trials']} n_success={shared_hypothesis['n_success']} n_valid={shared_hypothesis['n_valid']}")
    if "pairwise_rotation_deg" in shared_hypothesis:
        pr = shared_hypothesis["pairwise_rotation_deg"]
        print(f"  pairwise rotation median/p90/max: {pr['median']:.2f}/{pr['p90']:.2f}/{pr['max']:.2f} deg")
        ic8 = shared_hypothesis["inlier_counts_8px"]
        ic20 = shared_hypothesis["inlier_counts_20px"]
        print(f"  inlier@8px: min={ic8['min']} median={ic8['median']:.1f} max={ic8['max']} n_above_6={ic8['n_above_6']}")
        print(f"  inlier@20px: min={ic20['min']} median={ic20['median']:.1f} max={ic20['max']} n_above_6={ic20['n_above_6']}")

    # Part B: leave-one-out on shared 18
    print("\n--- Part B: leave-one-out on shared 18 ---")
    shared_ids_list = sorted(shared_ids)
    loo_results = []
    for remove_id in shared_ids_list:
        loo_ids = shared_ids - {remove_id}
        corrs_loo = _slice_by_landmark_ids(corrs_all, loo_ids)
        sweep = _inlier_sweep(corrs_loo, K)
        loo_results.append({
            "removed_landmark_id": int(remove_id),
            "n_remaining": int(corrs_loo.X_w.shape[1]),
            "inliers_by_threshold": sweep,
        })
        best = max(int(v) for v in sweep.values())
        print(f"  remove lm {remove_id:4d}: n={int(corrs_loo.X_w.shape[1])} best_inliers={best} sweep={sweep}")

    # Part C: add each extra one by one
    print("\n--- Part C: add each extra to shared 18 ---")
    add_results = []
    for add_id in extra_ids_sorted:
        add_ids = shared_ids | {add_id}
        corrs_add = _slice_by_landmark_ids(corrs_all, add_ids)
        sweep = _inlier_sweep(corrs_add, K)
        add_results.append({
            "added_landmark_id": int(add_id),
            "n_total": int(corrs_add.X_w.shape[1]),
            "inliers_by_threshold": sweep,
        })
        best = max(int(v) for v in sweep.values())
        print(f"  add lm {add_id:4d}: n={int(corrs_add.X_w.shape[1])} best_inliers={best} sweep={sweep}")

    # Full 22 sweep for comparison
    print("\n--- Full 22 correspondences sweep (reference) ---")
    full_sweep = _inlier_sweep(corrs_all, K)
    full_best = max(int(v) for v in full_sweep.values())
    print(f"  n=22 best_inliers={full_best} sweep={full_sweep}")

    # Part D: classification
    shared_best = max(int(v) for v in shared_sweep.values())
    loo_best_improvements = [
        (r["removed_landmark_id"], max(int(v) for v in r["inliers_by_threshold"].values()))
        for r in loo_results
    ]
    any_loo_improves = any(best > shared_best for _, best in loo_best_improvements)
    max_loo_improvement = max(best - shared_best for _, best in loo_best_improvements)

    extra_best_counts = [max(int(v) for v in r["inliers_by_threshold"].values()) for r in add_results]
    any_extra_worsens = any(b < shared_best for b in extra_best_counts)

    print("\n--- Classification ---")
    print(f"Shared-18 best achievable inliers: {shared_best}")
    print(f"Full-22 best achievable inliers: {full_best}")
    print(f"Max LOO improvement over shared-18 baseline: {max_loo_improvement}")
    print(f"Any single extra addition worsens outcome: {any_extra_worsens}")

    if shared_best == 0 and not any_loo_improves and not any_extra_worsens:
        classification = "shared_set_already_consensus_fragile"
    elif any_loo_improves and max_loo_improvement >= _MIN_INLIERS:
        classification = "one_or_two_shared_correspondences_poison_consensus"
    elif any_extra_worsens and not any_loo_improves:
        classification = "basis18_extra_correspondences_materially_worsen_failure"
    elif any_loo_improves and any_extra_worsens:
        classification = "mixed"
    else:
        classification = "shared_set_already_consensus_fragile"

    print(f"Classification: {classification}")

    result = {
        "n_shared": int(n_shared),
        "n_extra": int(len(extra_ids)),
        "shared_ids": sorted(shared_ids),
        "extra_ids": extra_ids_sorted,
        "part_a_shared18_inlier_sweep": shared_sweep,
        "part_a_hypothesis_analysis": shared_hypothesis,
        "part_b_leave_one_out": loo_results,
        "part_c_add_extras": add_results,
        "full_22_sweep": full_sweep,
        "shared18_best_inliers": int(shared_best),
        "full22_best_inliers": int(full_best),
        "max_loo_improvement": int(max_loo_improvement),
        "any_extra_worsens": bool(any_extra_worsens),
        "classification": classification,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nOutput written to {output_path}")


if __name__ == "__main__":
    main()
