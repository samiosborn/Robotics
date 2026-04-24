from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frontend_eth3d_common import frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg, load_runtime_cfg as _load_runtime_cfg
from core.checks import check_dir
from datasets.eth3d import load_eth3d_sequence
from geometry.camera import camera_centre, reprojection_errors_sq, world_to_camera_points
from geometry.pose import angle_between_translations
from geometry.pnp import (
    _pnp_inlier_mask_from_pose,
    _slice_pnp_correspondences,
    build_pnp_correspondences_with_stats,
    estimate_pose_pnp_ransac,
    pnp_inlier_spatial_coverage,
    pnp_local_displacement_consistency_mask,
)
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.tracking import track_against_keyframe

ROOT = Path(__file__).resolve().parents[1]


def _obs_count(lm: dict) -> int:
    obs = lm.get("obs", [])
    if not isinstance(obs, list):
        return 0
    return int(sum(1 for ob in obs if isinstance(ob, dict)))


def _grid_labels(x_cur: np.ndarray, image_shape: tuple[int, int], *, grid_cols: int = 4, grid_rows: int = 3) -> np.ndarray:
    H = int(image_shape[0])
    W = int(image_shape[1])
    x = np.asarray(x_cur[0, :], dtype=np.float64)
    y = np.asarray(x_cur[1, :], dtype=np.float64)
    cols = np.clip(np.floor((x / max(float(W), 1.0)) * int(grid_cols)).astype(np.int64), 0, int(grid_cols) - 1)
    rows = np.clip(np.floor((y / max(float(H), 1.0)) * int(grid_rows)).astype(np.int64), 0, int(grid_rows) - 1)
    return np.asarray([f"r{int(r)}c{int(c)}" for r, c in zip(rows, cols)], dtype=object)


def _format_counts(values, mask: np.ndarray) -> str:
    values = np.asarray(values, dtype=object).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size != mask.size:
        return "unaligned"
    subset = values[mask]
    if subset.size == 0:
        return "n=0"
    labels, counts = np.unique(subset, return_counts=True)
    order = np.argsort(-counts)
    return ", ".join(f"{labels[i]}:{int(counts[i])}" for i in order)


def _format_support_rate(values, support_mask: np.ndarray, *, filter_mask: np.ndarray | None = None) -> str:
    values = np.asarray(values, dtype=object).reshape(-1)
    support_mask = np.asarray(support_mask, dtype=bool).reshape(-1)
    if values.size != support_mask.size:
        return "unaligned"
    use = np.ones((values.size,), dtype=bool) if filter_mask is None else np.asarray(filter_mask, dtype=bool).reshape(-1)
    if use.size != values.size:
        return "unaligned"
    labels = np.unique(values[use])
    rows: list[str] = []
    for label in labels:
        sel = use & (values == label)
        n = int(np.sum(sel))
        if n == 0:
            continue
        support = int(np.sum(sel & support_mask))
        rows.append(f"{label}:{support}/{n} ({100.0 * support / max(n, 1):.0f}%)")
    return ", ".join(rows) if len(rows) > 0 else "n=0"


def _format_numeric_buckets(values, support_mask: np.ndarray, *, bins: list[tuple[str, np.ndarray]]) -> str:
    values = np.asarray(values).reshape(-1)
    support_mask = np.asarray(support_mask, dtype=bool).reshape(-1)
    if values.size != support_mask.size:
        return "unaligned"
    parts: list[str] = []
    for label, mask in bins:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        n = int(np.sum(mask))
        if n == 0:
            continue
        support = int(np.sum(mask & support_mask))
        parts.append(f"{label}:{support}/{n} ({100.0 * support / max(n, 1):.0f}%)")
    return ", ".join(parts) if len(parts) > 0 else "n=0"


def _residual_stats(errors_px: np.ndarray, mask: np.ndarray) -> str:
    errors_px = np.asarray(errors_px, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    vals = errors_px[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "n=0"
    return (
        f"n={int(vals.size)} median={float(np.median(vals)):.2f}px "
        f"p75={float(np.percentile(vals, 75)):.2f}px "
        f"p90={float(np.percentile(vals, 90)):.2f}px "
        f"max={float(np.max(vals)):.2f}px"
    )


def _coverage_stats(corrs, mask: np.ndarray, image_shape: tuple[int, int]) -> str:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    stats = pnp_inlier_spatial_coverage(
        corrs.x_cur,
        mask,
        image_shape,
        grid_cols=4,
        grid_rows=3,
    )
    return (
        f"count={int(np.sum(mask))} cells={int(stats.get('occupied_cells', 0))} "
        f"bbox_area_fraction={stats.get('bbox_area_fraction', None)} "
        f"max_cell_fraction={stats.get('max_cell_fraction', None)} "
        f"grid={stats.get('occupancy_grid', None)}"
    )


def _run_pnp(corrs, K: np.ndarray, pnp_cfg: dict, *, threshold_px: float) -> dict:
    R, t, inlier_mask, stats = estimate_pose_pnp_ransac(
        corrs,
        K,
        num_trials=int(pnp_cfg["num_trials"]),
        sample_size=int(pnp_cfg["sample_size"]),
        threshold_px=float(threshold_px),
        min_inliers=int(pnp_cfg["min_inliers"]),
        seed=int(pnp_cfg["ransac_seed"]),
        min_points=int(pnp_cfg["min_points"]),
        rank_tol=float(pnp_cfg["rank_tol"]),
        min_cheirality_ratio=float(pnp_cfg["min_cheirality_ratio"]),
        eps=float(pnp_cfg["eps"]),
        refit=bool(pnp_cfg["refit"]),
        refine_nonlinear=bool(pnp_cfg["refine_nonlinear"]),
        refine_max_iters=int(pnp_cfg["refine_max_iters"]),
        refine_damping=float(pnp_cfg["refine_damping"]),
        refine_step_tol=float(pnp_cfg["refine_step_tol"]),
        refine_improvement_tol=float(pnp_cfg["refine_improvement_tol"]),
    )
    N = int(corrs.X_w.shape[1])
    if inlier_mask is None:
        inlier_mask = np.zeros((N,), dtype=bool)
    else:
        inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    return {
        "R": None if R is None else np.asarray(R, dtype=np.float64),
        "t": None if t is None else np.asarray(t, dtype=np.float64).reshape(3),
        "inlier_mask": inlier_mask,
        "stats": stats if isinstance(stats, dict) else {},
        "ok": (R is not None) and (t is not None),
    }


def _pose_delta(ref_pose: dict, test_pose: dict) -> str:
    if not bool(ref_pose.get("ok", False)) or not bool(test_pose.get("ok", False)):
        return "pose_delta=unavailable"
    R_ref = np.asarray(ref_pose["R"], dtype=np.float64)
    t_ref = np.asarray(ref_pose["t"], dtype=np.float64).reshape(3)
    R = np.asarray(test_pose["R"], dtype=np.float64)
    t = np.asarray(test_pose["t"], dtype=np.float64).reshape(3)
    c_ref = camera_centre(R_ref, t_ref)
    c = camera_centre(R, t)
    return (
        f"rot={float(angle_between_rotmats(R_ref, R)):.2f}deg "
        f"trans_dir={float(angle_between_translations(t_ref, t)):.2f}deg "
        f"centre_dir={float(angle_between_translations(c_ref, c)):.2f}deg"
    )


def _subset_eval(label: str, corrs, subset_mask: np.ndarray, K: np.ndarray, pnp_cfg: dict, loose_pose: dict) -> None:
    subset_mask = np.asarray(subset_mask, dtype=bool).reshape(-1)
    corrs_sub = _slice_pnp_correspondences(corrs, subset_mask)
    n = int(corrs_sub.X_w.shape[1])
    if n < int(pnp_cfg["sample_size"]):
        print(f"  subset_rule {label}: keep={n} too_small_for_sample_size={int(pnp_cfg['sample_size'])}")
        return

    strict_pose = _run_pnp(corrs_sub, K, pnp_cfg, threshold_px=8.0)
    strict_count = int(strict_pose["stats"].get("n_inliers", 0))
    if bool(strict_pose["ok"]):
        support_mask, _ = _pnp_inlier_mask_from_pose(
            corrs.X_w,
            corrs.x_cur,
            K,
            strict_pose["R"],
            strict_pose["t"],
            threshold_px=8.0,
            eps=float(pnp_cfg["eps"]),
        )
        support_mask = np.asarray(support_mask, dtype=bool).reshape(-1)
        overlap = int(np.sum(support_mask & loose_pose["loose_support_mask"]))
        union = int(np.sum(support_mask | loose_pose["loose_support_mask"]))
        iou = float(overlap / union) if union > 0 else np.nan
        print(
            f"  subset_rule {label}: keep={n} strict_ok=True subset_strict_inliers={strict_count} "
            f"fullset_support@8={int(np.sum(support_mask))} fullset_overlap_with_loose12={overlap} "
            f"iou={iou:.3f} {_pose_delta(loose_pose, strict_pose)}"
        )
        print(f"    support_spread: {_coverage_stats(corrs, support_mask, loose_pose['image_shape'])}")
    else:
        print(
            f"  subset_rule {label}: keep={n} strict_ok=False "
            f"reason={strict_pose['stats'].get('reason', None)} inliers={strict_count}"
        )


def _loose_pose_summary(corrs, loose_pose: dict, meta: dict) -> None:
    errors_px = meta["errors_px"]
    loose_mask = meta["loose_support_mask"]
    strict_under_loose = meta["strict_under_loose_mask"]
    loose_only = loose_mask & ~strict_under_loose
    rejected = ~loose_mask

    print("residual_structure_under_loose12:")
    print(f"  all: {_residual_stats(errors_px, np.ones_like(loose_mask, dtype=bool))}")
    print(f"  <=12px support: {_residual_stats(errors_px, loose_mask)}")
    print(f"  <=8px under loose12: {_residual_stats(errors_px, strict_under_loose)}")
    print(f"  8-12px under loose12: {_residual_stats(errors_px, loose_only)}")
    print(f"  >12px rejected: {_residual_stats(errors_px, rejected)}")

    bins = [
        ("<=4", errors_px <= 4.0),
        ("4-8", (errors_px > 4.0) & (errors_px <= 8.0)),
        ("8-12", (errors_px > 8.0) & (errors_px <= 12.0)),
        ("12-20", (errors_px > 12.0) & (errors_px <= 20.0)),
        ("20-40", (errors_px > 20.0) & (errors_px <= 40.0)),
        (">40", errors_px > 40.0),
    ]
    print("  residual_bins:", ", ".join(f"{label}:{int(np.sum(mask))}" for label, mask in bins))
    print(f"  loose12_support_spread: {_coverage_stats(corrs, loose_mask, loose_pose['image_shape'])}")
    print(f"  strict8_under_loose12_spread: {_coverage_stats(corrs, strict_under_loose, loose_pose['image_shape'])}")
    print(f"  rejected_spread: {_coverage_stats(corrs, rejected, loose_pose['image_shape'])}")


def _population_summaries(meta: dict) -> None:
    loose_mask = meta["loose_support_mask"]
    rejected = ~loose_mask
    strict_under_loose = meta["strict_under_loose_mask"]
    loose_only = loose_mask & ~strict_under_loose
    finite_depth = np.isfinite(meta["depth_m"]) & (meta["depth_m"] > 0.0)

    print("population_breakdown:")
    print(f"  birth_source totals: {_format_counts(meta['birth_source'], np.ones_like(loose_mask, dtype=bool))}")
    print(f"  birth_source loose12: {_format_counts(meta['birth_source'], loose_mask)}")
    print(f"  birth_source rejected: {_format_counts(meta['birth_source'], rejected)}")
    print(f"  birth_source support_rate: {_format_support_rate(meta['birth_source'], loose_mask)}")

    print(f"  age_years? no, age_in_keyframes loose12: {_format_counts(meta['age_kf'], loose_mask)}")
    print(f"  age_in_keyframes rejected: {_format_counts(meta['age_kf'], rejected)}")
    print(f"  age support_rate: {_format_support_rate(meta['age_kf'], loose_mask)}")

    print(f"  obs_count loose12: {_format_counts(meta['obs_count'], loose_mask)}")
    print(f"  obs_count rejected: {_format_counts(meta['obs_count'], rejected)}")
    print(f"  obs_count support_rate: {_format_support_rate(meta['obs_count'], loose_mask)}")

    depth = meta["depth_m"]
    finite_depth_vals = depth[finite_depth]
    if finite_depth_vals.size >= 3:
        q33, q67 = np.percentile(finite_depth_vals, [33.3, 66.7])
        depth_bins = [
            (f"near<={q33:.2f}m", finite_depth & (depth <= q33)),
            (f"mid({q33:.2f},{q67:.2f}]", finite_depth & (depth > q33) & (depth <= q67)),
            (f"far>{q67:.2f}m", finite_depth & (depth > q67)),
        ]
        print(f"  depth support_rate: {_format_numeric_buckets(depth, loose_mask, bins=depth_bins)}")
    else:
        print("  depth support_rate: unavailable")

    print(f"  image_region loose12: {_format_counts(meta['region'], loose_mask)}")
    print(f"  image_region rejected: {_format_counts(meta['region'], rejected)}")
    print(f"  image_region support_rate: {_format_support_rate(meta['region'], loose_mask)}")

    for label, mask in [
        ("loose12", loose_mask),
        ("strict8_under_loose12", strict_under_loose),
        ("loose12_only", loose_only),
        ("rejected", rejected),
    ]:
        print(
            f"  {label} residual_by_birth: "
            f"bootstrap={_residual_stats(meta['errors_px'], mask & (meta['birth_source'] == 'bootstrap'))} "
            f"map_growth={_residual_stats(meta['errors_px'], mask & (meta['birth_source'] == 'map_growth'))}"
        )


def _metadata_arrays(corrs, seed: dict, K: np.ndarray, current_kf: int, image_shape: tuple[int, int], loose_pose: dict, *, eps: float) -> dict:
    lm_by_id = {
        int(lm["id"]): lm
        for lm in seed.get("landmarks", [])
        if isinstance(lm, dict) and "id" in lm
    }
    N = int(corrs.X_w.shape[1])
    birth_source = np.full((N,), "unknown", dtype=object)
    age_kf = np.full((N,), -1, dtype=np.int64)
    obs_count = np.full((N,), -1, dtype=np.int64)
    birth_kf = np.full((N,), -1, dtype=np.int64)
    for i, lm_id in enumerate(np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)):
        lm = lm_by_id.get(int(lm_id), {})
        birth_source[i] = str(lm.get("birth_source", "unknown"))
        birth_kf[i] = int(lm.get("birth_kf", -1))
        age_kf[i] = int(current_kf) - int(birth_kf[i])
        obs_count[i] = int(_obs_count(lm))

    R_loose = np.asarray(loose_pose["R"], dtype=np.float64)
    t_loose = np.asarray(loose_pose["t"], dtype=np.float64).reshape(3)
    X_c = world_to_camera_points(R_loose, t_loose, corrs.X_w)
    depth_m = np.asarray(X_c[2, :], dtype=np.float64).reshape(-1)
    d_sq = np.asarray(reprojection_errors_sq(K, R_loose, t_loose, corrs.X_w, corrs.x_cur), dtype=np.float64).reshape(-1)
    d_sq[~np.isfinite(d_sq)] = np.inf
    d_sq[depth_m <= float(eps)] = np.inf
    errors_px = np.sqrt(d_sq)
    loose_support_mask = np.isfinite(errors_px) & (errors_px <= 12.0)
    strict_under_loose_mask = np.isfinite(errors_px) & (errors_px <= 8.0)
    region = _grid_labels(corrs.x_cur, image_shape, grid_cols=4, grid_rows=3)

    return {
        "birth_source": birth_source,
        "birth_kf": birth_kf,
        "age_kf": age_kf.astype(object),
        "obs_count": obs_count.astype(object),
        "depth_m": depth_m,
        "errors_px": errors_px,
        "loose_support_mask": loose_support_mask,
        "strict_under_loose_mask": strict_under_loose_mask,
        "region": region,
    }


def _advance_frontend_to_frame(seed: dict, keyframe_feats, keyframe_index: int, seq, K: np.ndarray, frontend_kwargs: dict, *, target_frame: int) -> tuple[dict, object, int]:
    for frame_index in range(int(keyframe_index) + 1, int(target_frame)):
        cur_im, _, _ = seq.get(frame_index)
        out = process_frame_against_seed(
            K,
            seed,
            keyframe_feats,
            cur_im,
            feature_cfg=frontend_kwargs["feature_cfg"],
            F_cfg=frontend_kwargs["F_cfg"],
            keyframe_kf=keyframe_index,
            current_kf=frame_index,
            **frontend_kwargs["pnp_frontend_kwargs"],
        )
        seed = out["seed"]
        if bool(out.get("stats", {}).get("keyframe_promoted", False)):
            keyframe_feats = out["track_out"]["cur_feats"]
            keyframe_index = frame_index
    return seed, keyframe_feats, int(keyframe_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--frame_index", type=int, default=7)
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    frame_index = int(args.frame_index)

    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)
    pnp_cfg = dict(frontend_kwargs["pnp_frontend_kwargs"])
    dataset_cfg = cfg["dataset"]
    dataset_root = (ROOT / dataset_cfg["root"]).resolve()
    seq_name = str(dataset_cfg["seq"])
    check_dir(dataset_root, name="dataset_root")

    seq = load_eth3d_sequence(
        dataset_root,
        seq_name,
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    im0, _, _ = seq.get(0)
    im1, _, _ = seq.get(1)

    boot = bootstrap_from_two_frames(
        K,
        K,
        im0,
        im1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )
    if not bool(boot.get("ok", False)):
        raise RuntimeError(f"Bootstrap failed: {boot.get('stats', {}).get('reason', None)}")

    seed = boot["seed"]
    keyframe_feats = seed["feats1"]
    keyframe_index = 1

    seed, keyframe_feats, keyframe_index = _advance_frontend_to_frame(
        seed,
        keyframe_feats,
        keyframe_index,
        seq,
        K,
        frontend_kwargs,
        target_frame=frame_index,
    )

    cur_im, cur_ts, cur_id = seq.get(frame_index)
    image_shape = (int(np.asarray(cur_im).shape[0]), int(np.asarray(cur_im).shape[1]))
    track_out = track_against_keyframe(
        K,
        keyframe_feats,
        cur_im,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
    )
    corrs, corr_stats = build_pnp_correspondences_with_stats(
        seed,
        track_out,
        min_landmark_observations=int(pnp_cfg["min_landmark_observations"]),
        allow_bootstrap_landmarks_for_pose=bool(pnp_cfg["allow_bootstrap_landmarks_for_pose"]),
        min_post_bootstrap_observations_for_pose=int(pnp_cfg["min_post_bootstrap_observations_for_pose"]),
        enable_local_consistency_filter=False,
        enable_spatial_thinning_filter=False,
    )

    pose_8 = _run_pnp(corrs, K, pnp_cfg, threshold_px=8.0)
    pose_12 = _run_pnp(corrs, K, pnp_cfg, threshold_px=12.0)
    if not bool(pose_12["ok"]):
        raise RuntimeError(f"12px pose failed: {pose_12['stats'].get('reason', None)}")

    loose_meta = _metadata_arrays(
        corrs,
        seed,
        K,
        frame_index,
        image_shape,
        pose_12,
        eps=float(pnp_cfg["eps"]),
    )
    pose_12["loose_support_mask"] = loose_meta["loose_support_mask"]
    pose_12["image_shape"] = image_shape

    print(f"sequence={seq_name} frame_index={frame_index} frame_id={cur_id} timestamp={float(cur_ts)}")
    print(
        f"reference_keyframe_index={keyframe_index} "
        f"sample_size={int(pnp_cfg['sample_size'])} min_inliers={int(pnp_cfg['min_inliers'])} "
        f"strict_threshold_px=8.0 loose_threshold_px=12.0"
    )
    print(
        f"track_inliers={int(track_out.get('stats', {}).get('n_inliers', 0))} "
        f"eligible_corr={int(corrs.X_w.shape[1])} corr_raw={int(corr_stats.get('n_corr_raw', 0))} "
        f"bootstrap_used={int(corr_stats.get('n_corr_bootstrap_used', 0))} "
        f"map_growth_used={int(corr_stats.get('n_corr_post_bootstrap_used', 0))}"
    )
    print(
        f"strict8: ok={bool(pose_8['ok'])} n_inliers={int(pose_8['stats'].get('n_inliers', 0))} "
        f"reason={pose_8['stats'].get('reason', None)}"
    )
    print(
        f"loose12: ok={bool(pose_12['ok'])} n_inliers={int(pose_12['stats'].get('n_inliers', 0))} "
        f"reason={pose_12['stats'].get('reason', None)}"
    )
    if bool(pose_8["ok"]):
        overlap = int(np.sum(np.asarray(pose_8["inlier_mask"], dtype=bool) & np.asarray(pose_12["inlier_mask"], dtype=bool)))
        union = int(np.sum(np.asarray(pose_8["inlier_mask"], dtype=bool) | np.asarray(pose_12["inlier_mask"], dtype=bool)))
        print(f"strict8_vs_loose12: overlap={overlap} iou={float(overlap / union) if union > 0 else np.nan:.3f} {_pose_delta(pose_12, pose_8)}")

    _loose_pose_summary(corrs, pose_12, loose_meta)
    _population_summaries(loose_meta)

    print("subset_rule_tests:")
    _subset_eval("residual<=12_under_loose12", corrs, loose_meta["loose_support_mask"], K, pnp_cfg, pose_12)
    _subset_eval("residual<=10_under_loose12", corrs, np.isfinite(loose_meta["errors_px"]) & (loose_meta["errors_px"] <= 10.0), K, pnp_cfg, pose_12)
    _subset_eval("residual<=8_under_loose12", corrs, loose_meta["strict_under_loose_mask"], K, pnp_cfg, pose_12)

    xy_kf = np.asarray(track_out.get("xy_kf", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
    pair_to_track_idx = {
        (int(kf_idx), int(cur_idx)): i
        for i, (kf_idx, cur_idx) in enumerate(zip(
            np.asarray(track_out.get("kf_feat_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1),
            np.asarray(track_out.get("cur_feat_idx", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1),
        ))
    }
    corr_track_idx = np.asarray(
        [pair_to_track_idx.get((int(kf_idx), int(cur_idx)), -1) for kf_idx, cur_idx in zip(corrs.kf_feat_idx, corrs.cur_feat_idx)],
        dtype=np.int64,
    )
    valid_track = corr_track_idx >= 0
    xy_kf_corr = np.asarray(xy_kf[corr_track_idx[valid_track]], dtype=np.float64) if np.any(valid_track) else np.zeros((0, 2), dtype=np.float64)
    x_cur_corr = np.asarray(corrs.x_cur[:, valid_track].T, dtype=np.float64) if np.any(valid_track) else np.zeros((0, 2), dtype=np.float64)

    local_trials = [
        ("local80_n3_r12", 80.0, 3, 12.0),
        ("local80_n3_r8", 80.0, 3, 8.0),
        ("local120_n4_r12", 120.0, 4, 12.0),
        ("local120_n4_r8", 120.0, 4, 8.0),
    ]
    for label, radius_px, min_neighbours, max_residual_px in local_trials:
        keep = np.ones((corrs.X_w.shape[1],), dtype=bool)
        if np.any(valid_track):
            keep_valid, stats = pnp_local_displacement_consistency_mask(
                xy_kf_corr,
                x_cur_corr,
                radius_px=float(radius_px),
                min_neighbours=int(min_neighbours),
                max_median_residual_px=float(max_residual_px),
                min_keep=0,
            )
            keep = np.zeros((corrs.X_w.shape[1],), dtype=bool)
            keep[valid_track] = np.asarray(keep_valid, dtype=bool)
        else:
            stats = {"n_keep": 0, "n_removed": int(corrs.X_w.shape[1])}
        print(
            f"  local_consistency {label}: keep={int(np.sum(keep))} removed={int(np.sum(~keep))} "
            f"stats={stats}"
        )
        _subset_eval(label, corrs, keep, K, pnp_cfg, pose_12)


if __name__ == "__main__":
    main()
