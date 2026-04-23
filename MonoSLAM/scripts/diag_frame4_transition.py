# scripts/diag_frame4_transition.py
# Audit the frame-3 -> frame-4 state transition to find the PnP failure root cause.
# Runs the pipeline through frame 3, then inspects the frame-4 correspondence set.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from frontend_eth3d_common import (
    frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg,
    load_runtime_cfg as _load_runtime_cfg,
)
from core.checks import check_dir
from datasets.eth3d import load_eth3d_sequence
from geometry.camera import projection_matrix, reprojection_errors_sq, world_to_camera_points
from geometry.pnp import (
    PnPCorrespondences,
    _pnp_inlier_mask_from_pose,
    _slice_pnp_correspondences,
    build_pnp_correspondences_with_stats,
    estimate_pose_pnp,
    estimate_pose_pnp_ransac,
    pnp_inlier_spatial_coverage,
)
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.pnp_frontend import estimate_pose_from_seed
from slam.seed import seed_keyframe_pose
from slam.tracking import track_against_keyframe


def _reproj_err_px(K, R, t, X_w, x_cur):
    # Return per-point reprojection error in pixels (inf for behind-camera points)
    N = int(X_w.shape[1])
    X_c = world_to_camera_points(R, t, X_w)
    errs = np.asarray(reprojection_errors_sq(K, R, t, X_w, x_cur), dtype=np.float64).reshape(-1)
    errs[~np.isfinite(errs)] = np.inf
    errs[X_c[2, :] <= 1e-12] = np.inf
    return np.sqrt(errs)


def _print_err_summary(label, errs_px):
    fin = errs_px[np.isfinite(errs_px)]
    if fin.size == 0:
        print(f"  {label}: no finite errors")
        return
    print(
        f"  {label}: n={fin.size}  "
        f"median={np.median(fin):.2f}px  "
        f"p90={np.percentile(fin, 90):.2f}px  "
        f"@3px={int(np.sum(fin <= 3))}  "
        f"@8px={int(np.sum(fin <= 8))}  "
        f"@20px={int(np.sum(fin <= 20))}"
    )


def _obs_count(lm):
    obs = lm.get("obs", [])
    return int(sum(1 for o in obs if isinstance(o, dict)))


def _birth_source_counts(landmarks, landmark_ids):
    lm_by_id = {int(lm["id"]): lm for lm in landmarks if isinstance(lm, dict) and "id" in lm}
    counts = {}
    for lm_id in landmark_ids:
        lm = lm_by_id.get(int(lm_id), {})
        src = lm.get("birth_source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def _seed_snapshot(seed):
    lmid = np.asarray(seed.get("landmark_id_by_feat1", []), dtype=np.int64)
    landmarks = seed.get("landmarks", [])
    obs_by_kf = {}
    obs_by_kf_source = {}
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        src = lm.get("birth_source", "unknown")
        for ob in lm.get("obs", []):
            if not isinstance(ob, dict):
                continue
            kf = int(ob.get("kf", -1))
            obs_by_kf[kf] = obs_by_kf.get(kf, 0) + 1
            key = (kf, src)
            obs_by_kf_source[key] = obs_by_kf_source.get(key, 0) + 1

    return {
        "seed_id": id(seed),
        "keys": set(seed.keys()),
        "n_landmarks": int(len(landmarks)),
        "landmark_id_by_feat1_size": int(lmid.size),
        "landmark_id_by_feat1_mapped": int(np.sum(lmid >= 0)),
        "keyframe_kf": seed.get("keyframe_kf", None),
        "has_feats1": "feats1" in seed,
        "has_last_append_stats": "last_append_stats" in seed,
        "has_last_tracked_observation_append_stats": "last_tracked_observation_append_stats" in seed,
        "has_last_keyframe_promotion": "last_keyframe_promotion" in seed,
        "obs_by_kf": obs_by_kf,
        "obs_by_kf_source": obs_by_kf_source,
    }


def _print_seed_delta(before, after, current_kf):
    added_keys = sorted(after["keys"] - before["keys"])
    removed_keys = sorted(before["keys"] - after["keys"])
    before_obs = int(before["obs_by_kf"].get(int(current_kf), 0))
    after_obs = int(after["obs_by_kf"].get(int(current_kf), 0))
    print(f"  seed object reused: {before['seed_id'] == after['seed_id']}")
    print(f"  keys added={added_keys}  removed={removed_keys}")
    print(
        "  landmarks: "
        f"{before['n_landmarks']} -> {after['n_landmarks']}  "
        "landmark_id_by_feat1: "
        f"{before['landmark_id_by_feat1_size']}/{before['landmark_id_by_feat1_mapped']} -> "
        f"{after['landmark_id_by_feat1_size']}/{after['landmark_id_by_feat1_mapped']}"
    )
    print(f"  keyframe_kf: {before['keyframe_kf']} -> {after['keyframe_kf']}")
    print(f"  observations at kf={current_kf}: {before_obs} -> {after_obs}")
    src_counts = {}
    for (kf, src), count in after["obs_by_kf_source"].items():
        if int(kf) == int(current_kf):
            src_counts[src] = src_counts.get(src, 0) + int(count)
    print(f"  observations at kf={current_kf} by source: {src_counts}")


def _print_step_stats(label, out):
    stats = out.get("stats", {})
    append_stats = out.get("seed", {}).get("last_tracked_observation_append_stats", {})
    map_stats = out.get("map_growth_out").stats if out.get("map_growth_out", None) is not None else {}
    keyframe_stats = out.get("keyframe_out").stats if out.get("keyframe_out", None) is not None else {}
    print(f"\n=== {label} step stats ===")
    print(
        "  frontend: "
        f"track_inliers={stats.get('n_track_inliers')}  "
        f"pnp_corr={stats.get('n_pnp_corr')}  "
        f"pnp_inliers={stats.get('n_pnp_inliers')}  "
        f"new_added={stats.get('n_new_added')}  "
        f"promoted={stats.get('keyframe_promoted')}  "
        f"reason={stats.get('keyframe_reason')}"
    )
    print(
        "  append existing: "
        f"candidates={append_stats.get('n_append_candidates_existing')}  "
        f"pnp_inliers_added={append_stats.get('n_append_pnp_inliers_added')}  "
        f"extra_tested={append_stats.get('n_append_extra_reproj_tested')}  "
        f"extra_pass={append_stats.get('n_append_extra_reproj_pass')}  "
        f"total={append_stats.get('n_append_total')}  "
        f"linked_after={append_stats.get('n_landmarks_with_obs_current_kf_after_append')}"
    )
    print(
        "  map growth: "
        f"candidates={map_stats.get('n_candidates')}  "
        f"triangulated={map_stats.get('n_triangulated_valid')}  "
        f"added={map_stats.get('n_added')}  "
        f"reason={map_stats.get('reason')}"
    )
    print(
        "  keyframe: "
        f"linked_candidate={keyframe_stats.get('n_linked_landmarks_candidate')}  "
        f"promoted={keyframe_stats.get('promoted')}"
    )


def _spatial_summary(label, xy, image_shape):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim == 2 and xy.shape[0] == 2 and xy.shape[1] != 2:
        xy = xy.T
    if xy.ndim != 2 or xy.shape[1] != 2:
        xy = np.zeros((0, 2), dtype=np.float64)
    if xy.size == 0 or xy.shape[0] == 0:
        print(f"  {label}: n=0")
        return

    H, W = int(image_shape[0]), int(image_shape[1])
    x = xy[:, 0]
    y = xy[:, 1]
    bbox_w = float(np.max(x) - np.min(x))
    bbox_h = float(np.max(y) - np.min(y))
    bbox_area_frac = (bbox_w * bbox_h) / max(float(W * H), 1.0)
    cols = np.clip((x / max(float(W), 1.0) * 4).astype(np.int64), 0, 3)
    rows = np.clip((y / max(float(H), 1.0) * 3).astype(np.int64), 0, 2)
    cells = {(int(r), int(c)) for r, c in zip(rows, cols)}
    print(
        f"  {label}: n={xy.shape[0]}  "
        f"bbox_area_frac={bbox_area_frac:.3f}  "
        f"grid_cells_4x3={len(cells)}  "
        f"x=[{np.min(x):.1f},{np.max(x):.1f}]  "
        f"y=[{np.min(y):.1f},{np.max(y):.1f}]"
    )


def _point_spatial_stats(xy, image_shape, *, grid_cols=4, grid_rows=3):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim == 2 and xy.shape[0] == 2 and xy.shape[1] != 2:
        xy = xy.T
    if xy.ndim != 2 or xy.shape[1] != 2:
        xy = np.zeros((0, 2), dtype=np.float64)
    xy = xy[np.isfinite(xy).all(axis=1)]

    H, W = int(image_shape[0]), int(image_shape[1])
    grid = [[0 for _ in range(int(grid_cols))] for _ in range(int(grid_rows))]
    count = int(xy.shape[0])
    if count == 0:
        return {
            "count": 0,
            "bbox": None,
            "bbox_width_px": None,
            "bbox_height_px": None,
            "bbox_area_fraction": None,
            "occupied_cells": 0,
            "occupancy_grid": grid,
            "max_cell_fraction": None,
            "cell_ids": [],
            "min_pairwise_sep_px": None,
        }

    x = xy[:, 0]
    y = xy[:, 1]
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    bbox_w = max(0.0, xmax - xmin)
    bbox_h = max(0.0, ymax - ymin)
    bbox_area_fraction = float((bbox_w * bbox_h) / max(float(W * H), 1.0))

    cols = np.clip((x / max(float(W), 1.0) * int(grid_cols)).astype(np.int64), 0, int(grid_cols) - 1)
    rows = np.clip((y / max(float(H), 1.0) * int(grid_rows)).astype(np.int64), 0, int(grid_rows) - 1)
    for row, col in zip(rows, cols):
        grid[int(row)][int(col)] += 1

    occupied_cells = int(sum(1 for row in grid for value in row if int(value) > 0))
    max_cell_count = int(max(max(row) for row in grid))
    max_cell_fraction = float(max_cell_count / max(count, 1))
    cell_ids = sorted({(int(r), int(c)) for r, c in zip(rows, cols)})

    min_pairwise_sep_px = None
    if count >= 2:
        d_xy = xy[:, None, :] - xy[None, :, :]
        d = np.sqrt(np.sum(d_xy ** 2, axis=2))
        d[np.eye(count, dtype=bool)] = np.inf
        min_pairwise_sep_px = float(np.min(d))

    return {
        "count": int(count),
        "bbox": [xmin, ymin, xmax, ymax],
        "bbox_width_px": float(bbox_w),
        "bbox_height_px": float(bbox_h),
        "bbox_area_fraction": bbox_area_fraction,
        "occupied_cells": int(occupied_cells),
        "occupancy_grid": grid,
        "max_cell_fraction": max_cell_fraction,
        "cell_ids": [(int(r), int(c)) for r, c in cell_ids],
        "min_pairwise_sep_px": min_pairwise_sep_px,
    }


def _format_optional(value, digits=3):
    if value is None:
        return "None"
    return f"{float(value):.{int(digits)}f}"


def _support_cell_signature(summary):
    if not isinstance(summary, dict):
        return tuple()
    return tuple(tuple(int(v) for v in cell) for cell in summary.get("cell_ids", []))


def _sample_minimal_subset(rng, x_cur, sample_size, image_shape, *, policy):
    N = int(x_cur.shape[1])
    if policy == "baseline":
        return np.asarray(rng.choice(N, size=int(sample_size), replace=False), dtype=np.int64)

    if policy != "spatial_cells":
        raise ValueError(f"Unknown sampling policy: {policy}")

    xy = np.asarray(x_cur.T, dtype=np.float64)
    H, W = int(image_shape[0]), int(image_shape[1])
    cols = np.clip((xy[:, 0] / max(float(W), 1.0) * 4).astype(np.int64), 0, 3)
    rows = np.clip((xy[:, 1] / max(float(H), 1.0) * 3).astype(np.int64), 0, 2)

    cell_to_idx = {}
    for i in range(N):
        cell = (int(rows[i]), int(cols[i]))
        cell_to_idx.setdefault(cell, []).append(int(i))

    cell_keys = list(cell_to_idx.keys())
    rng.shuffle(cell_keys)

    selected = []
    used = np.zeros((N,), dtype=bool)
    for cell in cell_keys:
        candidates = np.asarray(cell_to_idx[cell], dtype=np.int64)
        candidates = candidates[~used[candidates]]
        if int(candidates.size) == 0:
            continue
        pick = int(candidates[int(rng.integers(int(candidates.size)))])
        selected.append(pick)
        used[pick] = True
        if len(selected) == int(sample_size):
            return np.asarray(selected, dtype=np.int64)

    remaining = np.flatnonzero(~used)
    need = int(sample_size) - len(selected)
    if need > 0:
        fill = np.asarray(rng.choice(remaining, size=need, replace=False), dtype=np.int64).reshape(-1)
        selected.extend(int(v) for v in fill.tolist())

    return np.asarray(selected, dtype=np.int64)


def _replay_minimal_sample_ransac(corrs, K, image_shape, pnp_kw, *, policy):
    X_w = np.asarray(corrs.X_w, dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur, dtype=np.float64)
    rng = np.random.default_rng(int(pnp_kw.get("ransac_seed", 0)))
    num_trials = int(pnp_kw.get("num_trials", 1000))
    sample_size = int(pnp_kw.get("sample_size", 6))
    threshold_px = float(pnp_kw.get("threshold_px", 8.0))
    min_points = int(pnp_kw.get("min_points", 6))
    rank_tol = float(pnp_kw.get("rank_tol", 1e-10))
    min_cheirality_ratio = float(pnp_kw.get("min_cheirality_ratio", 0.5))
    eps = float(pnp_kw.get("eps", 1e-12))

    spatial_tie_inlier_gap = 2
    spatial_tie_grid_cols = 4
    spatial_tie_grid_rows = 3

    x_min = float(np.min(x_cur[0, :]))
    y_min = float(np.min(x_cur[1, :]))
    x_span = max(float(np.max(x_cur[0, :]) - x_min), 1.0)
    y_span = max(float(np.max(x_cur[1, :]) - y_min), 1.0)
    x_cur_rank = np.asarray(x_cur, dtype=np.float64).copy()
    x_cur_rank[0, :] -= x_min
    x_cur_rank[1, :] -= y_min
    rank_image_shape = (int(np.ceil(y_span)) + 1, int(np.ceil(x_span)) + 1)

    def _support_rank(inlier_mask):
        coverage = pnp_inlier_spatial_coverage(
            x_cur_rank,
            inlier_mask,
            rank_image_shape,
            grid_cols=int(spatial_tie_grid_cols),
            grid_rows=int(spatial_tie_grid_rows),
        )
        bbox_area_fraction = coverage.get("bbox_area_fraction", None)
        max_cell_fraction = coverage.get("max_cell_fraction", None)
        return (
            int(coverage.get("occupied_cells", 0)),
            -np.inf if bbox_area_fraction is None else float(bbox_area_fraction),
            -np.inf if max_cell_fraction is None else -float(max_cell_fraction),
        )

    best_record = None
    best_count = 0
    best_mean_err = np.inf
    best_support_rank = (0, -np.inf, -np.inf)
    records = []
    n_model_success = 0

    for trial in range(num_trials):
        idx = _sample_minimal_subset(
            rng,
            x_cur,
            sample_size,
            image_shape,
            policy=policy,
        )
        corrs_sub = _slice_pnp_correspondences(corrs, idx)

        try:
            R_t, t_t, solve_stats = estimate_pose_pnp(
                corrs_sub,
                K,
                min_points=min_points,
                rank_tol=rank_tol,
                min_cheirality_ratio=min_cheirality_ratio,
                eps=eps,
            )
        except Exception:
            continue

        if R_t is None or t_t is None:
            continue

        n_model_success += 1
        mask_t, d_sq_t = _pnp_inlier_mask_from_pose(
            X_w,
            x_cur,
            K,
            R_t,
            t_t,
            threshold_px=threshold_px,
            eps=eps,
        )
        count_t = int(np.sum(mask_t))
        if count_t == 0:
            continue

        mean_err_t = float(np.mean(d_sq_t[mask_t]))
        support_rank_t = _support_rank(mask_t)
        sample_summary = _point_spatial_stats(x_cur[:, idx].T, image_shape)
        support_summary = _point_spatial_stats(x_cur[:, mask_t].T, image_shape)
        record = {
            "trial": int(trial),
            "sample_idx": np.asarray(idx, dtype=np.int64),
            "count": int(count_t),
            "mean_err_sq": float(mean_err_t),
            "support_rank": tuple(float(v) if np.isfinite(v) else v for v in support_rank_t),
            "sample_summary": sample_summary,
            "support_summary": support_summary,
            "support_signature": _support_cell_signature(support_summary),
            "A_rank": int(solve_stats.get("A_rank", -1)),
            "cheirality_ratio": solve_stats.get("cheirality_ratio", None),
            "reprojection_rmse_px": solve_stats.get("reprojection_rmse_px", None),
        }
        records.append(record)

        inlier_gap_t = int(best_count - count_t)
        better_support_t = support_rank_t > best_support_rank
        same_support_t = support_rank_t == best_support_rank
        keep_t = False
        if count_t > best_count:
            keep_t = True
        elif 0 <= inlier_gap_t <= int(spatial_tie_inlier_gap):
            if better_support_t:
                keep_t = True
            elif same_support_t and mean_err_t < best_mean_err:
                keep_t = True

        if keep_t:
            best_record = record
            best_count = count_t
            best_mean_err = mean_err_t
            best_support_rank = support_rank_t

    return {
        "policy": str(policy),
        "records": records,
        "best_record": best_record,
        "n_model_success": int(n_model_success),
        "num_trials": int(num_trials),
        "sample_size": int(sample_size),
        "threshold_px": float(threshold_px),
    }


def _print_record_summary(prefix, record):
    if record is None:
        print(f"{prefix}none")
        return

    sample = record["sample_summary"]
    support = record["support_summary"]
    print(
        f"{prefix}inliers={record['count']}  "
        f"support_cells={support['occupied_cells']}  "
        f"support_bbox_wh=({_format_optional(support['bbox_width_px'], 1)},"
        f"{_format_optional(support['bbox_height_px'], 1)})px  "
        f"sample_cells={sample['occupied_cells']}  "
        f"sample_bbox_wh=({_format_optional(sample['bbox_width_px'], 1)},"
        f"{_format_optional(sample['bbox_height_px'], 1)})px  "
        f"sample_min_sep={_format_optional(sample['min_pairwise_sep_px'], 1)}px  "
        f"sample_grid={sample['occupancy_grid']}  "
        f"A_rank={record['A_rank']}"
    )


def _print_best_group_summary(prefix, records, corrs, bbox):
    if len(records) == 0:
        print(f"{prefix}count=0")
        return

    sample_cells = np.asarray([r["sample_summary"]["occupied_cells"] for r in records], dtype=np.int64)
    bbox_w = np.asarray([r["sample_summary"]["bbox_width_px"] for r in records], dtype=np.float64)
    bbox_h = np.asarray([r["sample_summary"]["bbox_height_px"] for r in records], dtype=np.float64)
    min_sep = np.asarray(
        [
            np.nan if r["sample_summary"]["min_pairwise_sep_px"] is None else float(r["sample_summary"]["min_pairwise_sep_px"])
            for r in records
        ],
        dtype=np.float64,
    )
    ranks = sorted({int(r["A_rank"]) for r in records})

    all_inside = 0
    four_or_more_inside = 0
    sample_signature_hist = {}
    for r in records:
        sample_signature = tuple(tuple(int(v) for v in cell) for cell in r["sample_summary"].get("cell_ids", []))
        sample_signature_hist[sample_signature] = sample_signature_hist.get(sample_signature, 0) + 1

        sample_xy = np.asarray(corrs.x_cur[:, r["sample_idx"]].T, dtype=np.float64)
        inside = _point_mask_inside_bbox(sample_xy, bbox)
        n_inside = int(np.sum(inside))
        if n_inside == int(sample_xy.shape[0]):
            all_inside += 1
        if n_inside >= 4:
            four_or_more_inside += 1

    top_signature, top_signature_count = max(sample_signature_hist.items(), key=lambda kv: kv[1])
    print(
        f"{prefix}count={len(records)}  "
        f"sample_cells_median={np.median(sample_cells):.1f}  "
        f"sample_cells_max={int(np.max(sample_cells))}  "
        f"sample_bbox_w_median={np.median(bbox_w):.1f}px  "
        f"sample_bbox_h_median={np.median(bbox_h):.1f}px  "
        f"sample_min_sep_median={np.nanmedian(min_sep):.1f}px  "
        f"all6_inside_best_support_bbox={all_inside}/{len(records)}  "
        f"ge4_inside_best_support_bbox={four_or_more_inside}/{len(records)}  "
        f"top_sample_cells={top_signature} ({top_signature_count}/{len(records)})  "
        f"A_rank_values={ranks}"
    )


def _print_minimal_sample_failure_audit(label, corrs, K, pnp_kw, image_shape):
    strict_px = float(pnp_kw.get("threshold_px", 8.0))
    loose_px = 40.0
    sample_size = int(pnp_kw.get("sample_size", 6))
    num_trials = int(pnp_kw.get("num_trials", 1000))
    ransac_seed = int(pnp_kw.get("ransac_seed", 0))
    min_inliers = int(pnp_kw.get("min_inliers", 12))
    min_points = int(pnp_kw.get("min_points", 6))
    rank_tol = float(pnp_kw.get("rank_tol", 1e-10))
    min_cheirality_ratio = float(pnp_kw.get("min_cheirality_ratio", 0.5))
    eps = float(pnp_kw.get("eps", 1e-12))

    print(f"\n=== {label} minimal-sample audit ===")
    print(
        f"  strict_px={strict_px:.0f}  loose_px={loose_px:.0f}  "
        f"sample_size={sample_size}  trials={num_trials}  seed={ransac_seed}"
    )

    R_loose, t_loose, _, loose_stats = estimate_pose_pnp_ransac(
        corrs,
        K,
        num_trials=num_trials,
        sample_size=sample_size,
        threshold_px=loose_px,
        min_inliers=min_inliers,
        seed=ransac_seed,
        min_points=min_points,
        rank_tol=rank_tol,
        min_cheirality_ratio=min_cheirality_ratio,
        eps=eps,
        refit=bool(pnp_kw.get("refit", True)),
        refine_nonlinear=bool(pnp_kw.get("refine_nonlinear", True)),
        refine_max_iters=int(pnp_kw.get("refine_max_iters", 15)),
        refine_damping=float(pnp_kw.get("refine_damping", 1e-6)),
        refine_step_tol=float(pnp_kw.get("refine_step_tol", 1e-9)),
        refine_improvement_tol=float(pnp_kw.get("refine_improvement_tol", 1e-9)),
    )
    strict_under_loose = np.zeros((corrs.X_w.shape[1],), dtype=bool)
    if R_loose is not None and t_loose is not None:
        strict_under_loose, _ = _pnp_inlier_mask_from_pose(
            corrs.X_w,
            corrs.x_cur,
            K,
            R_loose,
            t_loose,
            threshold_px=strict_px,
            eps=eps,
        )
    loose_support = _point_spatial_stats(corrs.x_cur[:, strict_under_loose].T, image_shape)
    print(
        "  loose-reference pose: "
        f"ok={R_loose is not None}  "
        f"loose_inliers={int(loose_stats.get('n_inliers', 0)) if isinstance(loose_stats, dict) else 0}  "
        f"strict_inliers_under_loose={int(np.sum(strict_under_loose))}  "
        f"strict_support_cells={loose_support['occupied_cells']}  "
        f"strict_support_bbox_area_fraction={_format_optional(loose_support['bbox_area_fraction'], 3)}"
    )

    baseline = _replay_minimal_sample_ransac(corrs, K, image_shape, pnp_kw, policy="baseline")
    diverse = _replay_minimal_sample_ransac(corrs, K, image_shape, pnp_kw, policy="spatial_cells")

    for result in [baseline, diverse]:
        best = result["best_record"]
        print(f"\n  policy={result['policy']}:")
        print(
            f"    successful_minimal_solves={result['n_model_success']}/{result['num_trials']}  "
            f"best_strict_inliers={0 if best is None else best['count']}"
        )
        _print_record_summary("    best_hypothesis: ", best)

        records = result["records"]
        near_global_floor = max(int(np.sum(strict_under_loose)) - 10, 0)
        n_near_global = int(sum(1 for r in records if int(r['count']) >= near_global_floor))
        print(
            f"    near_global_hypotheses(>={near_global_floor} inliers)={n_near_global}/{len(records)}"
        )

        if best is None:
            continue

        best_signature = best["support_signature"]
        best_count = int(best["count"])
        best_group = [
            r for r in records
            if int(r["count"]) == int(best_count) and r["support_signature"] == best_signature
        ]
        best_bbox = best["support_summary"].get("bbox", None)
        print(
            f"    winning_support_signature={best_signature}  "
            f"winning_support_bbox={best_bbox}"
        )
        _print_best_group_summary("    winning_sample_group: ", best_group, corrs, best_bbox)


def _point_mask_inside_bbox(xy, bbox):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or bbox is None:
        return np.zeros((xy.shape[0] if xy.ndim > 0 else 0,), dtype=bool)

    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    return (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)


def _print_frame4_pose_support_audit(seed, track4, pose4, pnp_kw, image_shape):
    landmarks = seed.get("landmarks", [])
    lm_by_id = {int(lm["id"]): lm for lm in landmarks if isinstance(lm, dict) and "id" in lm}
    landmark_id_by_feat1 = np.asarray(seed.get("landmark_id_by_feat1", []), dtype=np.int64).reshape(-1)
    kf_feat_idx = np.asarray(track4.get("kf_feat_idx", []), dtype=np.int64).reshape(-1)
    cur_feat_idx = np.asarray(track4.get("cur_feat_idx", []), dtype=np.int64).reshape(-1)
    xy_cur = np.asarray(track4.get("xy_cur", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)

    if xy_cur.ndim != 2 or xy_cur.shape[1] != 2:
        xy_cur = np.zeros((0, 2), dtype=np.float64)

    M = int(kf_feat_idx.size)
    if int(cur_feat_idx.size) != M or int(xy_cur.shape[0]) != M:
        print("\n=== Frame-4 pose support audit ===")
        print("  unavailable: tracking arrays are not aligned")
        return

    min_landmark_observations = int(pnp_kw.get("min_landmark_observations", 2))
    allow_bootstrap = bool(pnp_kw.get("allow_bootstrap_landmarks_for_pose", True))
    min_post_bootstrap = int(pnp_kw.get("min_post_bootstrap_observations_for_pose", 3))

    in_range = (kf_feat_idx >= 0) & (kf_feat_idx < int(landmark_id_by_feat1.size))
    mapped = np.zeros((M,), dtype=bool)
    mapped_lm_ids = np.full((M,), -1, dtype=np.int64)
    mapped[in_range] = landmark_id_by_feat1[kf_feat_idx[in_range]] >= 0
    mapped_lm_ids[in_range] = landmark_id_by_feat1[kf_feat_idx[in_range]]

    valid_landmark = np.zeros((M,), dtype=bool)
    pose_eligible = np.zeros((M,), dtype=bool)
    obs_gated = np.zeros((M,), dtype=bool)
    source_by_track = np.full((M,), "unmapped", dtype=object)

    for i in range(M):
        if not bool(mapped[i]):
            continue

        lm = lm_by_id.get(int(mapped_lm_ids[i]), None)
        if lm is None:
            continue

        X_w = np.asarray(lm.get("X_w", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue

        valid_landmark[i] = True
        source = lm.get("birth_source", "unknown")
        source_by_track[i] = str(source)
        n_obs = _obs_count(lm)
        if source == "bootstrap":
            eligible = bool(allow_bootstrap) and n_obs >= int(min_landmark_observations)
        else:
            eligible = n_obs >= max(int(min_landmark_observations), int(min_post_bootstrap))

        pose_eligible[i] = bool(eligible)
        obs_gated[i] = not bool(eligible)

    corrs = pose4.get("corrs", None)
    pnp_inlier_mask = np.asarray(pose4.get("pnp_inlier_mask", np.zeros((0,), dtype=bool)), dtype=bool).reshape(-1)
    inlier_track = np.zeros((M,), dtype=bool)
    if corrs is not None and hasattr(corrs, "kf_feat_idx") and hasattr(corrs, "cur_feat_idx"):
        corr_kf = np.asarray(corrs.kf_feat_idx, dtype=np.int64).reshape(-1)
        corr_cur = np.asarray(corrs.cur_feat_idx, dtype=np.int64).reshape(-1)
        if int(pnp_inlier_mask.size) == int(corr_kf.size):
            inlier_pairs = {
                (int(corr_kf[i]), int(corr_cur[i]))
                for i in np.flatnonzero(pnp_inlier_mask)
            }
            for i in range(M):
                inlier_track[i] = (int(kf_feat_idx[i]), int(cur_feat_idx[i])) in inlier_pairs

    stats = pose4.get("stats", {}) if isinstance(pose4, dict) else {}
    coverage = stats.get("pnp_spatial_coverage", None)
    bbox = coverage.get("bbox", None) if isinstance(coverage, dict) else None
    inside_bbox = _point_mask_inside_bbox(xy_cur, bbox)
    outside_bbox = ~inside_bbox if int(inside_bbox.size) == M else np.ones((M,), dtype=bool)

    def _count(mask):
        return int(np.sum(mask))

    def _line(label, region_mask):
        not_linked = region_mask & (~mapped)
        mapped_not_valid = region_mask & mapped & (~valid_landmark)
        gated = region_mask & valid_landmark & obs_gated
        eligible = region_mask & pose_eligible
        eligible_outlier = eligible & (~inlier_track)
        print(
            f"  {label}: raw={_count(region_mask)}  "
            f"not_linked={_count(not_linked)}  "
            f"mapped_invalid={_count(mapped_not_valid)}  "
            f"obs_or_source_gated={_count(gated)}  "
            f"pose_eligible={_count(eligible)}  "
            f"accepted_8px_inliers={_count(region_mask & inlier_track)}  "
            f"eligible_rejected_by_8px={_count(eligible_outlier)}"
        )

    print("\n=== Frame-4 pose support audit ===")
    print(
        "  pose: "
        f"ok={pose4.get('ok', False)}  "
        f"reason={stats.get('reason', None)}  "
        f"n_corr={stats.get('n_corr', 0)}  "
        f"n_inliers={stats.get('n_pnp_inliers', 0)}  "
        f"spatial_reason={stats.get('pnp_spatial_gate_reason', None)}"
    )
    print(f"  accepted bbox: {bbox}")
    if isinstance(coverage, dict):
        print(
            "  accepted spread: "
            f"cells={coverage.get('occupied_cells', 0)}  "
            f"bbox_area_fraction={coverage.get('bbox_area_fraction', None)}  "
            f"grid={coverage.get('occupancy_grid', None)}"
        )

    _line("all raw tracks", np.ones((M,), dtype=bool))
    _line("inside accepted bbox", inside_bbox)
    _line("outside accepted bbox", outside_bbox)
    print(
        "  gated birth sources: "
        f"{_birth_source_counts(landmarks, mapped_lm_ids[obs_gated & valid_landmark])}"
    )


def _project(K, R, t, X_w):
    # Project (3,N) world points to (2,N) pixel coords
    P = projection_matrix(K, R, t)
    X_h = np.vstack([X_w, np.ones((1, X_w.shape[1]))])
    x_h = P @ X_h
    w = x_h[2, :]
    ok = np.abs(w) > 1e-12
    x = np.full((2, X_w.shape[1]), np.nan)
    x[0, ok] = x_h[0, ok] / w[ok]
    x[1, ok] = x_h[1, ok] / w[ok]
    return x


def _signed_residuals(K, R, t, X_w, x_cur):
    # Return per-point (dx, dy, err_px) under a given pose
    x_pred = _project(K, R, t, X_w)
    dx = x_cur[0] - x_pred[0]
    dy = x_cur[1] - x_pred[1]
    err = np.sqrt(dx ** 2 + dy ** 2)
    return dx, dy, err


def _rstat(label, dx, dy, err, prefix="  "):
    fin = np.isfinite(err)
    n = int(np.sum(fin))
    if n == 0:
        print(f"{prefix}{label}: n=0")
        return
    e = err[fin]; x = dx[fin]; y = dy[fin]
    print(
        f"{prefix}{label}: n={n}  "
        f"err median={np.median(e):.2f}  p90={np.percentile(e, 90):.2f}  "
        f"dx_mean={np.mean(x):+.2f}  dy_mean={np.mean(y):+.2f}  "
        f"@5px={int(np.sum(e <= 5))}  @10px={int(np.sum(e <= 10))}  @20px={int(np.sum(e <= 20))}"
    )


def _residual_structure_audit(corrs, K, pnp_kw, im):
    # Residual-structure audit: accepted vs rejected breakdown, spatial bias, depth bias,
    # and looser-threshold sweep.  Designed to discriminate between systematic bias,
    # competing pose groups, and stale landmark geometry.
    N = int(corrs.X_w.shape[1])
    H, W = int(im.shape[0]), int(im.shape[1])
    X_w = corrs.X_w
    x_cur = corrs.x_cur

    print(f"\n=== Frame-4 residual structure audit ({N} eligible correspondences) ===")

    thr = float(pnp_kw.get("threshold_px", 8.0))
    R_best, t_best, best_mask, _ = estimate_pose_pnp_ransac(
        corrs, K,
        num_trials=int(pnp_kw.get("num_trials", 1000)),
        sample_size=int(pnp_kw.get("sample_size", 6)),
        threshold_px=thr,
        min_inliers=int(pnp_kw.get("min_inliers", 12)),
        seed=int(pnp_kw.get("ransac_seed", 0)),
        refit=bool(pnp_kw.get("refit", True)),
        refine_nonlinear=bool(pnp_kw.get("refine_nonlinear", True)),
    )
    if R_best is None:
        print("  RANSAC returned no valid pose – residual audit skipped")
        return

    best_mask = np.asarray(best_mask, dtype=bool).reshape(-1)
    n_accepted = int(np.sum(best_mask))
    n_rejected = int(np.sum(~best_mask))
    print(f"  best RANSAC pose at {thr:.0f}px: accepted={n_accepted}  rejected={n_rejected}")

    dx, dy, err = _signed_residuals(K, R_best, t_best, X_w, x_cur)

    print(f"\n  residuals under best {thr:.0f}px RANSAC pose:")
    _rstat(f"accepted ({n_accepted})", dx[best_mask], dy[best_mask], err[best_mask])
    _rstat(f"rejected ({n_rejected})", dx[~best_mask], dy[~best_mask], err[~best_mask])

    # Spatial breakdown of rejected residuals by image half
    xy_rej = x_cur[:, ~best_mask].T
    dx_rej = dx[~best_mask]
    dy_rej = dy[~best_mask]
    err_rej = err[~best_mask]
    fin = np.isfinite(err_rej)

    print(f"\n  rejected by image half (W={W}, H={H}):")
    for label, mask_fn in [
        ("left  x<W/2", xy_rej[:, 0] < W / 2),
        ("right x>=W/2", xy_rej[:, 0] >= W / 2),
        ("top   y<H/2", xy_rej[:, 1] < H / 2),
        ("btm   y>=H/2", xy_rej[:, 1] >= H / 2),
    ]:
        m = mask_fn & fin
        n_q = int(np.sum(m))
        if n_q == 0:
            print(f"    {label}: n=0")
            continue
        print(
            f"    {label}: n={n_q}  "
            f"err median={np.median(err_rej[m]):.2f}px  "
            f"dx_mean={np.mean(dx_rej[m]):+.2f}  dy_mean={np.mean(dy_rej[m]):+.2f}"
        )

    # Depth breakdown of rejected residuals
    depths_all = world_to_camera_points(R_best, t_best, X_w)[2, :]
    depths_rej = depths_all[~best_mask]
    valid_d = np.isfinite(depths_rej) & (depths_rej > 0)
    if int(np.sum(valid_d)) > 1:
        med_d = float(np.median(depths_rej[valid_d]))
        near = valid_d & (depths_rej < med_d)
        far = valid_d & (depths_rej >= med_d)
        print(f"\n  rejected by depth (median={med_d:.2f}m):")
        _rstat(f"near (<{med_d:.2f}m, n={int(np.sum(near))})", dx_rej[near], dy_rej[near], err_rej[near], prefix="    ")
        _rstat(f"far  (>={med_d:.2f}m, n={int(np.sum(far))})", dx_rej[far], dy_rej[far], err_rej[far], prefix="    ")

    # Looser-threshold sweep – critical test for systematic bias vs fragmented geometry
    print(f"\n  looser-threshold RANSAC sweep (same {N} correspondences, same RANSAC seed):")
    for loose_px in [8, 12, 20, 40]:
        R_l, t_l, mask_l, stats_l = estimate_pose_pnp_ransac(
            corrs, K,
            num_trials=int(pnp_kw.get("num_trials", 1000)),
            sample_size=int(pnp_kw.get("sample_size", 6)),
            threshold_px=float(loose_px),
            min_inliers=int(pnp_kw.get("min_inliers", 12)),
            seed=int(pnp_kw.get("ransac_seed", 0)),
            refit=bool(pnp_kw.get("refit", True)),
            refine_nonlinear=bool(pnp_kw.get("refine_nonlinear", True)),
        )
        n_l = int(stats_l.get("n_inliers", 0))
        ok_l = R_l is not None
        overlap = 0
        if ok_l and mask_l is not None:
            mask_l_arr = np.asarray(mask_l, dtype=bool).reshape(-1)
            overlap = int(np.sum(best_mask & mask_l_arr))
        print(
            f"    @{loose_px:2d}px: ok={ok_l}  n_inliers={n_l}  "
            f"overlap_with_{thr:.0f}px_set={overlap}"
        )
        if ok_l and loose_px == 40:
            # Under the 40px pose, report signed bias for correspondences not in the 8px set
            dx_l, dy_l, err_l = _signed_residuals(K, R_l, t_l, X_w, x_cur)
            outside_8px = ~best_mask
            _rstat(
                f"      40px-pose residuals for {thr:.0f}px-rejected ({int(np.sum(outside_8px))})",
                dx_l[outside_8px], dy_l[outside_8px], err_l[outside_8px],
                prefix="      "
            )


def main():
    profile_path = ROOT / "configs" / "profiles" / "eth3d_c2.yaml"
    cfg, K = _load_runtime_cfg(profile_path)
    kwargs = _frontend_kwargs_from_cfg(cfg)
    pnp_kw = kwargs["pnp_frontend_kwargs"]

    dataset_cfg = cfg["dataset"]
    dataset_root = (ROOT / dataset_cfg["root"]).resolve()
    seq_name = str(dataset_cfg["seq"])
    check_dir(dataset_root, name="dataset_root")

    seq = load_eth3d_sequence(
        dataset_root, seq_name, normalise_01=True, dtype=np.float64, require_timestamps=True
    )
    max_frames = dataset_cfg.get("max_frames", None)
    n_eff = len(seq) if max_frames is None else min(len(seq), int(max_frames))

    im0, _, _ = seq.get(0)
    im1, _, _ = seq.get(1)
    im2, _, _ = seq.get(2)
    im3, _, _ = seq.get(3)
    im4, _, _ = seq.get(4)

    boot = bootstrap_from_two_frames(
        K, K, im0, im1,
        feature_cfg=kwargs["feature_cfg"],
        F_cfg=kwargs["F_cfg"],
        H_cfg=kwargs["H_cfg"],
        bootstrap_cfg=kwargs["bootstrap_cfg"],
    )
    assert boot["ok"], "Bootstrap failed"

    seed = boot["seed"]
    feats_kf = seed["feats1"]
    kf_idx = 1

    # --- Frame 2 ---
    out2 = process_frame_against_seed(
        K, seed, feats_kf, im2,
        feature_cfg=kwargs["feature_cfg"],
        F_cfg=kwargs["F_cfg"],
        keyframe_kf=kf_idx,
        current_kf=2,
        **pnp_kw,
    )
    assert out2["ok"], "Frame 2 failed"
    seed = out2["seed"]
    if out2["stats"]["keyframe_promoted"]:
        feats_kf = out2["track_out"]["cur_feats"]
        kf_idx = 2
    print(f"frame 2: ok=True  kf_promoted={out2['stats']['keyframe_promoted']}  kf_idx_now={kf_idx}")
    _print_step_stats("Frame 2", out2)

    # --- Frame 3 ---
    seed_before_frame3 = _seed_snapshot(seed)
    out3 = process_frame_against_seed(
        K, seed, feats_kf, im3,
        feature_cfg=kwargs["feature_cfg"],
        F_cfg=kwargs["F_cfg"],
        keyframe_kf=kf_idx,
        current_kf=3,
        **pnp_kw,
    )
    assert out3["ok"], "Frame 3 failed"
    seed = out3["seed"]
    if out3["stats"]["keyframe_promoted"]:
        feats_kf = out3["track_out"]["cur_feats"]
        kf_idx = 3
    print(f"frame 3: ok=True  kf_promoted={out3['stats']['keyframe_promoted']}  kf_idx_now={kf_idx}")
    _print_step_stats("Frame 3", out3)

    print(f"\n=== Frame-3 state mutation ===")
    _print_seed_delta(seed_before_frame3, _seed_snapshot(seed), current_kf=3)

    R3 = out3["R"]
    t3 = out3["t"]
    print(f"  frame-3 pose: R3_det={np.linalg.det(R3):.6f}  |t3|={np.linalg.norm(t3):.4f}")

    # --- Inspect seed state after frame 3 promotion ---
    landmarks = seed.get("landmarks", [])
    lmid_by_feat1 = np.asarray(seed.get("landmark_id_by_feat1", []), dtype=np.int64)
    n_feat1 = int(lmid_by_feat1.size)
    n_mapped = int(np.sum(lmid_by_feat1 >= 0))
    n_lm = int(len(landmarks))
    n_feats_kf = int(feats_kf.kps_xy.shape[0]) if hasattr(feats_kf, "kps_xy") else -1

    print(f"\n=== Seed state after frame-3 promotion ===")
    print(f"  n_landmarks={n_lm}")
    print(f"  landmark_id_by_feat1 size={n_feat1}  mapped={n_mapped}")
    print(f"  feats_kf keypoints={n_feats_kf}")

    # Check consistency: do lmid_by_feat1 size and feats_kf kp count match?
    print(f"  size match (lmid_by_feat1 vs feats_kf.kps_xy): {n_feat1 == n_feats_kf}")

    # Count obs at kf=3 per landmark
    obs_at_kf3 = {}
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        lm_id = int(lm.get("id", -1))
        for ob in lm.get("obs", []):
            if isinstance(ob, dict) and int(ob.get("kf", -1)) == 3:
                obs_at_kf3[lm_id] = obs_at_kf3.get(lm_id, 0) + 1

    print(f"  landmarks with obs at kf=3: {len(obs_at_kf3)}")
    print(f"  mapped landmark birth sources: {_birth_source_counts(landmarks, lmid_by_feat1[lmid_by_feat1 >= 0])}")

    # Verify that mapped entries in lmid_by_feat1 correspond to landmarks with obs at kf=3
    n_consistent = 0
    n_inconsistent = 0
    for feat_i, lm_id in enumerate(lmid_by_feat1):
        if lm_id < 0:
            continue
        if lm_id in obs_at_kf3:
            n_consistent += 1
        else:
            n_inconsistent += 1
    print(f"  landmark_id_by_feat1 entries -> obs at kf=3: consistent={n_consistent}  inconsistent={n_inconsistent}")

    # --- Build frame-4 track and correspondences ---
    print(f"\n=== Frame-4 tracking against kf_idx={kf_idx} ===")
    track4 = track_against_keyframe(
        K, feats_kf, im4,
        feature_cfg=kwargs["feature_cfg"],
        F_cfg=kwargs["F_cfg"],
    )
    n_track_inliers = int(track4["stats"].get("n_inliers", 0))
    print(f"  n_track_inliers={n_track_inliers}")

    corrs4, corr_stats4 = build_pnp_correspondences_with_stats(
        seed, track4,
        min_landmark_observations=pnp_kw.get("min_landmark_observations", 2),
        allow_bootstrap_landmarks_for_pose=pnp_kw.get("allow_bootstrap_landmarks_for_pose", True),
        min_post_bootstrap_observations_for_pose=pnp_kw.get("min_post_bootstrap_observations_for_pose", 3),
    )
    N4 = int(corrs4.X_w.shape[1])
    print(f"  n_pnp_corr={N4}  (bootstrap={corr_stats4['n_corr_bootstrap_born']}  post-boot={corr_stats4['n_corr_post_bootstrap_born']})")

    if N4 == 0:
        print("  No correspondences - cannot proceed")
        return

    X_w4 = corrs4.X_w
    x_cur4 = corrs4.x_cur

    pose4 = estimate_pose_from_seed(
        K,
        seed,
        track4,
        image_shape=(int(im4.shape[0]), int(im4.shape[1])),
        **pnp_kw,
    )
    _print_frame4_pose_support_audit(seed, track4, pose4, pnp_kw, im4.shape)

    # --- Observation count distribution ---
    lm_by_id = {int(lm["id"]): lm for lm in landmarks if isinstance(lm, dict) and "id" in lm}
    obs_counts = []
    for lm_id in corrs4.landmark_ids:
        lm = lm_by_id.get(int(lm_id), {})
        obs_counts.append(_obs_count(lm))
    obs_counts = np.array(obs_counts)
    print(f"  obs count distribution: min={obs_counts.min()}  median={np.median(obs_counts):.0f}  max={obs_counts.max()}")

    # --- Reprojection under frame-3 pose ---
    print(f"\n=== Reprojection errors under frame-3 pose (T_WC3) ===")
    errs_R3 = _reproj_err_px(K, R3, t3, X_w4, x_cur4)
    _print_err_summary("frame-4 corrs @ frame-3 pose", errs_R3)

    # --- Check KF pose stored in seed ---
    R_kf, t_kf = seed_keyframe_pose(seed)
    errs_kf = _reproj_err_px(K, R_kf, t_kf, X_w4, x_cur4)
    print(f"  (seed T_WC1 is frame-3 pose: {np.allclose(R_kf, R3) and np.allclose(t_kf, t3)})")
    _print_err_summary("frame-4 corrs @ seed keyframe pose", errs_kf)

    # --- Full-data DLT on frame-4 correspondences ---
    print(f"\n=== Full-data DLT on {N4} frame-4 correspondences ===")
    R4_dlt, t4_dlt, dlt_stats4 = estimate_pose_pnp(corrs4, K, min_points=6)
    if R4_dlt is None:
        print(f"  DLT FAILED: reason={dlt_stats4.get('reason')}  rank={dlt_stats4.get('A_rank', 'N/A')}  cheirality={dlt_stats4.get('cheirality_ratio', 'N/A'):.3f}")
    else:
        rmse = float(dlt_stats4.get("reprojection_rmse_px", np.nan))
        cheirality = float(dlt_stats4.get("cheirality_ratio", np.nan))
        print(f"  DLT OK: RMSE={rmse:.2f}px  cheirality={cheirality:.3f}")
        errs_dlt = _reproj_err_px(K, R4_dlt, t4_dlt, X_w4, x_cur4)
        _print_err_summary("frame-4 corrs @ DLT pose", errs_dlt)

    # --- Check 3D point geometry ---
    print(f"\n=== 3D point cloud sanity check ===")
    depths_R3 = world_to_camera_points(R3, t3, X_w4)[2, :]
    n_front = int(np.sum(depths_R3 > 0))
    print(f"  points in front of frame-3 camera: {n_front}/{N4}")
    print(f"  X_w range: x=[{X_w4[0].min():.2f},{X_w4[0].max():.2f}]  y=[{X_w4[1].min():.2f},{X_w4[1].max():.2f}]  z=[{X_w4[2].min():.2f},{X_w4[2].max():.2f}]")
    print(f"  depth_R3 range: [{depths_R3.min():.2f},{depths_R3.max():.2f}]  median={np.median(depths_R3):.2f}")

    # --- Feature-index mapping audit ---
    print(f"\n=== Feature-index mapping audit ===")
    kf_feat_idx4 = np.asarray(track4.get("kf_feat_idx", []), dtype=np.int64)
    cur_feat_idx4 = np.asarray(track4.get("cur_feat_idx", []), dtype=np.int64)
    print(f"  track4 kf_feat_idx: size={kf_feat_idx4.size}  min={kf_feat_idx4.min() if kf_feat_idx4.size > 0 else 'N/A'}  max={kf_feat_idx4.max() if kf_feat_idx4.size > 0 else 'N/A'}")
    print(f"  landmark_id_by_feat1 size={n_feat1}")
    if kf_feat_idx4.size > 0:
        in_range = (kf_feat_idx4 >= 0) & (kf_feat_idx4 < n_feat1)
        mapped = np.zeros(kf_feat_idx4.size, dtype=bool)
        mapped[in_range] = lmid_by_feat1[kf_feat_idx4[in_range]] >= 0
        print(f"  kf_feat_idx in range: {int(in_range.sum())}/{kf_feat_idx4.size}")
        print(f"  kf_feat_idx mapped to landmark: {int(mapped.sum())}/{kf_feat_idx4.size}")
        _spatial_summary("raw frame-4 tracks", track4.get("xy_cur", np.zeros((0, 2))), im4.shape)
        _spatial_summary("mapped frame-4 tracks", track4.get("xy_cur", np.zeros((0, 2)))[mapped], im4.shape)
        _spatial_summary("unmapped frame-4 tracks", track4.get("xy_cur", np.zeros((0, 2)))[~mapped], im4.shape)

    # --- Check corrs landmark_ids point to real landmarks ---
    n_valid_lm = int(sum(1 for lm_id in corrs4.landmark_ids if int(lm_id) in lm_by_id))
    print(f"  corrs landmark_ids with valid entry: {n_valid_lm}/{N4}")
    _spatial_summary("frame-4 PnP correspondences", corrs4.x_cur.T, im4.shape)

    # --- Observation feat-index consistency check ---
    # For each correspondence, verify the obs at kf=3 has the right feat index
    kf_feat_in_corrs = corrs4.kf_feat_idx
    n_feat_consistent = 0
    n_feat_inconsistent = 0
    for i in range(N4):
        lm_id = int(corrs4.landmark_ids[i])
        kf_fi = int(kf_feat_in_corrs[i])
        lm = lm_by_id.get(lm_id, None)
        if lm is None:
            n_feat_inconsistent += 1
            continue
        obs_kf3 = [ob for ob in lm.get("obs", []) if isinstance(ob, dict) and int(ob.get("kf", -1)) == 3]
        if len(obs_kf3) == 0:
            n_feat_inconsistent += 1
            continue
        feats_in_obs = {int(ob.get("feat", -1)) for ob in obs_kf3}
        if int(kf_fi) in feats_in_obs:
            n_feat_consistent += 1
        else:
            n_feat_inconsistent += 1

    print(
        "  correspondence feature-index consistency at kf=3: "
        f"consistent={n_feat_consistent}  inconsistent={n_feat_inconsistent}"
    )

    # --- Birth source breakdown for frame-4 corrs ---
    birth_counts = {}
    for lm_id in corrs4.landmark_ids:
        lm = lm_by_id.get(int(lm_id), {})
        src = lm.get("birth_source", "unknown")
        birth_counts[src] = birth_counts.get(src, 0) + 1
    print(f"\n  birth source breakdown: {birth_counts}")

    # --- Residual structure audit ---
    _residual_structure_audit(corrs4, K, pnp_kw, im4)
    _print_minimal_sample_failure_audit("Frame 4", corrs4, K, pnp_kw, im4.shape)

    # --- Frame-3 control audit ---
    pose3 = out3.get("pose_out", {}) if isinstance(out3, dict) else {}
    corrs3 = pose3.get("corrs", None) if isinstance(pose3, dict) else None
    if isinstance(corrs3, PnPCorrespondences) and int(corrs3.X_w.shape[1]) > 0:
        _print_minimal_sample_failure_audit("Frame 3", corrs3, K, pnp_kw, im3.shape)


if __name__ == "__main__":
    main()
