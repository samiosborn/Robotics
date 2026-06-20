# scripts/diag_rescue_candidate_quality.py
# Compares support-conditioned rescue-time candidate quality signals across
# bad canonical-pose fallback frames and useful late fallback frames.
#
# Candidate signals (all available at rescue time from the accepted candidate
# and its support set):
#   1. Residual median within accepted inlier set
#   2. Residual p90 within accepted inlier set
#   3. Fraction of accepted inliers at or below 12 px (retention at tighter threshold)
#   4. Fraction of accepted inliers above 16 px (heavy tail near acceptance ceiling)
#   5. Bootstrap minimal-sample pose rotation dispersion (6-point subsets)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from frontend_eth3d_common import ROOT, frontend_kwargs_from_cfg, load_runtime_cfg
from datasets.loader import load_sequence
from geometry.camera import reprojection_errors_sq
from geometry.pnp import PnPCorrespondences, _build_pnp_dlt_matrix, estimate_pose_pnp_ransac
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_active_keyframe_kf


_TARGET_FRAMES: dict[str, dict[int, str]] = {
    "eth3d": {
        12: "bad canonical pose",
        16: "bad canonical pose",
        17: "load-bearing good refresh",
        18: "neutral",
    },
    "kitti": {
        17: "load-bearing good refresh",
        18: "neutral",
        20: "neutral",
    },
}


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(w) for k, w in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(w) for w in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    return v


def _inlier_arrays(
    pose_out: dict,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    if not isinstance(pose_out, dict) or pose_out.get("corrs") is None:
        return None, None, np.zeros((3, 0)), np.zeros((2, 0))
    corrs = pose_out["corrs"]
    n = int(corrs.X_w.shape[1])
    raw = pose_out.get("pnp_inlier_mask", None)
    mask = np.asarray(raw if raw is not None else np.zeros(n, dtype=bool), dtype=bool).reshape(-1)
    if int(mask.size) != n:
        mask = np.zeros(n, dtype=bool)
    R = np.asarray(pose_out["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(pose_out["t"], dtype=np.float64).reshape(3)
    X_w = np.asarray(corrs.X_w[:, mask], dtype=np.float64)
    x_cur = np.asarray(corrs.x_cur[:, mask], dtype=np.float64)
    return R, t, X_w, x_cur


def _residuals_px(K, R, t, X_w, x_cur, *, eps: float = 1e-12) -> np.ndarray:
    if int(X_w.shape[1]) == 0:
        return np.zeros(0, dtype=np.float64)
    depth = (np.asarray(R) @ np.asarray(X_w) + np.asarray(t).reshape(3, 1))[2, :]
    err_sq = np.asarray(reprojection_errors_sq(K, R, t, X_w, x_cur), dtype=np.float64).reshape(-1)
    valid = np.isfinite(depth) & (np.asarray(depth) > eps) & np.isfinite(err_sq) & (err_sq >= 0.0)
    out = np.full(int(err_sq.size), np.nan, dtype=np.float64)
    out[valid] = np.sqrt(err_sq[valid])
    return out


def _residual_shape(K, R, t, X_w, x_cur) -> dict[str, Any]:
    res = _residuals_px(K, R, t, X_w, x_cur)
    finite = res[np.isfinite(res)]
    n = int(finite.size)
    if n == 0:
        return {"n_inliers": 0}
    result: dict[str, Any] = {
        "n_inliers": n,
        "median_px": float(np.median(finite)),
        "p90_px": float(np.percentile(finite, 90.0)),
        "max_px": float(np.max(finite)),
    }
    for thr in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0):
        key = int(thr)
        result[f"n_above_{key}px"] = int(np.sum(finite > thr))
        result[f"frac_above_{key}px"] = float(np.mean(finite > thr))
        result[f"n_at_most_{key}px"] = int(np.sum(finite <= thr))
        result[f"frac_at_most_{key}px"] = float(np.mean(finite <= thr))
    return result


def _dlt_condition(K, X_w: np.ndarray, x_cur: np.ndarray) -> float | None:
    n = int(X_w.shape[1])
    if n < 6:
        return None
    try:
        K_arr = np.asarray(K, dtype=np.float64)
        K_inv = np.linalg.inv(K_arr)
        ones = np.ones((1, n), dtype=np.float64)
        x_hat = K_inv @ np.vstack([np.asarray(x_cur, dtype=np.float64), ones])
        A = _build_pnp_dlt_matrix(np.asarray(X_w, dtype=np.float64), x_hat)
        _, s, _ = np.linalg.svd(A, full_matrices=False)
        s_abs = np.abs(s)
        s_max = float(s_abs[0])
        s_min = float(s_abs[-1])
        if s_max < 1e-12:
            return None
        return float(s_min / s_max)
    except Exception:
        return None


def _bootstrap_dispersion(K, X_w: np.ndarray, x_cur: np.ndarray, *, n_samples: int = 50, rng_seed: int = 42) -> dict[str, Any]:
    n = int(X_w.shape[1])
    if n < 6:
        return {"attempted": False, "reason": "too_few_inliers", "n_inliers": n}
    rng = np.random.default_rng(int(rng_seed))
    R_list: list[np.ndarray] = []
    n_failed = 0
    for _ in range(int(n_samples)):
        idx = rng.choice(n, size=6, replace=False)
        X_s = np.asarray(X_w[:, idx], dtype=np.float64)
        x_s = np.asarray(x_cur[:, idx], dtype=np.float64)
        dummy = np.arange(6, dtype=np.int64)
        corrs_s = PnPCorrespondences(
            X_w=X_s,
            x_cur=x_s,
            landmark_ids=dummy.copy(),
            cur_feat_idx=dummy.copy(),
            kf_feat_idx=dummy.copy(),
        )
        try:
            result = estimate_pose_pnp_ransac(
                corrs_s,
                np.asarray(K, dtype=np.float64),
                num_trials=1,
                sample_size=6,
                threshold_px=20.0,
                min_inliers=4,
                seed=0,
            )
            if bool(result.get("ok", False)):
                R_s = np.asarray(result["R"], dtype=np.float64).reshape(3, 3)
                R_list.append(R_s)
            else:
                n_failed += 1
        except Exception:
            n_failed += 1
    n_valid = int(len(R_list))
    if n_valid < 4:
        return {
            "attempted": True,
            "n_valid_samples": n_valid,
            "n_failed_samples": int(n_failed),
            "pairwise_median_deg": None,
        }
    degs: list[float] = []
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            degs.append(float(np.degrees(angle_between_rotmats(R_list[i], R_list[j]))))
    return {
        "attempted": True,
        "n_valid_samples": int(n_valid),
        "n_failed_samples": int(n_failed),
        "pairwise_median_deg": float(np.median(degs)),
        "pairwise_p90_deg": float(np.percentile(degs, 90.0)),
        "pairwise_max_deg": float(np.max(degs)),
    }


def _rescue_stage_label(pose_stats: dict) -> str:
    if not bool(pose_stats.get("pnp_support_rescue_succeeded", False)):
        return "not_rescue"
    thr = pose_stats.get("pnp_support_rescue_loose_threshold_px", None)
    if bool(pose_stats.get("pnp_support_rescue_loose_localisation_fallback_succeeded", False)):
        return f"loose_{float(thr):.0f}px" if thr is not None else "loose_localisation_only"
    if bool(pose_stats.get("pnp_support_rescue_second_stage_succeeded", False)):
        return "stage2_strict_refit"
    return str(pose_stats.get("pnp_support_rescue_reason", "rescued"))


def _analyse_frame(K, frame_index: int, out: dict, active_before: int, label: str) -> dict[str, Any]:
    stats = out.get("stats") or {}
    pose_out = out.get("pose_out") or {}
    pose_stats = pose_out.get("stats") or {}
    is_rescue = bool(stats.get("localisation_only_rescue_frame", False))
    stage = _rescue_stage_label(pose_stats)
    loose_thr = pose_stats.get("pnp_support_rescue_loose_threshold_px", None)
    loose_n = int(pose_stats.get("pnp_support_rescue_loose_inliers", 0))
    n_corr = int(stats.get("n_pnp_corr", 0))
    refresh = bool(stats.get("guarded_support_refresh_triggered", False))
    R, t, X_w, x_cur = _inlier_arrays(pose_out)
    n_inliers = int(X_w.shape[1]) if X_w is not None else 0
    residual_shape: dict[str, Any] = {}
    dlt_cond: float | None = None
    bootstrap: dict[str, Any] = {}
    if R is not None and n_inliers > 0:
        residual_shape = _residual_shape(K, R, t, X_w, x_cur)
        dlt_cond = _dlt_condition(K, X_w, x_cur)
        bootstrap = _bootstrap_dispersion(K, X_w, x_cur)
    return {
        "frame_index": int(frame_index),
        "label": str(label),
        "pipeline_ok": True,
        "is_rescue": bool(is_rescue),
        "rescue_stage": str(stage),
        "active_basis_before": int(active_before),
        "loose_threshold_px": loose_thr,
        "loose_inlier_count": int(loose_n),
        "final_pnp_inliers": int(n_inliers),
        "final_pnp_correspondences": int(n_corr),
        "refresh_triggered": bool(refresh),
        "residual_shape": residual_shape,
        "dlt_condition_sigma_min_over_max": float(dlt_cond) if dlt_cond is not None else None,
        "bootstrap_stability": bootstrap,
    }


def _replay(profile_path: Path, dataset_key: str, stop_frame: int) -> list[dict[str, Any]]:
    cfg, K = load_runtime_cfg(profile_path)
    kw = frontend_kwargs_from_cfg(cfg)
    pnp_cfg = kw["pnp_frontend_kwargs"]
    ds_cfg = cfg["dataset"]
    run_cfg = cfg.get("run", {})
    boot_cfg = run_cfg.get("bootstrap", {})
    i0 = int(boot_cfg.get("i0", 0))
    i1 = int(boot_cfg.get("i1", 1))
    seq = load_sequence(
        str(ds_cfg["name"]),
        (ROOT / str(ds_cfg["root"])).resolve(),
        str(ds_cfg["seq"]),
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )
    img0, _, _ = seq.get(i0)
    img1, _, _ = seq.get(i1)
    boot = bootstrap_from_two_frames(
        K,
        K,
        img0,
        img1,
        feature_cfg=kw["feature_cfg"],
        F_cfg=kw["F_cfg"],
        H_cfg=kw["H_cfg"],
        bootstrap_cfg=kw["bootstrap_cfg"],
    )
    if not bool(boot.get("ok", False)):
        raise RuntimeError(f"Bootstrap failed for {dataset_key}")
    seed = boot["seed"]
    target_frames = _TARGET_FRAMES[str(dataset_key)]
    results: list[dict[str, Any]] = []
    for frame_index in range(i1 + 1, min(int(stop_frame) + 1, len(seq))):
        active_before = int(get_active_keyframe_kf(seed))
        img, _, _ = seq.get(int(frame_index))
        out = process_frame_against_seed(
            K,
            seed,
            img,
            feature_cfg=kw["feature_cfg"],
            F_cfg=kw["F_cfg"],
            current_kf=int(frame_index),
            **pnp_cfg,
        )
        seed = out["seed"]
        if int(frame_index) not in target_frames:
            continue
        label = str(target_frames[int(frame_index)])
        if not bool(out.get("ok", False)):
            results.append({
                "frame_index": int(frame_index),
                "label": label,
                "pipeline_ok": False,
            })
            continue
        row = _analyse_frame(K, int(frame_index), out, int(active_before), label)
        results.append(row)
    return results


def _build_comparison_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        rs = row.get("residual_shape") or {}
        bs = row.get("bootstrap_stability") or {}
        table.append({
            "label": row.get("label"),
            "frame_index": row.get("frame_index"),
            "rescue_stage": row.get("rescue_stage"),
            "n_inliers": rs.get("n_inliers"),
            "median_px": rs.get("median_px"),
            "p90_px": rs.get("p90_px"),
            "max_px": rs.get("max_px"),
            "frac_at_most_12px": rs.get("frac_at_most_12px"),
            "frac_above_14px": rs.get("frac_above_14px"),
            "frac_above_16px": rs.get("frac_above_16px"),
            "dlt_condition": row.get("dlt_condition_sigma_min_over_max"),
            "bootstrap_pairwise_median_deg": bs.get("pairwise_median_deg"),
            "bootstrap_pairwise_p90_deg": bs.get("pairwise_p90_deg"),
        })
    return table


def _classify_signals(table: list[dict[str, Any]]) -> dict[str, Any]:
    bad_rows = [r for r in table if r.get("label") == "bad canonical pose"]
    good_rows = [r for r in table if r.get("label") == "load-bearing good refresh"]
    neutral_rows = [r for r in table if r.get("label") == "neutral"]

    def _separates(signal_key: str, bad_gt_good: bool) -> bool:
        bad_vals = [r.get(signal_key) for r in bad_rows if r.get(signal_key) is not None]
        good_vals = [r.get(signal_key) for r in good_rows if r.get(signal_key) is not None]
        if not bad_vals or not good_vals:
            return False
        if bad_gt_good:
            return float(min(bad_vals)) > float(max(good_vals))
        return float(max(bad_vals)) < float(min(good_vals))

    residual_median_sep = _separates("median_px", bad_gt_good=True)
    residual_p90_sep = _separates("p90_px", bad_gt_good=True)
    retention_12px_sep = _separates("frac_at_most_12px", bad_gt_good=False)
    heavy_tail_14px_sep = _separates("frac_above_14px", bad_gt_good=True)
    heavy_tail_16px_sep = _separates("frac_above_16px", bad_gt_good=True)
    bootstrap_sep = _separates("bootstrap_pairwise_median_deg", bad_gt_good=True)

    residual_shape_separates = bool(residual_median_sep or residual_p90_sep or retention_12px_sep or heavy_tail_14px_sep or heavy_tail_16px_sep)
    bootstrap_separates = bool(bootstrap_sep)

    return {
        "bad_count": int(len(bad_rows)),
        "good_count": int(len(good_rows)),
        "neutral_count": int(len(neutral_rows)),
        "residual_median_separates": bool(residual_median_sep),
        "residual_p90_separates": bool(residual_p90_sep),
        "retention_at_12px_separates": bool(retention_12px_sep),
        "heavy_tail_above_14px_separates": bool(heavy_tail_14px_sep),
        "heavy_tail_above_16px_separates": bool(heavy_tail_16px_sep),
        "residual_shape_family_separates": bool(residual_shape_separates),
        "bootstrap_stability_separates": bool(bootstrap_sep),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth3d_profile", default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"))
    parser.add_argument("--kitti_profile", default=str(ROOT / "configs" / "profiles" / "kitti_odometry_00.yaml"))
    parser.add_argument("--eth3d_stop_frame", type=int, default=20)
    parser.add_argument("--kitti_stop_frame", type=int, default=21)
    parser.add_argument("--out", default="/tmp/rescue_candidate_quality.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    eth3d_rows = _replay(
        Path(args.eth3d_profile).expanduser().resolve(),
        "eth3d",
        int(args.eth3d_stop_frame),
    )
    kitti_rows = _replay(
        Path(args.kitti_profile).expanduser().resolve(),
        "kitti",
        int(args.kitti_stop_frame),
    )
    all_rows = eth3d_rows + kitti_rows
    table = _build_comparison_table(all_rows)
    classification = _classify_signals(table)

    result = {
        "event": "rescue_candidate_quality",
        "rows": all_rows,
        "comparison_table": table,
        "classification": classification,
    }
    text = json.dumps(_jsonable(result), indent=2)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    if bool(args.quiet):
        print(f"wrote {out_path}")
        print(json.dumps(_jsonable({"comparison_table": table, "classification": classification}), indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
