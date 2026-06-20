# scripts/diag_40px_resolve_stability.py
# Tests stability and safety of a 40px RANSAC re-solve on full rescue correspondences.
# Runs 8 RANSAC seeds per target frame; reports solution cluster tightness, local
# residual quality, and full-history replacement impact for bad frames.
#
# ETH3D targets: frame 12 (bad canonical pose), frame 16 (bad canonical pose),
#                frame 17 (load-bearing good refresh).
# KITTI targets: frame 17 (load-bearing good refresh), frame 18 (neutral),
#                frame 20 (neutral).

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from frontend_eth3d_common import ROOT
from frontend_eth3d_common import frontend_kwargs_from_cfg as _fkw
from frontend_eth3d_common import load_runtime_cfg as _load_cfg
from datasets.loader import load_sequence
from geometry.camera import camera_centre, reprojection_errors_sq
from geometry.pose import angle_between_translations
from geometry.pnp import _pnp_inlier_mask_from_pose, estimate_pose_pnp_ransac
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_pose_for_kf

_THRESH_40PX = 40.0
_TRIALS_40PX = 5000
_SEEDS = [0, 1, 2, 3, 7, 42, 123, 999]

_ETH3D_STOP = 19
_KITTI_STOP = 21
_ETH3D_LIVE_FRAME = 19

_ETH3D_TARGETS = {
    12: "bad_canonical_pose",
    16: "bad_canonical_pose",
    17: "good_refresh",
}
_KITTI_TARGETS = {
    17: "good_refresh",
    18: "neutral",
    20: "neutral",
}
_ETH3D_BAD_FRAMES = frozenset([12, 16])


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(val) for val in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        fv = float(v)
        return fv if np.isfinite(fv) else None
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    return v


def _residuals(K: np.ndarray, pose: tuple, X_w: np.ndarray, xy: np.ndarray, *, eps: float) -> np.ndarray:
    R, t = pose
    if int(X_w.shape[1]) == 0:
        return np.zeros((0,), dtype=np.float64)
    depth = (R @ X_w + t.reshape(3, 1))[2, :]
    err_sq = np.asarray(reprojection_errors_sq(K, R, t, X_w, xy), dtype=np.float64).reshape(-1)
    valid = np.isfinite(depth) & (depth > float(eps)) & np.isfinite(err_sq) & (err_sq >= 0.0)
    out = np.full(int(err_sq.size), np.nan, dtype=np.float64)
    out[valid] = np.sqrt(err_sq[valid])
    return out


def _summary(errors: np.ndarray) -> dict:
    arr = np.asarray(errors, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"count": 0, "median_px": None, "p90_px": None, "max_px": None,
                "squared_error": 0.0, "above_8_count": 0}
    above8 = int(np.sum(arr > 8.0))
    return {
        "count": n,
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90.0)),
        "max_px": float(np.max(arr)),
        "squared_error": float(np.sum(arr * arr)),
        "above_8_count": above8,
    }


def _inlier_counts(corrs, pose: tuple | None, K: np.ndarray, eps: float) -> dict:
    if pose is None:
        return {"at_8px": 0, "at_12px": 0, "at_20px": 0, "at_40px": 0}
    R, t = pose
    counts = {}
    for px in (8.0, 12.0, 20.0, 40.0):
        mask, _ = _pnp_inlier_mask_from_pose(
            corrs.X_w, corrs.x_cur, K, R, t, threshold_px=px, eps=eps,
        )
        counts[f"at_{int(px)}px"] = int(np.sum(np.asarray(mask, dtype=bool)))
    return counts


def _pose_delta(a: tuple, b: tuple) -> dict:
    R_a, t_a = a
    R_b, t_b = b
    return {
        "rotation_delta_deg": float(np.degrees(angle_between_rotmats(R_a, R_b))),
        "camera_centre_distance": float(
            np.linalg.norm(camera_centre(R_b, t_b) - camera_centre(R_a, t_a))
        ),
    }


def _sq_reduction(base: float, cf: float) -> float | None:
    if base <= 0.0:
        return None
    return float((base - cf) / base)


def _run_40px_seed(corrs, K: np.ndarray, seed: int, cfg: dict) -> tuple | None:
    n = int(corrs.X_w.shape[1])
    sample_size = int(cfg.get("sample_size", 6))
    if n < sample_size:
        return None
    try:
        R, t, _, _ = estimate_pose_pnp_ransac(
            corrs, K,
            num_trials=_TRIALS_40PX,
            sample_size=sample_size,
            threshold_px=_THRESH_40PX,
            min_inliers=int(cfg.get("min_inliers", 8)),
            seed=int(seed),
            min_points=int(cfg["min_points"]),
            rank_tol=float(cfg["rank_tol"]),
            min_cheirality_ratio=float(cfg["min_cheirality_ratio"]),
            eps=float(cfg["eps"]),
            refit=bool(cfg["refit"]),
            refine_nonlinear=bool(cfg["refine_nonlinear"]),
            refine_max_iters=int(cfg["refine_max_iters"]),
            refine_damping=float(cfg["refine_damping"]),
            refine_step_tol=float(cfg["refine_step_tol"]),
            refine_improvement_tol=float(cfg["refine_improvement_tol"]),
        )
    except Exception:
        return None
    if R is None or t is None:
        return None
    return (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3))


def _stability_test(corrs, K: np.ndarray, cfg: dict, *, eps: float) -> tuple[dict, tuple | None]:
    per_seed = []
    successful: list[tuple[int, tuple, dict]] = []

    for seed_val in _SEEDS:
        pose = _run_40px_seed(corrs, K, seed_val, cfg)
        ok = pose is not None
        if ok:
            errs = _residuals(K, pose, corrs.X_w, corrs.x_cur, eps=eps)
            s = _summary(errs)
            ic = _inlier_counts(corrs, pose, K, eps)
            per_seed.append({"seed": int(seed_val), "ok": True, "summary": s, "inlier_counts": ic})
            successful.append((seed_val, pose, s))
        else:
            per_seed.append({"seed": int(seed_val), "ok": False})

    n_ok = len(successful)
    cluster: dict = {"n_successful": n_ok, "success_rate": float(n_ok / len(_SEEDS))}

    if n_ok >= 2:
        medians = [float(s["median_px"]) for _, _, s in successful if s["median_px"] is not None]
        p90s = [float(s["p90_px"]) for _, _, s in successful if s["p90_px"] is not None]
        above8s = [int(s["above_8_count"]) for _, _, s in successful]
        rot_deltas, cc_dists = [], []
        for i in range(n_ok):
            for j in range(i + 1, n_ok):
                d = _pose_delta(successful[i][1], successful[j][1])
                rot_deltas.append(float(d["rotation_delta_deg"]))
                cc_dists.append(float(d["camera_centre_distance"]))
        cluster.update({
            "median_px_min": float(min(medians)) if medians else None,
            "median_px_max": float(max(medians)) if medians else None,
            "median_px_mean": float(np.mean(medians)) if medians else None,
            "median_px_std": float(np.std(medians)) if len(medians) > 1 else 0.0,
            "p90_px_min": float(min(p90s)) if p90s else None,
            "p90_px_max": float(max(p90s)) if p90s else None,
            "above_8_min": int(min(above8s)) if above8s else 0,
            "above_8_max": int(max(above8s)) if above8s else 0,
            "max_pairwise_rotation_delta_deg": float(max(rot_deltas)) if rot_deltas else 0.0,
            "max_pairwise_cc_distance": float(max(cc_dists)) if cc_dists else 0.0,
            "mean_pairwise_rotation_delta_deg": float(np.mean(rot_deltas)) if rot_deltas else 0.0,
            "tight": bool(
                float(max(rot_deltas)) < 2.0 and float(max(cc_dists)) < 0.5
            ) if rot_deltas else None,
        })
    elif n_ok == 1:
        _, _, s_one = successful[0]
        cluster.update({
            "median_px_min": s_one["median_px"],
            "median_px_max": s_one["median_px"],
            "max_pairwise_rotation_delta_deg": 0.0,
            "max_pairwise_cc_distance": 0.0,
            "tight": None,
        })

    best_pose: tuple | None = None
    best_seed: int | None = None
    best_summary: dict | None = None
    if successful:
        best_idx = int(np.argmin([
            float(s["median_px"]) if s["median_px"] is not None else np.inf
            for _, _, s in successful
        ]))
        best_seed, best_pose, best_summary = successful[best_idx]

    return {
        "per_seed": per_seed,
        "cluster": cluster,
        "best_seed": best_seed,
        "best_summary_on_corrs": best_summary,
    }, best_pose


def _frame_bundle(
    seed: dict,
    live_ids: list[int],
    kf: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    lm_by_id = {
        int(lm["id"]): lm
        for lm in seed.get("landmarks", [])
        if isinstance(lm, dict) and "id" in lm
    }
    X_cols: list[np.ndarray] = []
    xy_cols: list[np.ndarray] = []
    used: list[int] = []
    for lm_id in sorted(set(int(v) for v in live_ids)):
        lm = lm_by_id.get(int(lm_id))
        if not isinstance(lm, dict):
            continue
        X_w = np.asarray(lm.get("X_w"), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue
        for obs in lm.get("obs", []):
            if not isinstance(obs, dict) or int(obs.get("kf", -1)) != kf:
                continue
            xy = np.asarray(obs.get("xy"), dtype=np.float64).reshape(-1)
            if xy.size == 2 and np.isfinite(xy).all():
                X_cols.append(X_w.reshape(3, 1))
                xy_cols.append(xy.reshape(2, 1))
                used.append(int(lm_id))
                break
    if not X_cols:
        return np.zeros((3, 0), dtype=np.float64), np.zeros((2, 0), dtype=np.float64), []
    return np.hstack(X_cols), np.hstack(xy_cols), used


def _full_history(
    seed: dict,
    live_ids: list[int],
    K: np.ndarray,
    *,
    eps: float,
) -> list[dict]:
    lm_by_id = {
        int(lm["id"]): lm
        for lm in seed.get("landmarks", [])
        if isinstance(lm, dict) and "id" in lm
    }
    rows: list[dict] = []
    for lm_id in sorted(set(int(v) for v in live_ids)):
        lm = lm_by_id.get(int(lm_id))
        if not isinstance(lm, dict):
            continue
        X_w = np.asarray(lm.get("X_w"), dtype=np.float64).reshape(-1)
        if X_w.size != 3 or not np.isfinite(X_w).all():
            continue
        for obs in lm.get("obs", []):
            if not isinstance(obs, dict):
                continue
            obs_kf = int(obs.get("kf", -1))
            if obs_kf < 0 or obs_kf not in seed.get("poses", {}):
                continue
            xy = np.asarray(obs.get("xy"), dtype=np.float64).reshape(-1)
            if xy.size != 2 or not np.isfinite(xy).all():
                continue
            try:
                pose = get_pose_for_kf(seed, obs_kf, context="stability_diag_history")
            except Exception:
                continue
            err = _residuals(K, pose, X_w.reshape(3, 1), xy.reshape(2, 1), eps=eps)
            if int(err.size) == 1 and np.isfinite(err[0]):
                rows.append({"landmark_id": int(lm_id), "kf": obs_kf, "error_px": float(err[0])})
    return rows


def _replace_kf_residuals(rows: list[dict], kf: int, by_id: dict[int, float]) -> np.ndarray:
    out = []
    for row in rows:
        if int(row["kf"]) == kf and int(row["landmark_id"]) in by_id:
            out.append(by_id[int(row["landmark_id"])])
        else:
            out.append(float(row["error_px"]))
    return np.asarray(out, dtype=np.float64)


def _replay_and_capture(
    profile_path: Path,
    targets: dict[int, str],
    stop_frame: int,
    *,
    live_frame: int | None = None,
) -> tuple[np.ndarray, dict, dict, dict[int, dict], list[int], int | None, list[int]]:
    cfg, K = _load_cfg(profile_path)
    fkw = _fkw(cfg)
    pnp_cfg = fkw["pnp_frontend_kwargs"]
    dataset_cfg = cfg["dataset"]
    run_cfg = cfg.get("run", {})
    dataset_root = (ROOT / str(dataset_cfg["root"])).resolve()

    seq = load_sequence(
        str(dataset_cfg["name"]),
        dataset_root,
        str(dataset_cfg["seq"]),
        normalise_01=True,
        dtype=np.float64,
        require_timestamps=True,
    )

    boot_cfg = run_cfg.get("bootstrap", {})
    i0 = int(boot_cfg.get("i0", 0))
    i1 = int(boot_cfg.get("i1", 1))

    im0, _, _ = seq.get(i0)
    im1, _, _ = seq.get(i1)
    boot = bootstrap_from_two_frames(
        K, K, im0, im1,
        feature_cfg=fkw["feature_cfg"],
        F_cfg=fkw["F_cfg"],
        H_cfg=fkw["H_cfg"],
        bootstrap_cfg=fkw["bootstrap_cfg"],
    )
    if not bool(boot.get("ok", False)):
        raise RuntimeError("Bootstrap failed")

    seed = boot["seed"]
    captured: dict[int, dict] = {}
    live_ids: list[int] = []
    first_failure: int | None = None
    accepted: list[int] = []

    for fi in range(i1 + 1, stop_frame + 1):
        im, _, _ = seq.get(fi)
        out = process_frame_against_seed(
            K, seed, im,
            feature_cfg=fkw["feature_cfg"],
            F_cfg=fkw["F_cfg"],
            current_kf=fi,
            **pnp_cfg,
        )
        ok = bool(out.get("ok", False))
        if ok:
            accepted.append(fi)
        elif first_failure is None:
            first_failure = fi

        po = out.get("pose_out") or {}

        if fi in targets:
            corrs = po.get("corrs")
            raw_mask = po.get("pnp_inlier_mask")
            R_acc = po.get("R")
            t_acc = po.get("t")
            captured[fi] = {
                "frame_index": fi,
                "label": targets[fi],
                "pipeline_ok": ok,
                "corrs": corrs,
                "loose_mask": (
                    None if raw_mask is None
                    else np.asarray(raw_mask, dtype=bool).reshape(-1)
                ),
                "accepted_pose": (
                    (
                        np.asarray(R_acc, dtype=np.float64),
                        np.asarray(t_acc, dtype=np.float64).reshape(3),
                    )
                    if R_acc is not None and t_acc is not None
                    else None
                ),
            }

        if live_frame is not None and fi == live_frame:
            corrs_live = po.get("corrs")
            if corrs_live is not None:
                live_ids = [
                    int(v)
                    for v in np.asarray(corrs_live.landmark_ids, dtype=np.int64).reshape(-1)
                ]

        seed = out["seed"]

    return K, pnp_cfg, seed, captured, sorted(set(live_ids)), first_failure, accepted


def _analyse_frame(
    fi: int,
    cap: dict,
    K: np.ndarray,
    pnp_cfg: dict,
    seed: dict,
    live_ids: list[int],
    *,
    bad_frames: frozenset[int],
    dataset_key: str,
) -> dict:
    eps = float(pnp_cfg["eps"])
    corrs = cap.get("corrs")
    accepted_pose = cap.get("accepted_pose")
    label = cap.get("label", "unknown")
    ok = bool(cap.get("pipeline_ok", False))

    result: dict = {
        "dataset": dataset_key,
        "frame_index": fi,
        "label": label,
        "pipeline_ok": ok,
        "n_corrs": int(corrs.X_w.shape[1]) if corrs is not None else 0,
        "n_loose_inliers": (
            int(np.sum(cap["loose_mask"])) if cap.get("loose_mask") is not None else 0
        ),
    }

    if corrs is None or not ok or accepted_pose is None:
        result["error"] = "no_correspondences_or_failed_frame"
        return result

    n_corrs = int(corrs.X_w.shape[1])

    # Accepted pose on full correspondences
    err_acc = _residuals(K, accepted_pose, corrs.X_w, corrs.x_cur, eps=eps)
    acc_summary = _summary(err_acc)
    acc_ic = _inlier_counts(corrs, accepted_pose, K, eps)
    result["accepted_pose_on_corrs"] = {"summary": acc_summary, "inlier_counts": acc_ic}

    # 40px stability test
    stab, best_pose = _stability_test(corrs, K, pnp_cfg, eps=eps)
    result["stability_40px"] = stab

    if best_pose is not None:
        err_best = _residuals(K, best_pose, corrs.X_w, corrs.x_cur, eps=eps)
        best_summary = _summary(err_best)
        best_ic = _inlier_counts(corrs, best_pose, K, eps)
        result["best_40px_on_corrs"] = {"summary": best_summary, "inlier_counts": best_ic}
        result["delta_40px_vs_accepted"] = _pose_delta(accepted_pose, best_pose)
        acc_sq = float(acc_summary["squared_error"])
        best_sq = float(best_summary["squared_error"])
        result["local_sq_reduction_on_corrs"] = _sq_reduction(acc_sq, best_sq)
    else:
        result["best_40px_on_corrs"] = None
        result["delta_40px_vs_accepted"] = None
        result["local_sq_reduction_on_corrs"] = None

    if fi not in bad_frames or not live_ids:
        return result

    # Live-bundle evaluation for bad frames only
    X_w_live, xy_live, used_live = _frame_bundle(seed, live_ids, fi)
    n_matched = int(len(used_live))
    live_acc_errs = _residuals(K, accepted_pose, X_w_live, xy_live, eps=eps)
    live_acc_s = _summary(live_acc_errs)

    live_best_s: dict | None = None
    live_sq_red: float | None = None
    if best_pose is not None and n_matched > 0:
        live_best_errs = _residuals(K, best_pose, X_w_live, xy_live, eps=eps)
        live_best_s = _summary(live_best_errs)
        live_sq_red = _sq_reduction(
            float(live_acc_s["squared_error"]),
            float(live_best_s["squared_error"]),
        )

    result["live_bundle"] = {
        "n_live_ids": int(len(live_ids)),
        "n_matched_at_kf": n_matched,
        "accepted_pose": live_acc_s,
        "best_40px": live_best_s,
        "sq_reduction": live_sq_red,
    }

    # Full-history replacement for bad frames
    history = _full_history(seed, live_ids, K, eps=eps)
    base_arr = np.asarray([r["error_px"] for r in history], dtype=np.float64)
    base_s = _summary(base_arr)
    base_sq = float(base_s["squared_error"])
    base_above8 = int(base_s["above_8_count"])

    if best_pose is not None and n_matched > 0:
        live_best_errs_hist = _residuals(K, best_pose, X_w_live, xy_live, eps=eps)
        by_id_best = {
            int(lid): float(e)
            for lid, e in zip(used_live, live_best_errs_hist)
            if np.isfinite(e)
        }
        cf_arr = _replace_kf_residuals(history, fi, by_id_best)
        cf_s = _summary(cf_arr)
        history_result = {
            "n_history_rows": int(len(history)),
            "baseline": base_s,
            "best_40px": cf_s,
            "sq_reduction": _sq_reduction(base_sq, float(cf_s["squared_error"])),
            "above8_reduction": int(base_above8 - cf_s["above_8_count"]),
        }
    else:
        history_result = {
            "n_history_rows": int(len(history)),
            "baseline": base_s,
            "best_40px": None,
            "sq_reduction": None,
            "above8_reduction": None,
        }

    result["history_replacement"] = history_result
    return result


def _print_frame_summary(r: dict) -> None:
    fi = int(r.get("frame_index", -1))
    ds = str(r.get("dataset", "?"))
    label = str(r.get("label", "?"))
    ok = bool(r.get("pipeline_ok", False))
    n_corrs = int(r.get("n_corrs", 0))
    n_loose = int(r.get("n_loose_inliers", 0))
    error = r.get("error", None)

    print(f"\n  [{ds} frame {fi}] label={label} pipeline_ok={ok} n_corrs={n_corrs} n_loose={n_loose}")

    if error is not None:
        print(f"    ERROR: {error}")
        return

    acc = r.get("accepted_pose_on_corrs", {})
    acc_s = acc.get("summary", {}) if isinstance(acc, dict) else {}
    acc_ic = acc.get("inlier_counts", {}) if isinstance(acc, dict) else {}
    print(f"    accepted: median={acc_s.get('median_px', 'N/A')} "
          f"p90={acc_s.get('p90_px', 'N/A')} "
          f"above8={acc_s.get('above_8_count', 'N/A')} "
          f"8px-inliers={acc_ic.get('at_8px', 'N/A')}/{n_corrs}")

    stab = r.get("stability_40px", {})
    if isinstance(stab, dict):
        cl = stab.get("cluster", {})
        best_s = stab.get("best_summary_on_corrs", {})
        best_seed = stab.get("best_seed", None)
        n_ok = int(cl.get("n_successful", 0))
        rate = cl.get("success_rate", None)
        tight = cl.get("tight", None)
        max_rot = cl.get("max_pairwise_rotation_delta_deg", None)
        max_cc = cl.get("max_pairwise_cc_distance", None)
        med_min = cl.get("median_px_min", None)
        med_max = cl.get("median_px_max", None)
        med_std = cl.get("median_px_std", None)
        print(f"    40px stability: {n_ok}/{len(_SEEDS)} seeds succeeded  rate={rate}")
        print(f"      cluster: tight={tight}  max_rot_delta={max_rot:.3f}deg  max_cc_dist={max_cc:.4f}"
              if max_rot is not None and max_cc is not None
              else f"      cluster: tight={tight}")
        if best_s and isinstance(best_s, dict):
            print(f"      best (seed={best_seed}): median={best_s.get('median_px', 'N/A'):.2f}px  "
                  f"p90={best_s.get('p90_px', 'N/A'):.2f}px  "
                  f"above8={best_s.get('above_8_count', 'N/A')}")
        if med_min is not None and med_max is not None:
            print(f"      median range: {med_min:.2f}–{med_max:.2f}px  std={med_std:.3f}px"
                  if med_std is not None else f"      median range: {med_min:.2f}–{med_max:.2f}px")

    best_40 = r.get("best_40px_on_corrs", None)
    if isinstance(best_40, dict):
        b40_s = best_40.get("summary", {})
        b40_ic = best_40.get("inlier_counts", {})
        delta = r.get("delta_40px_vs_accepted", {})
        local_red = r.get("local_sq_reduction_on_corrs", None)
        print(f"    best 40px pose: median={b40_s.get('median_px', 'N/A'):.2f}px  "
              f"above8={b40_s.get('above_8_count', 'N/A')}  "
              f"8px-inliers={b40_ic.get('at_8px', 'N/A')}/{n_corrs}  "
              f"sq_red={local_red:.3f}" if local_red is not None
              else f"    best 40px pose: median={b40_s.get('median_px', 'N/A')}  "
                   f"above8={b40_s.get('above_8_count', 'N/A')}")
        if isinstance(delta, dict):
            print(f"      vs accepted: rot_delta={delta.get('rotation_delta_deg', None):.2f}deg  "
                  f"cc_dist={delta.get('camera_centre_distance', None):.4f}"
                  if delta.get("rotation_delta_deg") is not None else "")

    lb = r.get("live_bundle", None)
    if isinstance(lb, dict):
        lb_acc = lb.get("accepted_pose", {})
        lb_best = lb.get("best_40px", None)
        lb_red = lb.get("sq_reduction", None)
        n_mat = int(lb.get("n_matched_at_kf", 0))
        print(f"    live bundle (n={n_mat}): "
              f"accepted median={lb_acc.get('median_px', 'N/A'):.2f}px  "
              f"above8={lb_acc.get('above_8_count', 'N/A')}")
        if isinstance(lb_best, dict):
            print(f"      40px median={lb_best.get('median_px', 'N/A'):.2f}px  "
                  f"above8={lb_best.get('above_8_count', 'N/A')}  "
                  f"sq_red={lb_red:.3f}" if lb_red is not None else "")

    hist = r.get("history_replacement", None)
    if isinstance(hist, dict):
        base = hist.get("baseline", {})
        best_h = hist.get("best_40px", None)
        sq_red = hist.get("sq_reduction", None)
        above8_red = hist.get("above8_reduction", None)
        n_rows = int(hist.get("n_history_rows", 0))
        print(f"    history ({n_rows} rows): baseline sq={base.get('squared_error', 'N/A'):.0f}  "
              f"above8={base.get('above_8_count', 'N/A')}")
        if isinstance(best_h, dict):
            print(f"      40px replacement: sq_red={sq_red:.3f}  above8_reduction={above8_red}"
                  if sq_red is not None else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eth3d_profile",
        default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"),
    )
    parser.add_argument(
        "--kitti_profile",
        default=str(ROOT / "configs" / "profiles" / "kitti_odometry_00.yaml"),
    )
    parser.add_argument("--out", default="/tmp/diag_40px_resolve_stability.json")
    args = parser.parse_args()

    eth3d_profile = Path(args.eth3d_profile).expanduser().resolve()
    kitti_profile = Path(args.kitti_profile).expanduser().resolve()

    print("=== ETH3D replay ===")
    K_eth3d, pnp_eth3d, seed_eth3d, cap_eth3d, live_ids, first_fail_eth3d, acc_eth3d = (
        _replay_and_capture(
            eth3d_profile,
            _ETH3D_TARGETS,
            _ETH3D_STOP,
            live_frame=_ETH3D_LIVE_FRAME,
        )
    )
    print(f"  ETH3D: first_failure={first_fail_eth3d}  accepted={acc_eth3d}")
    print(f"  live_ids from frame {_ETH3D_LIVE_FRAME}: n={len(live_ids)}")

    print("=== KITTI replay ===")
    K_kitti, pnp_kitti, seed_kitti, cap_kitti, _, first_fail_kitti, acc_kitti = (
        _replay_and_capture(
            kitti_profile,
            _KITTI_TARGETS,
            _KITTI_STOP,
            live_frame=None,
        )
    )
    print(f"  KITTI: first_failure={first_fail_kitti}  accepted={acc_kitti}")

    print("=== Analysis ===")
    eth3d_results = {}
    for fi, cap in sorted(cap_eth3d.items()):
        r = _analyse_frame(
            fi, cap, K_eth3d, pnp_eth3d, seed_eth3d, live_ids,
            bad_frames=_ETH3D_BAD_FRAMES,
            dataset_key="eth3d",
        )
        eth3d_results[fi] = r
        _print_frame_summary(r)

    kitti_results = {}
    for fi, cap in sorted(cap_kitti.items()):
        r = _analyse_frame(
            fi, cap, K_kitti, pnp_kitti, seed_kitti, [],
            bad_frames=frozenset(),
            dataset_key="kitti",
        )
        kitti_results[fi] = r
        _print_frame_summary(r)

    result = {
        "event": "40px_resolve_stability_test",
        "eth3d_profile": str(eth3d_profile),
        "kitti_profile": str(kitti_profile),
        "seeds_tested": _SEEDS,
        "threshold_px": float(_THRESH_40PX),
        "num_trials": int(_TRIALS_40PX),
        "eth3d": {
            "first_failure": first_fail_eth3d,
            "accepted_frames": acc_eth3d,
            "live_ids_count": int(len(live_ids)),
            "frames": {str(fi): r for fi, r in eth3d_results.items()},
        },
        "kitti": {
            "first_failure": first_fail_kitti,
            "accepted_frames": acc_kitti,
            "frames": {str(fi): r for fi, r in kitti_results.items()},
        },
    }

    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
