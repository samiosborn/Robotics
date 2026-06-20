# scripts/diag_frame16_proxy_poses.py
# Compares rescue-time canonical pose proxy candidates for ETH3D kf=16.
# Candidates: frame-15 pose, seeded-40px RANSAC on full correspondences,
# trimmed-50%-refit on accepted loose inliers, pruned-<15px refit on same.
# Evaluation: residuals on 22 live frame-19 landmarks' kf=16 observations
# plus analytical full-history replacement.
# All candidates are evaluated at rescue time (no future information used
# except oracle, which is labelled retrospective).

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
from geometry.pnp import (
    _pnp_inlier_mask_from_pose,
    _slice_pnp_correspondences,
    estimate_pose_pnp_ransac,
)
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_pose_for_kf

_TARGET_KF = 16
_LIVE_FRAME = 19
_ORACLE_NEXT_KF = 17
_SEEDED_PX = 40.0
_REFIT_PX = 12.0
_PRUNE_THRESH_PX = 15.0
_TRIM_KEEP_FRACTION = 0.50
_REFIT_TRIALS = 5000
_REFIT_MIN_INLIERS = 6


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


def _normalise_R(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    if float(np.linalg.det(U @ Vt)) < 0.0:
        U[:, -1] *= -1.0
    return U @ Vt


def _read_pose(seed: dict, kf: int) -> tuple[np.ndarray, np.ndarray]:
    R, t = get_pose_for_kf(seed, kf, context="proxy_pose_diag")
    return (
        np.asarray(R, dtype=np.float64).reshape(3, 3).copy(),
        np.asarray(t, dtype=np.float64).reshape(3).copy(),
    )


def _extrapolate_pose(
    pose_from: tuple[np.ndarray, np.ndarray],
    pose_through: tuple[np.ndarray, np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    R_f, t_f = pose_from
    R_g, t_g = pose_through
    C_f = camera_centre(R_f, t_f)
    C_g = camera_centre(R_g, t_g)
    R_rel = _normalise_R(R_g @ R_f.T)
    theta = float(np.arccos(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)))
    if theta <= 1e-12:
        R_inc = np.eye(3, dtype=np.float64)
    elif abs(float(np.sin(theta))) <= 1e-12:
        R_inc = _normalise_R((1.0 - alpha) * np.eye(3, dtype=np.float64) + alpha * R_rel)
    else:
        K_ax = (R_rel - R_rel.T) / (2.0 * np.sin(theta))
        th_a = alpha * theta
        R_inc = (
            np.eye(3, dtype=np.float64)
            + np.sin(th_a) * K_ax
            + (1.0 - np.cos(th_a)) * (K_ax @ K_ax)
        )
    R_out = _normalise_R(R_inc @ R_f)
    C_out = C_f + alpha * (C_g - C_f)
    return R_out, -R_out @ C_out


def _direction_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        return float(np.degrees(angle_between_translations(a, b)))
    except Exception:
        return None


def _pose_delta(a: tuple, b: tuple) -> dict:
    R_a, t_a = a
    R_b, t_b = b
    return {
        "rotation_delta_deg": float(np.degrees(angle_between_rotmats(R_a, R_b))),
        "translation_direction_delta_deg": _direction_deg(t_a, t_b),
        "camera_centre_distance": float(
            np.linalg.norm(camera_centre(R_b, t_b) - camera_centre(R_a, t_a))
        ),
    }


def _residuals(
    K: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    X_w: np.ndarray,
    xy: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    R, t = pose
    if int(X_w.shape[1]) == 0:
        return np.zeros((0,), dtype=np.float64)
    depth = (R @ X_w + t.reshape(3, 1))[2, :]
    err_sq = np.asarray(reprojection_errors_sq(K, R, t, X_w, xy), dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(depth)
        & (depth > float(eps))
        & np.isfinite(err_sq)
        & (err_sq >= 0.0)
    )
    out = np.full(int(err_sq.size), np.nan, dtype=np.float64)
    out[valid] = np.sqrt(err_sq[valid])
    return out


def _summary(errors: np.ndarray) -> dict:
    arr = np.asarray(errors, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "count": 0,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "squared_error": 0.0,
            "above_8_count": 0,
            "above_8_fraction": None,
        }
    above_8 = int(np.sum(arr > 8.0))
    return {
        "count": n,
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90.0)),
        "max_px": float(np.max(arr)),
        "squared_error": float(np.sum(arr * arr)),
        "above_8_count": above_8,
        "above_8_fraction": float(above_8 / n),
    }


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
                pose = _read_pose(seed, obs_kf)
            except Exception:
                continue
            err = _residuals(K, pose, X_w.reshape(3, 1), xy.reshape(2, 1), eps=eps)
            if int(err.size) == 1 and np.isfinite(err[0]):
                rows.append({"landmark_id": int(lm_id), "kf": obs_kf, "error_px": float(err[0])})
    return rows


def _replace_kf_residuals(
    rows: list[dict],
    kf: int,
    by_id: dict[int, float],
) -> np.ndarray:
    out = []
    for row in rows:
        if int(row["kf"]) == kf and int(row["landmark_id"]) in by_id:
            out.append(by_id[int(row["landmark_id"])])
        else:
            out.append(float(row["error_px"]))
    return np.asarray(out, dtype=np.float64)


def _sq_reduction(base: float, cf: float) -> float | None:
    if base <= 0.0:
        return None
    return float((base - cf) / base)


def _run_pnp(
    corrs,
    K: np.ndarray,
    *,
    threshold_px: float,
    num_trials: int,
    min_inliers: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(corrs.X_w.shape[1])
    sample_size = int(cfg.get("sample_size", 6))
    if n < sample_size:
        return None
    try:
        R, t, _, _ = estimate_pose_pnp_ransac(
            corrs,
            K,
            num_trials=num_trials,
            sample_size=sample_size,
            threshold_px=float(threshold_px),
            min_inliers=min_inliers,
            seed=int(cfg["ransac_seed"]),
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


def _inlier_counts(corrs, pose, K: np.ndarray, eps: float) -> dict:
    if pose is None:
        return {"at_8px": 0, "at_12px": 0, "at_20px": 0}
    R, t = pose
    counts = {}
    for px in (8.0, 12.0, 20.0):
        mask, _ = _pnp_inlier_mask_from_pose(
            corrs.X_w, corrs.x_cur, K, R, t,
            threshold_px=px, eps=eps,
        )
        counts[f"at_{int(px)}px"] = int(np.sum(np.asarray(mask, dtype=bool)))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=str(ROOT / "configs" / "profiles" / "eth3d_c2.yaml"),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
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
    timestamps: dict[int, float] = {}
    live_ids: list[int] = []
    frame16_corrs = None
    frame16_loose_mask: np.ndarray | None = None
    frame16_bad_R: np.ndarray | None = None
    frame16_bad_t: np.ndarray | None = None
    first_failure: int | None = None
    accepted: list[int] = []

    for frame_index in range(i1 + 1, _LIVE_FRAME + 1):
        im, ts, _ = seq.get(frame_index)
        timestamps[frame_index] = float(ts)
        out = process_frame_against_seed(
            K,
            seed,
            im,
            feature_cfg=fkw["feature_cfg"],
            F_cfg=fkw["F_cfg"],
            current_kf=frame_index,
            **pnp_cfg,
        )

        if bool(out.get("ok", False)):
            accepted.append(frame_index)
        elif first_failure is None:
            first_failure = frame_index

        po = out.get("pose_out") or {}

        if frame_index == _TARGET_KF:
            frame16_corrs = po.get("corrs")
            raw_mask = po.get("pnp_inlier_mask")
            if raw_mask is not None:
                frame16_loose_mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
            r16 = po.get("R")
            t16 = po.get("t")
            if r16 is not None and t16 is not None:
                frame16_bad_R = np.asarray(r16, dtype=np.float64).reshape(3, 3).copy()
                frame16_bad_t = np.asarray(t16, dtype=np.float64).reshape(3).copy()

        if frame_index == _LIVE_FRAME:
            corrs19 = po.get("corrs")
            if corrs19 is not None:
                live_ids = [
                    int(v)
                    for v in np.asarray(corrs19.landmark_ids, dtype=np.int64).reshape(-1)
                ]

        seed = out["seed"]

    live_ids = sorted(set(live_ids))
    eps = float(pnp_cfg["eps"])

    pose_15 = _read_pose(seed, 15)
    pose_16_bad = (frame16_bad_R, frame16_bad_t) if frame16_bad_R is not None else None
    pose_17 = _read_pose(seed, _ORACLE_NEXT_KF)

    ts15 = timestamps[15]
    ts16 = timestamps[_TARGET_KF]
    ts17 = timestamps[_ORACLE_NEXT_KF]

    # Retrospective oracle interpolation between frames 15 and 17
    dt_15_17 = float(ts17 - ts15)
    alpha_oracle = float((ts16 - ts15) / dt_15_17) if dt_15_17 > 1e-12 else 0.5
    pose_16_oracle = _extrapolate_pose(pose_15, pose_17, alpha_oracle)

    # Seeded 40px RANSAC on the full 28 correspondences
    pose_16_seeded: tuple | None = None
    seeded_meta: dict = {}
    if frame16_corrs is not None:
        n_corr = int(frame16_corrs.X_w.shape[1])
        seeded_result = _run_pnp(
            frame16_corrs,
            K,
            threshold_px=_SEEDED_PX,
            num_trials=_REFIT_TRIALS,
            min_inliers=int(pnp_cfg.get("min_inliers", 8)),
            cfg=pnp_cfg,
        )
        seeded_meta["n_corr_input"] = n_corr
        seeded_meta["succeeded"] = seeded_result is not None
        if seeded_result is not None:
            pose_16_seeded = seeded_result
            seeded_meta["inlier_counts_on_full_corrs"] = _inlier_counts(
                frame16_corrs, seeded_result, K, eps
            )

    # Trim and prune candidates from the 24 accepted loose inliers
    pose_16_trim50: tuple | None = None
    pose_16_prune15: tuple | None = None
    trim50_meta: dict = {}
    prune15_meta: dict = {}

    if (
        frame16_corrs is not None
        and frame16_loose_mask is not None
        and frame16_bad_R is not None
    ):
        corrs_loose = _slice_pnp_correspondences(frame16_corrs, frame16_loose_mask)
        n_loose = int(corrs_loose.X_w.shape[1])
        sample_size = int(pnp_cfg.get("sample_size", 6))

        resid_24 = _residuals(K, (frame16_bad_R, frame16_bad_t), corrs_loose.X_w, corrs_loose.x_cur, eps=eps)
        resid_safe = np.where(np.isfinite(resid_24), resid_24, np.inf)

        # Trimmed 50%: keep the n_keep landmarks with lowest residual under bad pose
        n_keep = max(sample_size, int(np.ceil(n_loose * _TRIM_KEEP_FRACTION)))
        sorted_idx = np.argsort(resid_safe)
        keep_50 = sorted_idx[:n_keep]
        trim50_meta["n_loose_total"] = n_loose
        trim50_meta["n_kept"] = int(n_keep)
        trim50_meta["kept_residuals_under_bad_pose_median"] = float(np.median(resid_safe[keep_50]))
        trim50_meta["kept_residuals_under_bad_pose_max"] = float(np.max(resid_safe[keep_50]))

        corrs_trim = _slice_pnp_correspondences(corrs_loose, keep_50)
        trim_result = _run_pnp(
            corrs_trim,
            K,
            threshold_px=_REFIT_PX,
            num_trials=_REFIT_TRIALS,
            min_inliers=_REFIT_MIN_INLIERS,
            cfg=pnp_cfg,
        )
        trim50_meta["succeeded"] = trim_result is not None
        if trim_result is not None:
            pose_16_trim50 = trim_result
            trim50_meta["inlier_counts_on_loose"] = _inlier_counts(corrs_loose, trim_result, K, eps)

        # Pruned < 15px: keep inliers with residual below threshold under bad pose
        keep_lt = resid_safe < _PRUNE_THRESH_PX
        n_lt = int(np.sum(keep_lt))
        prune15_meta["n_loose_total"] = n_loose
        prune15_meta["n_kept"] = n_lt
        prune15_meta["prune_threshold_px"] = float(_PRUNE_THRESH_PX)
        if n_lt >= sample_size:
            corrs_lt = _slice_pnp_correspondences(corrs_loose, keep_lt)
            lt_result = _run_pnp(
                corrs_lt,
                K,
                threshold_px=_REFIT_PX,
                num_trials=_REFIT_TRIALS,
                min_inliers=_REFIT_MIN_INLIERS,
                cfg=pnp_cfg,
            )
            prune15_meta["succeeded"] = lt_result is not None
            if lt_result is not None:
                pose_16_prune15 = lt_result
                prune15_meta["inlier_counts_on_loose"] = _inlier_counts(corrs_loose, lt_result, K, eps)
        else:
            prune15_meta["succeeded"] = False
            prune15_meta["reason"] = "too_few_correspondences_after_pruning"

    # Frame-16 kf=16 observation bundle for the 22 live frame-19 landmarks
    X_w16, xy_16, used16 = _frame_bundle(seed, live_ids, _TARGET_KF)

    def _eval(pose):
        if pose is None or pose[0] is None:
            return None
        return _residuals(K, pose, X_w16, xy_16, eps=eps)

    err_bad = _eval(pose_16_bad)
    err_pose15 = _eval(pose_15)
    err_seeded = _eval(pose_16_seeded)
    err_trim50 = _eval(pose_16_trim50)
    err_prune15 = _eval(pose_16_prune15)
    err_oracle = _eval(pose_16_oracle)

    def _by_id(errs):
        if errs is None:
            return {}
        return {int(lid): float(e) for lid, e in zip(used16, errs) if np.isfinite(e)}

    by_bad = _by_id(err_bad)
    by_pose15 = _by_id(err_pose15)
    by_seeded = _by_id(err_seeded)
    by_trim50 = _by_id(err_trim50)
    by_prune15 = _by_id(err_prune15)
    by_oracle = _by_id(err_oracle)

    # Full canonical history (340 rows)
    history = _full_history(seed, live_ids, K, eps=eps)
    base_arr = np.asarray([r["error_px"] for r in history], dtype=np.float64)
    base_sum = _summary(base_arr)
    base_sq = float(base_sum["squared_error"])
    base_above8 = int(base_sum["above_8_count"])

    def _hist_cf(by_id: dict) -> dict:
        arr = _replace_kf_residuals(history, _TARGET_KF, by_id)
        s = _summary(arr)
        return {
            "summary": s,
            "sq_error_reduction": _sq_reduction(base_sq, float(s["squared_error"])),
            "above8_reduction": int(base_above8 - s["above_8_count"]),
        }

    def _cand_row(
        label: str,
        errs,
        candidate_pose,
        metadata: dict | None = None,
    ) -> dict:
        local_s = _summary(errs if errs is not None else np.zeros((0,), dtype=np.float64))
        row: dict = {
            "label": label,
            "local_frame16_on_live19_bundle": local_s,
        }
        if candidate_pose is not None and pose_16_bad is not None and pose_16_bad[0] is not None:
            row["delta_vs_accepted_bad_pose"] = _pose_delta(pose_16_bad, candidate_pose)
        if metadata:
            row["metadata"] = metadata
        return row

    local_sq_bad = float(_summary(err_bad)["squared_error"]) if err_bad is not None else None

    def _local_sq_red(errs):
        if errs is None or local_sq_bad is None or local_sq_bad <= 0.0:
            return None
        s = float(_summary(errs)["squared_error"])
        return float((local_sq_bad - s) / local_sq_bad)

    candidates = [
        {
            **_cand_row("accepted_bad_rescue_pose", err_bad, None),
            "local_sq_reduction_vs_bad": 0.0,
            "history_cf": _hist_cf(by_bad),
        },
        {
            **_cand_row("frame15_pose", err_pose15, pose_15,
                        {"source": "previous_accepted_canonical_pose", "kf": 15}),
            "local_sq_reduction_vs_bad": _local_sq_red(err_pose15),
            "history_cf": _hist_cf(by_pose15),
        },
        {
            **_cand_row("seeded_40px_ransac", err_seeded, pose_16_seeded, seeded_meta),
            "local_sq_reduction_vs_bad": _local_sq_red(err_seeded),
            "history_cf": _hist_cf(by_seeded) if pose_16_seeded is not None else None,
        },
        {
            **_cand_row("trimmed_50pct_refit_12px", err_trim50, pose_16_trim50, trim50_meta),
            "local_sq_reduction_vs_bad": _local_sq_red(err_trim50),
            "history_cf": _hist_cf(by_trim50) if pose_16_trim50 is not None else None,
        },
        {
            **_cand_row("pruned_lt15px_refit_12px", err_prune15, pose_16_prune15, prune15_meta),
            "local_sq_reduction_vs_bad": _local_sq_red(err_prune15),
            "history_cf": _hist_cf(by_prune15) if pose_16_prune15 is not None else None,
        },
        {
            **_cand_row("oracle_interp_15_17", err_oracle, pose_16_oracle,
                        {"alpha": float(alpha_oracle), "retrospective_only": True}),
            "local_sq_reduction_vs_bad": _local_sq_red(err_oracle),
            "history_cf": _hist_cf(by_oracle),
        },
    ]

    result = {
        "event": "frame16_proxy_pose_comparison",
        "profile": str(profile_path),
        "run_summary": {
            "first_failure": first_failure,
            "accepted_count": int(len(accepted)),
        },
        "live_landmarks": {
            "count": int(len(live_ids)),
            "frame16_bundle_matched": int(len(used16)),
        },
        "history_baseline": base_sum,
        "n_history_rows": int(len(history)),
        "candidates": candidates,
    }

    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.out is not None:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
