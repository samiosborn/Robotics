# scripts/diag_canonical_pose_storage.py
# Diagnostic counterfactual: keeping frame-16 rescue acceptance and refresh unchanged,
# replace only the stored canonical pose at kf=16 with a constant-velocity extrapolation
# from accepted frames 14 and 15.
# Measures impact on frame-19 live landmark history residuals.

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
from geometry.rotation import angle_between_rotmats
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.keyframe_state import get_pose_for_kf

_EXTRAP_FROM_KF = 14
_EXTRAP_THROUGH_KF = 15
_TARGET_KF = 16
_ORACLE_NEXT_KF = 17
_LIVE_FRAME = 19


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
    R, t = get_pose_for_kf(seed, kf, context="canonical_pose_cf")
    return (
        np.asarray(R, dtype=np.float64).reshape(3, 3).copy(),
        np.asarray(t, dtype=np.float64).reshape(3).copy(),
    )


def _extrapolate_pose(
    pose_from: tuple[np.ndarray, np.ndarray],
    pose_through: tuple[np.ndarray, np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Rodrigues-scaled rotation increment; linear camera-centre extrapolation
    R_f, t_f = pose_from
    R_g, _ = pose_through
    C_f = camera_centre(R_f, t_f)
    C_g = camera_centre(R_g, pose_through[1])
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
    # Collect kf observations for the given landmark ids
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
    # All canonical observation residuals for the live landmarks
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

        if frame_index == _LIVE_FRAME:
            po = out.get("pose_out") or {}
            corrs = po.get("corrs")
            if corrs is not None:
                live_ids = [
                    int(v) for v in np.asarray(corrs.landmark_ids, dtype=np.int64).reshape(-1)
                ]

        seed = out["seed"]

    live_ids = sorted(set(live_ids))
    eps = float(pnp_cfg["eps"])

    pose_14 = _read_pose(seed, _EXTRAP_FROM_KF)
    pose_15 = _read_pose(seed, _EXTRAP_THROUGH_KF)
    pose_16_bad = _read_pose(seed, _TARGET_KF)
    pose_17 = _read_pose(seed, _ORACLE_NEXT_KF)

    ts14 = timestamps[_EXTRAP_FROM_KF]
    ts15 = timestamps[_EXTRAP_THROUGH_KF]
    ts16 = timestamps[_TARGET_KF]
    ts17 = timestamps[_ORACLE_NEXT_KF]

    # Constant-velocity extrapolation: apply the 14→15 step to reach 16
    dt_step = float(ts15 - ts14)
    alpha_extrap = float((ts16 - ts14) / dt_step) if dt_step > 1e-12 else 2.0
    pose_16_extrap = _extrapolate_pose(pose_14, pose_15, alpha_extrap)

    # Retrospective oracle: time-interpolate between 15 and 17 (requires future frame)
    dt_15_17 = float(ts17 - ts15)
    alpha_oracle = float((ts16 - ts15) / dt_15_17) if dt_15_17 > 1e-12 else 0.5
    pose_16_oracle = _extrapolate_pose(pose_15, pose_17, alpha_oracle)

    # Frame-16 observation bundle for the live frame-19 landmarks
    X_w16, xy_16, used16 = _frame_bundle(seed, live_ids, _TARGET_KF)

    err_bad = _residuals(K, pose_16_bad, X_w16, xy_16, eps=eps)
    err_extrap = _residuals(K, pose_16_extrap, X_w16, xy_16, eps=eps)
    err_oracle = _residuals(K, pose_16_oracle, X_w16, xy_16, eps=eps)

    extrap_by_id = {int(lid): float(e) for lid, e in zip(used16, err_extrap) if np.isfinite(e)}
    oracle_by_id = {int(lid): float(e) for lid, e in zip(used16, err_oracle) if np.isfinite(e)}

    # Full canonical history: all frames' observations for the 22 live landmarks
    history = _full_history(seed, live_ids, K, eps=eps)
    base_errs = np.asarray([r["error_px"] for r in history], dtype=np.float64)
    extrap_errs = _replace_kf_residuals(history, _TARGET_KF, extrap_by_id)
    oracle_errs = _replace_kf_residuals(history, _TARGET_KF, oracle_by_id)

    base_sum = _summary(base_errs)
    extrap_sum = _summary(extrap_errs)
    oracle_sum = _summary(oracle_errs)
    base_sq = float(base_sum["squared_error"])

    result = {
        "event": "canonical_pose_storage_counterfactual",
        "profile": str(profile_path),
        "run_summary": {
            "first_failure": first_failure,
            "accepted_count": int(len(accepted)),
        },
        "live_landmarks": {
            "count": int(len(live_ids)),
            "frame16_bundle_matched": int(len(used16)),
        },
        "extrapolation_params": {
            "from_kf": _EXTRAP_FROM_KF,
            "through_kf": _EXTRAP_THROUGH_KF,
            "target_kf": _TARGET_KF,
            "alpha_extrap": float(alpha_extrap),
            "dt_step_s": float(dt_step),
            "dt_target_s": float(ts16 - ts14),
        },
        "oracle_params": {
            "alpha_oracle": float(alpha_oracle),
            "dt_15_17_s": float(dt_15_17),
        },
        "frame16_local": {
            "accepted_rescue_pose": _summary(err_bad),
            "past_extrap_pose": {
                **_summary(err_extrap),
                "delta_vs_accepted": _pose_delta(pose_16_bad, pose_16_extrap),
            },
            "oracle_interp_pose": {
                **_summary(err_oracle),
                "delta_vs_accepted": _pose_delta(pose_16_bad, pose_16_oracle),
            },
        },
        "history_counterfactual": {
            "n_rows": int(len(history)),
            "baseline": base_sum,
            "past_extrap_kf16": {
                "summary": extrap_sum,
                "sq_error_reduction": _sq_reduction(base_sq, float(extrap_sum["squared_error"])),
                "above8_reduction": int(base_sum["above_8_count"] - extrap_sum["above_8_count"]),
            },
            "oracle_kf16": {
                "summary": oracle_sum,
                "sq_error_reduction": _sq_reduction(base_sq, float(oracle_sum["squared_error"])),
                "above8_reduction": int(base_sum["above_8_count"] - oracle_sum["above_8_count"]),
            },
        },
    }

    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.out is not None:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
