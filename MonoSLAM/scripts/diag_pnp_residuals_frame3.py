# scripts/diag_pnp_residuals_frame3.py
# Audit PnP convention consistency at frame 3.
# Tests: (1) bootstrap reprojection quality, (2) full-data DLT residuals,
# (3) residuals under the alternative convention (transpose-R projection).
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from frontend_eth3d_common import (
    ROOT,
    frontend_kwargs_from_cfg as _frontend_kwargs_from_cfg,
    load_runtime_cfg as _load_runtime_cfg,
)
from core.checks import check_dir
from datasets.eth3d import load_eth3d_sequence
from geometry.camera import projection_matrix, world_to_camera_points
from geometry.pnp import build_pnp_correspondences_with_stats, estimate_pose_pnp, PnPCorrespondences
from slam.frame_pipeline import process_frame_against_seed
from slam.frontend import bootstrap_from_two_frames
from slam.seed import seed_keyframe_pose
from slam.tracking import track_against_keyframe


def _project(K, R, t, X_w):
    # Project (3,N) world points to (2,N) pixel coords using K @ [R|t]
    P = projection_matrix(K, R, t)
    X_h = np.vstack([X_w, np.ones((1, X_w.shape[1]))])
    x_h = P @ X_h
    w = x_h[2, :]
    ok = np.abs(w) > 1e-12
    x = np.full((2, X_w.shape[1]), np.nan)
    x[0, ok] = x_h[0, ok] / w[ok]
    x[1, ok] = x_h[1, ok] / w[ok]
    return x


def _residual_stats(res):
    # Summarise a 1-D residual array (observed minus predicted)
    fin = res[np.isfinite(res)]
    if fin.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "p25": np.nan, "p75": np.nan, "n": 0}
    return {
        "mean": float(np.mean(fin)),
        "median": float(np.median(fin)),
        "std": float(np.std(fin)),
        "p25": float(np.percentile(fin, 25)),
        "p75": float(np.percentile(fin, 75)),
        "n": int(fin.size),
    }


def _print_stats(label, stats):
    print(f"  {label}: mean={stats['mean']:+.2f}  median={stats['median']:+.2f}  "
          f"std={stats['std']:.2f}  p25={stats['p25']:+.2f}  p75={stats['p75']:+.2f}  n={stats['n']}")


def main():
    profile_path = ROOT / "configs" / "profiles" / "eth3d_c2.yaml"
    cfg, K = _load_runtime_cfg(profile_path)
    frontend_kwargs = _frontend_kwargs_from_cfg(cfg)

    dataset_cfg = cfg["dataset"]
    dataset_root = (ROOT / dataset_cfg["root"]).resolve()
    seq_name = str(dataset_cfg["seq"])
    check_dir(dataset_root, name="dataset_root")

    seq = load_eth3d_sequence(dataset_root, seq_name, normalise_01=True, dtype=np.float64, require_timestamps=True)

    im0, _, _ = seq.get(0)
    im1, _, _ = seq.get(1)
    im2, _, _ = seq.get(2)
    im3, _, _ = seq.get(3)

    # Bootstrap
    boot = bootstrap_from_two_frames(
        K, K, im0, im1,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        H_cfg=frontend_kwargs["H_cfg"],
        bootstrap_cfg=frontend_kwargs["bootstrap_cfg"],
    )
    if not boot["ok"]:
        print("Bootstrap failed"); return

    seed = boot["seed"]
    keyframe_1_feats = seed["feats1"]

    # --- TEST 1: Bootstrap reprojection quality ---
    # Project X_w using R1, t1 and compare with frame-1 obs stored in seed
    print("=== Test 1: Bootstrap reprojection quality (project X_w with R1, t1 vs frame-1 obs) ===")
    R1, t1 = seed_keyframe_pose(seed)
    landmarks = seed.get("landmarks", [])
    boot_res_x, boot_res_y = [], []
    for lm in landmarks:
        X_w = np.asarray(lm.get("X_w", []), dtype=np.float64).reshape(3, 1)
        obs = lm.get("obs", [])
        if len(obs) < 2:
            continue
        xy1_obs = np.asarray(obs[1].get("xy", []), dtype=np.float64).reshape(2)
        x_pred = _project(K, R1, t1, X_w)
        if not np.isfinite(x_pred).all():
            continue
        boot_res_x.append(float(xy1_obs[0] - x_pred[0, 0]))
        boot_res_y.append(float(xy1_obs[1] - x_pred[1, 0]))

    print(f"  {len(boot_res_x)} bootstrap landmarks checked")
    _print_stats("X residual (obs - pred)", _residual_stats(np.array(boot_res_x)))
    _print_stats("Y residual (obs - pred)", _residual_stats(np.array(boot_res_y)))

    # --- Process frame 2 to get seed_after_frame2 ---
    out_frame2 = process_frame_against_seed(
        K, seed, keyframe_1_feats, im2,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
        keyframe_kf=1,
        current_kf=2,
        **frontend_kwargs["pnp_frontend_kwargs"],
    )
    print(f"\nFrame 2: ok={out_frame2['ok']}, keyframe_promoted={out_frame2['stats']['keyframe_promoted']}")

    seed_after_frame2 = out_frame2["seed"]
    keyframe_2_feats = out_frame2["track_out"]["cur_feats"]

    # --- Build frame-3 correspondences ---
    track_out = track_against_keyframe(
        K, keyframe_2_feats, im3,
        feature_cfg=frontend_kwargs["feature_cfg"],
        F_cfg=frontend_kwargs["F_cfg"],
    )
    corrs, _ = build_pnp_correspondences_with_stats(
        seed_after_frame2, track_out,
        min_landmark_observations=2,
        allow_bootstrap_landmarks_for_pose=True,
        min_post_bootstrap_observations_for_pose=3,
    )
    N = int(corrs.X_w.shape[1])
    print(f"\nFrame-3 correspondences: N={N}")
    if N == 0:
        print("No correspondences – cannot proceed."); return

    X_w = corrs.X_w
    x_cur = corrs.x_cur

    # --- TEST 2: Full-data DLT on all N correspondences ---
    print("\n=== Test 2: Full-data DLT on all frame-3 correspondences ===")
    R_dlt, t_dlt, dlt_stats = estimate_pose_pnp(corrs, K, min_points=6)
    if R_dlt is None:
        print(f"  DLT failed: reason={dlt_stats.get('reason')}")
        print(f"  rank={dlt_stats.get('A_rank', 'N/A')}, cheirality={dlt_stats.get('cheirality_ratio', 'N/A')}")
    else:
        print(f"  DLT succeeded: RMSE={dlt_stats.get('reprojection_rmse_px', float('nan')):.2f}px  "
              f"cheirality={dlt_stats.get('cheirality_ratio', float('nan')):.3f}")
        x_pred = _project(K, R_dlt, t_dlt, X_w)
        res = x_cur - x_pred
        _print_stats("X residual (obs - pred, current convention)", _residual_stats(res[0]))
        _print_stats("Y residual (obs - pred, current convention)", _residual_stats(res[1]))
        abs_err = np.sqrt(res[0]**2 + res[1]**2)
        print(f"  |error| median={np.nanmedian(abs_err):.2f}px  p90={np.nanpercentile(abs_err, 90):.2f}px  "
              f"inliers@5px={int(np.sum(abs_err < 5))}  inliers@10px={int(np.sum(abs_err < 10))}")

    # --- TEST 3: Alternative convention – transpose-R projection ---
    # Tests whether swapping R for R.T in the projection brings residuals down materially.
    # A large drop would indicate R is stored in the wrong orientation.
    print("\n=== Test 3: Alternative convention – transpose-R projection ===")
    if R_dlt is not None:
        R_T = R_dlt.T
        x_pred_alt = _project(K, R_T, t_dlt, X_w)
        res_alt = x_cur - x_pred_alt
        _print_stats("X residual (obs - pred, R transposed)", _residual_stats(res_alt[0]))
        _print_stats("Y residual (obs - pred, R transposed)", _residual_stats(res_alt[1]))
        abs_err_alt = np.sqrt(res_alt[0]**2 + res_alt[1]**2)
        print(f"  |error| median={np.nanmedian(abs_err_alt):.2f}px  p90={np.nanpercentile(abs_err_alt, 90):.2f}px  "
              f"inliers@5px={int(np.sum(abs_err_alt < 5))}  inliers@10px={int(np.sum(abs_err_alt < 10))}")

    # --- TEST 4: Project X_w with KF2 pose (seed after frame-2 promotion) ---
    # Shows whether the KF2 pose is consistent with frame-3 observations (i.e., how much motion).
    print("\n=== Test 4: Residuals under KF2 pose (T_WC1 after frame-2 promotion) ===")
    R_kf2, t_kf2 = seed_keyframe_pose(seed_after_frame2)
    x_pred_kf2 = _project(K, R_kf2, t_kf2, X_w)
    res_kf2 = x_cur - x_pred_kf2
    _print_stats("X residual (frame-3 obs - KF2 projection)", _residual_stats(res_kf2[0]))
    _print_stats("Y residual (frame-3 obs - KF2 projection)", _residual_stats(res_kf2[1]))
    abs_kf2 = np.sqrt(res_kf2[0]**2 + res_kf2[1]**2)
    print(f"  |error| median={np.nanmedian(abs_kf2):.2f}px  p90={np.nanpercentile(abs_kf2, 90):.2f}px")

    # --- TEST 5: DLT rank and conditioning ---
    print("\n=== Test 5: DLT rank and numerical conditioning ===")
    print(f"  A_rank={dlt_stats.get('A_rank', 'N/A')}  (need >= 11)")
    svs = dlt_stats.get("A_singular_values", None)
    if svs is not None:
        svs = np.asarray(svs, dtype=float)
        print(f"  singular values: max={svs[0]:.4g}  s10={svs[10]:.4g}  s11={svs[11] if len(svs) > 11 else 'N/A'}  "
              f"condition_ratio={svs[0]/max(svs[10], 1e-30):.2g}")
    print(f"  n_corr={N}  x_range=[{x_cur[0].min():.1f}, {x_cur[0].max():.1f}]  "
          f"y_range=[{x_cur[1].min():.1f}, {x_cur[1].max():.1f}]")


if __name__ == "__main__":
    main()
