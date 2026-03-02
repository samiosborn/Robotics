# scripts/run_runsac_8_point.py
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from synthetic.two_view import generate_two_view_scene
from geometry.distances import sampson_distances_sq
from geometry.fundamental import estimate_fundamental, estimate_fundamental_ransac
from geometry.camera import Camera
from geometry.evaluation import evaluate_two_view_model
from utils.compat import install_two_view_compat_shims

def main():
    parser = argparse.ArgumentParser()
    parser.parse_known_args()

    install_two_view_compat_shims()

    # Generate scene (with noise and outliers)
    noise_sigma = 2
    outlier_ratio = 0.2
    n_points = 50
    scene_data = generate_two_view_scene(
        n_points=n_points,
        outlier_ratio=outlier_ratio,
        noise_sigma=noise_sigma,
    )

    # Unpack scene
    R_true = scene_data["R"]
    t_true = scene_data["t"]
    X_true = scene_data["X"]
    x1_clean = scene_data["x1_clean"]
    x2_clean = scene_data["x2_clean"]
    x1 = scene_data["x1"]
    x2 = scene_data["x2"]
    true_inlier_mask = scene_data["inlier_mask"]
    K1 = scene_data["K1"]
    K2 = scene_data["K2"]

    # --- CAMERA 1 ---
    cam1 = Camera(K1, np.eye(3), np.zeros(3))

    # --- BASELINE ---
    # Baseline deterministic 8-point algo
    F_baseline = estimate_fundamental(x1, x2)
    # Evaluate model
    evaluate_two_view_model("Baseline: ", F_baseline, x1, x2, K1, K2, x1_clean, x2_clean, X_true, R_true, t_true, cam1, true_inlier_mask, pred_inlier_mask=None)

    # --- RANSAC ---
    # RANSAC 8-point algo
    num_trials = 2000
    threshold_sq = 36
    F_ransac, ransac_pred_inlier_mask_1 = estimate_fundamental_ransac(
        x1,
        x2,
        num_trials=num_trials,
        threshold=np.sqrt(threshold_sq),
    )
    n1 = int(ransac_pred_inlier_mask_1.sum())
    # Evaluate model
    evaluate_two_view_model("RANSAC: ", F_ransac, x1, x2, K1, K2, x1_clean, x2_clean, X_true, R_true, t_true, cam1, true_inlier_mask, ransac_pred_inlier_mask_1)

    # --- RANSAC REFIT ---
    # Refit on best RANSAC inliers
    F_ransac_refit_1 = estimate_fundamental(x1[:, ransac_pred_inlier_mask_1], x2[:, ransac_pred_inlier_mask_1])
    # Redo mask
    ransac_pred_inlier_mask_2 = sampson_distances_sq(x1, x2, F_ransac_refit_1) < threshold_sq
    n2 = int(ransac_pred_inlier_mask_2.sum())
    # Keep new mask only if it doesn't shrink too much
    if n2 < 0.8 * n1:
        print(f"LO-RANSAC: mask shrank {n1}->{n2}. Keeping mask 1.")
        ransac_pred_inlier_mask_2 = ransac_pred_inlier_mask_1
        F_ransac_refit_2 = F_ransac_refit_1
    else:
        F_ransac_refit_2 = estimate_fundamental(x1[:, ransac_pred_inlier_mask_2], x2[:, ransac_pred_inlier_mask_2])

    # Evaluate model
    evaluate_two_view_model("Refit RANSAC: ", F_ransac_refit_2, x1, x2, K1, K2, x1_clean, x2_clean, X_true, R_true, t_true, cam1, true_inlier_mask, ransac_pred_inlier_mask_2)


if __name__ == "__main__":
    main()
