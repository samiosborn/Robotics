# scripts/demo_fundamental_8point.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from synthetic.two_view import generate_two_view_scene
from geometry.camera import Camera
from geometry.distances import sampson_rmse
from geometry.fundamental import estimate_fundamental
from geometry.pose import angle_between_translations, pose_from_fundamental
from geometry.rotation import angle_between_rotmats
from geometry.triangulation import triangulate_points


# Evaluate and print two-view model quality metrics
def evaluate_two_view_model(
    tag,
    F_est,
    x1,
    x2,
    K1,
    K2,
    x1_clean=None,
    x2_clean=None,
    X_true=None,
    R_true=None,
    t_true=None,
    cam1=None,
    true_inlier_mask=None,
    pred_inlier_mask=None,
):
    # Evaluate predicted inliers against true inliers
    if (true_inlier_mask is not None) and (pred_inlier_mask is not None):
        true = np.asarray(true_inlier_mask, dtype=bool)
        pred = np.asarray(pred_inlier_mask, dtype=bool)

        tp = int(np.sum(true & pred))
        fp = int(np.sum((~true) & pred))
        fn = int(np.sum(true & (~pred)))

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)

        print(tag, f"Inlier precision={prec:.3f}, recall={rec:.3f}, tp={tp}, fp={fp}, fn={fn}")

    # Build default full mask
    N = int(x1.shape[1])
    all_mask = np.ones(N, dtype=bool)

    # Choose evaluation set
    use_pred = (pred_inlier_mask is not None) and (int(np.sum(pred_inlier_mask)) >= 8)
    if use_pred:
        use_mask = np.asarray(pred_inlier_mask, dtype=bool)
    else:
        if pred_inlier_mask is not None:
            print(tag, "Predicted inlier mask too small so evaluating on all correspondences")
        use_mask = all_mask

    x1_use = x1[:, use_mask]
    x2_use = x2[:, use_mask]

    # Build camera 1 if not provided
    if cam1 is None:
        cam1 = Camera(K1, np.eye(3), np.zeros(3))

    # Recover pose from F
    try:
        R_est, t_est, E_est, cheir_ratio, best_mask = pose_from_fundamental(F_est, K1, K2, x1_use, x2_use)
    except Exception as e:
        print(tag, "Pose recovery failed:", e)
        return

    # Build camera 2
    cam2 = Camera(K2, R_est, t_est)

    # Triangulate on evaluation subset
    X_est = triangulate_points(cam1.P, cam2.P, x1_use, x2_use)

    # Reprojection errors on noisy observations
    print(tag, "Cam1 reproj RMSE (noisy):", np.round(cam1.reprojection_rmse(X_est, x1_use), 3))
    print(tag, "Cam2 reproj RMSE (noisy):", np.round(cam2.reprojection_rmse(X_est, x2_use), 3))

    # Reprojection errors on clean observations if provided
    if (x1_clean is not None) and (x2_clean is not None) and (true_inlier_mask is not None):
        clean_mask = use_mask & np.asarray(true_inlier_mask, dtype=bool)
        if int(np.sum(clean_mask)) >= 8:
            x1_clean_use = x1_clean[:, clean_mask]
            x2_clean_use = x2_clean[:, clean_mask]

            X_clean = triangulate_points(cam1.P, cam2.P, x1_clean_use, x2_clean_use)

            print(tag, "Cam1 reproj RMSE (clean):", np.round(cam1.reprojection_rmse(X_clean, x1_clean_use), 3))
            print(tag, "Cam2 reproj RMSE (clean):", np.round(cam2.reprojection_rmse(X_clean, x2_clean_use), 3))

    # GT 3D reprojection using true cameras if provided
    if (X_true is not None) and (x1_clean is not None) and (x2_clean is not None) and (R_true is not None) and (t_true is not None):
        X_gt = np.asarray(X_true, dtype=float)

        if X_gt.shape == (N, 3):
            X_gt = X_gt.T

        if X_gt.shape == (3, N):
            cam2_true = Camera(K2, R_true, t_true)
            print(tag, "Cam1 GT-3D reproj RMSE (clean):", np.round(cam1.reprojection_rmse(X_gt, x1_clean), 3))
            print(tag, "Cam2 GT-3D reproj RMSE (clean):", np.round(cam2_true.reprojection_rmse(X_gt, x2_clean), 3))

    # Pose errors against ground truth
    if R_true is not None:
        print(tag, "Rotation error (rad):", np.round(angle_between_rotmats(R_est, R_true), 3))

    if t_true is not None:
        print(tag, "Translation error (rad):", np.round(angle_between_translations(t_est, t_true), 3))

    # Sampson RMSE summaries
    print(tag, "Sampson RMSE (all):", np.round(sampson_rmse(x1, x2, F_est), 3))

    if true_inlier_mask is not None:
        true_mask = np.asarray(true_inlier_mask, dtype=bool)
        if int(np.sum(true_mask)) >= 8:
            print(tag, "Sampson RMSE (true inliers):", np.round(sampson_rmse(x1[:, true_mask], x2[:, true_mask], F_est), 3))

    if use_pred:
        print(tag, "Sampson RMSE (pred inliers):", np.round(sampson_rmse(x1_use, x2_use, F_est), 3))

    # Report cheirality summary
    print(tag, f"Cheirality ratio: {cheir_ratio:.3f}")
    print(tag, f"Cheirality inliers: {int(np.sum(best_mask))}/{int(best_mask.size)}")


def main():
    # Parse CLI
    parser = argparse.ArgumentParser()
    parser.parse_known_args()

    # Generate synthetic two-view scene data
    scene_data = generate_two_view_scene(n_points=10, outlier_ratio=0.0, noise_sigma=None)

    # Unpack scene data
    R_true = scene_data["R"]
    t_true = scene_data["t"]
    x1 = scene_data["x1"]
    x2 = scene_data["x2"]
    K1 = scene_data["K1"]
    K2 = scene_data["K2"]

    # Clean and GT fields if present
    x1_clean = scene_data.get("x1_clean", None)
    x2_clean = scene_data.get("x2_clean", None)
    X_true = scene_data.get("X", scene_data.get("X_true", None))
    true_inlier_mask = scene_data.get("inlier_mask", scene_data.get("true_inlier_mask", None))

    # Estimate fundamental matrix
    F_est = estimate_fundamental(x1, x2)

    # Print estimated F
    print("Estimated F:")
    print(np.round(F_est, 6))

    # Evaluate estimated two-view model
    evaluate_two_view_model(
        tag="[8POINT]",
        F_est=F_est,
        x1=x1,
        x2=x2,
        K1=K1,
        K2=K2,
        x1_clean=x1_clean,
        x2_clean=x2_clean,
        X_true=X_true,
        R_true=R_true,
        t_true=t_true,
        cam1=Camera(K1, np.eye(3), np.zeros(3)),
        true_inlier_mask=true_inlier_mask,
        pred_inlier_mask=None,
    )


if __name__ == "__main__":
    main()
