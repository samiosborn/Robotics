# scripts/run_8_point.py
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from synthetic.two_view import generate_two_view_scene
from geometry.essential import essential_from_fundamental, enforce_essential_constraints, decompose_essential
from geometry.distances import sampson_distances_sq
from geometry.fundamental import estimate_fundamental
from geometry.camera import Camera
from geometry.triangulation import triangulate_points, disambiguate_pose_cheirality
from geometry.rotation import angle_between_rotmats
from geometry.pose import angle_between_translations
from utils.compat import install_two_view_compat_shims


def main():
    parser = argparse.ArgumentParser()
    parser.parse_known_args()

    install_two_view_compat_shims()

    # Generate two-view scene data
    scene_data = generate_two_view_scene(n_points=10, outlier_ratio=0.0, noise_sigma=None)

    # Unpack
    R_true = scene_data["R"]
    t_true = scene_data["t"]
    x1 = scene_data["x1"]
    x2 = scene_data["x2"]
    K1 = scene_data["K1"]
    K2 = scene_data["K2"]

    # Estimate fundamental matrix
    F_est = estimate_fundamental(x1, x2)

    # Sampson distance
    d_sq = sampson_distances_sq(x1, x2, F_est)
    sampson_rmse = np.sqrt(np.mean(d_sq))
    print("Sampson distance RMSE: ", np.round(sampson_rmse, 3))

    # Estimate essential matrix
    E_est = essential_from_fundamental(F_est, K1, K2)

    # Enforce essential constraints
    E_est = enforce_essential_constraints(E_est)

    # Decompose essential matrix
    candidate_poses = decompose_essential(E_est)

    # Camera 1
    cam1 = Camera(K1, np.eye(3), np.zeros(3))

    # Projection matrix 1
    P1 = cam1.P

    # Select valid pose
    R_est, t_est, _ = disambiguate_pose_cheirality(candidate_poses, K1, K2, x1, x2)

    # Error in rotation
    R_error = angle_between_rotmats(R_est, R_true)
    print("Rotation error (radians): ", np.round(R_error, 3))

    # Error in translation
    t_error = angle_between_translations(t_est, t_true)
    print("Translation error (radians): ", np.round(t_error, 3))

    # Camera 2
    cam2 = Camera(K2, R_est, t_est)

    # Projection matrix 2
    P2 = cam2.P

    # Triangulation
    X_est = triangulate_points(P1, P2, x1, x2)

    # Reprojection
    x1_est = cam1.project(X_est)
    x2_est = cam2.project(X_est)

    # 2D Reprojection error
    x1_residual = x1 - x1_est
    x2_residual = x2 - x2_est
    x1_residual_RMS = np.sqrt(np.mean(np.sum(x1_residual ** 2, axis=0)))
    x2_residual_RMS = np.sqrt(np.mean(np.sum(x2_residual ** 2, axis=0)))
    print("Camera 1 RMS reprojection error: ", np.round(x1_residual_RMS, 3))
    print("Camera 2 RMS reprojection error: ", np.round(x2_residual_RMS, 3))


if __name__ == "__main__":
    main()
