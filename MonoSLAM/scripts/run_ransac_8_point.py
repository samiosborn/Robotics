# scripts/run_ransac_8_point.py
import numpy as np
from synthetic.two_view import generate_two_view_scene
from geometry.essential import essential_from_fundamental, enforce_essential_constraints, decompose_essential
from geometry.epipolar import sampson_distances_sq
from geometry.fundamental import estimate_fundamental, estimate_fundamental_ransac
from geometry.camera import Camera
from geometry.triangulation import triangulate_points, select_valid_pose
from geometry.rotation import angle_between_rotmats
from geometry.pose import angle_between_translations

# Generate scene (with noise and outliers)
scene_data = generate_two_view_scene(n_points=20, outlier_ratio=0.2, noise_sigma_pixels=5)

# Unpack scene
R_true = scene_data["R"]
t_true = scene_data["t"]
X = scene_data["X"]
x1_clean = scene_data["x1_clean"]
x2_clean = scene_data["x2_clean"]
x1 = scene_data["x1"]
x2 = scene_data["x2"]
inlier_mask = scene_data["inlier_mask"]
outliers_idx = scene_data["outliers_idx"]
K1 = scene_data["K1"]
K2 = scene_data["K2"]

# Baseline deterministic 8-point algo

# Estimate fundamental matrix via RANSAC (return inliers)

# Refit fundamental matrix on all inliers

# Epipolar error: Sampson RMSE, rotation error, translation error

# Return essential matrix from fundamental

# Enforce essential constraints

# Decompose essential matrix into candidate poses

# Select valid pose using cheirality 

# Build camera from K, R, t

# Get projection matrix from valid pose

# Triangulate to world frame using inliers

# Reproject points to image frame from inliers

# Per-point Euclidean reprojection RMSE on inliers (cam1 / cam2)
