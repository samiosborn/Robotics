# scripts/two_view_synthetic.py
import numpy as np
from synthetic.two_view import generate_two_view_scene
from geometry.epipolar import essential_from_pose
from geometry.essential import decompose_essential
from geometry.camera import Camera
from geometry.triangulation import select_valid_pose
from geometry.rotation import angle_between_rotmats
from geometry.pose import angle_between_translations

# Generate scene data
scene_data = generate_two_view_scene(n_points=1)

# Unpack
R_true = scene_data["R"]
t_true = scene_data["t"]
X = scene_data["X"]
x1 = scene_data["x1"]
x2 = scene_data["x2"]
K1 = scene_data["K1"]
K2 = scene_data["K2"]

# Essential matrix
E = essential_from_pose(R_true, t_true)

# Decompose essential matrix
candidate_poses = decompose_essential(E)

# Camera
cam1 = Camera(K1, np.eye(3), np.zeros(3))

# Projection matrix
P1 = cam1.P

# Select valid pose
R_est, t_est = select_valid_pose(candidate_poses, P1, x1, x2)

# Error in rotation
R_error = angle_between_rotmats(R_est, R_true)
print("Rotation error (radians): ", np.round(R_error, 3))

# Error in translation
t_error = angle_between_translations(t_est, t_true)
print("Translation error (radians): ", np.round(t_error, 3))
