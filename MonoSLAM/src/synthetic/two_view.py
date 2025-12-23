# src/synthetic/two_view.py
import numpy as np
from geometry.rotation import axis_angle_to_rotmat
from geometry.camera import Camera

# Generate two-view scene
def generate_two_view_scene(n_points=20, seed=42, max_angle_deg=15.0, K1=None, K2=None): 
    # Random number generator
    rng = np.random.default_rng(seed)
    # Intrinsics
    if K1 is None:
        K1 = np.eye(3)
    if K2 is None:
        K2 = np.eye(3)
    # Axis
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    # Angle
    angle = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
    # Rotation matrix
    R = axis_angle_to_rotmat(axis, angle)
    # Translation
    t = rng.uniform(low=0.5, high=1.0, size=3)
    # Encourage a mostly horizontal translation (X-axis)
    t[1: ] *= 0.2
    # Normalise translation
    t = t / np.linalg.norm(t)
    # Generate 3D points
    X = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    # Ensure positive depth (Z-axis)
    X[:, 2] += 3.0
    # Cameras
    cam1 = Camera(K1, np.eye(3), np.zeros(3))
    cam2 = Camera(K2, R, t)
    # Project to image coordinates 
    x1 = np.array([cam1.project(Xi) for Xi in X])
    x2 = np.array([cam2.project(Xi) for Xi in X])
    # Return dict
    return {
        "R": R,
        "t": t,
        "X": X,
        "x1": x1,
        "x2": x2,
        "K1": K1,
        "K2": K2
    }
