# src/synthetic/two_view.py
import numpy as np
from geometry.rotation import axis_angle_to_rotmat
from geometry.camera import Camera

# Generate two-view scene
def generate_two_view_scene(n_points=20, seed=42, max_angle_deg=15.0, K1=None, K2=None, outlier_ratio=0.2, noise_sigma=0.005): 
    # Random number generator
    rng = np.random.default_rng(seed)
    # Intrinsics
    if K1 is None:
        K1 = np.array([[800, 0, 320],
                    [0, 800, 240],
                    [0,   0,   1]], dtype=float)
    if K2 is None:
        K2 = K1.copy()
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
    X = rng.uniform(-1.0, 1.0, size=(3, n_points))
    # Ensure positive depth (Z-axis)
    X[2, :] += 3.0
    # Cameras
    cam1 = Camera(K1, np.eye(3), np.zeros(3))
    cam2 = Camera(K2, R, t)
    # Project to image coordinates 
    x1_clean = cam1.project(X)
    x2_clean = cam2.project(X)
    # No noise
    if noise_sigma is None or noise_sigma <= 0: 
        x1 = x1_clean.copy()
        x2 = x2_clean.copy()
    else: 
        # Gaussian noise
        x1 = x1_clean + rng.normal(scale=noise_sigma, size=x1_clean.shape)
        x2 = x2_clean + rng.normal(scale=noise_sigma, size=x2_clean.shape)
    # Initialise inlier mask
    inlier_mask = np.ones(n_points, dtype=bool)
    # Number of outliers
    outlier_ratio = float(np.clip(outlier_ratio, 0.0, 1.0))
    n_outliers = int(np.round(outlier_ratio * n_points))
    if n_outliers > 0: 
        # Outliers index
        outliers_idx = rng.choice(n_points, size=n_outliers, replace=False)
        # Inlier mask
        inlier_mask[outliers_idx] = False
        # Permutation as copy of idx
        perm = outliers_idx.copy()
        # Shuffle outliers index
        rng.shuffle(perm)
        # Shuffle outliers 
        x2[:, outliers_idx] = x2[:, perm]
    else:
        outliers_idx = np.array([], dtype=int)
    # Return dict
    return {
        "R": R,
        "t": t,
        "X": X,
        "x1_clean": x1_clean, "x2_clean": x2_clean,
        "x1": x1, "x2": x2,
        "inlier_mask": inlier_mask,
        "outliers_idx": outliers_idx,
        "K1": K1, "K2": K2
    }
