# scripts/run_ransac_8_point.py
import numpy as np
from synthetic.two_view import generate_two_view_scene
from geometry.epipolar import sampson_distances_sq, sampson_rmse
from geometry.fundamental import estimate_fundamental, estimate_fundamental_ransac
from geometry.camera import Camera
from geometry.triangulation import triangulate_points
from geometry.rotation import angle_between_rotmats
from geometry.pose import recover_pose_from_fundamental, angle_between_translations

# Evaluation helper
def _evaluate_model(tag, F_est, x1, x2, K1, K2, x1_clean=None, x2_clean=None, X_true=None, R_true=None, t_true=None, cam1=None, true_inlier_mask=None, pred_inlier_mask=None):
    # Eval of pred vs true inliers
    if (true_inlier_mask is not None) and (pred_inlier_mask is not None):
        true = np.asarray(true_inlier_mask, dtype=bool)
        pred = np.asarray(pred_inlier_mask, dtype=bool)
        tp = int(np.sum(true & pred))
        fp = int(np.sum(~true & pred))
        fn = int(np.sum(true & ~pred))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        print(tag, f"Inlier precision={prec:.3f}, recall={rec:.3f}, tp={tp}, fp={fp}, fn={fn}")
    # Masks
    N = x1.shape[1]
    all_mask = np.ones(N, dtype=bool)
    # True mask
    if true_inlier_mask is not None:
        true_inlier_mask = np.asarray(true_inlier_mask, dtype=bool)
        x1_true = x1[:, true_inlier_mask]
        x2_true = x2[:, true_inlier_mask]
    # Evaluation set
    use_pred = (pred_inlier_mask is not None) and (int(np.sum(pred_inlier_mask)) >= 8)
    if not use_pred:
        if pred_inlier_mask is not None:
            print(tag, "RANSAC returned too few inliers; evaluating on all correspondences instead.")
        use_mask = all_mask
    else:
        use_mask = np.asarray(pred_inlier_mask, dtype=bool)
    x1_use = x1[:, use_mask]
    x2_use = x2[:, use_mask]
    # Camera 1
    if cam1 is None:
        cam1 = Camera(K1, np.eye(3), np.zeros(3))
    P1 = cam1.P
    # Pose from F
    try:
        R_est, t_est, E_est, cheir_ratio, best_mask = recover_pose_from_fundamental(F_est, K1, K2, x1_use, x2_use)
    except Exception as e:
        print(tag, "Pose recovery failed:", e)
        return
    # Camera 2
    cam2 = Camera(K2, R_est, t_est)
    P2 = cam2.P
    # Triangulate on eval
    X_est = triangulate_points(P1, P2, x1_use, x2_use)
    # Noisy reprojection
    print(tag, "Cam1 reproj RMSE (noisy):", cam1.reprojection_rmse(X_est, x1_use))
    print(tag, "Cam2 reproj RMSE (noisy):", cam2.reprojection_rmse(X_est, x2_use))
    # Clean reprojection
    if (x1_clean is not None) and (x2_clean is not None) and (true_inlier_mask is not None):
        # Intersection
        clean_mask = use_mask & true_inlier_mask
        if int(np.sum(clean_mask)) >= 8:
            x1_clean_use = x1_clean[:, clean_mask]
            x2_clean_use = x2_clean[:, clean_mask]
            x1_clean_pred = x1[:, clean_mask]
            x2_clean_pred = x2[:, clean_mask]
            # Triangulate on that same subset for comparison
            X_clean = triangulate_points(P1, P2, x1_clean[:, clean_mask], x2_clean[:, clean_mask])
            print(tag, "Cam1 reproj RMSE (clean, true & pred):", cam1.reprojection_rmse(X_clean, x1_clean_use))
            print(tag, "Cam2 reproj RMSE (clean, true & pred):", cam2.reprojection_rmse(X_clean, x2_clean_use))
    # GT 3D RMSE reprojection using estimates cameras
    if (X_true is not None) and (x1_clean is not None) and (x2_clean is not None):
        Xgt = np.asarray(X_true, dtype=float)
        if Xgt.shape == (N, 3):
            Xgt = Xgt.T
        if Xgt.shape[0] == 3 and Xgt.shape[1] == N:
            print(tag, "Cam1 GT-3D reproj RMSE (clean):", cam1.reprojection_rmse(Xgt, x1_clean))
            cam2_gt = Camera(K2, R_true, t_true)
            print(tag, "Cam2 GT-3D reproj RMSE (clean):", cam2_gt.reprojection_rmse(Xgt, x2_clean))
    # Pose error
    if R_true is not None:
        print(tag, "Rotation error (rad):", np.round(angle_between_rotmats(R_est, R_true), 3))
    if t_true is not None:
        print(tag, "Translation error (rad):", np.round(angle_between_translations(t_est, t_true), 3))
    # Sampson RMSE
    print(tag, "Sampson RMSE (all):", np.round(sampson_rmse(x1, x2, F_est), 3))
    if true_inlier_mask is not None:
        print(tag, "Sampson RMSE (true inliers):", np.round(sampson_rmse(x1[:, true_inlier_mask], x2[:, true_inlier_mask], F_est), 3))
    if use_pred:
        print(tag, "Sampson RMSE (pred inliers):", np.round(sampson_rmse(x1_use, x2_use, F_est), 3))

# Generate scene (with noise and outliers)
noise_sigma=2
outlier_ratio=0.2
n_points=50
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
outliers_idx = scene_data["outliers_idx"]
K1 = scene_data["K1"]
K2 = scene_data["K2"]

# --- CAMERA 1 ---
cam1 = Camera(K1, np.eye(3), np.zeros(3))

# --- BASELINE ---
# Baseline deterministic 8-point algo
F_baseline = estimate_fundamental(x1, x2)
# Evaluate model
_evaluate_model("Baseline: ", F_baseline, x1, x2, K1, K2, x1_clean, x2_clean, X_true, R_true, t_true, cam1, true_inlier_mask, pred_inlier_mask=None)

# --- RANSAC ---
# RANSAC 8-point algo
num_trials=2000
threshold_sq=36
F_ransac, ransac_pred_inlier_mask_1, _ = estimate_fundamental_ransac(x1, x2, num_trials, threshold_sq)
n1 = int(ransac_pred_inlier_mask_1.sum())
# Evaluate model
_evaluate_model("RANSAC: ", F_ransac, x1, x2, K1, K2, x1_clean, x2_clean, X_true, R_true, t_true, cam1, true_inlier_mask, ransac_pred_inlier_mask_1)

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
_evaluate_model("Refit RANSAC: ", F_ransac_refit_2, x1, x2, K1, K2, x1_clean, x2_clean, X_true, R_true, t_true, cam1, true_inlier_mask, ransac_pred_inlier_mask_2)
