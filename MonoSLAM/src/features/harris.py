# src/features/harris.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from features.checks import check_2d_image, check_2d_pair_same_shape, check_positive
from features.gradients import compute_image_gradients, gaussian_kernel_1d, separable_filter
from features.nms import nms_2d

# Result of Harris
@dataclass(frozen=True)
class HarrisResult:
    # Keypoints [x, y, score]
    kps: np.ndarray
    # Response map (optional)
    response: np.ndarray | None


# Structure tensor of smoothed outer-products
def structure_tensor(Ix, Iy, sigma_i, truncate, border_mode, constant_value, dtype=np.float64, eps=1e-8):

    # --- CHECKS ---
    # Validate 2D gradients
    Ix = check_2d_image(Ix, "Ix")
    Iy = check_2d_image(Iy, "Iy")
    # Validate same shape
    Ix, Iy = check_2d_pair_same_shape(Ix, Iy, "Ix", "Iy")
    # Validate integration sigma
    sigma_i = check_positive(sigma_i, "sigma_i", eps)
    # Validate truncate
    truncate = check_positive(truncate, "truncate", eps)
    # Cast gradients to dtype
    Ix = np.asarray(Ix, dtype=dtype)
    Iy = np.asarray(Iy, dtype=dtype)

    # Outer-product terms
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Build Gaussian kernel for integration smoothing
    g = gaussian_kernel_1d(sigma_i, truncate=truncate, dtype=dtype, eps=eps)

    # Smooth tensor terms
    Sxx = separable_filter(Ixx, ky=g, kx=g, border_mode=border_mode, constant_value=constant_value, dtype=dtype)
    Syy = separable_filter(Iyy, ky=g, kx=g, border_mode=border_mode, constant_value=constant_value, dtype=dtype)
    Sxy = separable_filter(Ixy, ky=g, kx=g, border_mode=border_mode, constant_value=constant_value, dtype=dtype)

    return Sxx, Syy, Sxy


# Harris response
def harris_response(Sxx, Syy, Sxy, k, dtype=np.float64):

    # --- CHECKS ---
    # Validate 2D
    Sxx = check_2d_image(Sxx, "Sxx")
    Syy = check_2d_image(Syy, "Syy")
    Sxy = check_2d_image(Sxy, "Sxy")
    # Validate shapes match
    Sxx, Syy = check_2d_pair_same_shape(Sxx, Syy, "Sxx", "Syy")
    Sxx, Sxy = check_2d_pair_same_shape(Sxx, Sxy, "Sxx", "Sxy")
    # Validate k
    if (not np.isfinite(k)) or (k <= 0.0):
        raise ValueError(f"k must be finite and > 0, got {k}")
    # Cast to dtype
    Sxx = np.asarray(Sxx, dtype=dtype)
    Syy = np.asarray(Syy, dtype=dtype)
    Sxy = np.asarray(Sxy, dtype=dtype)

    # Determinant
    det = (Sxx * Syy) - (Sxy * Sxy)

    # Trace
    tr = Sxx + Syy

    # Harris response: R = det(M) - k * trace(M)^2
    R = det - (float(k) * (tr * tr))

    return R


# Shi-Tomasi score
def shi_tomasi_score(Sxx, Syy, Sxy, dtype=np.float64):

    # --- CHECKS ---
    # Validate 2D
    Sxx = check_2d_image(Sxx, "Sxx")
    Syy = check_2d_image(Syy, "Syy")
    Sxy = check_2d_image(Sxy, "Sxy")
    # Validate shapes match
    Sxx, Syy = check_2d_pair_same_shape(Sxx, Syy, "Sxx", "Syy")
    Sxx, Sxy = check_2d_pair_same_shape(Sxx, Sxy, "Sxx", "Sxy")
    # Cast to dtype
    Sxx = np.asarray(Sxx, dtype=dtype)
    Syy = np.asarray(Syy, dtype=dtype)
    Sxy = np.asarray(Sxy, dtype=dtype)

    # Trace
    tr = Sxx + Syy

    # Determinant
    det = (Sxx * Syy) - (Sxy * Sxy)

    # Discriminant tr^2 - 4 det (clamp for numerical safety)
    disc = np.maximum((tr * tr) - (4.0 * det), 0.0)

    # Minimum eigenvalue of 2x2 structure tensor
    lam_min = 0.5 * (tr - np.sqrt(disc))

    return lam_min


# Harris keypoints from gradients
def harris_keypoints_from_gradients(
    Ix,
    Iy,
    *,
    method="harris",
    sigma_i=1.5,
    k=0.04,
    truncate=3.0,
    threshold_type="relative",
    threshold_rel=0.01,
    nms_window=3,
    border_margin=16,
    max_keypoints=1000,
    border_mode="reflect",
    constant_value=0.0,
    dtype=np.float64,
    eps=1e-8,
    return_response=False,
) -> HarrisResult:

    # --- CHECKS ---
    # Validate 2D gradients
    Ix = check_2d_image(Ix, "Ix")
    Iy = check_2d_image(Iy, "Iy")
    # Validate same shape
    Ix, Iy = check_2d_pair_same_shape(Ix, Iy, "Ix", "Iy")
    # Validate sigma_i
    sigma_i = check_positive(sigma_i, "sigma_i", eps)
    # Validate truncate
    truncate = check_positive(truncate, "truncate", eps)
    # Validate threshold type
    if str(threshold_type).lower() != "relative":
        raise ValueError(f"Only threshold.type == 'relative' is supported; got {threshold_type}")
    # Validate threshold_rel
    if (not np.isfinite(threshold_rel)) or (threshold_rel <= 0.0):
        raise ValueError(f"threshold_rel must be finite and > 0, got {threshold_rel}")
    # Validate NMS window (odd >= 3)
    if (not isinstance(nms_window, (int, np.integer))) or (int(nms_window) < 3) or ((int(nms_window) % 2) != 1):
        raise ValueError(f"nms_window must be odd int >= 3, got {nms_window}")
    nms_window = int(nms_window)
    # Validate border margin
    if (not isinstance(border_margin, (int, np.integer))) or (int(border_margin) < 0):
        raise ValueError(f"border_margin must be int >= 0, got {border_margin}")
    border_margin = int(border_margin)
    # Validate max_keypoints
    if (not isinstance(max_keypoints, (int, np.integer))) or (int(max_keypoints) <= 0):
        raise ValueError(f"max_keypoints must be int > 0, got {max_keypoints}")
    max_keypoints = int(max_keypoints)

    # Compute structure tensor (smoothed)
    Sxx, Syy, Sxy = structure_tensor(
        Ix,
        Iy,
        sigma_i=sigma_i,
        truncate=truncate,
        border_mode=border_mode,
        constant_value=constant_value,
        dtype=dtype,
        eps=eps)

    # Get method
    m = str(method).strip().lower()

    # Compute score map
    if m == "harris":
        score = harris_response(Sxx, Syy, Sxy, k=k, dtype=dtype)
    elif m == "shi_tomasi":
        score = shi_tomasi_score(Sxx, Syy, Sxy, dtype=dtype)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'harris' or 'shi_tomasi'.")

    # Clamp to positive for relative thresholding stability
    score_pos = np.maximum(score, 0.0)

    # Compute max positive score
    max_pos = float(np.max(score_pos))

    # Early exit if no positive responses
    if max_pos <= 0.0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return HarrisResult(kps=empty, response=score if return_response else None)

    # Compute threshold as fraction of max positive response
    thr = float(threshold_rel) * max_pos

    # Convert NMS window size to radius
    radius = nms_window // 2

    # Run NMS to get top candidates
    nms = nms_2d(
        score_map=score_pos,
        radius=radius,
        threshold=thr,
        border_margin=border_margin,
        max_points=max_keypoints)

    # Early exit if NMS returns nothing
    if nms.scores.size == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return HarrisResult(kps=empty, response=score if return_response else None)

    # Pack 
    kps = np.stack(
        [
            nms.xs.astype(np.float64, copy=False),
            nms.ys.astype(np.float64, copy=False),
            nms.scores.astype(np.float64, copy=False),
        ],
        axis=1)

    return HarrisResult(kps=kps, response=score if return_response else None)


# Harris keypoints (from image)
def harris_keypoints(img, cfg) -> HarrisResult:

    # Validate cfg
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dict")

    # --- Unpack Configs ---
    # Unpack image config
    img_cfg = cfg["image"]
    eps = float(img_cfg.get("eps", 1e-8))
    dtype_name = img_cfg.get("dtype", "float64")
    dtype = getattr(np, dtype_name) if isinstance(dtype_name, str) else dtype_name
    # Unpack border config
    border_cfg = cfg.get("border", {})
    border_mode = border_cfg.get("mode", "reflect")
    constant_value = float(border_cfg.get("constant_value", 0.0))
    # Unpack gradients gaussian truncate
    gcfg = cfg["gradients"]
    gauss_cfg = gcfg.get("gaussian", {})
    truncate = float(gauss_cfg.get("truncate", 3.0))
    # Unpack harris config
    hcfg = cfg["harris"]
    method = str(hcfg.get("method", "harris"))
    sigma_i = float(hcfg.get("sigma_i", 1.5))
    k = float(hcfg.get("k", 0.04))
    # Unpack threshold block
    th_cfg = hcfg.get("threshold", {})
    threshold_type = str(th_cfg.get("type", "relative"))
    threshold_rel = float(th_cfg.get("rel", 0.01))
    # Unpack NMS block
    nms_cfg = hcfg.get("nms", {})
    nms_window = int(nms_cfg.get("window", 3))
    # Unpack keypoint selection params
    border_margin = int(hcfg.get("border_margin", 16))
    max_keypoints = int(hcfg.get("max_keypoints", 1000))
    # Debug output (optional)
    return_response = bool(hcfg.get("return_response", False))

    # Compute gradients
    g = compute_image_gradients(img, cfg)

    # Enforce Harris requirements (Ix and Iy must be present)
    if ("Ix" not in g) or ("Iy" not in g):
        raise ValueError("Harris requires gradients Ix and Iy. Set cfg['gradients']['return']['Ix']=true and ['Iy']=true.")
    
    # Extract gradients
    Ix = g["Ix"]
    Iy = g["Iy"]

    # Run Harris keypoints detection from gradients calculated
    return harris_keypoints_from_gradients(
        Ix,
        Iy,
        method=method,
        sigma_i=sigma_i,
        k=k,
        truncate=truncate,
        threshold_type=threshold_type,
        threshold_rel=threshold_rel,
        nms_window=nms_window,
        border_margin=border_margin,
        max_keypoints=max_keypoints,
        border_mode=border_mode,
        constant_value=constant_value,
        dtype=dtype,
        eps=eps,
        return_response=return_response)
