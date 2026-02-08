# src/features/gradients.py
import math
import numpy as np
from PIL import Image
from features.checks import check_2d_image, check_axis_01, check_kernel_1d_odd, check_2d_pair_same_shape, check_positive

# Image to greyscale
def img_to_grey(img, luminance_weights, eotf_params, assume_srgb=True, normalise_01=True, dtype=np.float64, eps=1e-8):
    # If PIL.Image 
    if isinstance(img, Image.Image): 
        if img.mode == "RGBA": 
            # Drop A from RGBA 
            img = img.convert("RGB")
        # Other supported types
        elif img.mode in {"I;16", "L", "RGB"}: 
            pass
        else: 
            raise ValueError("PIL modes supported only I;16, L, RGB, RGBA")
        # Convert to NumPy
        img = np.array(img)
    # If NumPy array
    elif isinstance(img, np.ndarray): 
        pass
    else: 
        # Not NumPy or accepted PIL.Image
        raise ValueError("Image: Only accepts PIL.Image or NumPy as array")

    # Validate shape
    if img.ndim == 2: 
        # Already greyscale
        pass
    elif img.ndim == 3: 
        if img.shape[2] == 3: 
            pass
        elif img.shape[2] == 4: 
            # Drop A if RGBA
            img = img[:, :, :3]
        else: 
            raise ValueError("NumPy Image must be dim (H,W), (H,W,3) or (H,W,4)")
    else: 
        raise ValueError("NumPy Image must be dim (H,W), (H,W,3) or (H,W,4)")

    # Normalise to [0,1]
    if normalise_01: 
        img = to_unit_interval(img, dtype, eps)

    # Electro-optical transfer function
    if assume_srgb: 
        img = eotf(img, eotf_params, dtype)

    # If 3D, convert to greyscale 
    if img.ndim == 3 and img.shape[2] == 3: 
        img = rgb_to_grey(img, luminance_weights)

    # Sanity check
    if img.ndim != 2:
        raise RuntimeError(f"Expected greyscale output (H,W); got shape {img.shape}")

    # Cast to dtype and return
    return np.array(img, dtype=dtype)

# RGB to greyscale
def rgb_to_grey(img, luminance_weights, dtype=np.float64, eps=1e-8): 
    # Unpack luminance_weights
    wR = luminance_weights["r"]
    wG = luminance_weights["g"]
    wB = luminance_weights["b"]
    # Pack
    w = np.array([wR, wG, wB], dtype=dtype)
    # Check
    s = float(np.sum(w))
    if abs(s - 1) > eps: 
        raise ValueError("Luminance weights do not sum to 1")
    # Dot product
    return np.tensordot(img, w, axes=([-1],[0]))

# Normalise to unit interval [0, 1]
def to_unit_interval(img, dtype=np.float64, eps=1e-8): 
    # Unsigned integers
    if img.dtype in {np.uint8, np.uint16, np.uint32}: 
        # Divide by max representable
        return np.array(img / np.iinfo(img.dtype).max, dtype=dtype)
    # Floats
    elif img.dtype in {np.float32, np.float64}: 
        # Clip to [0,1] (allow tiny tolerance)
        vmax = np.max(img)
        vmin = np.min(img)
        if vmin < -eps or vmax > (1 + eps): 
            raise ValueError("Float image values out of [0,1] bounds")
        else: 
            return np.array(np.clip(img, 0.0, 1.0), dtype=dtype)
        # Raise error if some data is beyond tolerance
    else:
        raise ValueError("Unsupported NumPy dtype, must be unsigned integer or float")

# Electro-optical transfer function 
def eotf(img, eotf_params, dtype=np.float64): 
    # Unpack
    b = float(eotf_params["breakpoint"])
    d = float(eotf_params["linear_divisor"])
    a = float(eotf_params["offset"]) 
    g = float(eotf_params["gamma"]) 
    # Copy
    img_copy = np.asarray(img, dtype=dtype).copy()
    # Linear part
    lin = img_copy <= b
    img_copy[lin] = img_copy[lin] / d
    # Non-linear
    img_copy[~lin] = ((img_copy[~lin] + a) / (1 + a))**g
    # Return
    return img_copy

# Gaussian kernel 1D
def gaussian_kernel_1d(sigma, truncate=3.0, dtype=np.float64, eps=1e-8):
    # Validate sigma > 0
    sigma = check_positive(sigma, "sigma", eps)
    # Compute radius
    r = math.ceil(truncate * sigma)
    # Build integer support x = [-r, ..., 0, ..., +r]
    x = np.arange(-r, r+1)
    # Compute kernel g(x)
    g = np.exp(-x**2 / (2*sigma**2))
    # Normalise g to 1
    s = np.sum(g)
    g = g / s
    # Return as NumPy array in dtype
    return np.array(g, dtype=dtype)

# Gaussian derivative kernel 1D
def gaussian_derivative_kernel_1d(sigma, truncate=3.0, dtype=np.float64, eps=1e-8): 
    # Validate sigma > 0
    sigma = check_positive(sigma, "sigma", eps)
    # Compute radius
    r = math.ceil(truncate * sigma)
    # Build integer support x = [-r, ..., 0, ..., +r]
    x = np.arange(-r, r+1)
    # Compute kernel g(x)
    g = np.exp(-x**2 / (2*sigma**2))
    # Normalise g to 1
    s = np.sum(g)
    g = g / s
    # Compute derivative dg(x)
    dg = -(x / sigma**2) * g
    # Zero sum of gradients
    dg -= dg.mean()
    # Return as NumPy array in dtype
    return np.array(dg, dtype=dtype)

# Pad 1D axis
def pad_1d_axis(im, pad, axis, border_mode, constant_value=0.0): 
    # Check 2D image
    im = check_2d_image(im)
    # Check axis
    axis = check_axis_01(axis)
    # Check pad length
    if not isinstance(pad, (int, np.integer)):
        raise ValueError(f"pad must be an integer; got {type(pad)}")
    pad = int(pad)
    if pad < 0:
        raise ValueError(f"pad must be >= 0; got {pad}")
    if pad == 0:
        return im
    # Border mode
    mode = str(border_mode).lower()
    if mode not in {"reflect", "constant", "edge"}:
        raise ValueError(f"Unsupported padding mode '{mode}'. Use either: reflect, constant, or edge.")
    # Build pad widths: pad only along chosen axis
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad, pad)
    # Constant mode
    if mode == "constant":
        return np.pad(im, pad_width=pad_width, mode="constant", constant_values=constant_value)
    # Reflect or edge mode
    else:
        return np.pad(im, pad_width=pad_width, mode=mode)

# Correlation 1D
def correlate1d(im, k, axis, border_mode="reflect", constant_value=0.0, dtype=np.float64):
    # Checks
    im = check_2d_image(im)
    axis = check_axis_01(axis)
    k = check_kernel_1d_odd(k)
    k = np.asarray(k, dtype=dtype)
    # Pad
    pad = k.size // 2
    pim = pad_1d_axis(im, pad=pad, axis=axis, border_mode=border_mode, constant_value=constant_value)
    # Pre-allocate
    out = np.empty_like(im, dtype=dtype)
    L = k.size
    # Correlate
    if axis == 1:
        # Slide along x (columns)
        for j in range(im.shape[1]):
            window = pim[:, j:j + L]
            out[:, j] = window @ k
    else:
        # Slide along y (rows)
        for i in range(im.shape[0]):
            window = pim[i:i + L, :] 
            out[i, :] = (k[:, None] * window).sum(axis=0)

    return out

# Convolution 1D
def convolve1d(im, k, axis, border_mode="reflect", constant_value=0.0, dtype=np.float64):
    # Flip kernel
    k = np.asarray(k)
    k_flipped = k[::-1]
    return correlate1d(im, k_flipped, axis, border_mode=border_mode, constant_value=constant_value, dtype=dtype)

# Separable filter
def separable_filter(im, ky, kx, border_mode="reflect", constant_value=0.0, dtype=np.float64):
    # Checks
    im = check_2d_image(im)
    ky = check_kernel_1d_odd(ky, "ky")
    kx = check_kernel_1d_odd(kx, "kx")
    # First pass: y-direction (rows)
    tmp = convolve1d(im, ky, axis=0, border_mode=border_mode, constant_value=constant_value, dtype=dtype)
    # Second pass: x-direction (cols)
    out = convolve1d(tmp, kx, axis=1, border_mode=border_mode, constant_value=constant_value, dtype=dtype)
    return out

# Derivative-of-Gaussian (dog) gradients
def gradients_dog(im, sigma_d, truncate=3.0, border_mode="reflect", constant_value=0.0, dtype=np.float64, eps=1e-8,):
    # Checks
    im = check_2d_image(im)
    sigma_d = check_positive(sigma_d, "sigma_d", eps)
    # Cast image to working dtype
    im = np.asarray(im, dtype=dtype)

    # Build Gaussian and Gaussian-derivative kernels (1D, odd-length)
    g  = gaussian_kernel_1d(sigma_d, truncate=truncate, dtype=dtype, eps=eps)
    dg = gaussian_derivative_kernel_1d(sigma_d, truncate=truncate, dtype=dtype, eps=eps)

    # Apply separable derivative-of-Gaussian:
    # Ix = d/dx (G * I)  -> smooth in y with G, differentiate in x with dG
    Ix = separable_filter(im, ky=g,  kx=dg, border_mode=border_mode, constant_value=constant_value, dtype=dtype)
    # Iy = d/dy (G * I)  -> differentiate in y with dG, smooth in x with G
    Iy = separable_filter(im, ky=dg, kx=g,  border_mode=border_mode, constant_value=constant_value, dtype=dtype)

    return Ix, Iy

# Sobel gradients
def gradients_sobel(im, sigma_d, sobel_ksize=3, sobel_scale=0.125, truncate=3.0, border_mode="reflect", constant_value=0.0, dtype=np.float64, eps=1e-8):
    # Checks
    im = check_2d_image(im)
    im = np.asarray(im, dtype=dtype)
    if sobel_ksize != 3:
        raise ValueError(f"Only sobel ksize=3 is supported; got {sobel_ksize}")

    # Pre-smoothing
    if sigma_d is not None and float(sigma_d) > eps:
        g = gaussian_kernel_1d(sigma_d, truncate=truncate, dtype=dtype, eps=eps)
        im = separable_filter(im, ky=g, kx=g, border_mode=border_mode, constant_value=constant_value, dtype=dtype)

    # Separable Sobel kernels
    s = np.array([1.0, 2.0, 1.0], dtype=dtype)
    d = np.array([-1.0, 0.0, 1.0], dtype=dtype)

    # Gx = s(y) * d(x)
    tmp = correlate1d(im, s, axis=0, border_mode=border_mode, constant_value=constant_value, dtype=dtype)
    Ix  = correlate1d(tmp, d, axis=1, border_mode=border_mode, constant_value=constant_value, dtype=dtype)

    # Gy = d(y) * s(x)
    tmp = correlate1d(im, d, axis=0, border_mode=border_mode, constant_value=constant_value, dtype=dtype)
    Iy  = correlate1d(tmp, s, axis=1, border_mode=border_mode, constant_value=constant_value, dtype=dtype)

    # Scale outputs so magnitudes are comparable to central differences
    Ix *= float(sobel_scale)
    Iy *= float(sobel_scale)

    return Ix, Iy

# Magnitude of gradient
def gradient_magnitude(Ix, Iy, dtype=np.float64, eps=1e-8):
    # Checks
    Ix, Iy = check_2d_pair_same_shape(Ix, Iy, "Ix", "Iy")
    Ix = np.asarray(Ix, dtype=dtype)
    Iy = np.asarray(Iy, dtype=dtype)
    # Magnitude
    mag = np.hypot(Ix, Iy)
    if eps is not None and eps > 0:
        mag = np.maximum(mag, eps)
    return mag

# Orientation of gradient
def gradient_orientation(Ix, Iy, dtype=np.float64):
    # Checks
    Ix, Iy = check_2d_pair_same_shape(Ix, Iy, "Ix", "Iy")
    Ix = np.asarray(Ix, dtype=dtype)
    Iy = np.asarray(Iy, dtype=dtype)
    # Angle (radians)
    ori = np.arctan2(Iy, Ix)
    return ori

# Compute image gradients
def compute_image_gradients(img, cfg):
    # --- Unpack Configs ---
    # Unpack image config
    img_cfg = cfg["image"]
    eps = float(img_cfg.get("eps", 1e-8))
    dtype_name = img_cfg.get("dtype", "float64")
    dtype = getattr(np, dtype_name) if isinstance(dtype_name, str) else dtype_name
    luminance_weights = img_cfg["luminance_weights"]
    assume_srgb = bool(img_cfg.get("assume_srgb", True))
    normalise_01 = bool(img_cfg.get("normalise_01", True))
    eotf_params = img_cfg.get("eotf", {})
    # Unpack border config
    border_cfg = cfg.get("border", {})
    border_mode = border_cfg.get("mode", "reflect")
    constant_value = float(border_cfg.get("constant_value", 0.0))
    # Unpack gradients config
    gcfg = cfg["gradients"]
    method = str(gcfg.get("method", "dog")).lower()
    gauss_cfg = gcfg.get("gaussian", {})
    truncate = float(gauss_cfg.get("truncate", 3.0))

    # --- Return Params ---
    ret_cfg = gcfg.get("return", {})
    want_Ix = bool(ret_cfg.get("Ix", True))
    want_Iy = bool(ret_cfg.get("Iy", True))
    want_mag = bool(ret_cfg.get("mag", True))
    want_ori = bool(ret_cfg.get("ori", True))

    # Convert input to greyscale float image
    im = img_to_grey(
        img,
        luminance_weights=luminance_weights,
        eotf_params=eotf_params,
        assume_srgb=assume_srgb,
        normalise_01=normalise_01,
        dtype=dtype,
        eps=eps)
    im = check_2d_image(im, "im")
    im = np.asarray(im, dtype=dtype)

    # Compute Ix, Iy
    if method == "dog":
        dog_cfg = gcfg.get("dog", {})
        sigma_d = float(dog_cfg.get("sigma_d", 1.0))
        Ix, Iy = gradients_dog(
            im,
            sigma_d=sigma_d,
            truncate=truncate,
            border_mode=border_mode,
            constant_value=constant_value,
            dtype=dtype,
            eps=eps)
    elif method == "sobel":
        sob_cfg = gcfg.get("sobel", {})
        sigma_pre = sob_cfg.get("sigma_pre", 1.0)
        ksize = int(sob_cfg.get("ksize", 3))
        scale = float(sob_cfg.get("scale", 0.125))
        Ix, Iy = gradients_sobel(
            im,
            sigma_d=sigma_pre,
            sobel_ksize=ksize,
            sobel_scale=scale,
            truncate=truncate,
            border_mode=border_mode,
            constant_value=constant_value,
            dtype=dtype,
            eps=eps)
    else:
        raise ValueError(f"Unknown gradients method '{method}'. Use 'dog' or 'sobel'.")

    # Package outputs
    out = {}

    if want_Ix:
        out["Ix"] = Ix
    if want_Iy:
        out["Iy"] = Iy

    if want_mag or want_ori:
        # Compute once; reuse
        if want_mag:
            out["mag"] = gradient_magnitude(Ix, Iy, dtype=dtype, eps=eps)
        if want_ori:
            out["ori"] = gradient_orientation(Ix, Iy, dtype=dtype)

    return out
