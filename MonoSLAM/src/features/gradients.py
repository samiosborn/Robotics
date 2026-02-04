# src/features/gradients.py
import numpy as np
from PIL import Image

# Image to greyscale
def img_to_grey(img, luminance_weights, assume_srgb=True, normalise_01=True, dtype=np.float64, eps=1e-8):
    # If PIL.Image 
    if isinstance(img, Image.Image): 
        if img.mode == "RGBA": 
            # Drop A from RGBA 
            img = img.convert("RGB")
        # Convert to NumPy
        img = np.array(img)
    # If NumPy array
    if isinstance(img, np.array): 
        # Validate shape
        if img.ndim == 2: 
            # Fine
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
        # Inverse gamma
        if assume_srgb: 
            img = inv_gamma(img, dtype)
        # If 3D, convert to greyscale 
        if img.ndim == 3 and img.shape[2] == 3: 
            img = rgb_to_grey(img, luminance_weights)
        # Cast to float64
        img = np.array(img, dtype=dtype)
        # Return 2D array (H, W)
        return img
    else: 
        # Not NumPy or accepted PIL.Image
        raise ValueError("Image: Only accepts PIL.Image or NumPy as array")

# RGB to greyscale
def rgb_to_grey(img, luminance_weights): 
    # Dot product
    raise NotImplementedError

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
            raise ValueError("Image dtype float64 values are out of [0, 1] bounds")
        else: 
            return np.array(np.min(1, np.max(0, img)), dtype=dtype)
        # Raise error if some data is beyond tolerance
    else:
        raise ValueError("Unsupported NumPy dtype, must be unsigned integer or float")

# Inverse gamma 
def inv_gamma(img, dtype=np.float64): 
    # Per channel inverse gamma 
    raise NotImplementedError

# Gaussian kernel 1D
def gaussian_kernel_1d(sigma, truncate=3.0, force_odd=True):
    # Validate sigma > 0
    # Compute radius = ceil(truncate * sigma)
    # Build x = [-radius, ..., +radius]
    # Compute kernel g[x] = exp(-x^2/(2*sigma^2))
    # Normalise sum(g)=1
    # Return g as float64 1D array
    raise NotImplementedError

# Gaussian derivative kernel 1D
def gaussian_derivative_kernel_1d(sigma, truncate=3.0, force_odd=True):
    # Validate sigma > 0
    # Build same support x as gaussian_kernel_1d
    # Compute derivative kernel dg[x] = -(x/sigma^2) * g[x]
    # Enforce sum(dg)=0 numerically
    # Return dg as float64 1D array
    raise NotImplementedError

# Convolution 1D
def convolve1d(im, k, axis, border_mode="reflect"):
    # Pad im along axis
    # Flip kernel
    # Perform 1D convolution along axis (axis=0 for y, axis=1 for x)
    # Return filtered image same shape as im
    raise NotImplementedError

# Correlation 1D
def correlate1d(im, k, axis, border_mode="reflect"): 
    # Pad im along axis
    # Perform 1D correlation along axis (axis=0 for y, axis=1 for x)
    # Return filtered image same shape as im
    raise NotImplementedError

# Separable filter
def separable_filter(im, ky, kx, border_mode="reflect"):
    # Apply convolve1d along y with ky then along x with kx
    # Apply padding for both passes
    # Return filtered image
    raise NotImplementedError

# Derivative-of-Gaussian (dog) gradients
def gradients_dog(im, sigma_d, truncate=3.0, border_mode="reflect"):
    # Build g = gaussian_kernel_1d(sigma_d)
    # Build dg = gaussian_derivative_kernel_1d(sigma_d)
    # Compute Ix = separable_filter(im, ky=g, kx=dg)
    # Compute Iy = separable_filter(im, ky=dg, kx=g)
    # Return Ix, Iy
    raise NotImplementedError

# Sobel gradients
def gradients_sobel(im, sigma_d, sobel_ksize=3, sobel_scale=0.125, truncate=3.0, border_mode="reflect"):
    # Smooth image with Gaussian at sigma_d using separable_filter (g,g)
    # Apply Sobel kernels (prefer separable form: [1 2 1] and [-1 0 1]) with padding
    # Scale outputs so magnitudes are comparable to central differences
    # Return Ix, Iy
    raise NotImplementedError

# Magnitude of gradient
def gradient_magnitude(Ix, Iy, eps=1e-8):
    # Compute mag = sqrt(Ix^2 + Iy^2)
    # Return mag
    raise NotImplementedError

# Orientation of gradient 
def gradient_orientation(Ix, Iy):
    # Compute ori = arctan2(Iy, Ix) in radians (range [-pi, pi])
    # Return ori
    raise NotImplementedError

# Compute image gradients
def compute_image_gradients(img, cfg):
    # Unpack cfg
    # Convert PIL input to greyscale
    # Implement padding
    # Calc gradients with magnitude and orientation
    # Return an output dict (Ix, Iy, mag, ori)
    raise NotImplementedError
