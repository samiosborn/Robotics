# src/features/gradients.py
import numpy as np

# Compute image gradients
def compute_image_gradients(img, cfg):
    # Unpack cfg
    # Convert PIL input to greyscale
    # Implement padding
    # Calc gradients with magnitude and orientation
    # Return an output dict (Ix, Iy, mag, ori)
    raise NotImplementedError

# Image to greyscale
def img_to_grey(img, luminance_weights, normalise_01=True, dtype=np.float64):
    # Accept PIL.Image (either 2D, RGB, RGBA, I;16)
    # Convert to greyscale (float64 in [0,1] using luminance_weights)
    # Correct datatype
    # Normalise
    # Return 2D array (H, W)
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
def gradient_magnitude(Ix, Iy, eps=1e-12):
    # Compute mag = sqrt(Ix^2 + Iy^2)
    # Return mag
    raise NotImplementedError

# Orientation of gradient 
def gradient_orientation(Ix, Iy):
    # Compute ori = arctan2(Iy, Ix) in radians (range [-pi, pi])
    # Return ori
    raise NotImplementedError
