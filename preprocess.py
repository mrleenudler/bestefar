"""
Preprocessing functions for image downscaling, grayscale conversion, blur, and gradient computation.
"""

import cv2
import numpy as np


def downscale_max_side(img_bgr, max_side):
    """
    Downscale image so that max side is <= max_side.
    
    Args:
        img_bgr: Input image (BGR or grayscale)
        max_side: Maximum side length in pixels
    
    Returns:
        img_ds: Downscaled image
        scale: Scale factor (1.0 if no downscaling)
    """
    h, w = img_bgr.shape[:2]
    max_dim = max(h, w)
    scale = max_side / max_dim if max_dim > max_side else 1.0
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_ds = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_ds = img_bgr.copy()
    
    return img_ds, scale


def to_gray(img_bgr):
    """
    Convert BGR image to grayscale.
    
    Args:
        img_bgr: Input image (BGR or grayscale)
    
    Returns:
        gray: Grayscale image
    """
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    return gray


def gaussian_blur(gray, sigma):
    """
    Apply Gaussian blur to grayscale image.
    
    Args:
        gray: Grayscale image
        sigma: Gaussian blur sigma
    
    Returns:
        blur: Blurred image
    """
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    return blur


def compute_gradients(blur):
    """
    Compute Scharr gradients and magnitude.
    
    Args:
        blur: Blurred grayscale image
    
    Returns:
        gx: Gradient in x direction (float32)
        gy: Gradient in y direction (float32)
        mag: Gradient magnitude (float32)
        ux: Normalized gradient x (unit vector)
        uy: Normalized gradient y (unit vector)
    """
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    
    eps = 1e-6
    mag_safe = mag + eps
    ux = gx / mag_safe
    uy = gy / mag_safe
    
    return gx, gy, mag, ux, uy


def suppress_axis_normals(ux, uy, mag, thresh_deg):
    """
    Suppress gradients that are nearly horizontal or vertical.
    
    Args:
        ux, uy: Normalized gradient components
        mag: Gradient magnitude
        thresh_deg: Threshold in degrees (±N degrees from horizontal/vertical)
    
    Returns:
        mag_suppressed: Magnitude with suppressed axis-aligned gradients set to 0
    """
    if thresh_deg <= 0:
        return mag.copy()
    
    angle_threshold_rad = np.deg2rad(thresh_deg)
    angle_threshold_tan = np.tan(angle_threshold_rad)
    eps = 1e-6
    
    # Check if gradient is nearly horizontal or vertical
    nearly_horizontal = np.abs(uy) / (np.abs(ux) + eps) < angle_threshold_tan
    nearly_vertical = np.abs(ux) / (np.abs(uy) + eps) < angle_threshold_tan
    
    mag_suppressed = mag.copy()
    mag_suppressed[nearly_horizontal | nearly_vertical] = 0.0
    
    return mag_suppressed

