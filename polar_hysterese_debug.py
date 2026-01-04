"""
Polar transform + hysteresis ring detection prototype (debug only).
This module is isolated and does not affect the main pipeline.
"""

import cv2
import numpy as np
import time
from pathlib import Path

import preprocess
import nms
import debug_tools


def compute_mag_and_nms(gray_u8, cfg):
    """
    Compute gradient magnitude and NMS magnitude from grayscale image.
    
    Args:
        gray_u8: Grayscale image (uint8)
        cfg: Config dictionary
    
    Returns:
        mag_raw_f32: Raw gradient magnitude (float32)
        mag_nms_f32: NMS-thinned gradient magnitude (float32)
    """
    # Edge blur
    blur_sigma = cfg.get('outer_circle_blur_sigma', 2.0)
    blur = preprocess.gaussian_blur(gray_u8, blur_sigma)
    
    # Compute gradients
    gx, gy, mag_raw, ux, uy = preprocess.compute_gradients(blur)
    
    # NMS
    mag_nms = nms.nms_gradient_magnitude(gx, gy, mag_raw)
    
    return mag_raw, mag_nms


def warp_polar(image_f32, center_xy, r_max, theta_samples, r_samples):
    """
    Warp image to polar coordinates.
    
    Args:
        image_f32: Input image (float32)
        center_xy: Center point (cx, cy)
        r_max: Maximum radius
        theta_samples: Number of angle samples (columns)
        r_samples: Number of radius samples (rows)
    
    Returns:
        polar_f32: Polar image (float32), shape (r_samples, theta_samples)
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dsize = (theta_samples, r_samples)  # (width, height) = (theta, radius)
    
    polar = cv2.warpPolar(
        image_f32,
        dsize,
        center=(cx, cy),
        maxRadius=r_max,
        flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR
    )
    
    return polar


def radial_profile_from_polar(P, agg='sum', smooth_sigma=3.0):
    """
    Compute radial profile (1D histogram) from polar image.
    
    Args:
        P: Polar image (float32), shape (r_samples, theta_samples) or (theta_samples, r_samples)
        agg: Aggregation method ('sum' or 'mean')
        smooth_sigma: Gaussian smoothing sigma (in bins)
    
    Returns:
        H: Radial profile (1D float array)
        is_r_first: True if radius axis is first dimension
    """
    # Detect orientation: assume r_samples is typically larger than theta_samples
    # or check if first dimension matches expected r_samples
    # For now, assume standard: (r_samples, theta_samples) = (rows, cols)
    # where rows = radius, cols = theta
    
    if P.shape[0] > P.shape[1]:
        # Likely (r_samples, theta_samples): radius is axis 0
        is_r_first = True
        H_raw = np.sum(P, axis=1) if agg == 'sum' else np.mean(P, axis=1)
    else:
        # Likely (theta_samples, r_samples): radius is axis 1
        is_r_first = False
        H_raw = np.sum(P, axis=0) if agg == 'sum' else np.mean(P, axis=0)
    
    # Smooth with 1D Gaussian
    if smooth_sigma > 0:
        try:
            from scipy import ndimage
            H = ndimage.gaussian_filter1d(H_raw.astype(np.float64), sigma=smooth_sigma).astype(np.float32)
        except ImportError:
            # Fallback: use OpenCV GaussianBlur (1D approximation)
            # Reshape to 2D for cv2.GaussianBlur, then reshape back
            H_2d = H_raw.reshape(1, -1).astype(np.float32)
            H_2d_smooth = cv2.GaussianBlur(H_2d, (0, 0), sigmaX=smooth_sigma, sigmaY=0)
            H = H_2d_smooth.ravel().astype(np.float32)
    else:
        H = H_raw
    
    return H, is_r_first


def pick_outer_peak(H, ignore_right_margin_bins=10):
    """
    Pick the outermost (rightmost) peak in radial profile.
    
    Args:
        H: Radial profile (1D array)
        ignore_right_margin_bins: Number of bins to ignore at the right edge
    
    Returns:
        r_peak_idx: Index of the peak
    """
    if len(H) <= ignore_right_margin_bins:
        ignore_right_margin_bins = 0
    
    H_trimmed = H[:-ignore_right_margin_bins] if ignore_right_margin_bins > 0 else H
    r_peak_idx = int(np.argmax(H_trimmed))
    
    return r_peak_idx


def polar_nms_and_hysteresis(P_band, cfg):
    """
    Apply NMS and hysteresis in polar coordinates.
    
    Args:
        P_band: Polar magnitude image restricted to a radius band (float32)
        cfg: Config dictionary
    
    Returns:
        edge_mask_band: Binary mask of detected edges (bool array, same shape as P_band)
    """
    # Compute gradients in polar coordinates
    # Gx: gradient along theta (horizontal in polar image)
    # Gy: gradient along radius (vertical in polar image)
    gx = cv2.Sobel(P_band, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(P_band, cv2.CV_32F, 0, 1, ksize=3)
    
    # NMS on polar gradients
    P_thin = nms.nms_gradient_magnitude(gx, gy, P_band)
    
    # Hysteresis thresholding
    # Find thresholds within band where P_thin > 0
    vals = P_thin[P_thin > 0]
    if len(vals) == 0:
        return np.zeros_like(P_band, dtype=bool)
    
    high_percentile = cfg.get('polar_hyst_high_percentile', 98.0)
    low_frac = cfg.get('polar_hyst_low_frac', 0.5)
    
    t_high = np.percentile(vals, high_percentile)
    t_low = low_frac * t_high
    
    # Strong and weak masks
    strong_mask = (P_thin >= t_high)
    weak_mask = (P_thin >= t_low) & (P_thin < t_high)
    
    # Hysteresis: keep weak components that are 8-connected to strong
    weak_uint8 = weak_mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(weak_uint8, connectivity=8)
    
    # Find labels that intersect with strong mask
    strong_labels = np.unique(labels[strong_mask])
    strong_labels = strong_labels[strong_labels > 0]  # Exclude background (label 0)
    
    # Build edge mask: strong edges + weak edges connected to strong
    edge_mask = np.isin(labels, strong_labels)
    edge_mask = edge_mask | strong_mask  # Include all strong edges
    
    return edge_mask


def ridge_tracking_polar(P, r_pass1_peak_px, r_max, cfg):
    """
    Ridge tracking in polar image: start at top row at r_pass1_peak_px, track downward.
    In polar image: ringene er VERTIKALE => radius varierer langs X (kolonner), theta langs Y (rader).
    
    Args:
        P: Polar magnitude image (float32), shape (theta_rows, radius_cols) from cv2.warpPolar
           theta_rows = høyde (Y), radius_cols = bredde (X)
        r_pass1_peak_px: Pass 1 peak radius in pixels (downscaled coordinates)
        r_max: Maximum radius used in warpPolar
        cfg: Config dictionary
    
    Returns:
        ridge_mask: Binary mask (uint8) with ridge points set to 255, shape (theta_rows, radius_cols)
        ridge_points: List of (y, x) tuples for ridge points (y=theta_row, x=radius_col)
    """
    # Correct axis interpretation: P.shape = (theta_rows, radius_cols) = (høyde, bredde)
    theta_rows, radius_cols = P.shape
    
    # Convert r_pass1_peak_px to radius column index x0 (NOT y0)
    # In polar: x = (r / r_max) * (radius_cols - 1) maps radius to column
    x0 = int(round((r_pass1_peak_px / r_max) * (radius_cols - 1)))
    x0 = max(0, min(radius_cols - 1, x0))
    
    # Define radius band in X around x0
    half = cfg.get('polar_band_halfwidth_px', 10)
    band_x_lo = max(0, x0 - half)
    band_x_hi = min(radius_cols - 1, x0 + half)
    
    # Debug logging
    print(f"DEBUG ridge_tracking: P.shape={P.shape}, r_max={r_max:.2f}, r_pass1_peak_px={r_pass1_peak_px:.2f}")
    print(f"DEBUG ridge_tracking: x0={x0}, band_x_lo={band_x_lo}, band_x_hi={band_x_hi}")
    
    # Start at top row (y=0, theta=0)
    start_y = cfg.get('polar_ridge_start_row', 0)
    ridge_points = []
    ridge_mask = np.zeros((theta_rows, radius_cols), dtype=np.uint8)
    
    # First row: find strongest point in radius band [band_x_lo, band_x_hi]
    x = band_x_lo + int(np.argmax(P[start_y, band_x_lo:band_x_hi+1]))
    x = max(band_x_lo, min(band_x_hi, x))  # Clamp to band
    
    ridge_points.append((start_y, x))
    ridge_mask[start_y, x] = 255
    
    # Track downward in theta (Y-axis)
    for y in range(start_y + 1, theta_rows):
        x_prev = x
        # Candidates: three pixels immediately below previous pixel (x-1, x, x+1)
        candidates = [x_prev - 1, x_prev, x_prev + 1]
        # Clamp candidates to radius band
        candidates = [max(band_x_lo, min(band_x_hi, c)) for c in candidates]
        # Remove duplicates
        candidates = list(dict.fromkeys(candidates))
        
        # Find strongest candidate in this row
        candidate_values = [P[y, c] for c in candidates]
        best_idx = np.argmax(candidate_values)
        x = candidates[best_idx]
        
        ridge_points.append((y, x))
        ridge_mask[y, x] = 255
    
    return ridge_mask, ridge_points


def visualize_polar_outputs(P_raw, P_nms, H, r_peak_idx, ridge_mask, out_dir, prefix="polar"):
    """
    Visualize and save polar transform outputs.
    
    Args:
        P_raw: Raw polar magnitude (float32)
        P_nms: NMS polar magnitude (float32, optional, can be None)
        H: Radial profile (1D array)
        r_peak_idx: Selected peak index (for profile visualization)
        ridge_mask: Ridge tracking mask (uint8, same shape as P_raw)
        out_dir: Output directory
        prefix: Filename prefix
    """
    # 1) Polar raw magnitude (normalized to 0..255)
    P_vis = (P_raw / (np.max(P_raw) + 1e-6) * 255).astype(np.uint8)
    debug_tools.save_visualization(P_vis, "01_Polar_mag_raw", None, out_dir)
    
    # 2) Radial profile with peak marker
    # Create a simple plot using cv2
    profile_height = 300
    profile_width = len(H)
    if profile_width > 800:
        # Downsample for display
        step = profile_width // 800
        H_display = H[::step]
        r_peak_display = r_peak_idx // step
        profile_width = len(H_display)
    else:
        H_display = H
        r_peak_display = r_peak_idx
    
    profile_img = np.zeros((profile_height, profile_width), dtype=np.uint8)
    
    # Normalize H for display
    H_max = np.max(H_display) if np.max(H_display) > 0 else 1.0
    H_norm = (H_display / H_max * (profile_height - 20)).astype(int)
    
    # Draw profile as polyline
    points = np.column_stack([np.arange(profile_width), profile_height - 10 - H_norm])
    cv2.polylines(profile_img, [points], isClosed=False, color=255, thickness=1)
    
    # Mark peak
    if 0 <= r_peak_display < profile_width:
        peak_y = profile_height - 10 - H_norm[r_peak_display]
        cv2.circle(profile_img, (r_peak_display, peak_y), 5, 255, 2)
        cv2.line(profile_img, (r_peak_display, 0), (r_peak_display, profile_height), 128, 1)
    
    debug_tools.save_visualization(profile_img, "02_Polar_radial_profile_with_peak", None, out_dir)
    
    # 3) Polar ridge overlay (green on normalized P) - replaces hysteresis overlay
    P_overlay = cv2.cvtColor(P_vis, cv2.COLOR_GRAY2BGR)
    P_overlay[ridge_mask > 0] = (0, 255, 0)  # Green (BGR)
    debug_tools.save_visualization(P_overlay, "03_Polar_hysterese_overlay_green", None, out_dir)


def run(img_bgr, center_xy, out_dir, cfg, filename="unknown"):
    """
    Run polar transform + hysteresis ring detection prototype.
    
    Args:
        img_bgr: Input image (BGR)
        center_xy: Center point (cx, cy) in downscaled coordinates
        out_dir: Output directory for visualizations
        cfg: Config dictionary
        filename: Filename for logging
    """
    start_total = time.perf_counter()
    
    # 1) Downscale and convert to gray
    img_down, scale = preprocess.downscale_max_side(img_bgr, cfg.get('outer_circle_max_side', 1200))
    gray = preprocess.to_gray(img_down)
    h, w = gray.shape
    
    # Use provided center or fallback to image center
    if center_xy is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = float(center_xy[0]), float(center_xy[1])
    
    # 2) Compute mag and NMS
    start_mag = time.perf_counter()
    mag_raw, mag_nms = compute_mag_and_nms(gray, cfg)
    time_mag_ms = (time.perf_counter() - start_mag) * 1000.0
    
    # 3) Determine r_max
    r_max = None
    # Use r_pass1_peak_px if available (same radius as used in histogram)
    if 'r_pass1_peak_px' in cfg and cfg.get('r_pass1_peak_px') is not None:
        r_pass1 = cfg.get('r_pass1_peak_px')
        r_max = cfg.get('polar_r_max_frac', 1.10) * r_pass1
        print(f"DEBUG: Using r_max from r_pass1_peak_px: {r_max:.2f} (r_pass1={r_pass1:.2f})")
    # Fallback: try accepted peaks
    elif 'accepted_peaks' in cfg and cfg.get('accepted_peaks'):
        outermost_peak = max(cfg['accepted_peaks'], key=lambda p: p.get('r_peak', 0))
        r_outer = outermost_peak.get('r_peak', 0)
        if r_outer > 0:
            r_max = cfg.get('polar_r_max_frac', 1.10) * r_outer
            print(f"DEBUG: Using r_max from accepted_peaks: {r_max:.2f} (r_outer={r_outer:.2f})")
    
    # Fallback: use image dimensions
    if r_max is None or r_max <= 0:
        r_max = cfg.get('polar_r_max_frac', 1.10) * min(h, w) * 0.5
        print(f"DEBUG: Using r_max from image dimensions: {r_max:.2f}")
    
    # 4) Warp to polar
    theta_samples = cfg.get('polar_theta_samples', 720)
    r_samples = cfg.get('polar_r_samples', 600)
    
    start_warp = time.perf_counter()
    P_raw = warp_polar(mag_raw.astype(np.float32), (cx, cy), r_max, theta_samples, r_samples)
    # cv2.warpPolar returns shape (r_samples, theta_samples) = (rows, cols)
    # But in our visualization, ringene er VERTIKALE => radius varierer langs X (kolonner), theta langs Y (rader)
    # So we interpret: P.shape = (theta_rows, radius_cols) = (høyde, bredde)
    theta_rows, radius_cols = P_raw.shape
    time_warp_ms = (time.perf_counter() - start_warp) * 1000.0
    
    # 5) Radial profile (for visualization only, not used for ridge tracking)
    start_profile = time.perf_counter()
    H, is_r_first = radial_profile_from_polar(P_raw, agg='sum', smooth_sigma=3.0)
    r_peak_idx = pick_outer_peak(H, ignore_right_margin_bins=10)
    time_profile_ms = (time.perf_counter() - start_profile) * 1000.0
    
    # 6) Ridge tracking (replaces NMS + hysteresis)
    start_ridge = time.perf_counter()
    r_pass1_peak_px = cfg.get('r_pass1_peak_px')
    if r_pass1_peak_px is None or r_pass1_peak_px <= 0:
        # Fallback: use peak from radial profile
        r_pass1_peak_px = (r_peak_idx / len(H)) * r_max
        print(f"WARNING: r_pass1_peak_px not found in cfg, using fallback: {r_pass1_peak_px:.2f}")
    else:
        print(f"DEBUG: Using r_pass1_peak_px from cfg: {r_pass1_peak_px:.2f}, r_max={r_max:.2f}")
    
    ridge_mask, ridge_points = ridge_tracking_polar(P_raw, r_pass1_peak_px, r_max, cfg)
    time_ridge_ms = (time.perf_counter() - start_ridge) * 1000.0
    
    # 7) Visualize
    visualize_polar_outputs(P_raw, None, H, r_peak_idx, ridge_mask, out_dir)
    
    # 8) Log timing
    time_total_ms = (time.perf_counter() - start_total) * 1000.0
    log_line = f"polar_ridge: mag+nms={time_mag_ms:.2f}ms warp={time_warp_ms:.2f}ms H+peak={time_profile_ms:.2f}ms ridge={time_ridge_ms:.2f}ms total={time_total_ms:.2f}ms"
    debug_tools.log_operation_time(filename, "polar_ridge_debug", "total", time_total_ms / 1000.0)
    
    # Also write detailed log
    log_file = cfg.get('log_file_kronologisk', 'ytelse_kronologisk.txt')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{log_line}\n")
    
    print(f"Polar ridge tracking debug: {log_line}")

