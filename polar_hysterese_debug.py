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


def subpixel_peak_parabola(v_m: float, v_0: float, v_p: float) -> float:
    """
    Returner subpixel offset delta i [-0.5, 0.5] fra en 3-punkts paraboltilpasning.
    v_m = P[y, x-1], v_0 = P[y, x], v_p = P[y, x+1]
    """
    denom = 2.0 * (v_m - 2.0 * v_0 + v_p)
    if abs(denom) < 1e-12:
        return 0.0
    delta = (v_m - v_p) / denom
    # clamp for stabilitet
    if not np.isfinite(delta) or abs(delta) > 0.5:
        return 0.0
    return float(delta)


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


def ridge_tracking_polar_viterbi(P, r_pass1_peak_px, r_max, cfg):
    """
    Ridge tracking with Viterbi/DP for globally optimal smooth path.
    Finds path that maximizes magnitude while penalizing jumps and curvature.

    Args:
        P: Polar magnitude image (float32), shape (theta_rows, radius_cols)
        r_pass1_peak_px: Pass 1 peak radius in pixels (downscaled coordinates)
        r_max: Maximum radius used in warpPolar
        cfg: Config dictionary

    Returns:
        ridge_mask: Binary mask (uint8) with ridge points set to 255
        ridge_points_int: List of (y, x_int) tuples
        ridge_points_float: List of (y, x_float) tuples
        xs_float: Array of subpixel x positions
        start_y: Starting row
    """
    theta_rows, radius_cols = P.shape

    # Convert r_pass1_peak_px to radius column index
    x0 = int(round((r_pass1_peak_px / r_max) * (radius_cols - 1)))
    x0 = max(0, min(radius_cols - 1, x0))

    # Define radius band
    half = cfg.get('polar_band_halfwidth_px', 20)
    band_x_lo = max(0, x0 - half)
    band_x_hi = min(radius_cols - 1, x0 + half)
    band_width = band_x_hi - band_x_lo + 1

    start_y = cfg.get('polar_ridge_start_row', 0)

    # DP parameters
    lambda_smooth = cfg.get('polar_viterbi_smooth_weight', 1.0)  # Penalty for jumps
    lambda_curve = cfg.get('polar_viterbi_curve_weight', 0.5)    # Penalty for curvature

    # Normalize P in band for numerical stability
    P_band = P[start_y:, band_x_lo:band_x_hi+1]
    P_max = np.max(P_band) if np.max(P_band) > 0 else 1.0
    P_norm = P_band / (P_max + 1e-6)

    n_rows = theta_rows - start_y

    # DP tables: cost[y, x_rel] = minimum cost to reach (y, x_rel)
    # x_rel is relative to band_x_lo (0 to band_width-1)
    cost = np.full((n_rows, band_width), np.inf, dtype=np.float32)
    backtrack = np.zeros((n_rows, band_width), dtype=np.int32)

    # Initialize first row: cost = -magnitude (we minimize cost = maximize magnitude)
    cost[0, :] = -P_norm[0, :]

    # Forward pass: Viterbi DP
    for y in range(1, n_rows):
        for x_curr in range(band_width):
            # Data term: negative magnitude (higher magnitude = lower cost)
            data_cost = -P_norm[y, x_curr]

            # Find best previous state
            min_transition_cost = np.inf
            best_prev = 0

            for x_prev in range(band_width):
                # Transition cost: penalize jumps
                jump = abs(x_curr - x_prev)
                transition_cost = lambda_smooth * (jump ** 2)

                # Curvature cost: penalize deviation from linear motion
                if y >= 2:
                    # Get previous-previous position from backtrack
                    x_prevprev = backtrack[y-1, x_prev]
                    expected_x = 2 * x_prev - x_prevprev  # Linear extrapolation
                    curve = abs(x_curr - expected_x)
                    curvature_cost = lambda_curve * (curve ** 2)
                else:
                    curvature_cost = 0.0

                total_transition = transition_cost + curvature_cost
                total_cost = cost[y-1, x_prev] + total_transition

                if total_cost < min_transition_cost:
                    min_transition_cost = total_cost
                    best_prev = x_prev

            cost[y, x_curr] = data_cost + min_transition_cost
            backtrack[y, x_curr] = best_prev

    # Backward pass: extract optimal path
    xs_int = np.zeros(theta_rows, dtype=np.int32)
    xs_float = np.zeros(theta_rows, dtype=np.float32)

    # Find best ending state (minimum cost in last row)
    x_rel = int(np.argmin(cost[n_rows-1, :]))

    # Trace back
    path_rel = np.zeros(n_rows, dtype=np.int32)
    path_rel[n_rows-1] = x_rel

    for y in range(n_rows-2, -1, -1):
        x_rel = backtrack[y+1, x_rel]
        path_rel[y] = x_rel

    # Convert relative positions to absolute and apply subpixel refinement
    ridge_points_int = []
    ridge_points_float = []
    ridge_mask = np.zeros((theta_rows, radius_cols), dtype=np.uint8)
    deltas = []

    for i, y in enumerate(range(start_y, theta_rows)):
        x_rel = path_rel[i]
        x_int = band_x_lo + x_rel

        # Subpixel refinement
        xm = max(band_x_lo, x_int - 1)
        xp = min(band_x_hi, x_int + 1)
        if xm != x_int and xp != x_int:
            v_m = float(P[y, xm])
            v_0 = float(P[y, x_int])
            v_p = float(P[y, xp])
            delta = subpixel_peak_parabola(v_m, v_0, v_p)
            x_float = x_int + delta
            deltas.append(abs(delta))
        else:
            delta = 0.0
            x_float = float(x_int)
            deltas.append(0.0)

        xs_int[y] = x_int
        xs_float[y] = x_float
        ridge_points_int.append((y, x_int))
        ridge_points_float.append((y, x_float))
        ridge_mask[y, x_int] = 255

    # Debug logging
    deltas_arr = np.array(deltas)
    non_zero_count = np.sum(deltas_arr > 1e-6)
    mean_delta = float(np.mean(deltas_arr)) if len(deltas_arr) > 0 else 0.0
    max_delta = float(np.max(deltas_arr)) if len(deltas_arr) > 0 else 0.0

    xs_arr = np.array(xs_int[start_y:])
    hit_lo_edge = np.sum(xs_arr == band_x_lo)
    hit_hi_edge = np.sum(xs_arr == band_x_hi)
    max_jump = np.max(np.abs(np.diff(xs_arr))) if len(xs_arr) > 1 else 0

    print(f"DEBUG ridge_tracking_viterbi: P.shape={P.shape}, r_max={r_max:.2f}, r_pass1_peak_px={r_pass1_peak_px:.2f}")
    print(f"DEBUG ridge_tracking_viterbi: x0={x0}, band=[{band_x_lo}, {band_x_hi}], band_width={band_width}")
    print(f"DEBUG ridge_tracking_viterbi: lambda_smooth={lambda_smooth}, lambda_curve={lambda_curve}")
    print(f"DEBUG ridge_tracking_viterbi: edge hits: lo={hit_lo_edge}, hi={hit_hi_edge}, max_jump={max_jump}px")
    print(f"DEBUG ridge_tracking_viterbi: subpixel: mean(|delta|)={mean_delta:.4f}, max(|delta|)={max_delta:.4f}, non-zero={non_zero_count}/{len(deltas_arr)}")

    return ridge_mask, ridge_points_int, ridge_points_float, xs_float, start_y


# Keep old function as alias for now
def ridge_tracking_polar(P, r_pass1_peak_px, r_max, cfg):
    """Wrapper that calls Viterbi-based tracking."""
    return ridge_tracking_polar_viterbi(P, r_pass1_peak_px, r_max, cfg)


def fit_harmonics_x_of_theta(xs_float, start_y=0):
    """
    Fit harmonics x(θ) = C + B1*sin(θ) + D1*cos(θ) + B2*sin(2θ) + D2*cos(2θ) to subpixel ridge points.
    
    Args:
        xs_float: Array of x_float values (subpixel radius indices), len == theta_rows
        start_y: Starting y index
    
    Returns:
        (C, B1, D1, B2, D2, rmse, A1, A2, x_hat, ys) or None if fit fails
    """
    theta_rows = len(xs_float)
    ys = np.arange(start_y, theta_rows, dtype=np.float64)
    
    # θ over [0, 2π). Use theta_rows (not theta_rows-1) for even periodicity
    theta = 2.0 * np.pi * (ys / float(theta_rows))
    
    x = xs_float[start_y:].astype(np.float64)
    mask = np.isfinite(x)
    
    if np.sum(mask) < 5:
        return None
    
    theta_valid = theta[mask]
    x_valid = x[mask]
    
    # Build design matrix: [1, sin(θ), cos(θ), sin(2θ), cos(2θ)]
    s1 = np.sin(theta_valid)
    c1 = np.cos(theta_valid)
    s2 = np.sin(2.0 * theta_valid)
    c2 = np.cos(2.0 * theta_valid)
    
    X = np.column_stack([np.ones_like(theta_valid), s1, c1, s2, c2])  # N x 5
    
    try:
        beta = np.linalg.lstsq(X, x_valid, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    
    C, B1, D1, B2, D2 = [float(v) for v in beta]
    
    # Predictions for all valid points
    x_hat_valid = X @ beta
    residuals = x_valid - x_hat_valid
    rmse = float(np.sqrt(np.mean(residuals**2)))
    
    # Amplitudes
    A1 = float(np.sqrt(B1*B1 + D1*D1))
    A2 = float(np.sqrt(B2*B2 + D2*D2))
    
    # Predictions for all y values (for visualization)
    theta_full = 2.0 * np.pi * (ys / float(theta_rows))
    s1_full = np.sin(theta_full)
    c1_full = np.cos(theta_full)
    s2_full = np.sin(2.0 * theta_full)
    c2_full = np.cos(2.0 * theta_full)
    x_hat_full = C + B1*s1_full + D1*c1_full + B2*s2_full + D2*c2_full
    
    return (C, B1, D1, B2, D2, rmse, A1, A2, x_hat_full, ys)


def visualize_polar_outputs(P_raw, P_nms, H, r_peak_idx, ridge_mask, out_dir, prefix="polar", 
                             harm_params=None, start_y=0):
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
    
    # 3) Polar ridge overlay (green on normalized P) + sinus-fit (red)
    P_overlay = cv2.cvtColor(P_vis, cv2.COLOR_GRAY2BGR)
    
    # Draw green ridge first
    P_overlay[ridge_mask > 0] = (0, 255, 0)  # Green (BGR)
    
    # Draw red harmonic fit (k=1+k=2) if available
    if harm_params is not None:
        theta_rows, radius_cols = P_raw.shape
        x_hat = harm_params[8]  # x_hat_full
        ys = harm_params[9]     # ys
        
        for i, y in enumerate(ys):
            y_int = int(y)
            if 0 <= y_int < theta_rows:
                x_red = int(round(x_hat[i]))
                x_red = max(0, min(radius_cols - 1, x_red))
                P_overlay[y_int, x_red] = (0, 0, 255)  # Red (BGR)
    
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

    Returns:
        center_corrected: Corrected center (cx, cy) in downscaled coordinates after harmonic fitting
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
    
    ridge_mask, ridge_points_int, ridge_points_float, xs_float, start_y = ridge_tracking_polar(P_raw, r_pass1_peak_px, r_max, cfg)
    time_ridge_ms = (time.perf_counter() - start_ridge) * 1000.0
    
    # 6b) Fit harmonics (k=1+k=2) to subpixel ridge points
    harm_result = fit_harmonics_x_of_theta(xs_float, start_y)
    
    if harm_result is None:
        print("WARNING: Harmonic fit failed")
        harm_params = None
        center_corrected = (cx, cy)
    else:
        C, B1, D1, B2, D2, rmse, A1, A2, x_hat_full, ys = harm_result
        
        # Convert (B1, D1) to (dx, dy) in downscaled pixels
        scale = r_max / (radius_cols - 1)
        dx_px = D1 * scale
        dy_px = B1 * scale
        
        print(f"HARM FIT: rmse={rmse:.3f}, A1(k1)={A1:.3f}, A2(k2)={A2:.3f}, dx={dx_px:.2f}, dy={dy_px:.2f}")
        
        # Test both + and - corrections
        if A1 > 0.01:  # Only test if k=1 component is significant
            cand_plus = (cx + dx_px, cy + dy_px)
            cand_minus = (cx - dx_px, cy - dy_px)
            
            # Test + correction
            P_plus = warp_polar(mag_raw.astype(np.float32), cand_plus, r_max, theta_samples, r_samples)
            _, _, _, xs_float_plus, _ = ridge_tracking_polar(P_plus, r_pass1_peak_px, r_max, cfg)
            harm_plus = fit_harmonics_x_of_theta(xs_float_plus, start_y)
            A1_plus = harm_plus[6] if harm_plus is not None else float('inf')
            
            # Test - correction
            P_minus = warp_polar(mag_raw.astype(np.float32), cand_minus, r_max, theta_samples, r_samples)
            _, _, _, xs_float_minus, _ = ridge_tracking_polar(P_minus, r_pass1_peak_px, r_max, cfg)
            harm_minus = fit_harmonics_x_of_theta(xs_float_minus, start_y)
            A1_minus = harm_minus[6] if harm_minus is not None else float('inf')
            
            # Choose candidate with lowest A1
            if A1_plus < A1_minus and A1_plus < A1:
                center_corrected = cand_plus
                print(f"CENTER CORRECTION: chosen +, A1_original={A1:.3f}, A1_plus={A1_plus:.3f}, A1_minus={A1_minus:.3f}")
                harm_result = harm_plus
            elif A1_minus < A1:
                center_corrected = cand_minus
                print(f"CENTER CORRECTION: chosen -, A1_original={A1:.3f}, A1_plus={A1_plus:.3f}, A1_minus={A1_minus:.3f}")
                harm_result = harm_minus
            else:
                center_corrected = (cx, cy)
                print(f"CENTER CORRECTION: no improvement, keeping original, A1_original={A1:.3f}, A1_plus={A1_plus:.3f}, A1_minus={A1_minus:.3f}")
        else:
            center_corrected = (cx, cy)
        
        # Final refit with chosen center for visualization
        if center_corrected != (cx, cy):
            P_final = warp_polar(mag_raw.astype(np.float32), center_corrected, r_max, theta_samples, r_samples)
            ridge_mask, _, _, xs_float, start_y = ridge_tracking_polar(P_final, r_pass1_peak_px, r_max, cfg)
            harm_result = fit_harmonics_x_of_theta(xs_float, start_y)
            if harm_result is not None:
                C, B1, D1, B2, D2, rmse, A1, A2, x_hat_full, ys = harm_result
                print(f"HARM FIT (after correction): rmse={rmse:.3f}, A1(k1)={A1:.3f}, A2(k2)={A2:.3f}")
                # Update P_raw for visualization
                P_raw = P_final
        
        harm_params = harm_result
    
    # 7) Visualize
    visualize_polar_outputs(P_raw, None, H, r_peak_idx, ridge_mask, out_dir, 
                             harm_params=harm_params, start_y=start_y)
    
    # 8) Log timing
    time_total_ms = (time.perf_counter() - start_total) * 1000.0
    log_line = f"polar_ridge: mag+nms={time_mag_ms:.2f}ms warp={time_warp_ms:.2f}ms H+peak={time_profile_ms:.2f}ms ridge={time_ridge_ms:.2f}ms total={time_total_ms:.2f}ms"
    debug_tools.log_operation_time(filename, "polar_ridge_debug", "total", time_total_ms / 1000.0)

    # Also write detailed log
    log_file = cfg.get('log_file_kronologisk', 'ytelse_kronologisk.txt')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{log_line}\n")

    print(f"Polar ridge tracking debug: {log_line}")

    # Return corrected center (in downscaled coordinates)
    return center_corrected

