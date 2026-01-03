"""
Histogram building, smoothing, peak detection, and ring selection.
"""

import numpy as np


def build_radius_histogram(points_xy, center_xy, mag, uxuy, cfg, *, abs_align=True):
    """
    Build weighted radius histogram from points.
    
    Args:
        points_xy: Dict with 'x', 'y' (numpy arrays)
        center_xy: (cx, cy) center coordinates
        mag: Magnitude array (for weighting)
        uxuy: Dict with 'ux', 'uy' (for alignment)
        cfg: Config dictionary
        abs_align: If True, use absolute alignment (default True)
    
    Returns:
        hist: Histogram array
        bin_edges: Bin edge array
    """
    x = points_xy['x']
    y = points_xy['y']
    cx, cy = center_xy
    ux = uxuy['ux']
    uy = uxuy['uy']
    
    # Compute radii
    dx = x - cx
    dy = y - cy
    ri = np.sqrt(dx**2 + dy**2)
    
    # Radial alignment
    eps = 1e-6
    ri_safe = ri + eps
    vx = dx / ri_safe
    vy = dy / ri_safe
    
    if abs_align:
        ai = np.abs(vx * ux + vy * uy)
    else:
        ai = vx * ux + vy * uy
    
    # Filter by alignment
    align_min = cfg['outer_circle_align_min']
    keep_mask = ai >= align_min
    
    x_aligned = x[keep_mask]
    y_aligned = y[keep_mask]
    ri_aligned = ri[keep_mask]
    w_aligned = mag[keep_mask]
    ai_aligned = ai[keep_mask]
    
    # Build histogram
    new_w = int(np.max(x)) + 1
    new_h = int(np.max(y)) + 1
    rmax_search = 0.6 * min(new_h, new_w)
    bin_width = cfg['outer_circle_r_bin_px']
    bins = np.arange(0, rmax_search + bin_width, bin_width)
    
    weights = w_aligned * ai_aligned
    hist, bin_edges = np.histogram(ri_aligned, bins=bins, weights=weights)
    
    return hist, bin_edges


def smooth_hist(hist, sigma):
    """
    Smooth histogram using Gaussian filter.
    
    Args:
        hist: Histogram array
        sigma: Smoothing sigma
    
    Returns:
        hist_smooth: Smoothed histogram
    """
    try:
        from scipy import ndimage
        hist_smooth = ndimage.gaussian_filter1d(hist.astype(np.float32), sigma)
    except ImportError:
        # Fallback: simple box filter
        kernel_size = int(sigma * 2) * 2 + 1
        kernel = np.ones(kernel_size) / kernel_size
        hist_smooth = np.convolve(hist.astype(np.float32), kernel, mode='same')
    
    return hist_smooth


def find_peaks_1d(hist_s, cfg):
    """
    Find peaks in 1D histogram.
    
    Args:
        hist_s: Smoothed histogram
        cfg: Config dictionary
    
    Returns:
        peak_indices: Array of peak indices
    """
    peak_min = cfg['outer_circle_peak_score_min_frac'] * np.max(hist_s)
    peaks_keep = cfg['outer_circle_peaks_keep']
    
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist_s, height=peak_min)
    except ImportError:
        # Fallback: simple local maxima detection
        peaks = []
        for i in range(1, len(hist_s) - 1):
            if hist_s[i] > hist_s[i-1] and hist_s[i] > hist_s[i+1]:
                if hist_s[i] >= peak_min:
                    peaks.append(i)
        peaks = np.array(peaks)
    
    if len(peaks) == 0:
        # Fallback: use maximum
        peaks = np.array([np.argmax(hist_s)])
    
    # Sort by radius (largest first) and keep top N
    # Note: We need bin_edges for radius, but caller will handle that
    return peaks


def cluster_peaks(peak_indices, bin_edges, hist_s, cfg):
    """
    Cluster nearby peaks (double edges).
    
    Args:
        peak_indices: Array of peak indices
        bin_edges: Bin edges array
        hist_s: Smoothed histogram
        cfg: Config dictionary
    
    Returns:
        clustered_peaks: List of peak indices (after clustering)
        clustered_r: List of peak radii
    """
    r_peaks = (bin_edges[peak_indices] + bin_edges[peak_indices + 1]) / 2
    sorted_indices = np.argsort(r_peaks)[::-1]  # Largest first
    sorted_peaks = peak_indices[sorted_indices]
    sorted_r_peaks = r_peaks[sorted_indices]
    
    cluster_px = cfg['outer_circle_peaks_cluster_px']
    clustered_peaks = []
    clustered_r = []
    used = np.zeros(len(sorted_peaks), dtype=bool)
    
    for i in range(len(sorted_peaks)):
        if used[i]:
            continue
        cluster = [i]
        cluster_r = [sorted_r_peaks[i]]
        for j in range(i + 1, len(sorted_peaks)):
            if not used[j] and abs(sorted_r_peaks[i] - sorted_r_peaks[j]) < cluster_px:
                cluster.append(j)
                cluster_r.append(sorted_r_peaks[j])
                used[j] = True
        
        # Choose peak with highest value, or weighted average
        if len(cluster) == 1:
            clustered_peaks.append(sorted_peaks[i])
            clustered_r.append(sorted_r_peaks[i])
        else:
            # Use weighted average of radius
            cluster_vals = hist_s[sorted_peaks[cluster]]
            total_weight = np.sum(cluster_vals)
            if total_weight > 0:
                r_weighted = np.sum(np.array(cluster_r) * cluster_vals) / total_weight
                # Find closest bin
                best_idx = cluster[np.argmax(cluster_vals)]
                clustered_peaks.append(sorted_peaks[best_idx])
                clustered_r.append(r_weighted)
            else:
                clustered_peaks.append(sorted_peaks[i])
                clustered_r.append(sorted_r_peaks[i])
    
    return clustered_peaks, clustered_r


def peak_fwhm_band(hist_s, peak_idx, bin_edges):
    """
    Compute FWHM band for a peak.
    
    Args:
        hist_s: Smoothed histogram
        peak_idx: Peak index
        bin_edges: Bin edges array
    
    Returns:
        r_lo: Lower radius bound
        r_hi: Upper radius bound
        valid: True if FWHM width is sufficient
    """
    peak_val = hist_s[peak_idx]
    half = 0.5 * peak_val
    
    # Find FWHM: go left and right
    left = peak_idx
    while left > 0 and hist_s[left] >= half:
        left -= 1
    left += 1  # First bin where hist < half
    
    right = peak_idx
    while right < len(hist_s) - 1 and hist_s[right] >= half:
        right += 1
    right -= 1  # Last bin where hist >= half
    
    r_lo = bin_edges[left]
    r_hi = bin_edges[right + 1]
    
    return r_lo, r_hi, (right - left + 1)


def angular_coverage(points_xy, center_xy, mask):
    """
    Compute angular coverage for points in mask.
    
    Args:
        points_xy: Dict with 'x', 'y'
        center_xy: (cx, cy) center
        mask: Boolean mask for points to include
    
    Returns:
        coverage: Coverage fraction (0-1)
    """
    x = points_xy['x'][mask]
    y = points_xy['y'][mask]
    cx, cy = center_xy
    
    if len(x) < 10:
        return 0.0
    
    theta = np.arctan2(y - cy, x - cx)
    cov_bins = 120  # Default, should be from cfg but keeping simple
    theta_bins = np.floor((theta + np.pi) / (2 * np.pi) * cov_bins).astype(int)
    theta_bins = np.clip(theta_bins, 0, cov_bins - 1)
    unique_bins = len(np.unique(theta_bins))
    coverage = unique_bins / cov_bins
    
    return coverage


def select_accepted_peaks(points_xy, center_xy, hist_s, bin_edges, cfg):
    """
    Select accepted peaks based on FWHM and coverage.
    
    Args:
        points_xy: Dict with 'x', 'y'
        center_xy: (cx, cy) center
        hist_s: Smoothed histogram
        bin_edges: Bin edges array
        cfg: Config dictionary
    
    Returns:
        peak_info: List of dicts with 'r_peak', 'r_lo', 'r_hi', 'coverage'
    """
    # Find and cluster peaks
    peaks = find_peaks_1d(hist_s, cfg)
    clustered_peaks, clustered_r = cluster_peaks(peaks, bin_edges, hist_s, cfg)
    
    fwhm_min_bins = cfg['outer_circle_fwhm_min_bins']
    cov_min_frac = cfg['outer_circle_cov_min_frac']
    
    accepted_peaks = []
    peak_info = []
    
    x = points_xy['x']
    y = points_xy['y']
    cx, cy = center_xy
    
    # Compute radii for all points
    dx = x - cx
    dy = y - cy
    ri = np.sqrt(dx**2 + dy**2)
    
    for peak_idx, r_peak in zip(clustered_peaks, clustered_r):
        r_lo, r_hi, fwhm_width = peak_fwhm_band(hist_s, peak_idx, bin_edges)
        
        # Check FWHM width
        if fwhm_width < fwhm_min_bins:
            continue
        
        # Compute coverage
        mask = (ri >= r_lo) & (ri <= r_hi)
        if np.sum(mask) < 10:
            continue
        
        coverage = angular_coverage(points_xy, center_xy, mask)
        
        if coverage >= cov_min_frac:
            accepted_peaks.append(peak_idx)
            peak_info.append({
                'idx': peak_idx,
                'r_peak': r_peak,
                'r_lo': r_lo,
                'r_hi': r_hi,
                'coverage': coverage
            })
    
    return peak_info

