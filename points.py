"""
Point extraction and filtering functions.
"""

import numpy as np


def extract_edge_points(mag, ux, uy, cfg, mode="pass1"):
    """
    Extract edge points from gradient magnitude.
    
    Args:
        mag: Gradient magnitude array
        ux, uy: Normalized gradient components
        cfg: Config dictionary
        mode: "pass1" or "pass2" (for different percentile thresholds if needed)
    
    Returns:
        points: Dict with 'x', 'y', 'ux', 'uy', 'mag' (numpy arrays)
    """
    mag_floor = cfg['outer_circle_mag_floor']
    mag_nonzero = mag[mag > mag_floor]
    if len(mag_nonzero) == 0:
        return None
    
    percentile = cfg['outer_circle_mag_percentile']
    threshold = np.percentile(mag_nonzero, percentile)
    edge_mask = mag >= threshold
    
    # Extract edge points
    y_coords, x_coords = np.where(edge_mask)
    x = x_coords.astype(np.float32)
    y = y_coords.astype(np.float32)
    ux_vals = ux[y_coords, x_coords]
    uy_vals = uy[y_coords, x_coords]
    mag_vals = mag[y_coords, x_coords]
    
    # Get image dimensions
    h, w = mag.shape
    
    # Border margin filter
    margin = cfg['outer_circle_border_margin_frac'] * min(w, h)
    keep = (x >= margin) & (x < w - margin) & (y >= margin) & (y < h - margin)
    x = x[keep]
    y = y[keep]
    ux_vals = ux_vals[keep]
    uy_vals = uy_vals[keep]
    mag_vals = mag_vals[keep]
    
    # Random subsample if too many
    max_points = cfg['outer_circle_max_edge_points']
    if len(x) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(x), max_points, replace=False)
        x = x[indices]
        y = y[indices]
        ux_vals = ux_vals[indices]
        uy_vals = uy_vals[indices]
        mag_vals = mag_vals[indices]
    
    return {
        'x': x,
        'y': y,
        'ux': ux_vals,
        'uy': uy_vals,
        'mag': mag_vals
    }


def filter_points_by_radius_bands(points, center_xy, bands, cfg):
    """
    Filter points by radius bands.
    
    Args:
        points: Dict with 'x', 'y'
        center_xy: (cx, cy) center
        bands: List of dicts with 'r_lo', 'r_hi'
        cfg: Config dictionary
    
    Returns:
        mask: Boolean mask for points in bands
    """
    x = points['x']
    y = points['y']
    cx, cy = center_xy
    
    dx = x - cx
    dy = y - cy
    ri = np.sqrt(dx**2 + dy**2)
    
    # Find outermost ring
    if len(bands) == 0:
        return np.zeros(len(x), dtype=bool)
    
    r_outer = max(b['r_peak'] for b in bands)
    outer_cut_eps = cfg['outer_circle_pass2_outer_cut_eps']
    
    # Build mask: points within any band AND within outer + eps
    mask = np.zeros(len(x), dtype=bool)
    for band in bands:
        r_lo = band['r_lo']
        r_hi = band['r_hi']
        mask |= (ri >= r_lo) & (ri <= r_hi)
    
    # Also cut points outside outermost ring
    mask &= (ri <= r_outer + outer_cut_eps)
    
    return mask


def filter_points_by_alignment(points, center_xy, uxuy, min_align, abs_align=True):
    """
    Filter points by radial alignment.
    
    Args:
        points: Dict with 'x', 'y'
        center_xy: (cx, cy) center
        uxuy: Dict with 'ux', 'uy'
        min_align: Minimum alignment value
        abs_align: If True, use absolute alignment
    
    Returns:
        mask: Boolean mask for aligned points
    """
    x = points['x']
    y = points['y']
    cx, cy = center_xy
    ux = uxuy['ux']
    uy = uxuy['uy']
    
    dx = x - cx
    dy = y - cy
    ri = np.sqrt(dx**2 + dy**2)
    
    eps = 1e-6
    ri_safe = ri + eps
    vx = dx / ri_safe
    vy = dy / ri_safe
    
    if abs_align:
        ai = np.abs(vx * ux + vy * uy)
    else:
        ai = vx * ux + vy * uy
    
    return ai >= min_align

