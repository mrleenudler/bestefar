"""
Center voting using intersection-of-normals method.
"""

import numpy as np


def intersection_vote(points_xy, uxuy, cfg, image_shape=None, return_accumulator=True):
    """
    Vote for center using intersection-of-normals method.
    
    Args:
        points_xy: Dict with 'x', 'y' (numpy arrays)
        uxuy: Dict with 'ux', 'uy' (numpy arrays)
        cfg: Config dictionary
        image_shape: Tuple (H, W) for image dimensions. If None, uses max(x/y) from points (fallback).
        return_accumulator: If False, accumulator is still built but not returned (for performance)
    
    Returns:
        center_xy: (cx, cy) refined center (float)
        accumulator: 2D accumulator array (float32) or None if return_accumulator=False
        debug: Dict with statistics
    """
    x = points_xy['x']
    y = points_xy['y']
    ux = uxuy['ux']
    uy = uxuy['uy']
    w = points_xy.get('mag', np.ones(len(x), dtype=np.float32))  # Use magnitude as weight if available
    
    # Get image dimensions
    if image_shape is not None:
        H, W = image_shape  # (rows, cols)
    else:
        # Fallback: use max from points (may cause cropping issues)
        W = int(np.max(x)) + 1
        H = int(np.max(y)) + 1
    
    accf = np.zeros((H, W), dtype=np.float32)
    accf_flat = accf.ravel()
    
    M = cfg['outer_circle_center_pairs']
    rng = np.random.default_rng(42)
    parallel_eps = cfg['outer_circle_parallel_eps']
    max_dist_frac = cfg.get('outer_circle_max_center_distance_frac', None)
    win = cfg['outer_circle_center_win']
    cross_min = cfg.get('cross_min', 0.3)  # Should be set by caller (pass1 or pass2)
    
    n_points = len(x)
    stats = {
        'sampled_pairs': M,
        'parallel_reject': 0,
        'cross_reject': 0,
        'dist_reject': 0,
        'oob_reject': 0,
        'votes_cast': 0,
        'peak_value': 0.0
    }
    
    if n_points < 2:
        if return_accumulator:
            return (W/2.0, H/2.0), accf, stats
        else:
            return (W/2.0, H/2.0), None, stats
    
    # Generate random pairs (i != j)
    i_indices = rng.integers(0, n_points, size=M)
    j_indices = rng.integers(0, n_points, size=M)
    same_mask = i_indices == j_indices
    j_indices[same_mask] = (j_indices[same_mask] + 1) % n_points
    
    # Extract pairs
    x1, y1 = x[i_indices], y[i_indices]
    x2, y2 = x[j_indices], y[j_indices]
    ux1, uy1 = ux[i_indices], uy[i_indices]
    ux2, uy2 = ux[j_indices], uy[j_indices]
    w1, w2 = w[i_indices], w[j_indices]
    
    # Compute cross product
    cross = ux1 * uy2 - uy1 * ux2
    
    # Filter 1: Numerical stability (parallel_eps)
    valid_parallel = np.abs(cross) >= parallel_eps
    stats['parallel_reject'] = np.sum(~valid_parallel)
    
    # Filter 2: Quality filter (cross_min)
    valid_cross = np.abs(cross) >= cross_min
    stats['cross_reject'] = np.sum(valid_parallel & ~valid_cross)
    valid = valid_parallel & valid_cross
    
    # Filter to valid pairs
    x1 = x1[valid]
    y1 = y1[valid]
    x2 = x2[valid]
    y2 = y2[valid]
    ux1 = ux1[valid]
    uy1 = uy1[valid]
    ux2 = ux2[valid]
    uy2 = uy2[valid]
    w1 = w1[valid]
    w2 = w2[valid]
    cross = cross[valid]
    
    # Solve for intersection
    dx = x2 - x1
    dy = y2 - y1
    s = (dx * uy2 - dy * ux2) / cross
    cx = x1 + s * ux1
    cy = y1 + s * uy1
    
    # Filter by distance from image center
    if max_dist_frac is not None:
        img_center_x = W / 2.0
        img_center_y = H / 2.0
        dist_from_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
        max_dist = max_dist_frac * np.hypot(W, H)
        valid_distance = dist_from_center <= max_dist
        stats['dist_reject'] = np.sum(~valid_distance)
        cx = cx[valid_distance]
        cy = cy[valid_distance]
        w1 = w1[valid_distance]
        w2 = w2[valid_distance]
    else:
        valid_distance = np.ones(len(cx), dtype=bool)
    
    # Round to integer coordinates
    cx_int = np.round(cx).astype(int)
    cy_int = np.round(cy).astype(int)
    
    # Filter out-of-bounds
    valid_bounds = (cx_int >= 0) & (cx_int < W) & (cy_int >= 0) & (cy_int < H)
    stats['oob_reject'] = np.sum(~valid_bounds)
    
    cx_int = cx_int[valid_bounds]
    cy_int = cy_int[valid_bounds]
    vote_weights = np.minimum(w1[valid_bounds], w2[valid_bounds])
    
    # Vote
    if len(cx_int) > 0:
        idx = cy_int * W + cx_int
        np.add.at(accf_flat, idx, vote_weights)
        stats['votes_cast'] = len(cx_int)
    
    # Find peak
    peak_idx = np.argmax(accf)
    peak_y = peak_idx // W
    peak_x = peak_idx % W
    stats['peak_value'] = float(accf[peak_y, peak_x])
    
    # Subpixel refinement
    x_min = max(0, peak_x - win)
    x_max = min(W, peak_x + win + 1)
    y_min = max(0, peak_y - win)
    y_max = min(H, peak_y + win + 1)
    
    window = accf[y_min:y_max, x_min:x_max]
    y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
    
    if window.sum() > 0:
        # Weighted centroid
        cx_refined = np.sum(x_grid * window) / window.sum()
        cy_refined = np.sum(y_grid * window) / window.sum()
    else:
        cx_refined = float(peak_x)
        cy_refined = float(peak_y)
    
    if return_accumulator:
        return (cx_refined, cy_refined), accf, stats
    else:
        return (cx_refined, cy_refined), None, stats

