"""
Center refinement using radial variance minimization.
"""

import numpy as np


def refine_center_radial_variance(center0_xy, points_xy, cfg):
    """
    Refine center using hillclimbing to minimize radial variance.
    
    Args:
        center0_xy: Start center (cx, cy)
        points_xy: Dict with 'x', 'y' (numpy arrays)
        cfg: Config dictionary
    
    Returns:
        center_xy: (cx, cy) refined center
        best_variance: best variance achieved
        steps_taken: number of steps taken
    """
    x = points_xy['x']
    y = points_xy['y']
    
    if len(x) == 0:
        return center0_xy, float('inf'), 0
    
    max_steps = cfg.get('outermost_ring_refine_max_steps', 60)
    step_px = cfg.get('outermost_ring_refine_step_px', 1)
    max_radius_px = cfg.get('outermost_ring_refine_max_radius_px', 12)
    tiny_eps = 1e-12
    
    # Get image dimensions from points
    W = int(np.max(x)) + 1
    H = int(np.max(y)) + 1
    
    # Start from rounded pixel center
    c = np.array([round(center0_xy[0]), round(center0_xy[1])], dtype=np.float32)
    c_start_rounded = c.copy()
    
    # Compute initial variance
    dx = x - c[0]
    dy = y - c[1]
    ri = np.sqrt(dx**2 + dy**2)
    r_mean = np.mean(ri)
    best_variance = np.mean((ri - r_mean)**2)
    
    # 8-neighborhood offsets
    offsets = [(step_px, 0), (-step_px, 0), (0, step_px), (0, -step_px),
               (step_px, step_px), (-step_px, -step_px), (step_px, -step_px), (-step_px, step_px)]
    
    steps_taken = 0
    for step in range(max_steps):
        best_neighbor_variance = best_variance
        best_neighbor = None
        
        for dx_offset, dy_offset in offsets:
            c2 = c + np.array([dx_offset, dy_offset])
            
            # Check bounds
            if c2[0] < 0 or c2[0] >= W or c2[1] < 0 or c2[1] >= H:
                continue
            
            # Check max radius constraint
            dist_from_start = np.sqrt((c2[0] - c_start_rounded[0])**2 + (c2[1] - c_start_rounded[1])**2)
            if dist_from_start > max_radius_px:
                continue
            
            # Compute variance
            dx2 = x - c2[0]
            dy2 = y - c2[1]
            ri2 = np.sqrt(dx2**2 + dy2**2)
            r_mean2 = np.mean(ri2)
            variance2 = np.mean((ri2 - r_mean2)**2)
            
            if variance2 < best_neighbor_variance:
                best_neighbor_variance = variance2
                best_neighbor = c2.copy()
        
        if best_neighbor is not None and best_neighbor_variance < best_variance - tiny_eps:
            c = best_neighbor
            best_variance = best_neighbor_variance
            steps_taken += 1
        else:
            break  # Local minimum reached
    
    return tuple(c), best_variance, steps_taken

